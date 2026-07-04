from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def sinusoidal_time_embedding_1d(t: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"embed_dim must be even for time embedding, got {dim}")
    t_flat = t.reshape(-1).float()
    half = dim // 2
    device = t_flat.device
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    angles = t_flat[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb.to(dtype=t.dtype)


def init_conv1d_modules(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.orthogonal_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Linear):
            nn.init.orthogonal_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)

class TemporalConditionEncoder(nn.Module):
    """
    Encode a temporal condition sequence ``(B, T, C_cond)`` into a single
    window-level conditioning vector ``(B, cond_dim)``.

    This is meant for signals like cmd: (B, T, 3) or prev_action: (B, T, 29)
    histories whose values can change within a sliding window.
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        width = hidden_dim or cond_dim
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = width

        self.in_proj = nn.Conv1d(input_dim, width, kernel_size=1)
        blocks: list[nn.Module] = []
        padding = kernel_size // 2
        for _ in range(num_layers):
            blocks.append(
                nn.Sequential(
                    nn.GroupNorm(min(8, width), width),
                    nn.SiLU(),
                    nn.Conv1d(width, width, kernel_size=kernel_size, padding=padding),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_proj = nn.Sequential(
            nn.Linear(width, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        init_conv1d_modules(self)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        x = cond.transpose(1, 2)
        # x: (B, C, T)
        x = self.in_proj(x)
        for block in self.blocks:
            x = x + block(x)
        pooled = x.mean(dim=-1) # over time
        return self.out_proj(pooled)

class ConditionalResidualBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.cond_proj = nn.Linear(cond_dim, 2 * out_channels)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None]) + shift[:, :, None]
        h = self.conv2(self.dropout(self.act(h)))
        return h + self.skip(x)


class Downsample1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalUNet1D(nn.Module):
    """
    Conditional 1D U-Net for sequence generation.

    Inputs and outputs use shape ``(B, T, C)`` to match robotics trajectories.
    Conditioning is an optional dense vector ``cond`` of shape ``(B, T, dim)``.
    Optional scalar time ``t`` is embedded through a separate path and added to
    the residual conditioning signal, which makes this suitable for diffusion /
    flow / EqM over trajectories.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        *,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        cond_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.cond_dim = cond_dim
        self._cond_encoders: nn.ModuleDict = nn.ModuleDict()
        self._vector_cond_encoders: nn.ModuleDict = nn.ModuleDict()

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.in_conv = nn.Conv1d(input_dim, base_channels, kernel_size=3, padding=1)

        widths = [base_channels * mult for mult in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur_channels = base_channels
        for idx, width in enumerate(widths):
            self.down_blocks.append(ConditionalResidualBlock1d(cur_channels, width, cond_dim, dropout=dropout))
            cur_channels = width
            if idx < len(widths) - 1:
                self.downsamples.append(Downsample1d(cur_channels))

        self.mid_block1 = ConditionalResidualBlock1d(cur_channels, cur_channels, cond_dim, dropout=dropout)
        self.mid_block2 = ConditionalResidualBlock1d(cur_channels, cur_channels, cond_dim, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for idx in range(len(widths) - 1, -1, -1):
            skip_channels = widths[idx]
            self.up_blocks.append(
                ConditionalResidualBlock1d(cur_channels + skip_channels, skip_channels, cond_dim, dropout=dropout)
            )
            cur_channels = skip_channels
            if idx > 0:
                self.upsamples.append(Upsample1d(cur_channels))

        self.out_norm = nn.GroupNorm(min(8, cur_channels), cur_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(cur_channels, self.output_dim, kernel_size=3, padding=1)

        init_conv1d_modules(self)

    def _get_cond_encoder(self, input_dim: int) -> TemporalConditionEncoder:
        key = str(int(input_dim))
        if key not in self._cond_encoders:
            self._cond_encoders[key] = TemporalConditionEncoder(
                input_dim=input_dim, cond_dim=self.cond_dim
            )
        return cast(TemporalConditionEncoder, self._cond_encoders[key])

    def _get_vector_cond_encoder(self, input_dim: int) -> nn.Sequential:
        key = str(int(input_dim))
        if key not in self._vector_cond_encoders:
            self._vector_cond_encoders[key] = nn.Sequential(
                nn.Linear(input_dim, self.cond_dim),
                nn.SiLU(),
                nn.Linear(self.cond_dim, self.cond_dim),
            )
        return cast(nn.Sequential, self._vector_cond_encoders[key])

    def _build_condition(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        cond: Float[Tensor, "B T dim"] | Float[Tensor, "B dim"] | None,
        t: torch.Tensor | None,
    ) -> torch.Tensor:
        if cond is None:
            out = torch.zeros(batch_size, self.cond_dim, device=device, dtype=dtype)
        else:
            if cond.dim() == 2:
                input_dim = cond.shape[1]
                proj = self._get_vector_cond_encoder(input_dim).to(device=device, dtype=dtype)
                out = proj(cond)
            elif cond.dim() == 3:
                input_dim = cond.shape[2]
                proj = self._get_cond_encoder(input_dim).to(device=device, dtype=dtype)
                out = proj(cond)
            else:
                raise ValueError(f"cond must have shape (B, D) or (B, T, D), got {tuple(cond.shape)}")
        if t is not None:
            if t.dim() != 1 or t.shape[0] != batch_size:
                raise ValueError(f"t must have shape (B,), got {tuple(t.shape)}")
            t_embed = sinusoidal_time_embedding_1d(t.to(device=device, dtype=dtype), self.cond_dim)
            out = out + self.time_mlp(t_embed.to(dtype=dtype))
        return out

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B, T, C), got {tuple(x.shape)}")
        batch_size = x.shape[0]
        device, dtype = x.device, x.dtype
        x = x.transpose(1, 2)
        cond_vec = self._build_condition(batch_size, device, dtype, cond, t) # cond_vec: (B, cond_dim)

        h = self.in_conv(x)
        skips: list[torch.Tensor] = []
        for idx, block in enumerate(self.down_blocks):
            h = block(h, cond_vec)
            skips.append(h)
            if idx < len(self.downsamples):
                h = self.downsamples[idx](h)

        h = self.mid_block1(h, cond_vec)
        h = self.mid_block2(h, cond_vec)

        for idx, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond_vec)
            if idx < len(self.upsamples):
                h = self.upsamples[idx](h)

        return self.out_conv(self.out_act(self.out_norm(h))).transpose(1, 2)
