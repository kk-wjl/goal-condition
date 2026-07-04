from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def init_transformer_modules(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Embedding):
            nn.init.normal_(child.weight, mean=0.0, std=0.02)


class TemporalConditionEncoder(nn.Module):
    """
    Encode a temporal condition sequence ``(B, T, C_cond)`` into a single
    window-level conditioning vector ``(B, cond_dim)``.
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
        self.in_proj = nn.Conv1d(input_dim, width, kernel_size=1)
        padding = kernel_size // 2
        blocks: list[nn.Module] = []
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

        for child in self.modules():
            if isinstance(child, nn.Conv1d):
                nn.init.orthogonal_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(cond.transpose(1, 2))
        for block in self.blocks:
            x = x + block(x)
        return self.out_proj(x.mean(dim=-1))


class MLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(mlp_hidden, hidden_size)
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class DiTBlock1D(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_adaLN_on_self_attn: bool = True,
        use_adaLN_on_mlp: bool = True,
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_adaLN_on_self_attn = use_adaLN_on_self_attn
        self.use_adaLN_on_mlp = use_adaLN_on_mlp
        self.use_cross_attention = use_cross_attention

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = MLP(hidden_size, mlp_ratio=mlp_ratio, dropout=dropout)

        if use_cross_attention:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        else:
            self.norm_cross = None
            self.cross_attn = None

        modulation_chunks = 0
        if use_adaLN_on_self_attn:
            modulation_chunks += 3
        if use_adaLN_on_mlp:
            modulation_chunks += 3
        modulation_dim = modulation_chunks * hidden_size

        if modulation_dim > 0:
            self.modulation_act = nn.SiLU()
            self.modulation_proj = nn.Linear(hidden_size, modulation_dim)
            nn.init.zeros_(self.modulation_proj.weight)
            nn.init.zeros_(self.modulation_proj.bias)
        else:
            self.modulation_act = None
            self.modulation_proj = None


    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cond_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate_attn: torch.Tensor | None = None
        gate_mlp: torch.Tensor | None = None
        if self.modulation_proj is not None and self.modulation_act is not None:
            modulation_parts = list(self.modulation_proj(self.modulation_act(cond)).chunk(self.modulation_proj.out_features // self.hidden_size, dim=-1))
        else:
            modulation_parts = []

        if self.use_adaLN_on_self_attn:
            shift_attn, scale_attn, gate_attn = modulation_parts[:3]
            modulation_parts = modulation_parts[3:]
            h = modulate(self.norm1(x), shift_attn, scale_attn)
        else:
            h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        if self.use_adaLN_on_self_attn:
            x = x + cast(torch.Tensor, gate_attn)[:, None, :] * h
        else:
            x = x + h

        if self.use_cross_attention and cond_tokens is not None:
            norm_cross = cast(nn.LayerNorm, self.norm_cross)
            cross_attn = cast(nn.MultiheadAttention, self.cross_attn)
            h = norm_cross(x)
            h, _ = cross_attn(h, cond_tokens, cond_tokens, need_weights=False)
            x = x + h

        if self.use_adaLN_on_mlp:
            shift_mlp, scale_mlp, gate_mlp = modulation_parts[:3]
            h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        else:
            h = self.norm2(x)
        h = self.mlp(h)
        if self.use_adaLN_on_mlp:
            return x + cast(torch.Tensor, gate_mlp)[:, None, :] * h
        return x + h


class DiffusionTransformer1D(nn.Module):
    """
    DiT-style transformer backbone for sequence generation.

    Inputs and outputs use shape ``(B, T, C)`` to match robotics trajectories.
    Global conditioning can be provided as ``(B, D)`` or ``(B, T, D)``.
    Optional per-token conditioning can be provided through ``cond_tokens``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        *,
        hidden_size: int = 256,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        cond_dim: int | None = None,
        max_seq_len: int = 256,
        dropout: float = 0.0,
        use_adaLN_on_self_attn: bool = True,
        use_adaLN_on_mlp: bool = True,
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size must be divisible by num_heads, got {hidden_size} and {num_heads}")

        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim or hidden_size
        self.max_seq_len = max_seq_len
        self.use_cross_attention = use_cross_attention

        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        self.time_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.global_cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock1D(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_adaLN_on_self_attn=use_adaLN_on_self_attn,
                    use_adaLN_on_mlp=use_adaLN_on_mlp,
                    use_cross_attention=use_cross_attention,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, self.output_dim)

        self._global_vector_encoders: nn.ModuleDict = nn.ModuleDict()
        self._global_sequence_encoders: nn.ModuleDict = nn.ModuleDict()
        self._token_condition_projs: nn.ModuleDict = nn.ModuleDict()

        init_transformer_modules(self)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    # if cond: (B, D)
    def _get_global_vector_encoder(self, input_dim: int) -> nn.Module:
        key = str(int(input_dim))
        if key not in self._global_vector_encoders:
            self._global_vector_encoders[key] = nn.Sequential(
                nn.Linear(input_dim, self.cond_dim),
                nn.SiLU(),
                nn.Linear(self.cond_dim, self.cond_dim),
            )
        return self._global_vector_encoders[key]

    # if cond: (B, T, D)
    def _get_global_sequence_encoder(self, input_dim: int) -> TemporalConditionEncoder:
        key = str(int(input_dim))
        if key not in self._global_sequence_encoders:
            self._global_sequence_encoders[key] = TemporalConditionEncoder(
                input_dim=input_dim,
                cond_dim=self.cond_dim,
            )
        return cast(TemporalConditionEncoder, self._global_sequence_encoders[key])

    # if cond: token
    def _get_token_condition_proj(self, input_dim: int) -> nn.Module:
        key = str(int(input_dim))
        if key not in self._token_condition_projs:
            self._token_condition_projs[key] = nn.Sequential(
                nn.Linear(input_dim, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
            )
        return self._token_condition_projs[key]

    def _build_global_condition(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        cond: Float[Tensor, "B dim"] | Float[Tensor, "B T dim"] | None,
        t: torch.Tensor | None,
    ) -> torch.Tensor:
        if cond is None:
            out = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        elif cond.dim() == 2:
            proj = self._get_global_vector_encoder(cond.shape[-1]).to(device=device, dtype=dtype)
            out = self.global_cond_proj(proj(cond))
        elif cond.dim() == 3:
            proj = self._get_global_sequence_encoder(cond.shape[-1]).to(device=device, dtype=dtype)
            out = self.global_cond_proj(proj(cond))
        else:
            raise ValueError(f"cond must have shape (B, D) or (B, T, D), got {tuple(cond.shape)}")

        if t is not None:
            if t.dim() != 1 or t.shape[0] != batch_size:
                raise ValueError(f"t must have shape (B,), got {tuple(t.shape)}")
            t_embed = sinusoidal_time_embedding_1d(t.to(device=device, dtype=dtype), self.cond_dim)
            out = out + self.time_mlp(t_embed.to(dtype=dtype))
        return out

    def _build_cond_tokens(
        self,
        device: torch.device,
        dtype: torch.dtype,
        cond_tokens: Float[Tensor, "B S dim"] | Float[Tensor, "B dim"] | None,
    ) -> torch.Tensor | None:
        if cond_tokens is None:
            return None
        if cond_tokens.dim() == 2:
            cond_tokens = cond_tokens[:, None, :]
        if cond_tokens.dim() != 3:
            raise ValueError(f"cond_tokens must have shape (B, S, D) or (B, D), got {tuple(cond_tokens.shape)}")
        proj = self._get_token_condition_proj(cond_tokens.shape[-1]).to(device=device, dtype=dtype)
        return proj(cond_tokens)

    def forward(
        self,
        x: Float[Tensor, "B T input_dim"],
        cond: Float[Tensor, "B dim"] | Float[Tensor, "B T dim"] | None = None,
        t: torch.Tensor | None = None,
        cond_tokens: Float[Tensor, "B S dim"] | Float[Tensor, "B dim"] | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must have shape (B, T, C), got {tuple(x.shape)}")
        if x.shape[1] > self.max_seq_len:
            raise ValueError(f"sequence length {x.shape[1]} exceeds max_seq_len={self.max_seq_len}")

        batch_size, seq_len, _ = x.shape
        device, dtype = x.device, x.dtype

        tokens = self.input_proj(x)
        tokens = tokens + self.pos_embed[:, :seq_len, :].to(device=device, dtype=dtype)

        global_cond = self._build_global_condition(batch_size, device, dtype, cond, t)
        token_cond = self._build_cond_tokens(device, dtype, cond_tokens)

        for block in self.blocks:
            tokens = block(tokens, global_cond, token_cond)

        return self.output_proj(self.final_norm(tokens))


__all__ = [
    "DiTBlock1D",
    "DiffusionTransformer1D",
    "TemporalConditionEncoder",
    "sinusoidal_time_embedding_1d",
]
