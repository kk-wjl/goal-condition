"""
Diffusion models for motion generation.
"""

from __future__ import annotations

from jaxtyping import Float
from typing import cast
from torch import Tensor

import torch
import torch.nn as nn


class TrajectoryDiffusion_DDPM(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        *,
        seq_len: int,
        cond_steps: int,
        num_diffusion_steps: int,
        beta_start: float,
        beta_end: float,
    ) -> None:
        super().__init__()
        if cond_steps < 1:
            raise ValueError("cond_steps must be >= 1")
        if num_diffusion_steps < 2:
            raise ValueError("num_diffusion_steps must be >= 2")
        if not (0.0 < beta_start < beta_end < 1.0):
            raise ValueError(
                f"Need 0 < beta_start < beta_end < 1, got {beta_start}, {beta_end}"
            )

        self.network = network
        self.time_conditioning = bool(getattr(network, "time_conditioning", True))
        self.seq_len = seq_len
        self.cond_steps = cond_steps
        self.num_diffusion_steps = num_diffusion_steps

        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt((1.0 - alpha_bars).clamp_min(1e-12)),
        )
        self.register_buffer(
            "sqrt_recip_alphas",
            torch.sqrt((1.0 / alphas).clamp_max(1e12)),
        )
        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars).clamp_min(1e-12)
        posterior_var[0] = 0.0
        posterior_log_var = torch.log(posterior_var.clamp_min(1e-20))
        posterior_mean_coef1 = (
            betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars).clamp_min(1e-12)
        )
        posterior_mean_coef2 = (
            (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars).clamp_min(1e-12)
        )
        self.register_buffer("posterior_variance", posterior_var)
        self.register_buffer("posterior_log_variance", posterior_log_var)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def _time_to_unit_interval(
        self, timesteps: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        denom = max(self.num_diffusion_steps - 1, 1)
        return timesteps.to(dtype=dtype) / float(denom)

    def compute_loss(
        self,
        x0: Float[Tensor, "batch seq dim"],
        cmd: Float[Tensor, "B T dim"] | None = None,
    ) -> Float[Tensor, ""]:
        k = self.cond_steps
        if x0.ndim != 3 or x0.shape[1] != self.seq_len:
            raise ValueError(f"x0 must have shape (B, {self.seq_len}, D), got {tuple(x0.shape)}")
        if x0.shape[1] < k:
            raise ValueError(
                f"sequence length {x0.shape[1]} is shorter than cond_steps={k}"
            )

        batch_size = x0.shape[0]
        device, dtype = x0.device, x0.dtype
        timesteps = torch.randint(
            0,
            self.num_diffusion_steps,
            (batch_size,),
            device=device,
        )
        noise = torch.randn_like(x0)
        sqrt_ab = cast(torch.Tensor, self.sqrt_alpha_bars)[timesteps].to(
            device=device, dtype=dtype
        ).reshape(-1, 1, 1)
        sqrt_omb = cast(torch.Tensor, self.sqrt_one_minus_alpha_bars)[timesteps].to(
            device=device, dtype=dtype
        ).reshape(-1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_omb * noise
        x_t[:, :k] = x0[:, :k]

        t = self._time_to_unit_interval(timesteps, dtype=dtype)
        eps_pred = self.network(x_t, t)
        err = (eps_pred - noise) ** 2
        err[:, :k] = 0.0
        return err[:, k:, :].mean()

    @torch.inference_mode()
    def sample_cond_prefix(
        self,
        cond_prefix: Float[Tensor, "batch cond dim"],
        device: torch.device,
        cmd: Float[Tensor, "B T dim"] | None = None,
        *,
        num_steps: int,
        cfg_scale: float = 1.0,
        initial_noise: Float[Tensor, "batch seq dim"] | None = None,
    ) -> Float[Tensor, "batch seq dim"]:
        if cond_prefix.ndim != 3 or cond_prefix.shape[1] != self.cond_steps:
            raise ValueError(
                f"cond_prefix must have shape (N, {self.cond_steps}, D), got {tuple(cond_prefix.shape)}"
            )
        if num_steps < 1 or num_steps > self.num_diffusion_steps:
            raise ValueError(
                f"num_steps must be in [1, {self.num_diffusion_steps}], got {num_steps}"
            )
        if cfg_scale < 0.0:
            raise ValueError(f"cfg_scale must be >= 0, got {cfg_scale}")

        dtype = next(self.parameters()).dtype
        n, k, feat_dim = cond_prefix.shape
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        if initial_noise is not None:
            expected_shape = (n, self.seq_len, feat_dim)
            if tuple(initial_noise.shape) != expected_shape:
                raise ValueError(
                    f"initial_noise must have shape {expected_shape}, got {tuple(initial_noise.shape)}"
                )
            x_t = initial_noise.to(device=device, dtype=dtype).clone()
        else:
            x_t = torch.randn(n, self.seq_len, feat_dim, device=device, dtype=dtype)
        x_t[:, :k] = cond_prefix

        step_ids = torch.linspace(
            self.num_diffusion_steps - 1,
            0,
            num_steps,
            device=device,
            dtype=torch.float32,
        ).round().to(torch.long)
        step_ids = torch.unique_consecutive(step_ids)

        for t_idx in step_ids:
            t_batch_long = t_idx.expand(n)
            t_batch = self._time_to_unit_interval(t_batch_long, dtype=dtype).to(device=device)
            eps_pred = self.network(x_t, t_batch)
            eps_pred = eps_pred.clone()
            eps_pred[:, :k] = 0.0

            alpha_bar_t = cast(torch.Tensor, self.alpha_bars)[t_idx].to(device=device, dtype=dtype)
            sqrt_alpha_bar_t = cast(torch.Tensor, self.sqrt_alpha_bars)[t_idx].to(
                device=device, dtype=dtype
            )
            sqrt_one_minus_alpha_bar_t = cast(torch.Tensor, self.sqrt_one_minus_alpha_bars)[
                t_idx
            ].to(
                device=device, dtype=dtype
            )
            x0_hat = (
                x_t - sqrt_one_minus_alpha_bar_t.view(1, 1, 1) * eps_pred
            ) / sqrt_alpha_bar_t.view(1, 1, 1).clamp_min(1e-8)
            x0_hat[:, :k] = cond_prefix

            coef1 = cast(torch.Tensor, self.posterior_mean_coef1)[t_idx].to(
                device=device, dtype=dtype
            )
            coef2 = cast(torch.Tensor, self.posterior_mean_coef2)[t_idx].to(
                device=device, dtype=dtype
            )
            mean = coef1.view(1, 1, 1) * x0_hat + coef2.view(1, 1, 1) * x_t

            if int(t_idx.item()) == 0:
                x_t = mean
            else:
                log_var = cast(torch.Tensor, self.posterior_log_variance)[t_idx].to(
                    device=device, dtype=dtype
                )
                noise = torch.randn_like(x_t)
                x_t = mean + torch.exp(0.5 * log_var).view(1, 1, 1) * noise
            x_t[:, :k] = cond_prefix

        return x_t
       

class TrajectoryDiffusion_DDIM(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        *,
        seq_len: int,
        cond_steps: int,
        num_diffusion_steps: int,
        beta_start: float,
        beta_end: float,
    ) -> None:
        super().__init__()
        if cond_steps < 1:
            raise ValueError("cond_steps must be >= 1")
        if num_diffusion_steps < 2:
            raise ValueError("num_diffusion_steps must be >= 2")
        if not (0.0 < beta_start < beta_end < 1.0):
            raise ValueError(
                f"Need 0 < beta_start < beta_end < 1, got {beta_start}, {beta_end}"
            )

        self.network = network
        self.time_conditioning = bool(getattr(network, "time_conditioning", True))
        self.seq_len = seq_len
        self.cond_steps = cond_steps
        self.num_diffusion_steps = num_diffusion_steps

        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt((1.0 - alpha_bars).clamp_min(1e-12)),
        )
        self.register_buffer(
            "sqrt_recip_alphas",
            torch.sqrt((1.0 / alphas).clamp_max(1e12)),
        )
        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars).clamp_min(1e-12)
        posterior_var[0] = 0.0
        self.register_buffer("posterior_variance", posterior_var)

    def _time_to_unit_interval(
        self, timesteps: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        denom = max(self.num_diffusion_steps - 1, 1)
        return timesteps.to(dtype=dtype) / float(denom)

    def compute_loss(self, x0: Float[Tensor, "batch seq dim"], cmd: Float[Tensor, "B T dim"] | None = None) -> Float[Tensor, ""]:
        k = self.cond_steps
        if x0.ndim != 3 or x0.shape[1] != self.seq_len:
            raise ValueError(f"x0 must have shape (B, {self.seq_len}, D), got {tuple(x0.shape)}")
        if x0.shape[1] < k:
            raise ValueError(
                f"sequence length {x0.shape[1]} is shorter than cond_steps={k}"
            )

        batch_size = x0.shape[0]
        device, dtype = x0.device, x0.dtype
        timesteps = torch.randint(
            0,
            self.num_diffusion_steps,
            (batch_size,),
            device=device,
        )
        # x0.shape = (batch_size, T, D)
        noise = torch.randn_like(x0)
        sqrt_ab = cast(torch.Tensor, self.sqrt_alpha_bars)[timesteps].to(
            device=device, dtype=dtype
        ).reshape(-1, 1, 1)
        sqrt_omb = cast(torch.Tensor, self.sqrt_one_minus_alpha_bars)[timesteps].to(
            device=device, dtype=dtype
        ).reshape(-1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_omb * noise
        x_t[:, :k] = x0[:, :k]

        t = self._time_to_unit_interval(timesteps, dtype=dtype)
        eps_pred = self.network(x_t, t)
        err = (eps_pred - noise) ** 2
        err[:, :k] = 0.0
        return err[:, k:, :].mean()

    @torch.inference_mode()
    def sample_cond_prefix(
        self,
        cond_prefix: Float[Tensor, "batch cond dim"],
        device: torch.device,
        cmd: Float[Tensor, "B T dim"] | None = None,
        *,
        num_steps: int,
        eta: float | None = None,
        ddim_eta: float = 0.0,
        cfg_scale: float = 1.0,
        initial_noise: Float[Tensor, "batch seq dim"] | None = None,
    ) -> Float[Tensor, "batch seq dim"]:
        if cond_prefix.ndim != 3 or cond_prefix.shape[1] != self.cond_steps:
            raise ValueError(
                f"cond_prefix must have shape (N, {self.cond_steps}, D), got {tuple(cond_prefix.shape)}"
            )
        if num_steps < 1 or num_steps > self.num_diffusion_steps:
            raise ValueError(
                f"num_steps must be in [1, {self.num_diffusion_steps}], got {num_steps}"
            )
        if eta is not None:
            ddim_eta = eta
        if ddim_eta < 0.0:
            raise ValueError(f"ddim_eta must be >= 0, got {ddim_eta}")
        if cfg_scale < 0.0:
            raise ValueError(f"cfg_scale must be >= 0, got {cfg_scale}")

        dtype = next(self.parameters()).dtype
        n, k, feat_dim = cond_prefix.shape
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        if initial_noise is not None:
            expected_shape = (n, self.seq_len, feat_dim)
            if tuple(initial_noise.shape) != expected_shape:
                raise ValueError(
                    f"initial_noise must have shape {expected_shape}, got {tuple(initial_noise.shape)}"
                )
            x_t = initial_noise.to(device=device, dtype=dtype).clone()
        else:
            x_t = torch.randn(n, self.seq_len, feat_dim, device=device, dtype=dtype)
        x_t[:, :k] = cond_prefix

        step_ids = torch.linspace(
            self.num_diffusion_steps - 1,
            0,
            num_steps,
            device=device,
            dtype=torch.float32,
        ).round().to(torch.long)
        step_ids = torch.unique_consecutive(step_ids)

        for idx, t_idx in enumerate(step_ids):
            t_batch_long = t_idx.expand(n)
            t_batch = self._time_to_unit_interval(t_batch_long, dtype=dtype).to(device=device)
            eps_pred = self.network(x_t, t_batch)
            eps_pred[:, :k] = 0.0

            alpha_bar_t = cast(torch.Tensor, self.alpha_bars)[t_idx].to(device=device, dtype=dtype)
            sqrt_alpha_bar_t = cast(torch.Tensor, self.sqrt_alpha_bars)[t_idx].to(
                device=device, dtype=dtype
            )
            sqrt_one_minus_alpha_bar_t = cast(torch.Tensor, self.sqrt_one_minus_alpha_bars)[
                t_idx
            ].to(
                device=device, dtype=dtype
            )

            x0_hat = (
                x_t - sqrt_one_minus_alpha_bar_t.view(1, 1, 1) * eps_pred
            ) / sqrt_alpha_bar_t.view(1, 1, 1).clamp_min(1e-8)
            x0_hat[:, :k] = cond_prefix

            if idx == len(step_ids) - 1:
                x_t = x0_hat
                x_t[:, :k] = cond_prefix
                break

            next_t_idx = step_ids[idx + 1]
            alpha_bar_next = cast(torch.Tensor, self.alpha_bars)[next_t_idx].to(
                device=device, dtype=dtype
            )
            sigma = ddim_eta * torch.sqrt(
                ((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t).clamp_min(1e-12))
                * (1.0 - alpha_bar_t / alpha_bar_next.clamp_min(1e-12))
            ).clamp_min(0.0)
            dir_coeff = torch.sqrt((1.0 - alpha_bar_next - sigma**2).clamp_min(0.0))
            noise = torch.randn_like(x_t) if float(sigma.item()) > 0.0 else torch.zeros_like(x_t)
            x_t = (
                torch.sqrt(alpha_bar_next).view(1, 1, 1) * x0_hat
                + dir_coeff.view(1, 1, 1) * eps_pred
                + sigma.view(1, 1, 1) * noise
            )
            x_t[:, :k] = cond_prefix

        return x_t


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, buffer in model_buffers.items():
        ema_buffers[name].copy_(buffer)
