"""
Shared rectified / linear flow-matching building blocks (parameterization wrappers,
MSE objectives, Euler sampling, EMA).

Used by image examples (MNIST, STL-10) and can pair with any denoiser
``forward(x_t, t)`` or ``forward(x_t, t, y)`` (class-conditional + CFG).
"""

from __future__ import annotations

from typing import Literal
from typing import cast

import torch
import torch.nn as nn

PredictionType = Literal["x", "eps", "v"]
LossType = Literal["x", "eps", "v"]


def compute_flow_matching_loss(
    loss_type: LossType,
    x1: torch.Tensor,
    eps: torch.Tensor,
    predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    x1_hat, v_hat, eps_hat = predictions
    if loss_type == "x":
        return ((x1_hat - x1) ** 2).mean()
    if loss_type == "eps":
        return ((eps_hat - eps) ** 2).mean()
    if loss_type == "v":
        v_target = x1 - eps
        return ((v_hat - v_target) ** 2).mean()
    raise ValueError(f"Invalid loss type: {loss_type}")


class PredictionWrapper(nn.Module):
    """Bridge x / eps / v prediction targets for linear interpolation x_t = t x1 + (1-t) eps."""

    def __init__(self, network: nn.Module, pred_type: PredictionType):
        super().__init__()
        self.network = network
        self.pred_type = pred_type
        self.sample_shape = cast(tuple[int, ...], network.sample_shape)

    def reparameterize(
        self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        if y is None:
            return self.network(x_t, t)
        return self.network(x_t, t, y)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.reparameterize(x_t, t, y)
        expand_shape = (-1,) + (x_t.ndim - 1) * (1,)
        t_b = t.reshape(expand_shape)
        if self.pred_type == "x":
            x1_hat = pred
            v_hat = (x1_hat - x_t) / (1.0 - t_b)
            eps_hat = (x_t - t_b * x1_hat) / (1.0 - t_b)
        elif self.pred_type == "eps":
            eps_hat = pred
            v_hat = (x_t - eps_hat) / t_b
            x1_hat = x_t + (1.0 - t_b) * v_hat
        else:
            v_hat = pred
            x1_hat = x_t + (1.0 - t_b) * v_hat
            eps_hat = x_t - t_b * v_hat
        return x1_hat, v_hat, eps_hat

class LinearFlow:
    """Rectified flow with optional classifier-free guidance (conditional path)."""

    def __init__(
        self,
        model: nn.Module,
        *,
        noise_scale: float = 1.0,
        loss_type: LossType = "v",
        t_eps: float = 1e-2,
        conditional: bool = False,
        condition_dropout_prob: float = 0.0,
        condition_type: Literal["class", "vector"] = "class",
    ):
        self.model = model
        self.sample_shape = model.sample_shape
        self.noise_scale = noise_scale
        self.loss_type: LossType = loss_type
        self.t_eps = t_eps
        self.conditional = conditional
        self.condition_dropout_prob = condition_dropout_prob
        self.condition_type = condition_type

    def _sample_shape(self) -> tuple[int, ...]:
        return cast(tuple[int, ...], self.sample_shape)

    def _null_class_idx(self) -> int:
        inner = getattr(self.model, "network", None)
        backbone = getattr(inner, "backbone", None) if inner is not None else None
        if backbone is None or not hasattr(backbone, "null_class_idx"):
            raise TypeError("CFG / label dropout requires backbone.null_class_idx on the denoiser")
        return int(backbone.null_class_idx)

    def null_condition(self, y: torch.Tensor) -> torch.Tensor:
        if self.condition_type == "class":
            null = self._null_class_idx()
            return torch.full_like(y, null)
        return torch.zeros_like(y)

    def maybe_drop_condition(self, y: torch.Tensor) -> torch.Tensor:
        if self.condition_dropout_prob <= 0:
            return y
        drop_mask = torch.rand(y.shape[0], device=y.device) < self.condition_dropout_prob
        if not drop_mask.any():
            return y
        y_cond = y.clone()
        if self.condition_type == "class":
            y_cond[drop_mask] = self._null_class_idx()
            return y_cond
        y_cond[drop_mask] = 0.0
        return y_cond

    def compute_loss(
        self,
        x1: torch.Tensor,
        y: torch.Tensor | None = None,
        *,
        cond_steps: int | None = None,
    ) -> torch.Tensor:
        if self.conditional and y is None:
            raise ValueError("conditional flow: labels y are required for compute_loss")
        if not self.conditional and y is not None:
            raise ValueError("unconditional flow: did not expect labels y")

        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        t = t.clip(self.t_eps, 1.0 - self.t_eps)# t_eps is to avoid numerical issues at t=0 or t=1 where the interpolation degenerates to pure x1 or pure noise
        expand_shape = (-1,) + (x1.ndim - 1) * (1,)
        t_view = t.reshape(expand_shape)
        eps = torch.randn_like(x1) * self.noise_scale
        if cond_steps is not None:
            if not (1 <= cond_steps <= x1.shape[1]):
                raise ValueError(
                    f"cond_steps must be in [1, T], got cond_steps={cond_steps}, T={x1.shape[1]}"
                )
            eps[:, :cond_steps] = x1[:, :cond_steps]
        x_t = t_view * x1 + (1.0 - t_view) * eps
        if cond_steps is not None:
            x_t[:, :cond_steps] = x1[:, :cond_steps]

        if self.conditional:
            assert y is not None
            y_cond = self.maybe_drop_condition(y)
            predictions = self.model(x_t, t, y_cond)
        else:
            predictions = self.model(x_t, t, None)

        if cond_steps is None:
            return compute_flow_matching_loss(self.loss_type, x1, eps, predictions)
        k = cond_steps
        x1_hat, v_hat, eps_hat = predictions
        if self.loss_type == "x":
            err = (x1_hat - x1) ** 2
        elif self.loss_type == "eps":
            err = (eps_hat - eps) ** 2
        else:
            v_target = x1 - eps
            err = (v_hat - v_target) ** 2
        return err[:, k:, :].mean()

    @torch.inference_mode()
    def sample(self, num_samples: int, device: torch.device, num_steps: int) -> torch.Tensor:
        if self.conditional:
            raise TypeError("use sample_cfg(...) for class-conditional models")
        dtype = next(self.model.parameters()).dtype
        sample_shape = self._sample_shape()
        x_t = torch.randn((num_samples, *sample_shape), device=device, dtype=dtype)
        # x_t: (num_samples, T, D)
        x_t = x_t * self.noise_scale
        ts = torch.linspace(self.t_eps, 1.0 - self.t_eps, num_steps, device=device, dtype=dtype)
        dt = (
            ts[1] - ts[0]
            if num_steps > 1
            else torch.tensor(1.0 - 2 * self.t_eps, device=device, dtype=dtype)
        )
        for t_scalar in ts:
            t = torch.full((num_samples,), t_scalar.item(), device=device, dtype=dtype)
            _, v_hat, _ = self.model(x_t, t, None)
            x_t = x_t + v_hat * dt
        return x_t

    @torch.inference_mode()
    def sample_cond_prefix(
        self,
        cond_prefix: torch.Tensor, 
        device: torch.device,
        num_steps: int,
        y: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample a full window ``(N, T, D)`` with the first ``k`` timesteps fixed to
        ``cond_prefix`` (normalized space), matching the ``cond_steps`` loss path.
        """
        if self.conditional and y is None:
            raise TypeError("conditional flow: y is required for sample_cond_prefix")
        if not self.conditional and y is not None:
            raise TypeError("unconditional flow: did not expect y in sample_cond_prefix")
        dtype = next(self.model.parameters()).dtype
        k = cond_prefix.shape[1]
        n = cond_prefix.shape[0]
        if cond_prefix.ndim != 3:
            raise ValueError(f"cond_prefix must be (N, k, D), got shape {tuple(cond_prefix.shape)}")
        seq_len, dim = self._sample_shape()
        if cond_prefix.shape[2] != dim:
            raise ValueError(
                f"cond_prefix dim {cond_prefix.shape[2]} != sample_shape[1] {dim}"
            )
        if k > seq_len:
            raise ValueError(f"cond_steps k={k} exceeds seq_len={seq_len}")
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        if initial_noise is None:
            x_t = torch.randn((n, seq_len, dim), device=device, dtype=dtype)
        else:
            if initial_noise.shape != (n, seq_len, dim):
                raise ValueError(
                    f"initial_noise must have shape {(n, seq_len, dim)}, got {tuple(initial_noise.shape)}"
                )
            x_t = initial_noise.to(device=device, dtype=dtype)
        x_t = x_t * self.noise_scale
        x_t[:, :k] = cond_prefix
        ts = torch.linspace(self.t_eps, 1.0 - self.t_eps, num_steps, device=device, dtype=dtype)
        dt = (
            ts[1] - ts[0]
            if num_steps > 1
            else torch.tensor(1.0 - 2 * self.t_eps, device=device, dtype=dtype)
        )
        if y is not None:
            y = y.to(device=device, dtype=dtype)
        for t_scalar in ts:
            t = torch.full((n,), t_scalar.item(), device=device, dtype=dtype)
            _, v_hat, _ = self.model(x_t, t, y)
            v_hat = v_hat.clone()
            v_hat[:, :k] = 0.0
            x_t = x_t + v_hat * dt
            x_t[:, :k] = cond_prefix
        return x_t

    @torch.inference_mode()
    def sample_cfg(
        self,
        y: torch.Tensor,
        device: torch.device,
        num_steps: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        if not self.conditional:
            raise TypeError("sample_cfg requires a conditional flow (conditional=True)")
        dtype = next(self.model.parameters()).dtype
        n = y.shape[0]
        sample_shape = self._sample_shape()
        x_t = torch.randn((n, *sample_shape), device=device, dtype=dtype)
        x_t = x_t * self.noise_scale
        ts = torch.linspace(self.t_eps, 1.0 - self.t_eps, num_steps, device=device, dtype=dtype)
        dt = (
            ts[1] - ts[0]
            if num_steps > 1
            else torch.tensor(1.0 - 2 * self.t_eps, device=device, dtype=dtype)
        )
        if self.condition_type == "vector":
            y = y.to(device=device, dtype=dtype)
        else:
            y = y.to(device=device)
        uncond_y = self.null_condition(y)
        for t_scalar in ts:
            t = torch.full((n,), t_scalar.item(), device=device, dtype=dtype)
            _, v_cond, _ = self.model(x_t, t, y)
            _, v_uncond, _ = self.model(x_t, t, uncond_y)
            v_hat = v_uncond + cfg_scale * (v_cond - v_uncond)
            x_t = x_t + v_hat * dt
        return x_t

    @torch.inference_mode()
    def sample_cfg_cond_prefix(
        self,
        cond_prefix: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
        num_steps: int,
        cfg_scale: float,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.conditional:
            raise TypeError("sample_cfg_cond_prefix requires a conditional flow (conditional=True)")
        dtype = next(self.model.parameters()).dtype
        k = cond_prefix.shape[1]
        n = cond_prefix.shape[0]
        if cond_prefix.ndim != 3:
            raise ValueError(f"cond_prefix must be (N, k, D), got shape {tuple(cond_prefix.shape)}")
        seq_len, dim = self._sample_shape()
        if cond_prefix.shape[2] != dim:
            raise ValueError(
                f"cond_prefix dim {cond_prefix.shape[2]} != sample_shape[1] {dim}"
            )
        if k > seq_len:
            raise ValueError(f"cond_steps k={k} exceeds seq_len={seq_len}")
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        if self.condition_type == "vector":
            y = y.to(device=device, dtype=dtype)
        else:
            y = y.to(device=device)
        uncond_y = self.null_condition(y)
        if initial_noise is None:
            x_t = torch.randn((n, seq_len, dim), device=device, dtype=dtype)
        else:
            if initial_noise.shape != (n, seq_len, dim):
                raise ValueError(
                    f"initial_noise must have shape {(n, seq_len, dim)}, got {tuple(initial_noise.shape)}"
                )
            x_t = initial_noise.to(device=device, dtype=dtype)
        x_t = x_t * self.noise_scale
        x_t[:, :k] = cond_prefix
        ts = torch.linspace(self.t_eps, 1.0 - self.t_eps, num_steps, device=device, dtype=dtype)
        dt = (
            ts[1] - ts[0]
            if num_steps > 1
            else torch.tensor(1.0 - 2 * self.t_eps, device=device, dtype=dtype)
        )
        for t_scalar in ts:
            t = torch.full((n,), t_scalar.item(), device=device, dtype=dtype)
            _, v_cond, _ = self.model(x_t, t, y)
            _, v_uncond, _ = self.model(x_t, t, uncond_y)
            v_hat = v_uncond + cfg_scale * (v_cond - v_uncond)
            v_hat = v_hat.clone()
            v_hat[:, :k] = 0.0
            x_t = x_t + v_hat * dt
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
