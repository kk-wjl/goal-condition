"""
Simple conditional VAE wrapper.
posterior_encoder(target, condition) -> [mu, logvar]
prior_encoder(condition) -> [mu, logvar]
decoder(z, condition) -> reconstruction
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def split_gaussian_params(
    params: torch.Tensor,
    *,
    dim: int = -1,
    logvar_min: float | None = -10.0,
    logvar_max: float | None = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split concatenated Gaussian parameters into ``(mu, logvar)``."""
    if params.shape[dim] % 2 != 0:
        raise ValueError(
            f"Gaussian parameter size along dim={dim} must be even, got {params.shape[dim]}"
        )
    mu, logvar = params.chunk(2, dim=dim)
    if logvar_min is not None or logvar_max is not None:
        lo = logvar_min if logvar_min is not None else float("-inf")
        hi = logvar_max if logvar_max is not None else float("inf")
        logvar = logvar.clamp(min=lo, max=hi)
    return mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample from ``N(mu, exp(logvar))`` using the reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def diagonal_gaussian_kl(
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    *,
    reduce: str = "mean",
) -> torch.Tensor:
    """KL divergence between two diagonal Gaussians."""
    variance_ratio = torch.exp(posterior_logvar - prior_logvar)
    mean_term = (posterior_mu - prior_mu).square() * torch.exp(-prior_logvar)
    kl = 0.5 * (prior_logvar - posterior_logvar + variance_ratio + mean_term - 1.0)
    kl = kl.sum(dim=-1)
    if reduce == "none":
        return kl
    if reduce == "mean":
        return kl.mean()
    if reduce == "sum":
        return kl.sum()
    raise ValueError(f"Unsupported reduce={reduce!r}; expected 'none', 'mean', or 'sum'")


class ConditionVAE(nn.Module):
    """Generic conditional VAE wrapper with learned posterior, prior, and decoder."""

    def __init__(
        self,
        posterior_encoder: nn.Module,
        prior_encoder: nn.Module,
        decoder: nn.Module,
        *,
        logvar_min: float | None = -10.0,
        logvar_max: float | None = 8.0,
    ) -> None:
        super().__init__()
        self.posterior_encoder = posterior_encoder
        self.prior_encoder = prior_encoder
        self.decoder = decoder
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def encode_posterior(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        *extra_args: Any,
        **extra_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.posterior_encoder(target, condition, *extra_args, **extra_kwargs)
        return split_gaussian_params(
            params,
            logvar_min=self.logvar_min,
            logvar_max=self.logvar_max,
        )

    def encode_prior(
        self,
        condition: torch.Tensor,
        *extra_args: Any,
        **extra_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.prior_encoder(condition, *extra_args, **extra_kwargs)
        return split_gaussian_params(
            params,
            logvar_min=self.logvar_min,
            logvar_max=self.logvar_max,
        )

    def decode(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        *extra_args: Any,
        **extra_kwargs: Any,
    ) -> torch.Tensor:
        return self.decoder(z, condition, *extra_args, **extra_kwargs)

    def forward(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        *extra_args: Any,
        **extra_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mu, posterior_logvar = self.encode_posterior(
            target, condition, *extra_args, **extra_kwargs
        )
        prior_mu, prior_logvar = self.encode_prior(
            condition, *extra_args, **extra_kwargs
        )
        z = reparameterize(posterior_mu, posterior_logvar)
        reconstruction = self.decode(z, condition, *extra_args, **extra_kwargs)
        return reconstruction, posterior_mu, posterior_logvar, prior_mu, prior_logvar


__all__ = [
    "ConditionVAE",
    "diagonal_gaussian_kl",
    "reparameterize",
    "split_gaussian_params",
]
