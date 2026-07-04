"""Save and load training checkpoints (model, optimizer, RNG, config)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


def _config_to_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)
    raise TypeError(f"config must be a dataclass instance or dict, got {type(config)!r}")


def save_training_checkpoint(
    path: Path | str,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    config: Any,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": _config_to_dict(config),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_training_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    map_location: str | torch.device | None = "cpu",
) -> int:
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"].contiguous().cpu())
    if "numpy_rng_state" in payload:
        np.random.set_state(payload["numpy_rng_state"])
    return int(payload["epoch"])


def read_training_checkpoint_config(
    path: Path | str,
    *,
    map_location: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint at {path} does not contain a dict config payload")
    return config
