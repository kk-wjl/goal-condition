"""Hierarchical goal-conditioned trajectory generation on LAFAN1.

The high-level conditional VAE predicts intermediate root waypoints; the final goal
is supplied by the condition and appended unchanged. The low-level flow-matching
controller generates one local trajectory window from a prefix and local waypoint,
and rollout mode composes both levels into full trajectory generation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

_EXAMPLES_DIR = Path(__file__).resolve().parent

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from scripts.config_hierarchy import (
        ChunkConfig,
        LocalGoalLAFAN1Dataset,
        WAYPOINT_DIM,
        WaypointLAFAN1Dataset,
    )
except ModuleNotFoundError:
    from config_hierarchy import (
        ChunkConfig,
        LocalGoalLAFAN1Dataset,
        WAYPOINT_DIM,
        WaypointLAFAN1Dataset,
    )

from goal_condition.cvae import ConditionVAE, diagonal_gaussian_kl, split_gaussian_params
from goal_condition.datasets.lafan1 import (
    LAFAN1Dataset,
    POSE_BASE_DIM,
    ROOT_ROT_OFFSET,
    RobotName,
    rot6d_to_matrix,
)
from goal_condition.flow_matching import (
    LinearFlow,
    LossType,
    PredictionType,
    PredictionWrapper,
)
from goal_condition.modules.conditional_unet import ConditionalUNet1D
from goal_condition.modules.transformer import DiffusionTransformer1D
from goal_condition.utils.math import rot6d_from_matrix
from goal_condition.utils.checkpoint import (
    load_training_checkpoint,
    read_training_checkpoint_config,
    save_training_checkpoint,
)


Mode = Literal["train_high", "train_low", "rollout"]


@dataclass
class Config:
    mode: Mode = "train_high"
    data_root: str = "./data"
    robot: RobotName = "g1"
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    latent_dim: int = 32
    hidden_dim: int = 256
    beta: float = 0.01
    batch_size: int = 128
    train_epochs: int = 50
    lr: float = 3e-4
    val_fraction: float = 0.2
    val_every: int = 1
    num_val_samples: int = 256
    num_workers: int = 0
    num_threads: int = 1
    seed: int = 42
    use_wandb: bool = False
    seq_len: int = 32
    cond_steps: int = 4
    data_stride: int = 1
    use_dit: bool = False
    base_channels: int = 128
    cond_dim: int = 128
    max_seq_len: int = 256
    hidden_size: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_cross_attention: bool = False
    pred_type: PredictionType = "v"
    loss_type: LossType = "v"
    goal_loss_weight: float = 2.0
    noise_scale: float = 1.0
    t_eps: float = 1e-2
    sample_steps: int = 40
    condition_dropout_prob: float = 0.1
    num_plot_samples: int = 4
    num_rollout_samples: int = 8
    goal_reach_xy_threshold: float = 0.2
    goal_reach_yaw_threshold_rad: float = 0.3
    high_checkpoint: str | None = None
    low_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.chunk.prefix_frames != self.cond_steps:
            raise ValueError(
                "Expected chunk.prefix_frames == cond_steps, "
                f"got prefix_frames={self.chunk.prefix_frames}, cond_steps={self.cond_steps}"
            )
        if self.chunk.chunk_len - self.chunk.prefix_frames != (
            self.seq_len - self.cond_steps
        ) * (self.chunk.num_waypoints + 1):
            raise ValueError(
                "Expected chunk_len - prefix_frames == "
                "(seq_len - cond_steps) * (num_waypoints + 1), "
                f"got chunk_len={self.chunk.chunk_len}, prefix_frames={self.chunk.prefix_frames}, "
                f"seq_len={self.seq_len}, cond_steps={self.cond_steps}, "
                f"num_waypoints={self.chunk.num_waypoints}"
            )

# angular transformation
def _yaw_from_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
    rot = rot6d_to_matrix(rot6d)
    return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])


def _wrapped_yaw_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta = pred - target
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs()

class WaypointCVAE(ConditionVAE):
    """Conditional posterior, learned conditional prior, and waypoint decoder."""

    @staticmethod
    def _mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def __init__(
        self,
        condition_dim: int,
        num_waypoints: int,
        latent_dim: int,
        hidden_dim: int,
    ) -> None:
        waypoint_flat_dim = num_waypoints * WAYPOINT_DIM

        posterior = self._mlp(
            waypoint_flat_dim + condition_dim,
            2 * latent_dim,
            hidden_dim,
        )
        prior = self._mlp(condition_dim, 2 * latent_dim, hidden_dim)
        decoder = self._mlp(
            latent_dim + condition_dim,
            waypoint_flat_dim,
            hidden_dim,
        )
        super().__init__(
            posterior_encoder=posterior,
            prior_encoder=prior,
            decoder=decoder,
        )
        self.num_waypoints = num_waypoints
        self.latent_dim = latent_dim
        self._waypoint_flat_dim = waypoint_flat_dim
        self.posterior = posterior
        self.prior = prior
        self.decoder_network = decoder

    def encode_posterior(
        self,
        waypoints: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([waypoints.flatten(1), condition], dim=-1)
        return split_gaussian_params(self.posterior(inputs))

    def encode_prior(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return split_gaussian_params(self.prior(condition))

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_network(torch.cat([z, condition], dim=-1))
        decoded = decoded.view(-1, self.num_waypoints, WAYPOINT_DIM)
        yaw_vector = torch.nn.functional.normalize(decoded[..., 2:4], dim=-1, eps=1e-6)
        return torch.cat([decoded[..., :2], yaw_vector], dim=-1)


def _waypoint_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    goal: torch.Tensor,
    xy_scale: torch.Tensor,
) -> dict[str, float]:
    pred_xy = pred[..., :2] * xy_scale
    target_xy = target[..., :2] * xy_scale
    goal_xy = goal[..., :2] * xy_scale
    pred_yaw = torch.atan2(pred[..., 2], pred[..., 3])
    target_yaw = torch.atan2(target[..., 2], target[..., 3])
    goal_yaw = torch.atan2(goal[..., 2], goal[..., 3])
    return {
        "recon_waypoint_xy_error": float(torch.norm(pred_xy - target_xy, dim=-1).mean()),
        "recon_waypoint_yaw_error_rad": float(
            _wrapped_yaw_error(pred_yaw, target_yaw).mean()
        ),
        "last_waypoint_to_goal_xy_error": float(
            torch.norm(pred_xy[:, -1] - goal_xy, dim=-1).mean()
        ),
        "last_waypoint_to_goal_yaw_error_rad": float(
            _wrapped_yaw_error(pred_yaw[:, -1], goal_yaw).mean()
        ),
    }


@torch.no_grad()
def validate_high(
    model: WaypointCVAE,
    loader: DataLoader,
    dataset: WaypointLAFAN1Dataset,
    config: Config,
    device: torch.device,
    epoch: int,
    output_dir: Path,
) -> dict[str, float | int]:
    model.eval()
    val_dir = output_dir / "validation" / f"epoch_{epoch:03d}"
    val_dir.mkdir(parents=True, exist_ok=True)
    totals: dict[str, float] = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "kl": 0.0,
        "posterior_mu_abs": 0.0,
        "posterior_std": 0.0,
        "prior_mu_abs": 0.0,
        "prior_std": 0.0,
    }
    metric_totals: dict[str, float] = {}
    sample_count = 0
    xy_scale = dataset.base._root_pos_std[:2].to(device)

    for batch in loader:
        if sample_count >= config.num_val_samples:
            break
        condition = batch["condition"].to(device)
        target = batch["waypoints"].to(device)
        goal = batch["goal"].to(device)
        batch_size = min(condition.shape[0], config.num_val_samples - sample_count)
        condition, target, goal = condition[:batch_size], target[:batch_size], goal[:batch_size]

        posterior_mu, posterior_logvar = model.encode_posterior(target, condition)
        prior_mu, prior_logvar = model.encode_prior(condition)
        posterior = model.decode(posterior_mu, condition)
        reconstruction = torch.nn.functional.mse_loss(
            posterior[..., :2],
            target[..., :2],
        ) + torch.nn.functional.mse_loss(
            posterior[..., 2:4],
            target[..., 2:4],
        )
        kl = diagonal_gaussian_kl(posterior_mu, posterior_logvar, prior_mu, prior_logvar)
        loss = reconstruction + config.beta * kl

        values = {
            "loss": loss,
            "reconstruction": reconstruction,
            "kl": kl,
            "posterior_mu_abs": posterior_mu.abs().mean(),
            "posterior_std": torch.exp(0.5 * posterior_logvar).mean(),
            "prior_mu_abs": prior_mu.abs().mean(),
            "prior_std": torch.exp(0.5 * prior_logvar).mean(),
        }
        for key, value in values.items():
            totals[key] += float(value) * batch_size
        for key, value in _waypoint_metrics(posterior, target, goal, xy_scale).items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("Validation loader produced no samples")
    metrics: dict[str, float | int] = {
        "epoch": epoch,
        "val_loss": totals["loss"] / sample_count,
        "reconstruction_loss": totals["reconstruction"] / sample_count,
        "kl_loss": totals["kl"] / sample_count,
        "beta": config.beta,
        "recon_waypoint_xy_error": metric_totals["recon_waypoint_xy_error"] / sample_count,
        "recon_waypoint_yaw_error_rad": metric_totals["recon_waypoint_yaw_error_rad"] / sample_count,
        "last_waypoint_to_goal_xy_error": metric_totals["last_waypoint_to_goal_xy_error"] / sample_count,
        "last_waypoint_to_goal_yaw_error_rad": metric_totals["last_waypoint_to_goal_yaw_error_rad"] / sample_count,
        "posterior_mu_abs_mean": totals["posterior_mu_abs"] / sample_count,
        "posterior_std_mean": totals["posterior_std"] / sample_count,
        "prior_mu_abs_mean": totals["prior_mu_abs"] / sample_count,
        "prior_std_mean": totals["prior_std"] / sample_count,
        "num_samples": sample_count,
    }
    (val_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def train_high(config: Config, device: torch.device, resume: bool) -> None:
    base = LAFAN1Dataset(
        root=config.data_root,
        robot=config.robot,
        seq_len=config.chunk.chunk_len,
        stride=config.chunk.chunk_stride,
        download=True,
    )
    dataset_kwargs = {
        "chunk_len": config.chunk.chunk_len,
        "chunk_stride": config.chunk.chunk_stride,
        "prefix_frames": config.chunk.prefix_frames,
        "num_waypoints": config.chunk.num_waypoints,
        "val_fraction": config.val_fraction,
        "split_seed": config.seed,
    }
    train_dataset = WaypointLAFAN1Dataset(base, split="train", **dataset_kwargs)
    val_dataset = WaypointLAFAN1Dataset(base, split="val", **dataset_kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    model = WaypointCVAE(
        condition_dim=train_dataset.condition_dim,
        num_waypoints=config.chunk.num_waypoints,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    output_dir = _EXAMPLES_DIR / "outputs" / "FM_lafan1_hierarchy" / "high_level"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    (output_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(config), indent=2) + "\n"
    )
    start_epoch = 0
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No high-level checkpoint found at {checkpoint_path}")
        start_epoch = load_training_checkpoint(checkpoint_path, model, optimizer) + 1

    wandb_run = None
    if config.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project="goal-condition",
            name="FM_lafan1_goal_cond_hierarchy_high",
            config=dataclasses.asdict(config),
        )

    for epoch in range(start_epoch, config.train_epochs):
        model.train()
        losses: list[float] = []
        progress = tqdm(train_loader, desc=f"high epoch {epoch}")
        for batch in progress:
            condition = batch["condition"].to(device)
            target = batch["waypoints"].to(device)
            reconstruction, posterior_mu, posterior_logvar, prior_mu, prior_logvar = model(
                target,
                condition,
            )
            reconstruction_loss = torch.nn.functional.mse_loss(
                reconstruction[..., :2],
                target[..., :2],
            ) + torch.nn.functional.mse_loss(
                reconstruction[..., 2:4],
                target[..., 2:4],
            )
            kl_loss = diagonal_gaussian_kl(
                posterior_mu,
                posterior_logvar,
                prior_mu,
                prior_logvar,
            )
            loss = reconstruction_loss + config.beta * kl_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            progress.set_postfix(loss=f"{losses[-1]:.5f}")

        if epoch % config.val_every == 0 or epoch == config.train_epochs - 1:
            metrics = validate_high(
                model,
                val_loader,
                val_dataset,
                config,
                device,
                epoch,
                output_dir,
            )
            save_training_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                config=config,
                extra={"validation_metrics": metrics},
            )
            print(json.dumps(metrics, indent=2))
            if wandb_run is not None:
                wandb_run.log({"train_loss": np.mean(losses), **metrics}, step=epoch)
    if wandb_run is not None:
        wandb_run.finish()


class LocalGoalFlowBackbone(nn.Module):
    """Generate one local trajectory window from a prefix and local root goal."""

    def __init__(
        self,
        state_dim: int,
        *,
        use_dit: bool,
        base_channels: int,
        cond_dim: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        max_seq_len: int,
        dropout: float,
        use_cross_attention: bool,
    ) -> None:
        super().__init__()
        self.sample_shape: tuple[int, int] = (1, state_dim)
        self.use_dit = use_dit
        self.use_cross_attention = use_cross_attention
        if use_dit:
            self.network = DiffusionTransformer1D(
                input_dim=state_dim,
                output_dim=state_dim,
                hidden_size=hidden_size,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cond_dim=hidden_size,
                max_seq_len=max_seq_len,
                dropout=dropout,
                use_cross_attention=use_cross_attention,
            )
        else:
            self.network = ConditionalUNet1D(
                input_dim=state_dim,
                output_dim=state_dim,
                base_channels=base_channels,
                channel_mults=(1, 2, 4),
                cond_dim=cond_dim,
            )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        local_goal: torch.Tensor | None,
    ) -> torch.Tensor:
        if local_goal is None:
            raise ValueError("local_goal is required by the local-goal controller")
        if self.use_dit:
            if self.use_cross_attention:
                return self.network(x_t, cond=local_goal, t=t, cond_tokens=local_goal)
            return self.network(x_t, cond=local_goal, t=t)
        return self.network(x_t, cond=local_goal, t=t)


def build_local_model(config: Config, state_dim: int) -> nn.Module:
    backbone = LocalGoalFlowBackbone(
        state_dim,
        use_dit=config.use_dit,
        base_channels=config.base_channels,
        cond_dim=config.cond_dim,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        use_cross_attention=config.use_cross_attention,
    )
    backbone.sample_shape = (config.seq_len, state_dim)
    return PredictionWrapper(backbone, config.pred_type)


def compute_flow_loss(
    flow: LinearFlow,
    x1: torch.Tensor,
    local_goal: torch.Tensor,
    cond_steps: int,
    goal_loss_weight: float,
    *,
    drop_condition: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute flow loss plus a differentiable endpoint XY/yaw constraint."""
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    t = t.clamp(flow.t_eps, 1.0 - flow.t_eps)
    t_view = t.reshape((-1,) + (1,) * (x1.ndim - 1))
    eps = torch.randn_like(x1) * flow.noise_scale
    eps[:, :cond_steps] = x1[:, :cond_steps]
    x_t = t_view * x1 + (1.0 - t_view) * eps
    x_t[:, :cond_steps] = x1[:, :cond_steps]
    conditioned_goal = flow.maybe_drop_condition(local_goal) if drop_condition else local_goal
    x1_hat, v_hat, eps_hat = flow.model(x_t, t, conditioned_goal)

    if flow.loss_type == "x":
        error = (x1_hat - x1).square()
    elif flow.loss_type == "eps":
        error = (eps_hat - eps).square()
    else:
        error = (v_hat - (x1 - eps)).square()
    flow_loss = error[:, cond_steps:].mean()

    pred_final = x1_hat[:, -1]
    target_final = x1[:, -1]
    xy_loss = torch.nn.functional.mse_loss(pred_final[:, :2], target_final[:, :2])
    pred_yaw = _yaw_from_rot6d(pred_final[:, ROOT_ROT_OFFSET:POSE_BASE_DIM])
    target_yaw = _yaw_from_rot6d(target_final[:, ROOT_ROT_OFFSET:POSE_BASE_DIM])
    pred_heading = torch.stack([torch.sin(pred_yaw), torch.cos(pred_yaw)], dim=-1)
    target_heading = torch.stack([torch.sin(target_yaw), torch.cos(target_yaw)], dim=-1)
    endpoint_loss = xy_loss + torch.nn.functional.mse_loss(pred_heading, target_heading)
    return flow_loss + goal_loss_weight * endpoint_loss, flow_loss, endpoint_loss


@torch.no_grad()
def validate_local(
    flow: LinearFlow,
    loader: DataLoader,
    dataset: LocalGoalLAFAN1Dataset,
    config: Config,
    device: torch.device,
    epoch: int,
    output_dir: Path,
) -> dict[str, float | int]:
    flow.model.eval()
    val_dir = output_dir / "validation" / f"epoch_{epoch:03d}"
    val_dir.mkdir(parents=True, exist_ok=True)
    totals = {
        "loss": 0.0,
        "flow_loss": 0.0,
        "endpoint_loss": 0.0,
        "goal_xy": 0.0,
        "goal_yaw": 0.0,
        "trajectory_mse": 0.0,
    }
    generated_batches: list[torch.Tensor] = []
    sample_count = 0
    csv_count = 0

    for trajectory, goal_cond, _meta in loader:
        if sample_count >= config.num_val_samples:
            break
        batch_size = min(trajectory.shape[0], config.num_val_samples - sample_count)
        trajectory = trajectory[:batch_size].to(device)
        local_goal = goal_cond[:batch_size].to(device)
        target_local = dataset.make_relative(trajectory, yaw_only=True)
        target = dataset.normalize(target_local)
        total_loss, flow_loss, endpoint_loss = compute_flow_loss(
            flow,
            target,
            local_goal,
            config.cond_steps,
            config.goal_loss_weight,
            drop_condition=False,
        )
        generated = flow.sample_cond_prefix(
            target[:, : config.cond_steps],
            device,
            config.sample_steps,
            y=local_goal,
        )
        generated_local = dataset.denormalize(generated)
        pred_final = generated_local[:, -1]
        target_final = target_local[:, -1]
        xy_error = torch.norm(pred_final[:, :2] - target_final[:, :2], dim=-1)
        pred_yaw = _yaw_from_rot6d(pred_final[:, ROOT_ROT_OFFSET:POSE_BASE_DIM])
        target_yaw = _yaw_from_rot6d(target_final[:, ROOT_ROT_OFFSET:POSE_BASE_DIM])
        yaw_error = _wrapped_yaw_error(pred_yaw, target_yaw)

        totals["loss"] += float(total_loss) * batch_size
        totals["flow_loss"] += float(flow_loss) * batch_size
        totals["endpoint_loss"] += float(endpoint_loss) * batch_size
        totals["goal_xy"] += float(xy_error.mean()) * batch_size
        totals["goal_yaw"] += float(yaw_error.mean()) * batch_size
        totals["trajectory_mse"] += float(
            torch.nn.functional.mse_loss(generated, target)
        ) * batch_size
        generated_batches.append(generated_local.cpu())

        while csv_count < config.num_plot_samples and csv_count < sample_count + batch_size:
            local_index = csv_count - sample_count
            csv = dataset.trajectory_to_lafan1_csv_qpos(generated_local[local_index].cpu())
            np.savetxt(
                val_dir / f"rollout_{csv_count:03d}.csv",
                csv.numpy(),
                delimiter=",",
                fmt="%.8f",
            )
            csv_count += 1
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("Low-level validation loader produced no samples")
    generated_all = torch.cat(generated_batches, dim=0)
    velocity_metrics = dataset.compute_metrics(generated_all)
    metrics: dict[str, float | int] = {
        "epoch": epoch,
        "val_loss": totals["loss"] / sample_count,
        "val_flow_loss": totals["flow_loss"] / sample_count,
        "endpoint_loss": totals["endpoint_loss"] / sample_count,
        "goal_loss_weight": config.goal_loss_weight,
        "local_goal_xy_error": totals["goal_xy"] / sample_count,
        "local_goal_yaw_error_rad": totals["goal_yaw"] / sample_count,
        "trajectory_mse_normalized": totals["trajectory_mse"] / sample_count,
        "root_velocity_fd_mse": velocity_metrics["root_vel_fd_mse"],
        "joint_velocity_fd_mse": velocity_metrics["joint_vel_fd_mse"],
        "num_samples": sample_count,
    }
    (val_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def train_local(config: Config, device: torch.device, resume: bool) -> None:
    train_dataset = LocalGoalLAFAN1Dataset(
        root=config.data_root,
        split="train",
        robot=config.robot,
        seq_len=config.seq_len,
        cond_steps=config.cond_steps,
        stride=config.data_stride,
        val_fraction=config.val_fraction,
        split_seed=config.seed,
        download=True,
    )
    val_dataset = LocalGoalLAFAN1Dataset(
        root=config.data_root,
        split="val",
        robot=config.robot,
        seq_len=config.seq_len,
        cond_steps=config.cond_steps,
        stride=config.data_stride,
        val_fraction=config.val_fraction,
        split_seed=config.seed,
        download=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    model = build_local_model(config, train_dataset.state_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    flow = LinearFlow(
        model,
        noise_scale=config.noise_scale,
        loss_type=config.loss_type,
        t_eps=config.t_eps,
        conditional=True,
        condition_dropout_prob=config.condition_dropout_prob,
        condition_type="vector",
    )
    output_dir = _EXAMPLES_DIR / "outputs" / "FM_lafan1_hierarchy" / "low_level"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    (output_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(config), indent=2) + "\n"
    )
    start_epoch = 0
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No low-level checkpoint found at {checkpoint_path}")
        start_epoch = load_training_checkpoint(checkpoint_path, model, optimizer) + 1

    wandb_run = None
    if config.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project="goal-condition",
            name="FM_lafan1_goal_cond_hierarchy_low",
            config=dataclasses.asdict(config),
        )

    for epoch in range(start_epoch, config.train_epochs):
        model.train()
        losses: list[float] = []
        progress = tqdm(train_loader, desc=f"low epoch {epoch}")
        for trajectory, goal_cond, _meta in progress:
            trajectory = trajectory.to(device)
            local_goal = goal_cond.to(device)
            target = train_dataset.normalize(train_dataset.make_relative(trajectory, yaw_only=True))
            loss, flow_loss, endpoint_loss = compute_flow_loss(
                flow,
                target,
                local_goal,
                config.cond_steps,
                config.goal_loss_weight,
                drop_condition=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            progress.set_postfix(
                loss=f"{losses[-1]:.5f}",
                flow=f"{float(flow_loss.detach()):.5f}",
                goal=f"{float(endpoint_loss.detach()):.5f}",
            )

        if epoch % config.val_every == 0 or epoch == config.train_epochs - 1:
            metrics = validate_local(
                flow,
                val_loader,
                val_dataset,
                config,
                device,
                epoch,
                output_dir,
            )
            save_training_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                config=config,
                extra={"validation_metrics": metrics},
            )
            print(json.dumps(metrics, indent=2))
            if wandb_run is not None:
                wandb_run.log({"train_loss": np.mean(losses), **metrics}, step=epoch)
    if wandb_run is not None:
        wandb_run.finish()


def _config_from_checkpoint_dict(raw_config: dict[str, object]) -> Config:
    payload: dict[str, Any] = dict(raw_config)
    chunk_payload = payload.pop("chunk")
    if not isinstance(chunk_payload, dict):
        raise ValueError("Checkpoint config is missing a valid chunk config")
    chunk = ChunkConfig(**cast(dict[str, Any], chunk_payload))
    return Config(chunk=chunk, **payload)


def _yaw_to_rot6d(yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    matrix = torch.zeros(yaw.shape + (3, 3), device=yaw.device, dtype=yaw.dtype)
    matrix[..., 0, 0] = c
    matrix[..., 0, 1] = -s
    matrix[..., 1, 0] = s
    matrix[..., 1, 1] = c
    matrix[..., 2, 2] = 1.0
    return rot6d_from_matrix(matrix)


def _goal_frame_from_waypoint(
    anchor_frame: torch.Tensor,
    waypoint: torch.Tensor,
    xy_scale: torch.Tensor,
) -> torch.Tensor:
    goal_frame = anchor_frame.clone()
    goal_frame[..., 0] = waypoint[..., 0] * xy_scale[0]
    goal_frame[..., 1] = waypoint[..., 1] * xy_scale[1]
    goal_yaw = torch.atan2(waypoint[..., 2], waypoint[..., 3])
    goal_frame[..., ROOT_ROT_OFFSET:POSE_BASE_DIM] = _yaw_to_rot6d(goal_yaw)
    return goal_frame


def _convert_waypoints_between_datasets(
    waypoints: torch.Tensor,
    source_dataset: WaypointLAFAN1Dataset,
    target_dataset: LocalGoalLAFAN1Dataset,
) -> torch.Tensor:
    """Convert waypoint xy normalization from the high-level dataset into the low-level dataset."""
    xy_m = source_dataset.waypoint_xy_to_meters(waypoints[..., :2])
    target_xy_scale = target_dataset.base._root_pos_std[:2].to(
        device=waypoints.device,
        dtype=waypoints.dtype,
    )
    yaw_vec = torch.nn.functional.normalize(waypoints[..., 2:4], dim=-1, eps=1e-6)
    return torch.cat([xy_m / target_xy_scale, yaw_vec], dim=-1)


@torch.no_grad()
def rollout_local_trajectory(
    flow: LinearFlow,
    dataset: LocalGoalLAFAN1Dataset,
    config: Config,
    prefix: torch.Tensor,
    goals: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    rollout = prefix.to(device)
    xy_scale = dataset.base._root_pos_std[:2].to(device=device, dtype=rollout.dtype)
    flow_losses: list[float] = []
    goal_losses: list[float] = []

    for goal in goals:
        current_prefix = rollout[-config.cond_steps :]
        goal_frame = _goal_frame_from_waypoint(current_prefix[-1], goal.to(device), xy_scale)
        goal_cond = dataset.goal_condition_from_states(
            current_prefix[-1:].to(device),
            goal_frame.unsqueeze(0),
        )
        cond_prefix = dataset.normalize(
            dataset.make_relative(current_prefix.unsqueeze(0), yaw_only=True)
        )
        generated_norm = flow.sample_cond_prefix(
            cond_prefix,
            device,
            config.sample_steps,
            y=goal_cond,
        )
        _, flow_loss, endpoint_loss = compute_flow_loss(
            flow,
            generated_norm,
            goal_cond,
            config.cond_steps,
            config.goal_loss_weight,
            drop_condition=False,
        )
        flow_losses.append(float(flow_loss.detach()))
        goal_losses.append(float(endpoint_loss.detach()))

        generated_local = dataset.denormalize(generated_norm)
        generated_global = LAFAN1Dataset.accumulate_chunk_in_root_frame(
            generated_local,
            current_prefix[:1, :3],
            current_prefix[:1, ROOT_ROT_OFFSET:POSE_BASE_DIM],
            yaw_only=True,
        )[0]
        rollout = torch.cat([rollout, generated_global[config.cond_steps :]], dim=0)

    avg_flow_loss = float(np.mean(flow_losses)) if flow_losses else 0.0
    avg_goal_loss = float(np.mean(goal_losses)) if goal_losses else 0.0
    return rollout.cpu(), avg_flow_loss, avg_goal_loss


@torch.no_grad()
def run_rollout(config: Config, device: torch.device) -> None:
    high_checkpoint = Path(config.high_checkpoint) if config.high_checkpoint else (
        _EXAMPLES_DIR / "outputs" / "FM_lafan1_hierarchy" / "high_level" / "checkpoint.pt"
    )
    low_checkpoint = Path(config.low_checkpoint) if config.low_checkpoint else (
        _EXAMPLES_DIR / "outputs" / "FM_lafan1_hierarchy" / "low_level" / "checkpoint.pt"
    )
    if not high_checkpoint.is_file():
        raise FileNotFoundError(f"High-level checkpoint not found: {high_checkpoint}")
    if not low_checkpoint.is_file():
        raise FileNotFoundError(f"Low-level checkpoint not found: {low_checkpoint}")

    high_config = _config_from_checkpoint_dict(read_training_checkpoint_config(high_checkpoint))
    low_config = _config_from_checkpoint_dict(read_training_checkpoint_config(low_checkpoint))
    rollout_config = dataclasses.replace(high_config)
    for key in (
        "seq_len",
        "cond_steps",
        "data_stride",
        "use_dit",
        "base_channels",
        "cond_dim",
        "max_seq_len",
        "hidden_size",
        "depth",
        "num_heads",
        "mlp_ratio",
        "dropout",
        "use_cross_attention",
        "pred_type",
        "loss_type",
        "goal_loss_weight",
        "noise_scale",
        "t_eps",
        "sample_steps",
        "condition_dropout_prob",
        "num_plot_samples",
    ):
        setattr(rollout_config, key, getattr(low_config, key))
    rollout_config.data_root = config.data_root
    rollout_config.robot = config.robot
    rollout_config.num_rollout_samples = config.num_rollout_samples
    rollout_config.goal_reach_xy_threshold = config.goal_reach_xy_threshold
    rollout_config.goal_reach_yaw_threshold_rad = config.goal_reach_yaw_threshold_rad
    rollout_config.high_checkpoint = str(high_checkpoint)
    rollout_config.low_checkpoint = str(low_checkpoint)

    base = LAFAN1Dataset(
        root=rollout_config.data_root,
        robot=rollout_config.robot,
        seq_len=rollout_config.chunk.chunk_len,
        stride=rollout_config.chunk.chunk_stride,
        download=True,
    )
    rollout_dataset = WaypointLAFAN1Dataset(
        base,
        split="val",
        chunk_len=rollout_config.chunk.chunk_len,
        chunk_stride=rollout_config.chunk.chunk_stride,
        prefix_frames=rollout_config.chunk.prefix_frames,
        num_waypoints=rollout_config.chunk.num_waypoints,
        val_fraction=rollout_config.val_fraction,
        split_seed=rollout_config.seed,
    )
    local_dataset = LocalGoalLAFAN1Dataset(
        root=rollout_config.data_root,
        split="val",
        robot=rollout_config.robot,
        seq_len=rollout_config.seq_len,
        cond_steps=rollout_config.cond_steps,
        stride=rollout_config.data_stride,
        val_fraction=rollout_config.val_fraction,
        split_seed=rollout_config.seed,
        download=True,
    )

    high_model = WaypointCVAE(
        condition_dim=rollout_dataset.condition_dim,
        num_waypoints=rollout_config.chunk.num_waypoints,
        latent_dim=rollout_config.latent_dim,
        hidden_dim=rollout_config.hidden_dim,
    ).to(device)
    high_optimizer = optim.AdamW(high_model.parameters(), lr=rollout_config.lr)
    high_epoch = load_training_checkpoint(high_checkpoint, high_model, high_optimizer, map_location=device)
    high_model.eval()

    low_model = build_local_model(rollout_config, local_dataset.state_dim).to(device)
    low_optimizer = optim.AdamW(low_model.parameters(), lr=rollout_config.lr)
    low_epoch = load_training_checkpoint(low_checkpoint, low_model, low_optimizer, map_location=device)
    low_model.eval()
    flow = LinearFlow(
        low_model,
        noise_scale=rollout_config.noise_scale,
        loss_type=rollout_config.loss_type,
        t_eps=rollout_config.t_eps,
        conditional=True,
        condition_dropout_prob=rollout_config.condition_dropout_prob,
        condition_type="vector",
    )

    out_dir = _EXAMPLES_DIR / "outputs" / "FM_lafan1_hierarchy" / "rollout"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(rollout_config), indent=2) + "\n"
    )

    num_samples = min(rollout_config.num_rollout_samples, len(rollout_dataset))
    sample_records: list[dict[str, Any]] = []
    final_goal_xy_errors: list[float] = []
    final_goal_yaw_errors: list[float] = []
    goal_reached_flags: list[float] = []
    flow_losses: list[float] = []
    goal_losses: list[float] = []
    root_vel_fd_mses: list[float] = []
    joint_vel_fd_mses: list[float] = []

    for sample_index in range(num_samples):
        sample = rollout_dataset[sample_index]
        condition = cast(torch.Tensor, sample["condition"]).unsqueeze(0).to(device)
        prior_mu, prior_logvar = high_model.encode_prior(condition)
        eps = torch.randn_like(prior_mu)
        z = prior_mu + torch.exp(0.5 * prior_logvar) * eps
        predicted_waypoints = high_model.decode(z, condition)[0].cpu()
        predicted_waypoints_low = _convert_waypoints_between_datasets(
            predicted_waypoints,
            rollout_dataset,
            local_dataset,
        )
        target_goal_low = _convert_waypoints_between_datasets(
            cast(torch.Tensor, sample["goal"]).unsqueeze(0),
            rollout_dataset,
            local_dataset,
        )[0]
        goals = torch.cat([predicted_waypoints_low, target_goal_low.unsqueeze(0)], dim=0)

        clip_idx = int(cast(torch.Tensor, sample["clip_index"]))
        frame_start = int(cast(torch.Tensor, sample["frame_start"]))
        chunk = base._clips[clip_idx][frame_start : frame_start + rollout_config.chunk.chunk_len].clone()
        target_local = base.make_relative(chunk, yaw_only=True)
        prefix = target_local[: rollout_config.chunk.prefix_frames]

        rollout, flow_loss, goal_loss = rollout_local_trajectory(
            flow,
            local_dataset,
            rollout_config,
            prefix,
            goals,
            device=device,
        )

        final_goal_xy_error = float(torch.norm(rollout[-1, :2] - target_local[-1, :2]).item())
        pred_yaw = _yaw_from_rot6d(rollout[-1, ROOT_ROT_OFFSET:POSE_BASE_DIM])
        target_yaw = _yaw_from_rot6d(target_local[-1, ROOT_ROT_OFFSET:POSE_BASE_DIM])
        final_goal_yaw_error = float(_wrapped_yaw_error(pred_yaw, target_yaw).item())
        vel_metrics = base.compute_metrics(rollout)
        goal_reached = (
            final_goal_xy_error <= rollout_config.goal_reach_xy_threshold
            and final_goal_yaw_error <= rollout_config.goal_reach_yaw_threshold_rad
        )
        final_goal_xy_errors.append(final_goal_xy_error)
        final_goal_yaw_errors.append(final_goal_yaw_error)
        goal_reached_flags.append(float(goal_reached))
        flow_losses.append(flow_loss)
        goal_losses.append(goal_loss)
        root_vel_fd_mses.append(vel_metrics["root_vel_fd_mse"])
        joint_vel_fd_mses.append(vel_metrics["joint_vel_fd_mse"])

        csv_qpos = base.trajectory_to_lafan1_csv_qpos(rollout)
        csv_name = f"sample_{sample_index:03d}.csv"
        np.savetxt(out_dir / csv_name, csv_qpos.numpy(), delimiter=",", fmt="%.8f")
        final_goal_yaw = float(_yaw_from_rot6d(target_local[-1, ROOT_ROT_OFFSET:POSE_BASE_DIM]).item())
        sample_records.append(
            {
                "sample_index": sample_index,
                "clip_index": clip_idx,
                "clip_name": cast(str, sample["clip_name"]),
                "frame_start": frame_start,
                "rollout_len": int(rollout.shape[0]),
                "relative_goal_state": {
                    "x": float(target_local[-1, 0].item()),
                    "y": float(target_local[-1, 1].item()),
                    "yaw_rad": final_goal_yaw,
                },
                "final_goal_xy_error": final_goal_xy_error,
                "final_goal_yaw_error_rad": final_goal_yaw_error,
                "goal_reached": goal_reached,
                "flow_loss": flow_loss,
                "goal_loss": goal_loss,
                "root_vel_fd_mse": vel_metrics["root_vel_fd_mse"],
                "joint_vel_fd_mse": vel_metrics["joint_vel_fd_mse"],
                "csv": csv_name,
            }
        )

    if not sample_records:
        raise RuntimeError("No rollout samples were generated")

    metrics: dict[str, float | int | str] = {
        "high_checkpoint": str(high_checkpoint),
        "low_checkpoint": str(low_checkpoint),
        "high_epoch": high_epoch,
        "low_epoch": low_epoch,
        "num_samples": len(sample_records),
        "final_goal_xy_error": float(np.mean(final_goal_xy_errors)),
        "final_goal_yaw_error_rad": float(np.mean(final_goal_yaw_errors)),
        "goal_reach_rate": float(np.mean(goal_reached_flags)),
        "flow_loss": float(np.mean(flow_losses)),
        "goal_loss": float(np.mean(goal_losses)),
        "root_vel_fd_mse": float(np.mean(root_vel_fd_mses)),
        "joint_vel_fd_mse": float(np.mean(joint_vel_fd_mses)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "samples.json").write_text(json.dumps(sample_records, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def parse_args() -> tuple[Config, bool]:
    default_chunk = ChunkConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train_high", "train_low", "rollout"], default="train_high")
    parser.add_argument("--data-root", default=Config.data_root)
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default=Config.robot)
    parser.add_argument("--chunk-len", type=int, default=default_chunk.chunk_len)
    parser.add_argument("--chunk-stride", type=int, default=default_chunk.chunk_stride)
    parser.add_argument("--prefix-frames", type=int, default=default_chunk.prefix_frames)
    parser.add_argument("--num-waypoints", type=int, default=default_chunk.num_waypoints)
    parser.add_argument("--latent-dim", type=int, default=Config.latent_dim)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--beta", type=float, default=Config.beta)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--train-epochs", type=int, default=Config.train_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--val-fraction", type=float, default=Config.val_fraction)
    parser.add_argument("--val-every", type=int, default=Config.val_every)
    parser.add_argument("--num-val-samples", type=int, default=Config.num_val_samples)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--num-threads", type=int, default=Config.num_threads)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=Config.use_wandb)
    parser.add_argument("--seq-len", type=int, default=Config.seq_len)
    parser.add_argument("--cond-steps", type=int, default=Config.cond_steps)
    parser.add_argument("--data-stride", type=int, default=Config.data_stride)
    parser.add_argument("--use-dit", action=argparse.BooleanOptionalAction, default=Config.use_dit)
    parser.add_argument("--base-channels", type=int, default=Config.base_channels)
    parser.add_argument("--cond-dim", type=int, default=Config.cond_dim)
    parser.add_argument("--max-seq-len", type=int, default=Config.max_seq_len)
    parser.add_argument("--hidden-size", type=int, default=Config.hidden_size)
    parser.add_argument("--depth", type=int, default=Config.depth)
    parser.add_argument("--num-heads", type=int, default=Config.num_heads)
    parser.add_argument("--mlp-ratio", type=float, default=Config.mlp_ratio)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument(
        "--use-cross-attention",
        action=argparse.BooleanOptionalAction,
        default=Config.use_cross_attention,
    )
    parser.add_argument("--pred-type", choices=["x", "eps", "v"], default=Config.pred_type)
    parser.add_argument("--loss-type", choices=["x", "eps", "v"], default=Config.loss_type)
    parser.add_argument("--goal-loss-weight", type=float, default=Config.goal_loss_weight)
    parser.add_argument("--noise-scale", type=float, default=Config.noise_scale)
    parser.add_argument("--t-eps", type=float, default=Config.t_eps)
    parser.add_argument("--sample-steps", type=int, default=Config.sample_steps)
    parser.add_argument(
        "--condition-dropout-prob",
        type=float,
        default=Config.condition_dropout_prob,
    )
    parser.add_argument("--num-plot-samples", type=int, default=Config.num_plot_samples)
    parser.add_argument("--num-rollout-samples", type=int, default=Config.num_rollout_samples)
    parser.add_argument("--goal-reach-xy-threshold", type=float, default=Config.goal_reach_xy_threshold)
    parser.add_argument(
        "--goal-reach-yaw-threshold-rad",
        type=float,
        default=Config.goal_reach_yaw_threshold_rad,
    )
    parser.add_argument("--high-checkpoint", default=Config.high_checkpoint)
    parser.add_argument("--low-checkpoint", default=Config.low_checkpoint)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_kwargs = {key: value for key, value in vars(args).items() if key != "resume"}
    config_kwargs["chunk"] = ChunkConfig(
        chunk_len=args.chunk_len,
        chunk_stride=args.chunk_stride,
        prefix_frames=args.prefix_frames,
        num_waypoints=args.num_waypoints,
    )
    del config_kwargs["chunk_len"]
    del config_kwargs["chunk_stride"]
    del config_kwargs["prefix_frames"]
    del config_kwargs["num_waypoints"]
    config = Config(**config_kwargs)
    return config, args.resume


def main() -> None:
    config, resume = parse_args()
    torch.set_num_threads(max(config.num_threads, 1))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config.mode == "train_high":
        train_high(config, device, resume)
        return
    if config.mode == "train_low":
        train_local(config, device, resume)
        return
    if config.mode == "rollout":
        run_rollout(config, device)
        return
    raise NotImplementedError(f"Unsupported mode {config.mode!r}")


if __name__ == "__main__":
    main()
