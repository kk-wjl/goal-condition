"""
Flow Matching on full LAFAN1 chunks with DiT goal conditioning.

This experiment generates one full chunk at a time instead of stitching sliding
windows. The first ``prefix_frames`` are pinned during training and sampling.
The goal is the chunk's final root ``(x, y, yaw)`` expressed relative to frame 0.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_EXAMPLES_DIR = Path(__file__).resolve().parent
try:
    from scripts.config_chunk import ChunkGoalLAFAN1Dataset, FullChunkConfig
except ModuleNotFoundError:
    from config_chunk import ChunkGoalLAFAN1Dataset, FullChunkConfig

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from goal_condition.datasets.lafan1 import POSE_BASE_DIM, ROOT_ROT_OFFSET, RobotName, rot6d_to_matrix
from goal_condition.flow_matching import (
    LossType,
    PredictionType,
    PredictionWrapper,
    compute_flow_matching_loss,
)
from goal_condition.modules.conditional_unet import ConditionalUNet1D
from goal_condition.modules.transformer import DiffusionTransformer1D
from goal_condition.utils.checkpoint import (
    load_training_checkpoint,
    read_training_checkpoint_config,
    save_training_checkpoint,
)


@dataclass
class Config:
    data_root: str = "./data"
    robot: RobotName = "g1"
    chunk: FullChunkConfig = field(default_factory=FullChunkConfig)
    batch_size: int = 64
    use_dit: bool = False
    base_channels: int = 320
    cond_dim: int = 640
    max_seq_len: int = 256
    hidden_size: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_cross_attention: bool = False
    num_threads: int = 1
    seed: int = 42
    train_epochs: int = 50
    lr: float = 3e-4
    noise_scale: float = 1.0
    t_eps: float = 1e-2
    sample_steps: int = 80
    num_plot_samples: int = 8
    pred_type: PredictionType = "v"
    loss_type: LossType = "v"
    condition_dropout_prob: float = 0.1
    cfg_scale: float = 2.0
    goal_loss_weight: float = 0.05
    yaw_loss_weight: float = 1.0
    use_wandb: bool = False


class TrajectoryGoalFlowBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
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
        self.sample_shape: tuple[int, int] = (1, input_dim)
        self.use_dit = use_dit
        self.use_cross_attention = use_cross_attention
        if use_dit:
            self.network: nn.Module = DiffusionTransformer1D(
                input_dim=input_dim,
                output_dim=input_dim,
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
                input_dim=input_dim,
                output_dim=input_dim,
                base_channels=base_channels,
                channel_mults=(1, 2, 4),
                cond_dim=cond_dim,
            )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        goal_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        if goal_cond is None:
            raise ValueError("goal_cond is required for goal-conditioned flow matching")
        if self.use_dit:
            if self.use_cross_attention:
                return self.network(x_t, cond=goal_cond, t=t, cond_tokens=goal_cond)
            return self.network(x_t, cond=goal_cond, t=t)
        return self.network(x_t, cond=goal_cond, t=t)


def build_model(config: Config, state_dim: int) -> nn.Module:
    backbone = TrajectoryGoalFlowBackbone(
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
    backbone.sample_shape = (config.chunk.chunk_len, state_dim)
    return PredictionWrapper(backbone, config.pred_type)


def _assert_resume_compatible(checkpoint_path: Path, config: Config) -> None:
    ckpt_config = read_training_checkpoint_config(checkpoint_path)
    checks: dict[str, Any] = {
        "chunk_len": config.chunk.chunk_len,
        "prefix_frames": config.chunk.prefix_frames,
        "use_dit": config.use_dit,
        "base_channels": config.base_channels,
        "cond_dim": config.cond_dim,
        "hidden_size": config.hidden_size,
        "depth": config.depth,
        "num_heads": config.num_heads,
        "mlp_ratio": config.mlp_ratio,
        "max_seq_len": config.max_seq_len,
        "use_cross_attention": config.use_cross_attention,
        "pred_type": config.pred_type,
        "loss_type": config.loss_type,
    }
    for key, expected in checks.items():
        actual = ckpt_config.get(key)
        if actual != expected:
            raise ValueError(
                f"Checkpoint {checkpoint_path} was trained with {key}={actual}, "
                f"but current config requests {key}={expected}."
            )


def _yaw_error_rad(pred_rot6d: torch.Tensor, target_rot6d: torch.Tensor) -> torch.Tensor:
    pred_mat = rot6d_to_matrix(pred_rot6d)
    target_mat = rot6d_to_matrix(target_rot6d)
    pred_yaw = torch.atan2(pred_mat[..., 1, 0], pred_mat[..., 0, 0])
    target_yaw = torch.atan2(target_mat[..., 1, 0], target_mat[..., 0, 0])
    err = pred_yaw - target_yaw
    return torch.atan2(torch.sin(err), torch.cos(err)).abs()


def _yaw_sincos(rot6d: torch.Tensor) -> torch.Tensor:
    mat = rot6d_to_matrix(rot6d)
    yaw = torch.atan2(mat[..., 1, 0], mat[..., 0, 0])
    return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


def compute_goal_flow_loss(
    model: nn.Module,
    x1: torch.Tensor,
    goal_cond: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype).clip(config.t_eps, 1.0 - config.t_eps)
    expand_shape = (-1,) + (x1.ndim - 1) * (1,)
    t_view = t.reshape(expand_shape)
    eps = torch.randn_like(x1) * config.noise_scale
    k = config.chunk.prefix_frames
    eps[:, :k] = x1[:, :k]
    x_t = t_view * x1 + (1.0 - t_view) * eps
    x_t[:, :k] = x1[:, :k]
    if config.condition_dropout_prob > 0.0:
        drop_mask = torch.rand(goal_cond.shape[0], device=goal_cond.device) < config.condition_dropout_prob
        goal_model_cond = goal_cond.clone()
        goal_model_cond[drop_mask] = 0.0
    else:
        goal_model_cond = goal_cond
    predictions = model(x_t, t, goal_model_cond)
    x1_hat, _v_hat, _eps_hat = predictions
    flow_loss = compute_flow_matching_loss(config.loss_type, x1[:, k:], eps[:, k:], tuple(p[:, k:] for p in predictions))
    final_xy_loss = torch.mean((x1_hat[:, -1, :2] - x1[:, -1, :2]) ** 2)
    final_yaw_loss = torch.mean(
        (_yaw_sincos(x1_hat[:, -1, ROOT_ROT_OFFSET:POSE_BASE_DIM]) - _yaw_sincos(x1[:, -1, ROOT_ROT_OFFSET:POSE_BASE_DIM])) ** 2
    )
    goal_loss = final_xy_loss + config.yaw_loss_weight * final_yaw_loss
    total_loss = flow_loss + config.goal_loss_weight * goal_loss
    return total_loss, {
        "flow_loss": float(flow_loss.detach().item()),
        "goal_loss": float(goal_loss.detach().item()),
        "final_xy_loss": float(final_xy_loss.detach().item()),
        "final_yaw_loss": float(final_yaw_loss.detach().item()),
    }


@torch.no_grad()
def sample_goal_cfg(
    model: nn.Module,
    cond_prefix: torch.Tensor,
    goal_cond: torch.Tensor,
    *,
    device: torch.device,
    config: Config,
) -> torch.Tensor:
    dtype = next(model.parameters()).dtype
    n, k, dim = cond_prefix.shape
    x_t = torch.randn((n, config.chunk.chunk_len, dim), device=device, dtype=dtype) * config.noise_scale
    cond_prefix = cond_prefix.to(device=device, dtype=dtype)
    goal_cond = goal_cond.to(device=device, dtype=dtype)
    uncond_goal = torch.zeros_like(goal_cond)
    x_t[:, :k] = cond_prefix
    ts = torch.linspace(config.t_eps, 1.0 - config.t_eps, config.sample_steps, device=device, dtype=dtype)
    dt = ts[1] - ts[0] if config.sample_steps > 1 else torch.tensor(1.0 - 2 * config.t_eps, device=device, dtype=dtype)
    for t_scalar in ts:
        t = torch.full((n,), t_scalar.item(), device=device, dtype=dtype)
        if config.cfg_scale != 1.0:
            _, v_cond, _ = model(x_t, t, goal_cond)
            _, v_uncond, _ = model(x_t, t, uncond_goal)
            v_hat = v_uncond + config.cfg_scale * (v_cond - v_uncond)
        else:
            _, v_hat, _ = model(x_t, t, goal_cond)
        v_hat = v_hat.clone()
        v_hat[:, :k] = 0.0
        x_t = x_t + v_hat * dt
        x_t[:, :k] = cond_prefix
    return x_t


@torch.no_grad()
def save_validation_rollouts_csv(
    model: nn.Module,
    config: Config,
    device: torch.device,
    dataset: ChunkGoalLAFAN1Dataset,
    reference_indices: list[int],
    out_dir: Path,
    epoch: int,
) -> dict[str, float | int | str]:
    model.eval()
    n = min(config.num_plot_samples, len(reference_indices))
    selected = reference_indices[:n]
    chunks = torch.stack([dataset[idx][0] for idx in selected], dim=0)
    goals = torch.stack([dataset[idx][1] for idx in selected], dim=0)
    chunks_rel = dataset.make_relative(chunks)
    x_norm = dataset.normalize(chunks_rel)
    cond_prefix = x_norm[:, : config.chunk.prefix_frames].to(device)
    sample_norm = sample_goal_cfg(model, cond_prefix, goals, device=device, config=config)
    sample_rel = dataset.denormalize(sample_norm).cpu()
    target_rel = chunks_rel.cpu()
    final_goal_xy_error = torch.norm(sample_rel[:, -1, :2] - target_rel[:, -1, :2], dim=-1).mean()
    final_goal_yaw_error = _yaw_error_rad(
        sample_rel[:, -1, ROOT_ROT_OFFSET:POSE_BASE_DIM],
        target_rel[:, -1, ROOT_ROT_OFFSET:POSE_BASE_DIM],
    ).mean()

    val_dir = out_dir / "validation" / f"epoch_{epoch:03d}"
    val_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        csv_qpos = dataset.trajectory_to_lafan1_csv_qpos(sample_rel[i])
        np.savetxt(val_dir / f"rollout_{i:03d}.csv", csv_qpos.numpy(), delimiter=",", fmt="%.8f")

    meta: dict[str, float | int | str] = {
        "epoch": epoch,
        "chunk_len": config.chunk.chunk_len,
        "prefix_frames": config.chunk.prefix_frames,
        "num_rollouts": n,
        "csv_dir": str(val_dir),
        "sample_mean": float(sample_rel.mean().item()),
        "sample_std": float(sample_rel.std().item()),
        "sample_min": float(sample_rel.min().item()),
        "sample_max": float(sample_rel.max().item()),
        "final_goal_xy_error": float(final_goal_xy_error.item()),
        "final_goal_yaw_error_rad": float(final_goal_yaw_error.item()),
    }
    meta.update(dataset.compute_metrics(sample_rel))
    (out_dir / f"fm_lafan1_goal_cond_epoch_{epoch:03d}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    default_chunk = FullChunkConfig()
    parser = argparse.ArgumentParser(description="Full-chunk LAFAN1 goal-conditioned DiT Flow Matching.")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default=Config.robot)
    parser.add_argument("--chunk-len", type=int, default=default_chunk.chunk_len)
    parser.add_argument("--chunk-stride", type=int, default=default_chunk.chunk_stride)
    parser.add_argument("--prefix-frames", type=int, default=default_chunk.prefix_frames)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--use-dit", action=argparse.BooleanOptionalAction, default=Config.use_dit)
    parser.add_argument("--base-channels", type=int, default=Config.base_channels)
    parser.add_argument("--cond-dim", type=int, default=Config.cond_dim)
    parser.add_argument("--max-seq-len", type=int, default=Config.max_seq_len)
    parser.add_argument("--hidden-size", type=int, default=Config.hidden_size)
    parser.add_argument("--depth", type=int, default=Config.depth)
    parser.add_argument("--num-heads", type=int, default=Config.num_heads)
    parser.add_argument("--mlp-ratio", type=float, default=Config.mlp_ratio)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--use-cross-attention", action="store_true")
    parser.add_argument("--num-threads", type=int, default=Config.num_threads)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-epochs", type=int, default=Config.train_epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--noise-scale", type=float, default=Config.noise_scale)
    parser.add_argument("--t-eps", type=float, default=Config.t_eps)
    parser.add_argument("--sample-steps", type=int, default=Config.sample_steps)
    parser.add_argument("--num-plot-samples", type=int, default=Config.num_plot_samples)
    parser.add_argument("--pred-type", choices=["x", "eps", "v"], default=Config.pred_type)
    parser.add_argument("--loss-type", choices=["x", "eps", "v"], default=Config.loss_type)
    parser.add_argument("--condition-dropout-prob", type=float, default=Config.condition_dropout_prob)
    parser.add_argument("--cfg-scale", type=float, default=Config.cfg_scale)
    parser.add_argument("--goal-loss-weight", type=float, default=Config.goal_loss_weight)
    parser.add_argument("--yaw-loss-weight", type=float, default=Config.yaw_loss_weight)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = Config(
        data_root=args.data_root,
        robot=args.robot,
        chunk=FullChunkConfig(
            chunk_len=args.chunk_len,
            chunk_stride=args.chunk_stride,
            prefix_frames=args.prefix_frames,
        ),
        batch_size=args.batch_size,
        use_dit=args.use_dit,
        base_channels=args.base_channels,
        cond_dim=args.cond_dim,
        max_seq_len=args.max_seq_len,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_cross_attention=args.use_cross_attention,
        num_threads=args.num_threads,
        seed=args.seed,
        train_epochs=args.train_epochs,
        lr=args.lr,
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
        sample_steps=args.sample_steps,
        num_plot_samples=args.num_plot_samples,
        pred_type=args.pred_type,
        loss_type=args.loss_type,
        condition_dropout_prob=args.condition_dropout_prob,
        cfg_scale=args.cfg_scale,
        goal_loss_weight=args.goal_loss_weight,
        yaw_loss_weight=args.yaw_loss_weight,
        use_wandb=not args.no_wandb,
    )

    torch.set_num_threads(max(config.num_threads, 1))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = ChunkGoalLAFAN1Dataset(
        root=config.data_root,
        robot=config.robot,
        chunk_len=config.chunk.chunk_len,
        chunk_stride=config.chunk.chunk_stride,
        download=True,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = build_model(config, dataset.state_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    wandb_run = None
    if config.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project="goal-condition",
            name="FM_lafan1_goal_cond",
            config=dataclasses.asdict(config),
        )

    out_dir = Path(__file__).resolve().parent / "outputs" / "FM_lafan1_full_chunk"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoint.pt"

    start_epoch = 0
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but no checkpoint at {checkpoint_path}")
        _assert_resume_compatible(checkpoint_path, config)
        start_epoch = load_training_checkpoint(checkpoint_path, model, optimizer, map_location=device)
        start_epoch += 1
        print(f"Resumed from {checkpoint_path}; training from epoch {start_epoch}")

    ref_indices: list[int] = []
    for epoch in range(start_epoch, config.train_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        losses: list[float] = []
        flow_losses: list[float] = []
        goal_losses: list[float] = []

        for batch, goal_cond, meta in pbar:
            if not ref_indices:
                ref_indices = [int(x) for x in meta["sample_index"][: config.num_plot_samples].tolist()]
            x = dataset.normalize(dataset.make_relative(batch.to(device)))
            goal_cond = goal_cond.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, loss_parts = compute_goal_flow_loss(model, x, goal_cond, config)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            flow_losses.append(loss_parts["flow_loss"])
            goal_losses.append(loss_parts["goal_loss"])
            pbar.set_postfix(loss=f"{loss.item():.5f}", flow=f"{loss_parts['flow_loss']:.5f}")

        if not ref_indices:
            ref_indices = list(range(min(config.num_plot_samples, len(dataset))))
        metrics = save_validation_rollouts_csv(model, config, device, dataset, ref_indices, out_dir, epoch)
        save_training_checkpoint(
            checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            config=config,
        )
        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_flow_loss = float(np.mean(flow_losses)) if flow_losses else 0.0
        avg_goal_loss = float(np.mean(goal_losses)) if goal_losses else 0.0
        print(
            f"epoch {epoch}: loss={avg_loss:.6f}, flow={avg_flow_loss:.6f}, "
            f"goal={avg_goal_loss:.6f}, xy_err={metrics['final_goal_xy_error']:.6f}, "
            f"yaw_err={metrics['final_goal_yaw_error_rad']:.6f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": avg_loss,
                    "train/flow_loss": avg_flow_loss,
                    "train/goal_loss": avg_goal_loss,
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "val/final_goal_xy_error": float(metrics["final_goal_xy_error"]),
                    "val/final_goal_yaw_error_rad": float(metrics["final_goal_yaw_error_rad"]),
                    "val/sample_std": float(metrics["sample_std"]),
                    "val/root_vel_fd_mse": float(metrics["root_vel_fd_mse"]),
                    "val/joint_vel_fd_mse": float(metrics["joint_vel_fd_mse"]),
                },
                step=epoch,
            )

    final_meta = save_validation_rollouts_csv(
        model,
        config,
        device,
        dataset,
        ref_indices or list(range(min(config.num_plot_samples, len(dataset)))),
        out_dir,
        config.train_epochs,
    )
    (out_dir / "fm_lafan1_goal_cond_metrics.json").write_text(json.dumps(final_meta, indent=2))
    if wandb_run is not None:
        wandb_run.log(
            {
                "final/final_goal_xy_error": float(final_meta["final_goal_xy_error"]),
                "final/final_goal_yaw_error_rad": float(final_meta["final_goal_yaw_error_rad"]),
                "final/root_vel_fd_mse": float(final_meta["root_vel_fd_mse"]),
                "final/joint_vel_fd_mse": float(final_meta["joint_vel_fd_mse"]),
            },
            step=config.train_epochs,
        )
        wandb_run.finish()
    print(json.dumps(final_meta, indent=2))
    print(f"Saved validation CSV rollouts under {out_dir / 'validation'}")
    print(f"Latest summary: {out_dir / 'fm_lafan1_goal_cond_metrics.json'}")


if __name__ == "__main__":
    main()
