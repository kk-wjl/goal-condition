"""
Flow Matching on LAFAN1-style robot trajectories with sparse goal conditioning.

Training samples are short windows cut from a longer chunk. The chunk's final frame
defines a sparse goal ``(x, y, yaw, delta_t_seconds)`` relative to the current
window's last frame. At inference time, the same goal is re-evaluated in a
receding-horizon manner while sliding windows are stitched into a long rollout.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent
try:
    from scripts.config_sliding import SlidingGoalLAFAN1Dataset, SlidingWindowConfig
except ModuleNotFoundError:
    from config_sliding import SlidingGoalLAFAN1Dataset, SlidingWindowConfig

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from goal_condition.datasets.lafan1 import (
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
from goal_condition.utils.checkpoint import (
    load_training_checkpoint,
    read_training_checkpoint_config,
    save_training_checkpoint,
)


@dataclass
class Config:
    data_root: str = "./data"
    robot: RobotName = "g1"
    sliding: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    chunk_len: int = 180
    chunk_stride: int = 30
    min_goal_gap: int = 1
    batch_size: int = 128
    base_channels: int = 128
    cond_dim: int = 128
    time_conditioning: bool = True
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
    cfg_scale: float = 3.0
    use_wandb: bool = True


class TrajectoryGoalFlowBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        base_channels: int,
        cond_dim: int,
        time_conditioning: bool = True,
    ):
        super().__init__()
        self.sample_shape: tuple[int, int] = (1, input_dim)
        self.cond_dim = cond_dim
        self.time_conditioning = time_conditioning
        self.unet = ConditionalUNet1D(
            input_dim=input_dim,
            output_dim=input_dim,
            base_channels=base_channels,
            channel_mults=(1, 2, 4),
            cond_dim=cond_dim,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, goal_cond: torch.Tensor | None) -> torch.Tensor:
        if goal_cond is None:
            raise ValueError("goal_cond is required for goal-conditioned flow matching")
        if self.time_conditioning:
            return self.unet(x_t, cond=goal_cond, t=t)
        return self.unet(x_t, cond=goal_cond)


def build_model(config: Config, state_dim: int, seq_len: int) -> nn.Module:
    base_network = TrajectoryGoalFlowBackbone(
        input_dim=state_dim,
        base_channels=config.base_channels,
        cond_dim=config.cond_dim,
        time_conditioning=config.time_conditioning,
    )
    base_network.sample_shape = (seq_len, state_dim)
    return PredictionWrapper(base_network, config.pred_type)


def _assert_resume_compatible(checkpoint_path: Path, config: Config) -> None:
    ckpt_config = read_training_checkpoint_config(checkpoint_path)
    checks = {
        "time_conditioning": config.time_conditioning,
        "chunk_len": config.chunk_len,
        "chunk_stride": config.chunk_stride,
        "min_goal_gap": config.min_goal_gap,
    }
    for key, expected in checks.items():
        actual = ckpt_config.get(key)
        if actual != expected:
            raise ValueError(
                f"Checkpoint {checkpoint_path} was trained with {key}={actual}, "
                f"but current config requests {key}={expected}."
            )


def _goal_frames_in_chunk_root_frame(
    dataset: SlidingGoalLAFAN1Dataset,
    cond_prefix: torch.Tensor,
    goal_frames: torch.Tensor,
) -> torch.Tensor:
    pair = torch.stack([cond_prefix[:, 0], goal_frames], dim=1)
    return dataset.make_relative(pair)[:, 1]


@torch.no_grad()
def sliding_window_generate(
    flow: LinearFlow,
    dataset: SlidingGoalLAFAN1Dataset,
    config: Config,
    device: torch.device,
    cond_prefix: Float[Tensor, "batch cond dim"],
    goal_frames: Float[Tensor, "batch dim"],
    total_len: int,
    window_stride: int,
    cfg_scale: float | None = None,
) -> Float[Tensor, "batch total dim"]:
    k = config.sliding.cond_steps
    seq_len = config.sliding.seq_len
    if cond_prefix.ndim != 3 or cond_prefix.shape[1] != k:
        raise ValueError(
            f"cond_prefix must be (N, {k}, D), got {tuple(cond_prefix.shape)}"
        )
    if total_len < k:
        raise ValueError(f"total_len ({total_len}) must be >= cond_steps ({k})")
    max_stride = seq_len - k
    if not (1 <= window_stride <= max_stride):
        raise ValueError(
            f"window_stride must be in [1, {max_stride}], got {window_stride}"
        )

    dtype = next(flow.model.parameters()).dtype
    cond_prefix = cond_prefix.to(device=device, dtype=dtype)
    goal_frames = goal_frames.to(device=device, dtype=dtype)
    n, _, dim = cond_prefix.shape
    traj = torch.zeros(n, total_len, dim, device=device, dtype=dtype)
    cond0 = dataset.make_relative(cond_prefix.to(device=device, dtype=dtype))
    traj[:, :k] = cond0
    goal_frames_root = _goal_frames_in_chunk_root_frame(dataset, cond_prefix, goal_frames)

    ws = 0
    while True:
        if ws + k > total_len:
            break
        traj_ws = traj[:, ws : ws + k]
        cond_local = dataset.normalize(dataset.make_relative(traj_ws))
        root_pos_ref = traj[:, ws, :3]
        root_rot6d_ref = traj[:, ws, ROOT_ROT_OFFSET:POSE_BASE_DIM]
        current_frame = traj[:, ws + k - 1]
        remaining_frames = total_len - 1 - (ws + k - 1)
        goal_cond = dataset.goal_condition_from_states(
            current_frame,
            goal_frames_root,
            remaining_frames,
        ).to(device=device, dtype=dtype)

        if cfg_scale is not None and cfg_scale != 1.0:
            chunk = flow.sample_cfg_cond_prefix(
                cond_local,
                goal_cond,
                device,
                config.sample_steps,
                cfg_scale,
            )
        else:
            chunk = flow.sample_cond_prefix(
                cond_local,
                device,
                config.sample_steps,
                y=goal_cond,
            )
        n_write = min(seq_len, total_len - ws)
        chunk_phys = dataset.denormalize(chunk[:, :n_write])
        chunk_merged_phys = dataset.accumulate_chunk_in_root_frame(
            chunk_phys,
            root_pos_ref,
            root_rot6d_ref,
        )
        traj[:, ws : ws + n_write] = chunk_merged_phys
        if ws + n_write >= total_len:
            break
        ws += window_stride
    return traj


def _yaw_error_rad(pred_rot6d: torch.Tensor, target_rot6d: torch.Tensor) -> torch.Tensor:
    pred_mat = rot6d_to_matrix(pred_rot6d)
    target_mat = rot6d_to_matrix(target_rot6d)
    pred_yaw = torch.atan2(pred_mat[..., 1, 0], pred_mat[..., 0, 0])
    target_yaw = torch.atan2(target_mat[..., 1, 0], target_mat[..., 0, 0])
    err = pred_yaw - target_yaw
    return torch.atan2(torch.sin(err), torch.cos(err)).abs()


@torch.no_grad()
def save_validation_rollouts_csv(
    flow: LinearFlow,
    config: Config,
    device: torch.device,
    dataset: SlidingGoalLAFAN1Dataset,
    reference_indices: list[int],
    out_dir: Path,
    epoch: int,
) -> dict[str, float | int | str]:
    flow.model.eval()
    n = min(config.num_plot_samples, len(reference_indices))
    selected = reference_indices[:n]
    prefix_batch = torch.stack([dataset.get_chunk_prefix(idx) for idx in selected], dim=0)
    goal_batch = torch.stack([dataset.get_chunk_goal_frame(idx) for idx in selected], dim=0)
    total_len = config.chunk_len
    stride = config.sliding.val_window_stride
    assert stride is not None
    traj = sliding_window_generate(
        flow,
        dataset,
        config,
        device,
        prefix_batch,
        goal_batch,
        total_len,
        stride,
        cfg_scale=config.cfg_scale,
    )
    traj_denorm = traj.detach().cpu()
    goal_root = _goal_frames_in_chunk_root_frame(dataset, prefix_batch, goal_batch).cpu()
    final_goal_xy_error = torch.norm(traj_denorm[:, -1, :2] - goal_root[:, :2], dim=-1).mean()
    final_goal_yaw_error = _yaw_error_rad(
        traj_denorm[:, -1, ROOT_ROT_OFFSET:POSE_BASE_DIM],
        goal_root[:, ROOT_ROT_OFFSET:POSE_BASE_DIM],
    ).mean()

    val_dir = out_dir / "validation" / f"epoch_{epoch:03d}"
    val_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        csv_qpos = dataset.trajectory_to_lafan1_csv_qpos(traj_denorm[i])
        np.savetxt(val_dir / f"rollout_{i:03d}.csv", csv_qpos.numpy(), delimiter=",", fmt="%.8f")

    meta: dict[str, float | int | str] = {
        "epoch": epoch,
        "chunk_len": total_len,
        "val_window_stride": int(stride),
        "num_rollouts": n,
        "csv_dir": str(val_dir),
        "sample_mean": float(traj_denorm.mean().item()),
        "sample_std": float(traj_denorm.std().item()),
        "sample_min": float(traj_denorm.min().item()),
        "sample_max": float(traj_denorm.max().item()),
        "final_goal_xy_error": float(final_goal_xy_error.item()),
        "final_goal_yaw_error_rad": float(final_goal_yaw_error.item()),
    }
    meta.update(dataset.compute_metrics(traj_denorm))
    (out_dir / f"fm_lafan1_goal_cond_epoch_{epoch:03d}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="LAFAN1 goal-conditioned Flow Matching.")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default=Config.robot)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--base-channels", type=int, default=Config.base_channels)
    parser.add_argument("--cond-dim", type=int, default=Config.cond_dim)
    parser.add_argument("--chunk-len", type=int, default=Config.chunk_len)
    parser.add_argument("--chunk-stride", type=int, default=Config.chunk_stride)
    parser.add_argument("--min-goal-gap", type=int, default=Config.min_goal_gap)
    parser.add_argument(
        "--time-conditioning",
        action=argparse.BooleanOptionalAction,
        default=Config.time_conditioning,
        help="Enable scalar flow-time conditioning in the trajectory U-Net.",
    )
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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pt path (default: scripts/outputs/FM_lafan1_sliding_window/checkpoint.pt).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load weights, optimizer, and RNG from --checkpoint and continue.",
    )
    args = parser.parse_args()

    config = Config(
        data_root=args.data_root,
        robot=args.robot,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        cond_dim=args.cond_dim,
        chunk_len=args.chunk_len,
        chunk_stride=args.chunk_stride,
        min_goal_gap=args.min_goal_gap,
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
        time_conditioning=args.time_conditioning,
    )

    torch.set_num_threads(max(config.num_threads, 1))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = SlidingGoalLAFAN1Dataset(
        root=config.data_root,
        robot=config.robot,
        seq_len=config.sliding.seq_len,
        cond_steps=config.sliding.cond_steps,
        chunk_len=config.chunk_len,
        chunk_stride=config.chunk_stride,
        stride=config.sliding.stride,
        min_goal_gap=config.min_goal_gap,
        download=True,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    state_dim = dataset.state_dim

    model = build_model(config, state_dim, config.sliding.seq_len).to(device)
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

    wandb_run = None
    if config.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project="goal-condition",
            name="FM_lafan1_goal_cond",
            config=dataclasses.asdict(config),
        )

    out_dir = Path(__file__).resolve().parent / "outputs" / "FM_lafan1_sliding_window"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoint.pt"

    start_epoch = 0
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but no checkpoint at {checkpoint_path}")
        _assert_resume_compatible(checkpoint_path, config)
        start_epoch = load_training_checkpoint(checkpoint_path, model, optimizer)
        start_epoch += 1
        print(f"Resumed from {checkpoint_path}; training from epoch {start_epoch}")

    ref_indices: list[int] = []
    for epoch in range(start_epoch, config.train_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        losses: list[float] = []

        for batch, goal_cond, meta in pbar:
            if not ref_indices:
                ref_indices = [int(x) for x in meta["sample_index"][: config.num_plot_samples].tolist()]
            x = dataset.normalize(dataset.make_relative(batch.to(device)))
            goal_cond = goal_cond.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = flow.compute_loss(x, y=goal_cond, cond_steps=config.sliding.cond_steps)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        if not ref_indices:
            ref_indices = list(range(min(config.num_plot_samples, len(dataset))))
        metrics = save_validation_rollouts_csv(
            flow,
            config,
            device,
            dataset,
            ref_indices,
            out_dir,
            epoch,
        )
        save_training_checkpoint(
            checkpoint_path, epoch=epoch, model=model, optimizer=optimizer, config=config
        )
        if losses:
            avg_loss = float(np.mean(losses))
            print(
                f"epoch {epoch}: "
                f"loss={avg_loss:.6f}, "
                f"goal_xy_err={metrics['final_goal_xy_error']:.6f}, "
                f"goal_yaw_err_rad={metrics['final_goal_yaw_error_rad']:.6f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "val/final_goal_xy_error": float(metrics["final_goal_xy_error"]),
                        "val/final_goal_yaw_error_rad": float(metrics["final_goal_yaw_error_rad"]),
                        "val/sample_mean": float(metrics["sample_mean"]),
                        "val/sample_std": float(metrics["sample_std"]),
                        "val/root_vel_fd_mse": float(metrics["root_vel_fd_mse"]),
                        "val/joint_vel_fd_mse": float(metrics["joint_vel_fd_mse"]),
                    },
                    step=epoch,
                )

    final_meta = save_validation_rollouts_csv(
        flow,
        config,
        device,
        dataset,
        ref_indices or list(range(min(config.num_plot_samples, len(dataset)))),
        out_dir,
        config.train_epochs,
    )
    (out_dir / "fm_goal_cond_metrics.json").write_text(json.dumps(final_meta, indent=2))
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
    print(f"Latest summary: {out_dir / 'fm_goal_cond_metrics.json'}")


if __name__ == "__main__":
    main()
