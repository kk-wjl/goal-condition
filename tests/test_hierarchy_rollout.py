from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast

import numpy as np
import torch
import torch.optim as optim

from scripts.FM_lafan1_hierarchy import (
    WaypointCVAE,
    _goal_frame_from_waypoint,
    _wrapped_yaw_error,
    _yaw_from_rot6d,
    build_local_model,
    compute_flow_loss,
)
from scripts.config_hierarchy import ChunkConfig, LocalGoalLAFAN1Dataset, WAYPOINT_DIM, WaypointLAFAN1Dataset

from goal_condition.datasets.lafan1 import LAFAN1Dataset, POSE_BASE_DIM, ROOT_ROT_OFFSET, RobotName
from goal_condition.flow_matching import LinearFlow
from goal_condition.utils.checkpoint import load_training_checkpoint, read_training_checkpoint_config


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_HIGH_CHECKPOINT = (
    _ROOT / "scripts" / "outputs" / "FM_lafan1_hierarchy" / "high_level" / "checkpoint.pt"
)
_DEFAULT_LOW_CHECKPOINT = (
    _ROOT / "scripts" / "outputs" / "FM_lafan1_hierarchy" / "low_level" / "checkpoint.pt"
)
_DEFAULT_OUTPUT_DIR = _ROOT / "tests" / "outputs" / "hierarchy_rollout"
_MANUAL_GOALS_XY_YAW: tuple[tuple[float, float, float], ...] = (
    (5.0, 0.0, 0.0),
    (-5.0, 0.0, np.pi),
    (0.0, 5.0, np.pi / 2),
    (0.0, -5.0, -np.pi / 2),
    (5.0, 5.0, np.pi / 4),
    (5.0, -5.0, -np.pi / 4),
    (-5.0, -5.0, -3 * np.pi / 4),
    (-5.0, 5.0, 3 * np.pi / 4),
)

_NUMBERED_OUTPUT_DIR_RE = re.compile(r"^(?P<stem>.+)_(?P<index>\d+)$")


def _resolve_output_dir(output_dir: Path) -> Path:
    match = _NUMBERED_OUTPUT_DIR_RE.fullmatch(output_dir.name)
    if match is not None:
        return output_dir

    parent = output_dir
    stem = output_dir.name
    next_index = 0

    if parent.exists():
        for child in parent.iterdir():
            if not child.is_dir():
                continue
            child_match = _NUMBERED_OUTPUT_DIR_RE.fullmatch(child.name)
            if child_match is None or child_match.group("stem") != stem:
                continue
            next_index = max(next_index, int(child_match.group("index")) + 1)

    return parent / f"{stem}_{next_index:03d}"


def _load_raw_checkpoint_config(path: Path) -> dict[str, Any]:
    raw = dict(read_training_checkpoint_config(path))
    for deprecated_key in ("num_plot_cases", "num_prior_samples", "num_interp_steps"):
        raw.pop(deprecated_key, None)
    return raw


def _make_high_model_config(raw: dict[str, Any]) -> SimpleNamespace:
    chunk_raw = cast(dict[str, Any], raw["chunk"])
    return SimpleNamespace(
        data_root=cast(str, raw["data_root"]),
        robot=cast(str, raw["robot"]),
        chunk=ChunkConfig(**chunk_raw),
        latent_dim=int(raw["latent_dim"]),
        hidden_dim=int(raw["hidden_dim"]),
        lr=float(raw["lr"]),
        val_fraction=float(raw["val_fraction"]),
        seed=int(raw["seed"]),
        batch_size=int(raw["batch_size"]),
    )


def _make_low_model_config(raw: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        seq_len=int(raw["seq_len"]),
        cond_steps=int(raw["cond_steps"]),
        data_stride=int(raw["data_stride"]),
        use_dit=bool(raw["use_dit"]),
        base_channels=int(raw["base_channels"]),
        cond_dim=int(raw["cond_dim"]),
        max_seq_len=int(raw["max_seq_len"]),
        hidden_size=int(raw["hidden_size"]),
        depth=int(raw["depth"]),
        num_heads=int(raw["num_heads"]),
        mlp_ratio=float(raw["mlp_ratio"]),
        dropout=float(raw["dropout"]),
        use_cross_attention=bool(raw["use_cross_attention"]),
        pred_type=cast(str, raw["pred_type"]),
        loss_type=cast(str, raw["loss_type"]),
        goal_loss_weight=float(raw["goal_loss_weight"]),
        noise_scale=float(raw["noise_scale"]),
        t_eps=float(raw["t_eps"]),
        sample_steps=int(raw["sample_steps"]),
        condition_dropout_prob=float(raw["condition_dropout_prob"]),
        lr=float(raw["lr"]),
    )


def _convert_high_to_low_waypoint_space(
    waypoints: torch.Tensor,
    high_dataset: WaypointLAFAN1Dataset,
    low_dataset: LocalGoalLAFAN1Dataset,
) -> torch.Tensor:
    """Map high-level waypoint xy from high-dataset normalization into low-dataset normalization."""
    xy_m = high_dataset.waypoint_xy_to_meters(waypoints[..., :2])
    low_xy_scale = low_dataset.base._root_pos_std[:2].to(device=waypoints.device, dtype=waypoints.dtype)
    yaw_vec = torch.nn.functional.normalize(waypoints[..., 2:4], dim=-1, eps=1e-6)
    return torch.cat([xy_m / low_xy_scale, yaw_vec], dim=-1)


def _build_manual_goal_waypoint(
    goal_x: float,
    goal_y: float,
    goal_yaw: float,
    low_dataset: LocalGoalLAFAN1Dataset,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    low_xy_scale = low_dataset.base._root_pos_std[:2].to(device=device, dtype=dtype)
    return torch.tensor(
        [
            goal_x / float(low_xy_scale[0].item()),
            goal_y / float(low_xy_scale[1].item()),
            np.sin(goal_yaw),
            np.cos(goal_yaw),
        ],
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def _rollout_local_trajectory(
    flow: LinearFlow,
    dataset: LocalGoalLAFAN1Dataset,
    low_config: SimpleNamespace,
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
        current_prefix = rollout[-low_config.cond_steps :]
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
            low_config.sample_steps,
            y=goal_cond,
        )
        _, flow_loss, endpoint_loss = compute_flow_loss(
            flow,
            generated_norm,
            goal_cond,
            low_config.cond_steps,
            low_config.goal_loss_weight,
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
        rollout = torch.cat([rollout, generated_global[low_config.cond_steps :]], dim=0)

    avg_flow_loss = float(np.mean(flow_losses)) if flow_losses else 0.0
    avg_goal_loss = float(np.mean(goal_losses)) if goal_losses else 0.0
    return rollout.cpu(), avg_flow_loss, avg_goal_loss


@torch.no_grad()
def evaluate_rollout(
    *,
    high_checkpoint: Path,
    low_checkpoint: Path,
    output_dir: Path,
    num_samples: int,
    split: str,
    data_root: str | None,
    robot: RobotName | None,
    seed: int,
) -> dict[str, float | int | str]:
    high_raw = _load_raw_checkpoint_config(high_checkpoint)
    low_raw = _load_raw_checkpoint_config(low_checkpoint)
    high_config = _make_high_model_config(high_raw)
    low_config = _make_low_model_config(low_raw)

    if data_root is not None:
        high_config.data_root = data_root
    if robot is not None:
        high_config.robot = robot

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base = LAFAN1Dataset(
        root=high_config.data_root,
        robot=high_config.robot,
        seq_len=high_config.chunk.chunk_len,
        stride=high_config.chunk.chunk_stride,
        download=True,
    )
    rollout_dataset = WaypointLAFAN1Dataset(
        base,
        split=cast(str, split),
        chunk_len=high_config.chunk.chunk_len,
        chunk_stride=high_config.chunk.chunk_stride,
        prefix_frames=high_config.chunk.prefix_frames,
        num_waypoints=high_config.chunk.num_waypoints,
        val_fraction=high_config.val_fraction,
        split_seed=high_config.seed,
    )
    local_dataset = LocalGoalLAFAN1Dataset(
        root=high_config.data_root,
        split=cast(str, split),
        robot=high_config.robot,
        seq_len=low_config.seq_len,
        cond_steps=low_config.cond_steps,
        stride=low_config.data_stride,
        val_fraction=high_config.val_fraction,
        split_seed=high_config.seed,
        download=True,
    )

    high_model = WaypointCVAE(
        condition_dim=rollout_dataset.condition_dim,
        num_waypoints=high_config.chunk.num_waypoints,
        latent_dim=high_config.latent_dim,
        hidden_dim=high_config.hidden_dim,
    ).to(device)
    high_optimizer = optim.AdamW(high_model.parameters(), lr=high_config.lr)
    high_epoch = load_training_checkpoint(high_checkpoint, high_model, high_optimizer, map_location=device)
    high_model.eval()

    low_model = build_local_model(low_config, local_dataset.state_dim).to(device)
    low_optimizer = optim.AdamW(low_model.parameters(), lr=low_config.lr)
    low_epoch = load_training_checkpoint(low_checkpoint, low_model, low_optimizer, map_location=device)
    low_model.eval()
    flow = LinearFlow(
        low_model,
        noise_scale=low_config.noise_scale,
        loss_type=low_config.loss_type,
        t_eps=low_config.t_eps,
        conditional=True,
        condition_dropout_prob=low_config.condition_dropout_prob,
        condition_type="vector",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_records: list[dict[str, object]] = []
    final_goal_xy_errors: list[float] = []
    final_goal_yaw_errors: list[float] = []

    total_samples = min(num_samples, len(rollout_dataset), len(_MANUAL_GOALS_XY_YAW))

    for sample_index in range(total_samples):
        sample = rollout_dataset[sample_index]
        condition = cast(torch.Tensor, sample["condition"]).unsqueeze(0).to(device)
        target_waypoints = cast(torch.Tensor, sample["waypoints"])

        prior_mu, _prior_logvar = high_model.encode_prior(condition)
        predicted_waypoints = high_model.decode(prior_mu, condition)[0]
        predicted_waypoints_low = _convert_high_to_low_waypoint_space(
            predicted_waypoints,
            rollout_dataset,
            local_dataset,
        )
        goal_x, goal_y, goal_yaw = _MANUAL_GOALS_XY_YAW[sample_index]
        target_goal_low = _build_manual_goal_waypoint(
            goal_x,
            goal_y,
            goal_yaw,
            local_dataset,
            device=device,
            dtype=predicted_waypoints_low.dtype,
        )
        goals = torch.cat([predicted_waypoints_low, target_goal_low.unsqueeze(0)], dim=0)

        clip_idx = int(cast(torch.Tensor, sample["clip_index"]))
        frame_start = int(cast(torch.Tensor, sample["frame_start"]))
        chunk = base._clips[clip_idx][frame_start : frame_start + high_config.chunk.chunk_len].clone()
        target_local = base.make_relative(chunk, yaw_only=True)
        prefix = target_local[: high_config.chunk.prefix_frames]

        rollout, flow_loss, goal_loss = _rollout_local_trajectory(
            flow,
            local_dataset,
            low_config,
            prefix,
            goals,
            device=device,
        )

        target_goal_frame = _goal_frame_from_waypoint(
            prefix[-1].to(device),
            target_goal_low,
            local_dataset.base._root_pos_std[:2].to(device=device, dtype=prefix.dtype),
        ).cpu()
        final_goal_xy_error = float(torch.norm(rollout[-1, :2] - target_goal_frame[:2]).item())
        pred_yaw = _yaw_from_rot6d(rollout[-1, ROOT_ROT_OFFSET:POSE_BASE_DIM])
        target_yaw = _yaw_from_rot6d(target_goal_frame[ROOT_ROT_OFFSET:POSE_BASE_DIM])
        final_goal_yaw_error = float(_wrapped_yaw_error(pred_yaw, target_yaw).item())
        final_goal_xy_errors.append(final_goal_xy_error)
        final_goal_yaw_errors.append(final_goal_yaw_error)

        csv_qpos = base.trajectory_to_lafan1_csv_qpos(rollout)
        csv_name = f"sample_{sample_index:03d}.csv"
        np.savetxt(output_dir / csv_name, csv_qpos.numpy(), delimiter=",", fmt="%.8f")
        predicted_waypoints_cpu = predicted_waypoints.cpu()
        predicted_waypoints_low_cpu = predicted_waypoints_low.cpu()
        target_goal_low_cpu = target_goal_low.cpu()
        sample_records.append(
            {
                "sample_index": sample_index,
                "clip_index": clip_idx,
                "clip_name": cast(str, sample["clip_name"]),
                "frame_start": frame_start,
                "goal_xy": [goal_x, goal_y],
                "goal_yaw_rad": goal_yaw,
                "generated_waypoint_xy": rollout_dataset.waypoint_xy_to_meters(predicted_waypoints_cpu).cpu().tolist(),
                "target_waypoint_xy": rollout_dataset.waypoint_xy_to_meters(target_waypoints).cpu().tolist(),
                "generated_waypoint_low_space": predicted_waypoints_low_cpu.tolist(),
                "goal_low_space": target_goal_low_cpu.tolist(),
                "generated_waypoint_yaw_rad": torch.atan2(
                    predicted_waypoints_cpu[:, 2], predicted_waypoints_cpu[:, 3]
                ).cpu().tolist(),
                "target_waypoint_yaw_rad": torch.atan2(
                    target_waypoints[:, 2], target_waypoints[:, 3]
                ).cpu().tolist(),
                "predicted_waypoints": predicted_waypoints_cpu.tolist(),
                "rollout_len": int(rollout.shape[0]),
                "final_goal_xy_error": final_goal_xy_error,
                "final_goal_yaw_error_rad": final_goal_yaw_error,
                "flow_loss": flow_loss,
                "goal_loss": goal_loss,
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
        "split": split,
        "num_samples": len(sample_records),
        "mean_final_goal_xy_error": float(np.mean(final_goal_xy_errors)),
        "mean_final_goal_yaw_error_rad": float(np.mean(final_goal_yaw_errors)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "samples.json").write_text(json.dumps(sample_records, indent=2) + "\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom hierarchy rollout with hand-crafted goals.")
    parser.add_argument("--high-checkpoint", type=Path, default=_DEFAULT_HIGH_CHECKPOINT)
    parser.add_argument("--low-checkpoint", type=Path, default=_DEFAULT_LOW_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=len(_MANUAL_GOALS_XY_YAW))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args.output_dir.resolve())
    metrics = evaluate_rollout(
        high_checkpoint=args.high_checkpoint.resolve(),
        low_checkpoint=args.low_checkpoint.resolve(),
        output_dir=output_dir,
        num_samples=args.num_samples,
        split=args.split,
        data_root=args.data_root,
        robot=cast(RobotName | None, args.robot),
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


def test_custom_hierarchy_rollout_smoke() -> None:
    assert callable(evaluate_rollout)
    assert len(_MANUAL_GOALS_XY_YAW) == 8
    assert _resolve_output_dir(_DEFAULT_OUTPUT_DIR) == _DEFAULT_OUTPUT_DIR / "hierarchy_rollout_000"
    assert _resolve_output_dir(_DEFAULT_OUTPUT_DIR.parent / "hierarchy_rollout_007").name == "hierarchy_rollout_007"


if __name__ == "__main__":
    main()
