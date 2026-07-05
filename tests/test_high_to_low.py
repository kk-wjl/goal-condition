from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast

import numpy as np
import torch
import torch.optim as optim

from scripts.FM_lafan1_hierarchy import (
    _goal_frame_from_waypoint,
    _wrapped_yaw_error,
    _yaw_from_rot6d,
    build_local_model,
    compute_flow_loss,
)
from scripts.config_hierarchy import LocalGoalLAFAN1Dataset

from goal_condition.datasets.lafan1 import LAFAN1Dataset, POSE_BASE_DIM, ROOT_ROT_OFFSET, RobotName
from goal_condition.flow_matching import LinearFlow
from goal_condition.utils.checkpoint import load_training_checkpoint, read_training_checkpoint_config


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LOW_CHECKPOINT = (
    _ROOT / "scripts" / "outputs" / "FM_lafan1_hierarchy" / "low_level" / "checkpoint.pt"
)
_DEFAULT_OUTPUT_DIR = _ROOT / "tests" / "outputs" / "high_to_low"
_MANUAL_WAYPOINTS_XY_YAW: tuple[tuple[float, float, float], ...] = (
    (-0.00462860194966197, 0.0019111699657514691, -0.002605961635708809),
    (-0.004471862688660622, -0.0026719120796769857, -0.0008691298426128924),
    (-0.009759079664945602, -0.001222662627696991, 0.007955902256071568),
    (-0.017355872318148613, 0.01523163914680481, -0.04735814407467842),
    (-0.1613454520702362, 0.010113338008522987, 0.39698103070259094),
    (-0.6573478579521179, 0.022184256464242935, 0.3160480260848999),
    (-1.3060988187789917, 0.0025110915303230286, 0.3070533573627472),
    (-1.9133732318878174, -0.013475187122821808, 0.3731575608253479),
)
_FINAL_GOAL_XY_YAW: tuple[float, float, float] = (
    -2.194443702697754,
    0.021282322704792023,
    -0.07173439860343933,
)


def _load_raw_checkpoint_config(path: Path) -> dict[str, Any]:
    raw = dict(read_training_checkpoint_config(path))
    for deprecated_key in ("num_plot_cases", "num_prior_samples", "num_interp_steps"):
        raw.pop(deprecated_key, None)
    return raw


def _make_low_model_config(raw: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=cast(str, raw["data_root"]),
        robot=cast(str, raw["robot"]),
        val_fraction=float(raw["val_fraction"]),
        seed=int(raw["seed"]),
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


def _make_goal_waypoint(
    *,
    x_m: float,
    y_m: float,
    yaw_rad: float,
    xy_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.tensor(
        [
            x_m / float(xy_scale[0].item()),
            y_m / float(xy_scale[1].item()),
            np.sin(yaw_rad),
            np.cos(yaw_rad),
        ],
        dtype=xy_scale.dtype,
        device=xy_scale.device,
    )


@torch.no_grad()
def evaluate_low_level_waypoints(
    *,
    low_checkpoint: Path,
    output_dir: Path,
    split: str,
    sample_index: int,
    data_root: str | None,
    robot: RobotName | None,
) -> dict[str, float | int | str]:
    low_raw = _load_raw_checkpoint_config(low_checkpoint)
    config = _make_low_model_config(low_raw)
    if data_root is not None:
        config.data_root = data_root
    if robot is not None:
        config.robot = robot

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = LocalGoalLAFAN1Dataset(
        root=config.data_root,
        split=cast(str, split),
        robot=config.robot,
        seq_len=config.seq_len,
        cond_steps=config.cond_steps,
        stride=config.data_stride,
        val_fraction=config.val_fraction,
        split_seed=config.seed,
        download=True,
    )

    model = build_local_model(config, dataset.state_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    epoch = load_training_checkpoint(low_checkpoint, model, optimizer, map_location=device)
    model.eval()
    flow = LinearFlow(
        model,
        noise_scale=config.noise_scale,
        loss_type=config.loss_type,
        t_eps=config.t_eps,
        conditional=True,
        condition_dropout_prob=config.condition_dropout_prob,
        condition_type="vector",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = len(dataset)
    if not (0 <= sample_index < sample_count):
        raise IndexError(f"sample_index must be in [0, {sample_count}), got {sample_index}")

    trajectory, _goal_cond, meta = dataset[sample_index]
    target_local = dataset.make_relative(trajectory, yaw_only=True)
    prefix = target_local[: config.cond_steps]
    prefix_anchor = prefix[-1]
    xy_scale = dataset.base._root_pos_std[:2].to(device)

    goals_spec = [*_MANUAL_WAYPOINTS_XY_YAW, _FINAL_GOAL_XY_YAW]
    goal_waypoints = torch.stack(
        [
            _make_goal_waypoint(
                x_m=goal_x_m,
                y_m=goal_y_m,
                yaw_rad=goal_yaw_rad,
                xy_scale=xy_scale,
            )
            for goal_x_m, goal_y_m, goal_yaw_rad in goals_spec
        ],
        dim=0,
    )

    rollout = prefix.to(device)
    step_records: list[dict[str, object]] = []
    flow_losses: list[float] = []
    endpoint_losses: list[float] = []

    for goal_index, (goal_waypoint, goal_spec) in enumerate(zip(goal_waypoints, goals_spec)):
        goal_x_m, goal_y_m, goal_yaw_rad = goal_spec
        current_prefix = rollout[-config.cond_steps :]
        goal_frame = _goal_frame_from_waypoint(current_prefix[-1], goal_waypoint, xy_scale)
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
        flow_losses.append(float(flow_loss.item()))
        endpoint_losses.append(float(endpoint_loss.item()))

        generated_local = dataset.denormalize(generated_norm)
        generated_global = LAFAN1Dataset.accumulate_chunk_in_root_frame(
            generated_local,
            current_prefix[:1, :3],
            current_prefix[:1, ROOT_ROT_OFFSET:POSE_BASE_DIM],
            yaw_only=True,
        )[0]
        rollout = torch.cat([rollout, generated_global[config.cond_steps :]], dim=0)

        pred_final = generated_global[-1].cpu()
        pred_yaw = _yaw_from_rot6d(pred_final[ROOT_ROT_OFFSET:POSE_BASE_DIM])
        target_yaw = _yaw_from_rot6d(goal_frame[ROOT_ROT_OFFSET:POSE_BASE_DIM].cpu())
        step_records.append(
            {
                "goal_index": goal_index,
                "goal_type": "final_goal" if goal_index == len(goals_spec) - 1 else "waypoint",
                "goal_xy_m": [goal_x_m, goal_y_m],
                "goal_yaw_rad": goal_yaw_rad,
                "goal_waypoint": goal_waypoint.cpu().tolist(),
                "prefix_last_xy": current_prefix[-1, :2].cpu().tolist(),
                "generated_window_final_xy": pred_final[:2].cpu().tolist(),
                "generated_window_final_yaw_rad": float(pred_yaw.item()),
                "goal_xy_error": float(torch.norm(pred_final[:2] - goal_frame[:2].cpu()).item()),
                "goal_yaw_error_rad": float(_wrapped_yaw_error(pred_yaw, target_yaw).item()),
                "flow_loss": float(flow_loss.item()),
                "endpoint_loss": float(endpoint_loss.item()),
            }
        )

    final_goal_x_m, final_goal_y_m, final_goal_yaw_rad = _FINAL_GOAL_XY_YAW
    final_goal_waypoint = goal_waypoints[-1]
    final_goal_frame = _goal_frame_from_waypoint(prefix_anchor.to(device), final_goal_waypoint, xy_scale).cpu()
    pred_final = rollout[-1].cpu()
    final_pred_yaw = _yaw_from_rot6d(pred_final[ROOT_ROT_OFFSET:POSE_BASE_DIM])
    final_target_yaw = _yaw_from_rot6d(final_goal_frame[ROOT_ROT_OFFSET:POSE_BASE_DIM])
    final_goal_xy_error = float(torch.norm(pred_final[:2] - final_goal_frame[:2]).item())
    final_goal_yaw_error = float(_wrapped_yaw_error(final_pred_yaw, final_target_yaw).item())

    csv = dataset.trajectory_to_lafan1_csv_qpos(rollout.cpu())
    csv_name = "rollout.csv"
    np.savetxt(output_dir / csv_name, csv.numpy(), delimiter=",", fmt="%.8f")

    metrics: dict[str, float | int | str] = {
        "low_checkpoint": str(low_checkpoint),
        "epoch": epoch,
        "split": split,
        "sample_index": sample_index,
        "num_waypoints": len(_MANUAL_WAYPOINTS_XY_YAW),
        "rollout_len": int(rollout.shape[0]),
        "final_goal_xy_error": final_goal_xy_error,
        "final_goal_yaw_error_rad": final_goal_yaw_error,
        "mean_flow_loss": float(np.mean(flow_losses)),
        "mean_endpoint_loss": float(np.mean(endpoint_losses)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "samples.json").write_text(
        json.dumps(
            {
                "sample_index": sample_index,
                "prefix_xy": prefix[:, :2].cpu().tolist(),
                "waypoints_xy_yaw": [list(goal) for goal in _MANUAL_WAYPOINTS_XY_YAW],
                "final_goal_xy_yaw": list(_FINAL_GOAL_XY_YAW),
                "generated_final_xy": pred_final[:2].cpu().tolist(),
                "generated_final_yaw_rad": float(final_pred_yaw.item()),
                "step_records": step_records,
                "csv": csv_name,
            },
            indent=2,
        )
        + "\n"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the low-level checkpoint with fixed prefix and manual waypoints.")
    parser.add_argument("--low-checkpoint", type=Path, default=_DEFAULT_LOW_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_low_level_waypoints(
        low_checkpoint=args.low_checkpoint.resolve(),
        output_dir=args.output_dir.resolve(),
        split=args.split,
        sample_index=args.sample_index,
        data_root=args.data_root,
        robot=cast(RobotName | None, args.robot),
    )
    print(json.dumps(metrics, indent=2))


def test_high_to_low_smoke() -> None:
    assert len(_MANUAL_WAYPOINTS_XY_YAW) == 8
    assert callable(evaluate_low_level_waypoints)


if __name__ == "__main__":
    main()
