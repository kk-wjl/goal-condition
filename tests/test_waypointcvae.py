from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Literal, cast

import matplotlib
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.FM_lafan1_hierarchy import Config, WaypointCVAE, _waypoint_metrics
from scripts.config_hierarchy import ChunkConfig, WaypointLAFAN1Dataset

from goal_condition.datasets.lafan1 import LAFAN1Dataset, RobotName
from goal_condition.utils.checkpoint import (
    load_training_checkpoint,
    read_training_checkpoint_config,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CHECKPOINT = (
    _ROOT / "scripts" / "outputs" / "FM_lafan1_hierarchy" / "high_level" / "checkpoint.pt"
)
_DEFAULT_OUTPUT_DIR = _ROOT / "tests" / "outputs" / "waypoint"

_NUMBERED_OUTPUT_DIR_RE = re.compile(r"^(?P<stem>.+)_(?P<index>\d+)$")


def _load_checkpoint_config(path: Path) -> Config:
    raw_config = dict(read_training_checkpoint_config(path))
    for deprecated_key in ("num_plot_cases", "num_prior_samples", "num_interp_steps"):
        raw_config.pop(deprecated_key, None)
    chunk_cfg = ChunkConfig(**raw_config.pop("chunk"))
    return Config(chunk=chunk_cfg, **raw_config)


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


def _plot_waypoint_xy(
    path: Path,
    dataset: WaypointLAFAN1Dataset,
    prefix_xy: torch.Tensor,
    target: torch.Tensor,
    goal: torch.Tensor,
    generated: torch.Tensor,
    prior_samples: torch.Tensor,
) -> None:
    target_xy = dataset.waypoint_xy_to_meters(target).cpu().numpy()
    goal_xy = dataset.waypoint_xy_to_meters(goal).cpu().numpy()
    generated_xy = dataset.waypoint_xy_to_meters(generated).cpu().numpy()
    sampled_xy = dataset.waypoint_xy_to_meters(prior_samples).cpu().numpy()
    prefix_np = prefix_xy.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot(prefix_np[:, 0], prefix_np[:, 1], color="0.55", linewidth=2, label="prefix")
    ax.scatter(prefix_np[-1, 0], prefix_np[-1, 1], color="black", s=28, zorder=5)
    ax.scatter(goal_xy[0], goal_xy[1], marker="*", color="crimson", s=150, label="goal", zorder=6)
    ax.plot(
        np.r_[prefix_np[-1, 0], target_xy[:, 0], goal_xy[0]],
        np.r_[prefix_np[-1, 1], target_xy[:, 1], goal_xy[1]],
        "k--o",
        linewidth=1.5,
        markersize=4,
        label="expert",
    )
    for sample in sampled_xy:
        ax.plot(
            np.r_[prefix_np[-1, 0], sample[:, 0], goal_xy[0]],
            np.r_[prefix_np[-1, 1], sample[:, 1], goal_xy[1]],
            color="lightskyblue",
            marker="o",
            linewidth=1.0,
            alpha=0.35,
            markersize=3,
        )
    ax.plot(
        np.r_[prefix_np[-1, 0], generated_xy[:, 0], goal_xy[0]],
        np.r_[prefix_np[-1, 1], generated_xy[:, 1], goal_xy[1]],
        color="royalblue",
        marker="o",
        linewidth=2,
        markersize=4,
        label="generated",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("local y (m)")
    ax.set_title("Waypoint CVAE prior generation")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    split: Literal["train", "val"],
    num_samples: int,
    num_prior_samples: int,
    seed: int,
    output_dir: Path,
    data_root: str | None,
    robot: RobotName | None,
) -> dict[str, float | int | str]:
    config = _load_checkpoint_config(checkpoint_path)
    if data_root is not None:
        config.data_root = data_root
    if robot is not None:
        config.robot = robot

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base = LAFAN1Dataset(
        root=config.data_root,
        robot=config.robot,
        seq_len=config.chunk.chunk_len,
        stride=config.chunk.chunk_stride,
        download=True,
    )
    dataset = WaypointLAFAN1Dataset(
        base,
        split=split,
        chunk_len=config.chunk.chunk_len,
        chunk_stride=config.chunk.chunk_stride,
        prefix_frames=config.chunk.prefix_frames,
        num_waypoints=config.chunk.num_waypoints,
        val_fraction=config.val_fraction,
        split_seed=config.seed,
    )
    loader = DataLoader(dataset, batch_size=min(config.batch_size, num_samples), shuffle=False)

    model = WaypointCVAE(
        condition_dim=dataset.condition_dim,
        num_waypoints=config.chunk.num_waypoints,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    epoch = load_training_checkpoint(checkpoint_path, model, optimizer, map_location=device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_totals: dict[str, float] = {}
    sample_records: list[dict[str, object]] = []
    xy_scale = dataset.base._root_pos_std[:2].to(device)
    processed = 0

    for batch in loader:
        if processed >= num_samples:
            break
        batch_size = min(batch["condition"].shape[0], num_samples - processed)
        condition = batch["condition"][:batch_size].to(device)
        target = batch["waypoints"][:batch_size].to(device)
        goal = batch["goal"][:batch_size].to(device)

        prior_mu, prior_logvar = model.encode_prior(condition)
        prior_mean = model.decode(prior_mu, condition)
        posterior_mu, _posterior_logvar = model.encode_posterior(target, condition)
        posterior_mean = model.decode(posterior_mu, condition)

        for key, value in _waypoint_metrics(prior_mean, target, goal, xy_scale).items():
            generated_key = f"generated_{key}"
            metric_totals[generated_key] = metric_totals.get(generated_key, 0.0) + value * batch_size
        for key, value in _waypoint_metrics(posterior_mean, target, goal, xy_scale).items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * batch_size

        prior_std = torch.exp(0.5 * prior_logvar)
        eps = torch.randn(batch_size, num_prior_samples, config.latent_dim, device=device)
        z_samples = prior_mu[:, None, :] + prior_std[:, None, :] * eps
        sampled_paths = model.decode(
            z_samples.reshape(batch_size * num_prior_samples, config.latent_dim),
            condition[:, None, :].expand(-1, num_prior_samples, -1).reshape(
                batch_size * num_prior_samples,
                condition.shape[-1],
            ),
        ).view(batch_size, num_prior_samples, config.chunk.num_waypoints, -1)

        for i in range(batch_size):
            sample_index = int(batch["sample_index"][i])
            plot_name = f"sample_{sample_index:05d}.png"
            _plot_waypoint_xy(
                output_dir / plot_name,
                dataset,
                batch["prefix_xy"][i],
                target[i],
                goal[i],
                prior_mean[i],
                sampled_paths[i],
            )
            goal_xy = dataset.waypoint_xy_to_meters(goal[i]).cpu().tolist()
            goal_yaw = float(torch.atan2(goal[i, 2], goal[i, 3]).cpu())
            generated_yaw = torch.atan2(prior_mean[i, :, 2], prior_mean[i, :, 3]).cpu().tolist()
            target_yaw = torch.atan2(target[i, :, 2], target[i, :, 3]).cpu().tolist()
            sample_records.append(
                {
                    "sample_index": sample_index,
                    "clip_index": int(batch["clip_index"][i]),
                    "clip_name": batch["clip_name"][i],
                    "frame_start": int(batch["frame_start"][i]),
                    "goal_xyyaw": [goal_xy[0], goal_xy[1], goal_yaw],
                    "generated_waypoint_xy": dataset.waypoint_xy_to_meters(prior_mean[i]).cpu().tolist(),
                    "target_waypoint_xy": dataset.waypoint_xy_to_meters(target[i]).cpu().tolist(),
                    "generated_waypoint_yaw_rad": generated_yaw,
                    "target_waypoint_yaw_rad": target_yaw,
                    "plot": plot_name,
                }
            )
        processed += batch_size

    if processed == 0:
        raise RuntimeError("No waypoint samples were evaluated")

    metrics: dict[str, float | int | str] = {
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "split": split,
        "num_samples": processed,
    }
    for key, total in metric_totals.items():
        metrics[key] = total / processed

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "samples.json").write_text(json.dumps(sample_records, indent=2) + "\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Waypoint CVAE checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-prior-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--robot", choices=["g1", "h1", "h1_2"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_dir = _resolve_output_dir(args.output_dir.resolve())
    metrics = evaluate_checkpoint(
        checkpoint_path,
        split=cast(Literal["train", "val"], args.split),
        num_samples=args.num_samples,
        num_prior_samples=args.num_prior_samples,
        seed=args.seed,
        output_dir=output_dir,
        data_root=args.data_root,
        robot=cast(RobotName | None, args.robot),
    )
    print(json.dumps(metrics, indent=2))


def test_waypoint_eval_smoke() -> None:
    assert _DEFAULT_OUTPUT_DIR.name == "waypoint"
    assert _resolve_output_dir(_DEFAULT_OUTPUT_DIR) == _DEFAULT_OUTPUT_DIR / "waypoint_000"
    assert _resolve_output_dir(_DEFAULT_OUTPUT_DIR.parent / "waypoint_007").name == "waypoint_007"
    assert callable(evaluate_checkpoint)
    assert callable(parse_args)


if __name__ == "__main__":
    main()
