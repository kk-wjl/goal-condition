from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from goal_condition.datasets.lafan1 import (
    LAFAN1Dataset,
    POSE_BASE_DIM,
    ROOT_ROT_OFFSET,
    rot6d_to_matrix,
)

WAYPOINT_DIM = 4  # local x, local y, sin(yaw), cos(yaw)


@dataclass
class ChunkConfig:
    chunk_len: int = 256
    chunk_stride: int = 30
    prefix_frames: int = 4
    num_waypoints: int = 8

    def __post_init__(self) -> None:
        if self.chunk_len <= self.prefix_frames + self.num_waypoints:
            raise ValueError("chunk_len must leave room for prefix, waypoints, and goal")
        if self.chunk_stride < 1 or self.prefix_frames < 1 or self.num_waypoints < 1:
            raise ValueError("chunk_stride, prefix_frames, and num_waypoints must be positive")


class WaypointLAFAN1Dataset(Dataset[dict[str, torch.Tensor | str]]):
    """Extract intermediate waypoints and a fixed final goal from long chunks."""

    def __init__(
        self,
        base: LAFAN1Dataset,
        *,
        split: Literal["train", "val"],
        chunk_len: int,
        chunk_stride: int,
        prefix_frames: int,
        num_waypoints: int,
        val_fraction: float,
        split_seed: int,
    ) -> None:
        super().__init__()
        self.base = base
        self.chunk_len = chunk_len
        self.prefix_frames = prefix_frames
        self.num_waypoints = num_waypoints

        generator = torch.Generator().manual_seed(split_seed)
        permutation = torch.randperm(len(base._clips), generator=generator).tolist()
        num_val = max(1, round(len(permutation) * val_fraction))
        if len(permutation) > 1:
            num_val = min(num_val, len(permutation) - 1)
        selected = set(permutation[:num_val] if split == "val" else permutation[num_val:])
        if not selected:
            raise ValueError(f"No clips assigned to the {split} split")

        self.samples: list[tuple[int, int]] = []
        for clip_idx in sorted(selected):
            clip_len = base._clips[clip_idx].shape[0]
            for start in range(0, clip_len - chunk_len + 1, chunk_stride):
                self.samples.append((clip_idx, start))
        if not self.samples:
            raise ValueError(f"No {split} chunks of length {chunk_len} were found")

        anchor = prefix_frames - 1
        fractions = torch.arange(1, num_waypoints + 1, dtype=torch.float32)
        fractions = fractions / float(num_waypoints + 1)
        self.waypoint_indices = (
            anchor + fractions * float(chunk_len - 1 - anchor)
        ).round().long()

    @property
    def condition_dim(self) -> int:
        return self.prefix_frames * self.base.state_dim + WAYPOINT_DIM

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _yaw_from_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
        rot = rot6d_to_matrix(rot6d)
        return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])

    def _root_waypoints(self, local_chunk: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        frames = local_chunk[indices]
        xy = frames[:, :2] / self.base._root_pos_std[:2].clamp_min(1e-6)
        yaw = self._yaw_from_rot6d(frames[:, ROOT_ROT_OFFSET:POSE_BASE_DIM])
        return torch.cat([xy, torch.sin(yaw[:, None]), torch.cos(yaw[:, None])], dim=-1)

    def waypoint_xy_to_meters(self, waypoints: torch.Tensor) -> torch.Tensor:
        scale = self.base._root_pos_std[:2].to(waypoints)
        return waypoints[..., :2] * scale

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        clip_idx, start = self.samples[index]
        chunk = self.base._clips[clip_idx][start : start + self.chunk_len]
        local_chunk = self.base.make_relative(chunk, yaw_only=True)
        normalized_prefix = self.base.normalize(local_chunk[: self.prefix_frames])
        waypoints = self._root_waypoints(local_chunk, self.waypoint_indices)
        goal = self._root_waypoints(local_chunk, torch.tensor([self.chunk_len - 1]))[0]
        condition = torch.cat([normalized_prefix.flatten(), goal], dim=0)
        return {
            "condition": condition,
            "waypoints": waypoints,
            "goal": goal,
            "prefix_xy": local_chunk[: self.prefix_frames, :2],
            "sample_index": torch.tensor(index),
            "clip_index": torch.tensor(clip_idx),
            "frame_start": torch.tensor(start),
            "clip_name": self.base._clip_names[clip_idx],
        }


class LocalGoalLAFAN1Dataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]]):
    """Create fixed-length local windows conditioned on their own final frame."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: Literal["train", "val"],
        robot: Literal["g1", "h1", "h1_2"] = "g1",
        seq_len: int,
        cond_steps: int,
        stride: int,
        val_fraction: float,
        split_seed: int,
        download: bool = True,
    ) -> None:
        if not (1 <= cond_steps < seq_len):
            raise ValueError(f"Need 1 <= cond_steps < seq_len, got {cond_steps}, {seq_len}")
        if stride < 1:
            raise ValueError("stride must be positive")
        self.base = LAFAN1Dataset(
            root=root,
            robot=robot,
            seq_len=seq_len,
            stride=stride,
            download=download,
        )
        self.seq_len = int(seq_len)
        self.cond_steps = int(cond_steps)
        self.stride = int(stride)
        self.state_dim = self.base.state_dim
        self.fps = self.base.fps
        self.goal_dim = WAYPOINT_DIM

        generator = torch.Generator().manual_seed(split_seed)
        permutation = torch.randperm(len(self.base._clips), generator=generator).tolist()
        num_val = max(1, round(len(permutation) * val_fraction))
        if len(permutation) > 1:
            num_val = min(num_val, len(permutation) - 1)
        selected = set(permutation[:num_val] if split == "val" else permutation[num_val:])
        if not selected:
            raise ValueError(f"No clips assigned to the {split} split")

        self._samples: list[tuple[int, int]] = []
        for clip_idx in sorted(selected):
            clip_len = self.base._clips[clip_idx].shape[0]
            for start in range(0, clip_len - self.seq_len + 1, self.stride):
                self._samples.append((clip_idx, start))
        if not self._samples:
            raise ValueError(f"No {split} local-goal windows of length {self.seq_len} were found")

    def __len__(self) -> int:
        return len(self._samples)

    def make_relative(self, trajectory: torch.Tensor, *, yaw_only: bool = False) -> torch.Tensor:
        return self.base.make_relative(trajectory, yaw_only=yaw_only)

    def normalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.base.normalize(trajectory)

    def denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.base.denormalize(trajectory)

    def trajectory_to_lafan1_csv_qpos(self, traj: torch.Tensor) -> torch.Tensor:
        return self.base.trajectory_to_lafan1_csv_qpos(traj)

    def compute_metrics(self, trajectory: torch.Tensor) -> dict[str, float]:
        return self.base.compute_metrics(trajectory)

    def goal_condition_from_states(
        self,
        anchor_frames: torch.Tensor,
        goal_frames: torch.Tensor,
    ) -> torch.Tensor:
        if anchor_frames.shape != goal_frames.shape:
            raise ValueError(
                f"anchor_frames and goal_frames must match, got {anchor_frames.shape} vs {goal_frames.shape}"
            )
        if anchor_frames.ndim != 2:
            raise ValueError(
                f"anchor_frames and goal_frames must have shape (B, D), got {anchor_frames.shape}"
            )
        pair = torch.stack([anchor_frames, goal_frames], dim=1)
        rel = self.base.make_relative(pair, yaw_only=True)[:, 1]
        goal_xy = rel[:, :2]
        goal_xy_std = self.base._root_pos_std[:2].to(
            device=goal_xy.device,
            dtype=goal_xy.dtype,
        ).clamp_min(1e-6)
        goal_rot = rel[:, ROOT_ROT_OFFSET:POSE_BASE_DIM]
        goal_yaw = self._yaw_from_rot6d(goal_rot)
        return torch.cat(
            [
                goal_xy / goal_xy_std,
                torch.sin(goal_yaw).unsqueeze(-1),
                torch.cos(goal_yaw).unsqueeze(-1),
            ],
            dim=-1,
        )

    @staticmethod
    def _yaw_from_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
        rot = rot6d_to_matrix(rot6d)
        return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]:
        if index < 0 or index >= len(self._samples):
            raise IndexError(index)
        clip_idx, start = self._samples[index]
        end = start + self.seq_len
        window = self.base._clips[clip_idx][start:end].clone()
        goal_cond = self.goal_condition_from_states(
            window[self.cond_steps - 1].unsqueeze(0),
            window[-1].unsqueeze(0),
        )[0]
        meta: dict[str, int | str] = {
            "sample_index": index,
            "clip_index": clip_idx,
            "frame_start": start,
            "frame_end": end - 1,
            "clip_name": self.base._clip_names[clip_idx],
        }
        return window, goal_cond, meta
