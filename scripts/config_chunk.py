from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from goal_condition.datasets.lafan1 import (
    LAFAN1Dataset,
    POSE_BASE_DIM,
    ROOT_ROT_OFFSET,
    RobotName,
    rot6d_to_matrix,
)


@dataclass
class FullChunkConfig:
    chunk_len: int = 180
    chunk_stride: int = 16
    prefix_frames: int = 20

    def __post_init__(self) -> None:
        if self.chunk_len < 2:
            raise ValueError(f"chunk_len must be >= 2, got {self.chunk_len}")
        if self.chunk_stride < 1:
            raise ValueError(f"chunk_stride must be >= 1, got {self.chunk_stride}")
        if not (1 <= self.prefix_frames < self.chunk_len):
            raise ValueError(
                f"Need 1 <= prefix_frames < chunk_len, got {self.prefix_frames}, {self.chunk_len}"
            )


class ChunkGoalLAFAN1Dataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]]):
    def __init__(
        self,
        root: str | Path,
        *,
        robot: RobotName,
        chunk_len: int,
        chunk_stride: int,
        download: bool = True,
    ) -> None:
        self.base = LAFAN1Dataset(
            root=root,
            robot=robot,
            seq_len=chunk_len,
            stride=chunk_stride,
            download=download,
        )
        self.chunk_len = int(chunk_len)
        self.chunk_stride = int(chunk_stride)
        self.state_dim = self.base.state_dim
        self.fps = self.base.fps
        self._samples: list[tuple[int, int]] = []
        for clip_idx, clip in enumerate(self.base._clips):
            n_rows = clip.shape[0]
            if n_rows < self.chunk_len:
                continue
            for chunk_start in range(0, n_rows - self.chunk_len + 1, self.chunk_stride):
                self._samples.append((clip_idx, chunk_start))
        if not self._samples:
            raise ValueError("No full-chunk LAFAN1 samples were created.")

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

    def goal_condition_from_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        pair = torch.stack([chunk[0], chunk[-1]], dim=0).unsqueeze(0)
        rel_goal = self.base.make_relative(pair, yaw_only=True)[0, 1]
        goal_xy = rel_goal[:2]
        goal_xy_std = self.base._root_pos_std[:2].to(
            device=goal_xy.device,
            dtype=goal_xy.dtype,
        ).clamp_min(1e-6)
        goal_xy = goal_xy / goal_xy_std
        goal_rot = rel_goal[ROOT_ROT_OFFSET:POSE_BASE_DIM].unsqueeze(0)
        goal_mat = rot6d_to_matrix(goal_rot)[0]
        goal_yaw = torch.atan2(goal_mat[1, 0], goal_mat[0, 0])
        return torch.stack([goal_xy[0], goal_xy[1], torch.sin(goal_yaw), torch.cos(goal_yaw)])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]:
        if index < 0 or index >= len(self._samples):
            raise IndexError(index)
        clip_idx, chunk_start = self._samples[index]
        chunk_end = chunk_start + self.chunk_len
        chunk = self.base._clips[clip_idx][chunk_start:chunk_end].clone()
        goal_cond = self.goal_condition_from_chunk(chunk)
        meta: dict[str, int | str] = {
            "sample_index": index,
            "clip_index": clip_idx,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end - 1,
        }
        return chunk, goal_cond, meta
