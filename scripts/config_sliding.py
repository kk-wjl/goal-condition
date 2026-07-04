from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from goal_condition.datasets.lafan1 import (
    LAFAN1Dataset,
    POSE_BASE_DIM,
    ROOT_ROT_OFFSET,
    RobotName,
    rot6d_to_matrix,
)


@dataclass
class SlidingWindowConfig:
    seq_len: int = 32
    cond_steps: int = 4
    stride: int = 1
    val_total_len: int = 128
    val_window_stride: int | None = None

    def __post_init__(self) -> None:
        if not (1 <= self.cond_steps <= self.seq_len):
            raise ValueError(
                f"Need 1 <= cond-steps <= seq-len; got cond_steps={self.cond_steps}, seq_len={self.seq_len}"
            )
        if self.val_total_len < self.cond_steps:
            raise ValueError(
                f"val_total_len ({self.val_total_len}) must be >= cond_steps ({self.cond_steps})"
            )
        if self.val_window_stride is None:
            self.val_window_stride = self.seq_len - self.cond_steps


class SlidingGoalLAFAN1Dataset(LAFAN1Dataset):
    """Create sliding-window samples conditioned on the end of a longer chunk."""

    def __init__(
        self,
        root: str | Path,
        *,
        robot: RobotName = "g1",
        seq_len: int = 32,
        cond_steps: int = 4,
        chunk_len: int = 180,
        chunk_stride: int = 30,
        stride: int = 1,
        min_goal_gap: int = 10,
        fps: float = 30.0,
        dtype: torch.dtype = torch.float32,
        download: bool = True,
        rot6d: bool = True,
    ) -> None:
        if chunk_len < seq_len:
            raise ValueError(f"chunk_len must be >= seq_len, got {chunk_len} < {seq_len}")
        if not (1 <= cond_steps <= seq_len):
            raise ValueError(
                f"Need 1 <= cond_steps <= seq_len; got cond_steps={cond_steps}, seq_len={seq_len}"
            )
        if chunk_stride < 1:
            raise ValueError("chunk_stride must be >= 1")
        if min_goal_gap < 1:
            raise ValueError("min_goal_gap must be >= 1")
        super().__init__(
            root=root,
            robot=robot,
            seq_len=seq_len,
            stride=stride,
            fps=fps,
            dtype=dtype,
            download=download,
            rot6d=rot6d,
        )
        self.cond_steps = int(cond_steps)
        self.chunk_len = int(chunk_len)
        self.chunk_stride = int(chunk_stride)
        self.min_goal_gap = int(min_goal_gap)
        self.goal_dim = 4
        self._samples: list[tuple[int, int, int]] = []
        for clip_idx, clip in enumerate(self._clips):
            n_rows = clip.shape[0]
            if n_rows < self.chunk_len:
                continue
            for chunk_start in range(0, n_rows - self.chunk_len + 1, self.chunk_stride):
                chunk_end = chunk_start + self.chunk_len - 1
                for local_start in range(0, self.chunk_len - self.seq_len + 1, self.stride):
                    current_idx = chunk_start + local_start + self.seq_len - 1
                    delta_frames = chunk_end - current_idx
                    if delta_frames >= self.min_goal_gap:
                        self._samples.append((clip_idx, chunk_start, local_start))
        if not self._samples:
            raise ValueError(
                "No valid goal-conditioned samples were created. "
                "Try reducing chunk_len, seq_len, or min_goal_gap."
            )

    def __len__(self) -> int:
        return len(self._samples)

    def _goal_condition_from_frames(
        self,
        anchor_frame: torch.Tensor,
        goal_frame: torch.Tensor,
    ) -> torch.Tensor:
        pair = torch.stack([anchor_frame, goal_frame], dim=0).unsqueeze(0)
        rel = self.make_relative(pair, yaw_only=True)[0, 1]
        goal_xy = rel[:2]
        goal_rot = rel[ROOT_ROT_OFFSET:POSE_BASE_DIM].unsqueeze(0)
        goal_rot_mat = rot6d_to_matrix(goal_rot)[0]
        goal_yaw = torch.atan2(goal_rot_mat[1, 0], goal_rot_mat[0, 0]).unsqueeze(0)
        goal_xy_std = self._root_pos_std[:2].to(
            device=goal_xy.device,
            dtype=goal_xy.dtype,
        ).clamp_min(1e-6)
        return torch.cat([goal_xy / goal_xy_std, goal_yaw], dim=0)

    def goal_condition_from_states(
        self,
        anchor_frames: torch.Tensor,
        goal_frames: torch.Tensor,
        delta_frames: int | torch.Tensor,
    ) -> torch.Tensor:
        if anchor_frames.shape != goal_frames.shape:
            raise ValueError(
                f"anchor_frames and goal_frames must match, got {anchor_frames.shape} vs {goal_frames.shape}"
            )
        if anchor_frames.ndim != 2:
            raise ValueError(
                f"anchor_frames and goal_frames must have shape (B, D), got {anchor_frames.shape}"
            )
        goals = [
            self._goal_condition_from_frames(anchor_frames[i], goal_frames[i])
            for i in range(anchor_frames.shape[0])
        ]
        goal_tensor = torch.stack(goals, dim=0)
        if isinstance(delta_frames, int):
            delta = torch.full(
                (anchor_frames.shape[0], 1),
                float(delta_frames) / float(self.fps),
                device=goal_tensor.device,
                dtype=goal_tensor.dtype,
            )
        else:
            delta = delta_frames.to(
                device=goal_tensor.device,
                dtype=goal_tensor.dtype,
            ).reshape(-1, 1)
            delta = delta / float(self.fps)
        return torch.cat([goal_tensor, delta], dim=1)

    def get_chunk_goal_frame(self, index: int) -> torch.Tensor:
        clip_idx, chunk_start, _local_start = self._samples[index]
        chunk_end = chunk_start + self.chunk_len - 1
        return self._clips[clip_idx][chunk_end].clone()

    def get_chunk_prefix(self, index: int) -> torch.Tensor:
        clip_idx, chunk_start, _local_start = self._samples[index]
        return self._clips[clip_idx][chunk_start : chunk_start + self.cond_steps].clone()

    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]:
        if index < 0 or index >= len(self._samples):
            raise IndexError(index)

        clip_idx, chunk_start, local_start = self._samples[index]
        abs_start = chunk_start + local_start
        abs_end = abs_start + self.seq_len
        chunk_end = chunk_start + self.chunk_len - 1
        clip = self._clips[clip_idx]
        window = clip[abs_start:abs_end]
        delta_frames = chunk_end - (abs_end - 1)
        goal_cond = self.goal_condition_from_states(
            window[-1].unsqueeze(0),
            clip[chunk_end].unsqueeze(0),
            delta_frames,
        )[0]
        meta: dict[str, int | str] = {
            "sample_index": index,
            "clip_index": clip_idx,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "frame_start": abs_start,
            "delta_frames": delta_frames,
        }
        return window, goal_cond, meta
