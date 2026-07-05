"""
100STYLE motion dataset loader for retargeted G1 trajectories.

The raw dataset stores one ``qpos`` array per ``.npz`` clip under
``motions/<style>/*.npz`` with row layout
``[x, y, z, qw, qx, qy, qz, ...jpos]``.

This loader mirrors :class:`goal_condition.datasets.lafan1.LAFAN1Dataset`
as closely as possible, but:

- reads ``.npz`` clips instead of CSV files;
- extracts a ``style`` label from the parent directory;
- converts source root quaternions from ``wxyz`` storage to the same internal
  processed representation used by the LAFAN1 pipeline.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset

from goal_condition.utils.math import (
    quat_conjugate,
    quat_mul,
    quat_rotate,
    quat_to_rot6d,
    quat_wxyz_to_xyzw,
    rot6d_from_matrix,
    rot6d_to_matrix,
    rot6d_to_quat_wxyz,
    yaw_matrix,
    yaw_quat,
)
from .lafan1 import (
    CSV_QPOS_BASE_DIM,
    POSE_BASE_DIM,
    ROOT_LIN_VEL_DIM,
    ROOT_POS_DIM,
    ROOT_ROT_OFFSET,
)


STYLE100_REPO_DIRNAME = "any4hdmi-g1-100style"
STYLE100_MOTIONS_SUBDIR = "motions"
STYLE100_QPOS_DIM = 36
STYLE100_DEFAULT_FPS = 50.0

_STYLE100_NORM_STATS_CACHE_VERSION = 1

_STYLE100_NORM_STATS_KEYS = (
    "_jpos_mean",
    "_jpos_std",
    "_jvel_mean",
    "_jvel_std",
    "_root_pos_mean",
    "_root_pos_std",
    "_root_lin_vel_mean",
    "_root_lin_vel_std",
)


def _style100_base_with_clips(root: Path) -> Path | None:
    """Return the dataset base that contains ``motions/<style>/*.npz`` clips."""
    candidates = (root, root / STYLE100_REPO_DIRNAME)
    for base in candidates:
        motions_dir = base / STYLE100_MOTIONS_SUBDIR
        if motions_dir.is_dir() and any(motions_dir.glob("*/*.npz")):
            return base
    return None


def _style100_norm_stats_fingerprint(
    seq_len: int,
    stride: int,
    fps: float,
    rot6d: bool,
    clip_paths: list[Path],
) -> tuple[str, dict[str, Any]]:
    clips_meta: list[list[str | int]] = []
    for p in clip_paths:
        st = p.stat()
        clips_meta.append([str(p.relative_to(p.parents[1])), st.st_size, st.st_mtime_ns])
    payload: dict[str, Any] = {
        "cache_version": _STYLE100_NORM_STATS_CACHE_VERSION,
        "seq_len": seq_len,
        "stride": stride,
        "fps": fps,
        "rot6d": rot6d,
        "clips": clips_meta,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    return digest, payload


def _try_load_style100_norm_stats_cache(
    cache_path: Path,
    expected_digest: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor] | None:
    if not cache_path.is_file():
        return None
    try:
        text = cache_path.read_text(encoding="utf-8")
        blob = yaml.safe_load(text)
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    if blob.get("digest") != expected_digest:
        return None
    if int(blob.get("cache_version", 0)) != _STYLE100_NORM_STATS_CACHE_VERSION:
        return None
    raw_stats = blob.get("stats")
    if not isinstance(raw_stats, dict):
        return None
    stats: dict[str, torch.Tensor] = {}
    for k in _STYLE100_NORM_STATS_KEYS:
        row = raw_stats.get(k)
        if not isinstance(row, list) or not row:
            return None
        if not all(isinstance(x, (int, float)) for x in row):
            return None
        stats[k] = torch.tensor(row, dtype=dtype)
    return stats


def _save_style100_norm_stats_cache(
    cache_path: Path,
    digest: str,
    payload: dict[str, Any],
    stats: dict[str, torch.Tensor],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stats_yaml: dict[str, list[float]] = {
        k: v.detach().cpu().float().numpy().astype(np.float64).tolist() for k, v in stats.items()
    }
    blob: dict[str, Any] = {
        "cache_version": _STYLE100_NORM_STATS_CACHE_VERSION,
        "digest": digest,
        "payload": payload,
        "stats": stats_yaml,
    }
    cache_path.write_text(
        yaml.safe_dump(blob, sort_keys=False, default_flow_style=None, allow_unicode=False),
        encoding="utf-8",
    )


def _num_windows(n_rows: int, seq_len: int, stride: int) -> int:
    if n_rows < seq_len:
        return 0
    return (n_rows - seq_len) // stride + 1


class Styles100Dataset(Dataset[tuple[torch.Tensor, dict[str, int | str]]]):
    """
    Sliding-window access to retargeted 100STYLE G1 trajectories.
    Can be downloaded from https://huggingface.co/datasets/elijahgalahad/any4hdmi-100style 

    Parameters
    ----------
    root
        Directory that either contains ``motions/<style>/*.npz`` directly or is the parent of
        a cloned dataset tree ``any4hdmi-g1-100style/motions/<style>/*.npz``.
    seq_len
        Number of consecutive frames per sample.
    stride
        Frame stride between consecutive windows.
    fps
        Sampling rate used for finite-difference velocity computation. The dataset manifest uses
        50 Hz.
    dtype
        Output tensor dtype.
    rot6d
        If ``True`` (default), root orientation is stored as 6D rotation. Otherwise the internal
        representation keeps root quaternions in ``wxyz`` order.
    use_norm_stats_cache
        If ``True`` (default), read/write cached normalization statistics under
        ``motions/.goal_condition_cache`` when config and clip metadata match.
    """

    qpos_dim = STYLE100_QPOS_DIM
    n_joints = qpos_dim - CSV_QPOS_BASE_DIM
    state_dim = POSE_BASE_DIM + 2 * n_joints

    def __init__(
        self,
        root: str | Path,
        seq_len: int = 32,
        stride: int = 1,
        *,
        fps: float = STYLE100_DEFAULT_FPS,
        dtype: torch.dtype = torch.float32,
        rot6d: bool = True,
        use_norm_stats_cache: bool = True,
    ) -> None:
        super().__init__()
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if fps <= 0:
            raise ValueError("fps must be > 0")

        self.root = Path(root).expanduser().resolve()
        self.seq_len = seq_len
        self.stride = stride
        self.fps = float(fps)
        self.dtype = dtype
        self.rot6d = rot6d

        resolved = _style100_base_with_clips(self.root)
        if resolved is None:
            raise FileNotFoundError(
                "Could not find 100STYLE motions under "
                f"{self.root} or {self.root / STYLE100_REPO_DIRNAME}"
            )
        self.root = resolved
        self.motions_dir = self.root / STYLE100_MOTIONS_SUBDIR

        paths = sorted(self.motions_dir.glob("*/*.npz"))
        self._clips: list[torch.Tensor] = []
        self._clip_names: list[str] = []
        self._styles: list[str] = []
        windows_per_clip: list[int] = []

        valid_paths: list[Path] = []
        for p in paths:
            with np.load(p, allow_pickle=False) as data:
                if "qpos" not in data:
                    raise KeyError(f"Missing 'qpos' array in {p}")
                qpos = data["qpos"]
            if qpos.ndim != 2 or qpos.shape[1] != self.qpos_dim:
                raise ValueError(
                    f"{p}: expected qpos shape (T, {self.qpos_dim}), got {qpos.shape}"
                )
            n = int(qpos.shape[0])
            nw = _num_windows(n, seq_len, stride)
            if nw == 0:
                continue
            self._clips.append(self.process_data(qpos, self.fps))
            self._clip_names.append(p.name)
            self._styles.append(p.parent.name)
            windows_per_clip.append(nw)
            valid_paths.append(p)

        if not self._clips:
            raise ValueError(
                f"No windows of length {seq_len} in {self.motions_dir}; all clips are too short."
            )

        stats_digest, stats_payload = _style100_norm_stats_fingerprint(
            seq_len, stride, self.fps, rot6d, valid_paths
        )
        cache_dir = self.motions_dir / ".goal_condition_cache"
        cache_path = cache_dir / f"styles100_norm_stats_{stats_digest}.yaml"

        cached = None
        if use_norm_stats_cache:
            cached = _try_load_style100_norm_stats_cache(cache_path, stats_digest, dtype)

        if cached is not None:
            print(f"Loading cached statistics from {cache_path}")
            self._jpos_mean = cached["_jpos_mean"]
            self._jpos_std = cached["_jpos_std"]
            self._jvel_mean = cached["_jvel_mean"]
            self._jvel_std = cached["_jvel_std"]
            self._root_pos_mean = cached["_root_pos_mean"]
            self._root_pos_std = cached["_root_pos_std"]
            self._root_lin_vel_mean = cached["_root_lin_vel_mean"]
            self._root_lin_vel_std = cached["_root_lin_vel_std"]
        else:
            lo = POSE_BASE_DIM
            mid = lo + self.n_joints
            all_frames = torch.cat(self._clips, dim=0)
            jpos_block = all_frames[:, lo:mid]
            jvel_block = all_frames[:, mid:]
            eps = 1e-6
            self._jpos_mean = jpos_block.mean(dim=0)
            self._jpos_std = jpos_block.std(dim=0, correction=0).clamp_min(eps)
            self._jvel_mean = jvel_block.mean(dim=0)
            self._jvel_std = jvel_block.std(dim=0, correction=0).clamp_min(eps)

            rel_root_pos_rows: list[torch.Tensor] = []
            rel_root_vel_rows: list[torch.Tensor] = []
            for clip in tqdm(self._clips, desc="Computing statistics"):
                t_rows = clip.shape[0]
                for start in range(0, t_rows - seq_len + 1, stride):
                    chunk = clip[start : start + seq_len]
                    rel = self.make_relative(chunk)
                    pos, vel = rel[:, :ROOT_ROT_OFFSET].split([3, 3], dim=1)
                    rel_root_pos_rows.append(pos.reshape(-1, ROOT_POS_DIM))
                    rel_root_vel_rows.append(vel.reshape(-1, ROOT_LIN_VEL_DIM))
            all_rel_root_pos = torch.cat(rel_root_pos_rows, dim=0)
            all_rel_root_vel = torch.cat(rel_root_vel_rows, dim=0)
            self._root_pos_mean = all_rel_root_pos.mean(dim=0)
            self._root_pos_std = all_rel_root_pos.std(dim=0, correction=0).clamp_min(eps)
            self._root_lin_vel_mean = all_rel_root_vel.mean(dim=0)
            self._root_lin_vel_std = all_rel_root_vel.std(dim=0, correction=0).clamp_min(eps)

            if use_norm_stats_cache:
                _save_style100_norm_stats_cache(
                    cache_path,
                    stats_digest,
                    stats_payload,
                    {
                        "_jpos_mean": self._jpos_mean,
                        "_jpos_std": self._jpos_std,
                        "_jvel_mean": self._jvel_mean,
                        "_jvel_std": self._jvel_std,
                        "_root_pos_mean": self._root_pos_mean,
                        "_root_pos_std": self._root_pos_std,
                        "_root_lin_vel_mean": self._root_lin_vel_mean,
                        "_root_lin_vel_std": self._root_lin_vel_std,
                    },
                )

        self._window_offsets: list[int] = [0]
        for w in windows_per_clip:
            self._window_offsets.append(self._window_offsets[-1] + w)
        self._total_windows = self._window_offsets[-1]

    def __len__(self) -> int:
        return self._total_windows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, int | str]]:
        if index < 0 or index >= self._total_windows:
            raise IndexError(index)

        clip_idx = bisect.bisect_right(self._window_offsets, index) - 1
        win_in_clip = index - self._window_offsets[clip_idx]
        start = win_in_clip * self.stride
        end = start + self.seq_len

        clip = self._clips[clip_idx]
        chunk = clip[start:end]
        meta: dict[str, int | str] = {
            "clip_index": clip_idx,
            "clip_name": self._clip_names[clip_idx],
            "style": self._styles[clip_idx],
            "frame_start": start,
        }
        return chunk, meta

    def process_data(self, qpos: np.ndarray, fps: float) -> torch.Tensor:
        """
        Convert raw qpos rows to the processed representation used by the LAFAN1 pipeline.

        Source layout is ``[x, y, z, qw, qx, qy, qz, ...jpos]``.
        """
        x = torch.as_tensor(qpos, dtype=torch.float32)
        root_pos = x[:, :3]
        wxyz = x[:, 3:7]
        wxyz = wxyz / wxyz.norm(dim=1, keepdim=True).clamp_min(1e-8)
        if self.rot6d:
            root_rot = quat_to_rot6d(wxyz)
        else:
            root_rot = wxyz
        jpos = x[:, CSV_QPOS_BASE_DIM:]
        t_rows, j = jpos.shape
        root_lin_vel = torch.zeros((t_rows, 3), dtype=x.dtype, device=x.device)
        jvel = torch.zeros((t_rows, j), dtype=x.dtype, device=x.device)
        if t_rows >= 2:
            fp = float(fps)
            jvel[0] = (jpos[1] - jpos[0]) * fp
            jvel[1:] = (jpos[1:] - jpos[:-1]) * fp
            root_lin_vel[0] = (root_pos[1] - root_pos[0]) * fp
            root_lin_vel[1:] = (root_pos[1:] - root_pos[:-1]) * fp
        return torch.cat([root_pos, root_lin_vel, root_rot, jpos, jvel], dim=1)

    def make_relative(
        self,
        trajectory: torch.Tensor,
        xy_only: bool = True,
        yaw_only: bool = False,
    ) -> torch.Tensor:
        device = trajectory.device
        root_pos = trajectory[..., :3]
        if xy_only:
            root_pos_rel = root_pos - root_pos[..., 0:1, :] * torch.tensor(
                [1.0, 1.0, 0.0],
                device=device,
                dtype=trajectory.dtype,
            )
        else:
            root_pos_rel = root_pos - root_pos[..., 0:1, :]
        root_lin_vel = trajectory[..., ROOT_POS_DIM:ROOT_ROT_OFFSET]
        if self.rot6d:
            jstates = trajectory[..., POSE_BASE_DIM:]
            root_rot6d = trajectory[..., ROOT_ROT_OFFSET:POSE_BASE_DIM]
            R = rot6d_to_matrix(root_rot6d)
            if yaw_only:
                R0 = yaw_matrix(R[..., 0:1, :, :])
            else:
                R0 = R[..., 0:1, :, :]
            R0_inv = R0.transpose(-1, -2)
            R_rel = torch.matmul(R0_inv, R)
            root_rot6d_rel = rot6d_from_matrix(R_rel)
            root_pos_rel = torch.matmul(R0_inv, root_pos_rel.unsqueeze(-1)).squeeze(-1)
            root_lin_vel_rel = torch.matmul(R0_inv, root_lin_vel.unsqueeze(-1)).squeeze(-1)
            result = torch.cat([root_pos_rel, root_lin_vel_rel, root_rot6d_rel, jstates], dim=-1)
        else:
            jstates = trajectory[..., ROOT_ROT_OFFSET + 4 :]
            root_wxyz = trajectory[..., ROOT_ROT_OFFSET : ROOT_ROT_OFFSET + 4]
            root_wxyz_0 = root_wxyz[..., 0:1, :]
            if yaw_only:
                root_wxyz_0 = yaw_quat(root_wxyz_0)
            root_wxyz_0_inv = quat_conjugate(root_wxyz_0).expand_as(root_wxyz)
            root_wxyz_rel = quat_mul(root_wxyz_0_inv, root_wxyz)
            root_pos_rel = quat_rotate(root_wxyz_0_inv, root_pos_rel)
            root_lin_vel_rel = quat_rotate(root_wxyz_0_inv, root_lin_vel)
            result = torch.cat([root_pos_rel, root_lin_vel_rel, root_wxyz_rel, jstates], dim=-1)
        return result

    def normalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        lo = POSE_BASE_DIM
        mid = lo + self.n_joints
        out = trajectory.clone()
        device, dtype = trajectory.device, trajectory.dtype
        rp_mean = self._root_pos_mean.to(device=device, dtype=dtype)
        rp_std = self._root_pos_std.to(device=device, dtype=dtype)
        out[..., :ROOT_POS_DIM] = (out[..., :ROOT_POS_DIM] - rp_mean) / rp_std.clamp_min(1e-6)
        rv_mean = self._root_lin_vel_mean.to(device=device, dtype=dtype)
        rv_std = self._root_lin_vel_std.to(device=device, dtype=dtype)
        out[..., ROOT_POS_DIM:ROOT_ROT_OFFSET] = (
            out[..., ROOT_POS_DIM:ROOT_ROT_OFFSET] - rv_mean
        ) / rv_std.clamp_min(1e-6)
        jpos_mean = self._jpos_mean.to(device=device, dtype=dtype)
        jpos_std = self._jpos_std.to(device=device, dtype=dtype)
        out[..., lo:mid] = (out[..., lo:mid] - jpos_mean) / jpos_std.clamp_min(1e-6)
        jvel_mean = self._jvel_mean.to(device=device, dtype=dtype)
        jvel_std = self._jvel_std.to(device=device, dtype=dtype)
        out[..., mid:] = (out[..., mid:] - jvel_mean) / jvel_std.clamp_min(1e-6)
        return out

    def denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        lo = POSE_BASE_DIM
        mid = lo + self.n_joints
        out = trajectory.clone()
        device, dtype = trajectory.device, trajectory.dtype
        rp_mean = self._root_pos_mean.to(device=device, dtype=dtype)
        rp_std = self._root_pos_std.to(device=device, dtype=dtype)
        out[..., :ROOT_POS_DIM] = out[..., :ROOT_POS_DIM] * rp_std + rp_mean
        rv_mean = self._root_lin_vel_mean.to(device=device, dtype=dtype)
        rv_std = self._root_lin_vel_std.to(device=device, dtype=dtype)
        out[..., ROOT_POS_DIM:ROOT_ROT_OFFSET] = (
            out[..., ROOT_POS_DIM:ROOT_ROT_OFFSET] * rv_std + rv_mean
        )
        jpos_mean = self._jpos_mean.to(device=device, dtype=dtype)
        jpos_std = self._jpos_std.to(device=device, dtype=dtype)
        jvel_mean = self._jvel_mean.to(device=device, dtype=dtype)
        jvel_std = self._jvel_std.to(device=device, dtype=dtype)
        out[..., lo:mid] = out[..., lo:mid] * jpos_std + jpos_mean
        out[..., mid:] = out[..., mid:] * jvel_std + jvel_mean
        return out

    def trajectory_to_100style_qpos(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Convert a processed trajectory back to raw 100STYLE qpos layout.

        Output row layout is ``[x, y, z, qw, qx, qy, qz, ...jpos]``.
        """
        if self.rot6d:
            expected = POSE_BASE_DIM + 2 * self.n_joints
        else:
            expected = ROOT_ROT_OFFSET + 4 + 2 * self.n_joints
        if traj.shape[-1] != expected:
            raise ValueError(
                f"trajectory last dim must be {expected} (state_dim), got {traj.shape[-1]}"
            )
        pos = traj[..., :3]
        if self.rot6d:
            rot6d = traj[..., ROOT_ROT_OFFSET:POSE_BASE_DIM]
            quat_wxyz = rot6d_to_quat_wxyz(rot6d)
            jpos = traj[..., POSE_BASE_DIM : POSE_BASE_DIM + self.n_joints]
        else:
            quat_wxyz = traj[..., ROOT_ROT_OFFSET : ROOT_ROT_OFFSET + 4]
            jpos = traj[..., ROOT_ROT_OFFSET + 4 : ROOT_ROT_OFFSET + 4 + self.n_joints]
        return torch.cat([pos, quat_wxyz, jpos], dim=-1)

    def trajectory_to_lafan1_qpos(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Convert a processed trajectory to the LAFAN1-style export layout.

        Output row layout is ``[x, y, z, qx, qy, qz, qw, ...jpos]``.
        """
        qpos_wxyz = self.trajectory_to_100style_qpos(traj)
        pos = qpos_wxyz[..., :3]
        quat_wxyz = qpos_wxyz[..., 3:7]
        quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
        tail = qpos_wxyz[..., 7:]
        return torch.cat([pos, quat_xyzw, tail], dim=-1)

    def compute_metrics(self, trajectory: torch.Tensor) -> dict[str, float]:
        if trajectory.shape[-1] != self.state_dim:
            raise ValueError(
                f"trajectory last dim must be {self.state_dim}, got {trajectory.shape[-1]}"
            )
        if trajectory.shape[-2] < 2:
            return {
                "root_vel_fd_mse": 0.0,
                "joint_vel_fd_mse": 0.0,
            }

        fp = float(self.fps)
        root_pos = trajectory[..., :, :ROOT_POS_DIM]
        root_vel = trajectory[..., :, ROOT_POS_DIM:ROOT_ROT_OFFSET]
        jpos = trajectory[..., :, POSE_BASE_DIM : POSE_BASE_DIM + self.n_joints]
        jvel = trajectory[..., :, POSE_BASE_DIM + self.n_joints :]

        root_vel_fd = torch.zeros_like(root_vel)
        root_vel_fd[..., 0, :] = (root_pos[..., 1, :] - root_pos[..., 0, :]) * fp
        root_vel_fd[..., 1:, :] = (root_pos[..., 1:, :] - root_pos[..., :-1, :]) * fp

        joint_vel_fd = torch.zeros_like(jvel)
        joint_vel_fd[..., 0, :] = (jpos[..., 1, :] - jpos[..., 0, :]) * fp
        joint_vel_fd[..., 1:, :] = (jpos[..., 1:, :] - jpos[..., :-1, :]) * fp

        return {
            "root_vel_fd_mse": float(torch.mean((root_vel - root_vel_fd) ** 2).item()),
            "joint_vel_fd_mse": float(torch.mean((jvel - joint_vel_fd) ** 2).item()),
        }
