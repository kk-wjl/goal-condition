"""
Robot motion datasets (e.g. LAFAN1 retargeted mocap).

CSV rows are qpos only: ``[x, y, z, qw, qx, qy, qz, ...jpos]`` (free flyer + actuated joints),
as in the LAFAN1 retargeting layout ``<root>/<robot>/*.csv``. The root quaternion is converted
to **6D rotation** (Zhou et al.: first two columns of ``R``, flattened). Root linear velocity
and joint velocities are appended per frame from finite differences times ``fps``. Layout per
frame: ``[x, y, z, vx, vy, vz, rot6d(6), jpos..., jvel...]``.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset
from goal_condition.utils.math import (
    yaw_quat,
    yaw_matrix,
    quat_conjugate,
    quat_mul,
    quat_rotate,
    quat_to_rot6d,
    quat_wxyz_to_xyzw,
    rot6d_from_matrix,
    rot6d_to_matrix,
    rot6d_to_quat_wxyz,
)

RobotName = Literal["g1", "h1", "h1_2"]

LAFAN1_HF_REPO_ID = "lvhaidong/LAFAN1_Retargeting_Dataset"
# Same name as the Hugging Face / git clone top-level folder (``g1/``, ``h1/``, … live under it).
LAFAN1_REPO_DIRNAME = "LAFAN1_Retargeting_Dataset"

# Columns in CSV before joint positions: position (3) + quaternion (xyzw, 4).
CSV_QPOS_BASE_DIM = 7
# Processed trajectory layout per frame: position (3) + root linear velocity (3)
# + root rot6d (6) + jpos + jvel.
ROOT_POS_DIM = 3
ROOT_LIN_VEL_DIM = 3
ROOT_ROT6D_DIM = 6
ROOT_ROT_OFFSET = ROOT_POS_DIM + ROOT_LIN_VEL_DIM
POSE_BASE_DIM = ROOT_ROT_OFFSET + ROOT_ROT6D_DIM


def _lafan1_base_with_clips(root: Path, robot: RobotName) -> Path | None:
    """Return ``base`` if ``base/<robot>/*.csv`` exists; try flat layout then clone/hub layout."""
    for base in (root, root / LAFAN1_REPO_DIRNAME):
        clip = base / robot
        if clip.is_dir() and any(clip.glob("*.csv")):
            return base
    return None


_LAFAN1_NORM_STATS_CACHE_VERSION = 1

_LAFAN1_NORM_STATS_KEYS = (
    "_jpos_mean",
    "_jpos_std",
    "_jvel_mean",
    "_jvel_std",
    "_root_pos_mean",
    "_root_pos_std",
    "_root_lin_vel_mean",
    "_root_lin_vel_std",
)


def _lafan1_norm_stats_fingerprint(
    robot: RobotName,
    seq_len: int,
    stride: int,
    fps: float,
    rot6d: bool,
    clip_paths: list[Path],
) -> tuple[str, dict[str, Any]]:
    """Stable hash over config + clip identity (name, size, mtime) for norm-stat caching."""
    clips_meta: list[list[str | int]] = []
    for p in clip_paths:
        st = p.stat()
        clips_meta.append([p.name, st.st_size, st.st_mtime_ns])
    payload: dict[str, Any] = {
        "cache_version": _LAFAN1_NORM_STATS_CACHE_VERSION,
        "robot": robot,
        "seq_len": seq_len,
        "stride": stride,
        "fps": fps,
        "rot6d": rot6d,
        "clips": clips_meta,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    return digest, payload


def _try_load_lafan1_norm_stats_cache(
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
    if int(blob.get("cache_version", 0)) != _LAFAN1_NORM_STATS_CACHE_VERSION:
        return None
    raw_stats = blob.get("stats")
    if not isinstance(raw_stats, dict):
        return None
    stats: dict[str, torch.Tensor] = {}
    for k in _LAFAN1_NORM_STATS_KEYS:
        row = raw_stats.get(k)
        if not isinstance(row, list) or not row:
            return None
        if not all(isinstance(x, (int, float)) for x in row):
            return None
        stats[k] = torch.tensor(row, dtype=dtype)
    return stats


def _save_lafan1_norm_stats_cache(
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
        "cache_version": _LAFAN1_NORM_STATS_CACHE_VERSION,
        "digest": digest,
        "payload": payload,
        "stats": stats_yaml,
    }
    cache_path.write_text(
        yaml.safe_dump(blob, sort_keys=False, default_flow_style=None, allow_unicode=False),
        encoding="utf-8",
    )


def _ensure_lafan1_robot_files(root: Path, robot: RobotName, *, download: bool) -> None:
    if _lafan1_base_with_clips(root, robot) is not None:
        return
    clip_dir = root / robot
    if not download:
        if not clip_dir.is_dir():
            raise FileNotFoundError(f"Missing robot folder: {clip_dir}")
        raise ValueError(f"No CSV clips found in {clip_dir}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "LAFAN1Dataset with download=True requires `huggingface_hub`. "
            "Install it with: pip install huggingface_hub"
        ) from e
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(  # pyright: ignore[reportCallIssue]
        repo_id=LAFAN1_HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=[f"{robot}/*.csv"],
        local_dir_use_symlinks=False,
    )
    if not clip_dir.is_dir() or not any(clip_dir.glob("*.csv")):
        raise RuntimeError(
            f"Download finished but no CSV files were found under {clip_dir}"
        )


class LAFAN1Dataset(Dataset):
    """
    Sliding-window access to LAFAN1-style retargeted trajectories for one robot.
    Can be downloaded from https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset

    All clips are loaded into memory at construction (``float32``) to avoid disk I/O
    during training. Per-joint mean and population standard deviation are computed over
    all frames for jpos and jvel. Root position and root linear velocity mean/std are
    computed over **relative** windows (same ``seq_len`` / ``stride`` as :meth:`__getitem__`,
    via :meth:`make_relative`), matching the tensor layout seen after ``make_relative`` then
    ``normalize`` in typical training code. Root orientation (rot6d or quaternion) is not
    z-scored.
    Note that the original data's quaternions are in xyzw order. We convert them to wxyz order
    to align with MuJoCo's convention.

    Parameters
    ----------
    root
        Directory that either contains ``<robot>/*.csv`` directly **or** is the parent of a
        git-cloned / hub-style tree ``LAFAN1_Retargeting_Dataset/<robot>/*.csv``.
    robot
        Which robot / DoF layout to load.
    seq_len
        Number of consecutive frames per sample, shape ``(seq_len, state_dim)`` with
        ``state_dim = 12 + 2 * n_joints`` (pos + root_lin_vel + rot6d + jpos + jvel;
        ``n_joints`` from CSV).
    stride
        Frame stride between consecutive windows within the same clip.
    fps
        Sampling rate used to turn jpos finite differences into jvel (rad/s or deg/s per
        your CSV convention). LAFAN1 mocap is commonly 30 Hz.
    dtype
        Output tensor dtype (default ``float32``).
    download
        If ``True`` (default), fetch missing ``{robot}/*.csv`` from Hugging Face
        (``lvhaidong/LAFAN1_Retargeting_Dataset``) into ``root``. If ``False``,
        ``root/<robot>`` must already contain CSV clips.
    use_norm_stats_cache
        If ``True`` (default), read/write cached joint / root linear norm statistics under
        ``<robot>/.goal_condition_cache/lafan1_norm_stats_<sha256>.yaml`` when ``robot``,
        ``seq_len``, ``stride``, ``fps``, ``rot6d``, and on-disk clip files (name, size, mtime)
        match a previous run.
    """

    QPOS_DIM: dict[RobotName, int] = {
        "h1": 26,
        "h1_2": 34,
        "g1": 36,
    }

    def __init__(
        self,
        root: str | Path,
        robot: RobotName = "g1",
        seq_len: int = 32,
        stride: int = 1,
        *,
        fps: float = 30.0,
        dtype: torch.dtype = torch.float32,
        download: bool = True,
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
        self.robot: RobotName = robot
        self.seq_len = seq_len
        self.stride = stride
        self.fps = float(fps)
        self.dtype = dtype
        self.rot6d = rot6d

        qpos_dim = self.QPOS_DIM[robot]
        n_joints = qpos_dim - CSV_QPOS_BASE_DIM
        if n_joints < 1:
            raise ValueError(f"robot {robot!r}: expected at least one joint column in qpos")
        self.qpos_dim = qpos_dim
        self.n_joints = n_joints
        self.state_dim = POSE_BASE_DIM + 2 * n_joints

        resolved = _lafan1_base_with_clips(self.root, robot)
        if resolved is not None:
            self.root = resolved
        else:
            _ensure_lafan1_robot_files(self.root, robot, download=download)

        clip_dir = self.root / robot
        paths = sorted(clip_dir.glob("*.csv"))

        self._clips: list[torch.Tensor] = []
        self._clip_names: list[str] = []
        windows_per_clip: list[int] = []

        for p in paths:
            data = np.loadtxt(p, delimiter=",")
            n = data.shape[0]
            nw = _num_windows(n, seq_len, stride)
            if nw == 0:
                continue
            self._clips.append(self.process_data(data, self.fps))
            self._clip_names.append(p.name)
            windows_per_clip.append(nw)

        if not self._clips:
            raise ValueError(
                f"No windows of length {seq_len} in {clip_dir}; all clips are too short."
            )

        stats_digest, stats_payload = _lafan1_norm_stats_fingerprint(
            robot, seq_len, stride, self.fps, rot6d, paths
        )
        cache_dir = clip_dir / ".goal_condition_cache"
        cache_path = cache_dir / f"lafan1_norm_stats_{stats_digest}.yaml"

        cached = None
        if use_norm_stats_cache:
            cached = _try_load_lafan1_norm_stats_cache(cache_path, stats_digest, dtype)

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
            mid = lo + n_joints
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
                _save_lafan1_norm_stats_cache(
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
            "frame_start": start,
        }
        return chunk, meta
    
    def process_data(self, qpos: np.ndarray, fps: float) -> torch.Tensor:
        """
        Root position, velocity, orientation as rot6d (from normalized quaternion wxyz); append jvel from jpos.
        """
        x = torch.as_tensor(qpos, dtype=torch.float32)
        root_pos = x[:, :3]
        xyzw = x[:, 3:7]
        wxyz = xyzw[:, [3, 0, 1, 2]]
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
        """
        Express a trajectory in the first-frame root coordinate system.

        Root translation is offset by frame 0 and then rotated by the inverse of the
        frame-0 root orientation. If ``xy_only`` is ``True`` (default), only ``x/y`` are
        offset in world space before this rotation (``z`` keeps its absolute value);
        if ``xy_only`` is ``False``, all ``x/y/z`` components are offset.

        Defaulting to ``xy_only=True`` keeps the global height anchor and tends to reduce
        long-horizon rollout drift/accumulated vertical bias when windows are stitched
        autoregressively.
        If ``yaw_only=True``, the frame-0 reference orientation is projected to yaw-only
        before local-frame conversion, which avoids chaining pitch/roll in the stitched
        reference frame and can reduce unnatural long-horizon tilting.

        This function defines the local-frame convention used by training targets and
        by rollout composition utilities.
        """
        device = trajectory.device
        root_pos = trajectory[..., :3]
        if xy_only:
            root_pos_rel = root_pos - root_pos[..., 0:1, :] * torch.tensor([1.0, 1.0, 0.0], device=device)
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
            jstates = trajectory[..., ROOT_ROT_OFFSET + 4:]
            root_wxyz = trajectory[..., ROOT_ROT_OFFSET:ROOT_ROT_OFFSET + 4]
            root_wxyz_0 = root_wxyz[..., 0:1, :]
            if yaw_only:
                root_wxyz_0 = yaw_quat(root_wxyz_0)
            root_wxyz_0_inv = quat_conjugate(root_wxyz_0).expand_as(root_wxyz)
            root_wxyz_rel = quat_mul(root_wxyz_0_inv, root_wxyz)
            root_pos_rel = quat_rotate(root_wxyz_0_inv, root_pos_rel)
            root_lin_vel_rel = quat_rotate(root_wxyz_0_inv, root_lin_vel)
            result = torch.cat([root_pos_rel, root_lin_vel_rel, root_wxyz_rel, jstates], dim=-1)
        return result

    @staticmethod
    def accumulate_chunk_in_root_frame(
        chunk_local: torch.Tensor,
        root_pos_ref: torch.Tensor,
        root_rot6d_ref: torch.Tensor,
        xy_only: bool = True,
        yaw_only: bool = False,
    ) -> torch.Tensor:
        """
        Map a chunk expressed **relative to its first frame** (same convention as
        :meth:`make_relative` output) into the **rollout root frame** used by ``traj``:
        positions and rotations are relative to timestep 0 of the full trajectory.

        Joint blocks (jpos, jvel) are copied unchanged.

        Parameters
        ----------
        chunk_local
            ``(..., T, D)`` with root pos/rot6d relative to frame 0 of this chunk.
        root_pos_ref, root_rot6d_ref
            Global-encoding root pose at that chunk's frame 0, ``(..., 3)`` and ``(..., 6)``.
        xy_only
            Must match ``xy_only`` used by :meth:`make_relative`. With ``xy_only=True`` (default),
            only x/y translation is anchored to frame 0.
        yaw_only
            Must match ``yaw_only`` used by :meth:`make_relative`. If ``True``, compose with
            the yaw-only component of ``root_rot6d_ref``.
        """
        pos_l = chunk_local[..., :ROOT_POS_DIM]
        vel_l = chunk_local[..., ROOT_POS_DIM:ROOT_ROT_OFFSET]
        rot6d_l = chunk_local[..., ROOT_ROT_OFFSET:POSE_BASE_DIM]
        tail = chunk_local[..., POSE_BASE_DIM:]
        if yaw_only:
            R_ref = yaw_matrix(rot6d_to_matrix(root_rot6d_ref))
        else:
            R_ref = rot6d_to_matrix(root_rot6d_ref)
        R_l = rot6d_to_matrix(rot6d_l)
        root_pos_anchor = root_pos_ref.unsqueeze(-2)
        if xy_only:
            root_pos_anchor = root_pos_anchor * torch.tensor(
                [1.0, 1.0, 0.0],
                device=root_pos_ref.device,
                dtype=root_pos_ref.dtype,
            )
        pos_g = (
            torch.matmul(R_ref.unsqueeze(-3), pos_l.unsqueeze(-1)).squeeze(-1)
            + root_pos_anchor
        )
        vel_g = torch.matmul(R_ref.unsqueeze(-3), vel_l.unsqueeze(-1)).squeeze(-1)
        R_g = torch.matmul(R_ref.unsqueeze(-3), R_l)
        rot6d_g = rot6d_from_matrix(R_g)
        return torch.cat([pos_g, vel_g, rot6d_g, tail], dim=-1)

    def trajectory_to_lafan1_csv_qpos(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Convert a processed trajectory (root pos, root_lin_vel, rot6d, jpos, jvel) to the
        retargeting CSV
        **qpos** row layout: ``[x, y, z, qx, qy, qz, qw]`` (translation + quaternion **xyzw** +
        ``jpos``). **jvel is dropped** — original clips do not store it.
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
            quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
            jpos = traj[..., POSE_BASE_DIM:POSE_BASE_DIM + self.n_joints]
        else:
            quat_wxyz = traj[..., ROOT_ROT_OFFSET:ROOT_ROT_OFFSET + 4]
            quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
            jpos = traj[..., ROOT_ROT_OFFSET + 4:ROOT_ROT_OFFSET + 4 + self.n_joints]
        return torch.cat([pos, quat_xyzw, jpos], dim=-1)

    def normalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Z-score root position/velocity (relative space), jpos, and jvel; root orientation unchanged."""
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
        jpos_mean, jpos_std = self._jpos_mean.to(device=device, dtype=dtype), self._jpos_std.to(device=device, dtype=dtype)
        out[..., lo:mid] = (out[..., lo:mid] - jpos_mean) / jpos_std.clamp_min(1e-6)
        jvel_mean, jvel_std = self._jvel_mean.to(device=device, dtype=dtype), self._jvel_std.to(device=device, dtype=dtype)
        out[..., mid:] = (out[..., mid:] - jvel_mean) / jvel_std.clamp_min(1e-6)
        return out

    def denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`normalize` for root position/velocity, jpos, and jvel."""
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
        jpos_mean, jpos_std = self._jpos_mean.to(device=device, dtype=dtype), self._jpos_std.to(device=device, dtype=dtype)
        jvel_mean, jvel_std = self._jvel_mean.to(device=device, dtype=dtype), self._jvel_std.to(device=device, dtype=dtype)
        out[..., lo:mid] = out[..., lo:mid] * jpos_std + jpos_mean
        out[..., mid:] = out[..., mid:] * jvel_std + jvel_mean
        return out
    
    def compute_metrics(self, trajectory: torch.Tensor) -> dict[str, float]:
        """
        Compute finite-difference consistency metrics for root and joint velocities.

        This compares provided velocity channels against the finite-difference estimate
        from position channels using the dataset FPS convention.
        """
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
        jpos = trajectory[..., :, POSE_BASE_DIM:POSE_BASE_DIM + self.n_joints]
        jvel = trajectory[..., :, POSE_BASE_DIM + self.n_joints:]

        root_vel_fd = torch.zeros_like(root_vel)
        root_vel_fd[..., 0, :] = (root_pos[..., 1, :] - root_pos[..., 0, :]) * fp
        root_vel_fd[..., 1:, :] = (root_pos[..., 1:, :] - root_pos[..., :-1, :]) * fp

        joint_vel_fd = torch.zeros_like(jvel)
        joint_vel_fd[..., 0, :] = (jpos[..., 1, :] - jpos[..., 0, :]) * fp
        joint_vel_fd[..., 1:, :] = (jpos[..., 1:, :] - jpos[..., :-1, :]) * fp

        root_vel_fd_mse = torch.mean((root_vel - root_vel_fd) ** 2).item()
        joint_vel_fd_mse = torch.mean((jvel - joint_vel_fd) ** 2).item()
        return {
            "root_vel_fd_mse": float(root_vel_fd_mse),
            "joint_vel_fd_mse": float(joint_vel_fd_mse),
        }


def _num_windows(n_rows: int, seq_len: int, stride: int) -> int:
    if n_rows < seq_len:
        return 0
    return (n_rows - seq_len) // stride + 1
