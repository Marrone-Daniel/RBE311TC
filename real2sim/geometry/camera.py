from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from real2sim.io_utils import load_mapping


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    data = load_mapping(path)
    if "intrinsics" in data and isinstance(data["intrinsics"], dict):
        data = data["intrinsics"]
    if "fx" not in data and "fu" in data:
        data["fx"] = data["fu"]
    if "fy" not in data and "fv" in data:
        data["fy"] = data["fv"]
    if "cx" not in data and "pu" in data:
        data["cx"] = data["pu"]
    if "cy" not in data and "pv" in data:
        data["cy"] = data["pv"]
    required = ("width", "height", "fx", "fy", "cx", "cy")
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"Missing camera intrinsic keys in {path}: {missing}")
    return CameraIntrinsics(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
    )


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise RuntimeError(f"Quaternion must have 4 values, got {q.shape}")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise RuntimeError("Quaternion norm is zero")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_camera_pose(path: str | Path) -> tuple[np.ndarray, list[str]]:
    path = Path(path)
    warnings: list[str] = []
    if not path.exists():
        warnings.append(f"{path.name} not found; using identity T_world_camera.")
        return np.eye(4, dtype=np.float64), warnings
    data = load_mapping(path)
    matrix = data.get("T_world_camera") or data.get("matrix")
    if matrix is not None:
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.shape != (4, 4):
            raise RuntimeError(f"Camera pose matrix must be 4x4, got {arr.shape}")
        return arr, warnings

    position = data.get("position") or data.get("translation") or [0.0, 0.0, 0.0]
    rotation = data.get("rotation_matrix")
    if rotation is not None:
        rot = np.asarray(rotation, dtype=np.float64)
        if rot.shape != (3, 3):
            raise RuntimeError(f"rotation_matrix must be 3x3, got {rot.shape}")
    elif "quaternion_wxyz" in data:
        rot = _quat_wxyz_to_matrix(np.asarray(data["quaternion_wxyz"], dtype=np.float64))
    elif "quaternion_xyzw" in data:
        x, y, z, w = np.asarray(data["quaternion_xyzw"], dtype=np.float64)
        rot = _quat_wxyz_to_matrix(np.asarray([w, x, y, z], dtype=np.float64))
    else:
        rot = np.eye(3, dtype=np.float64)
        warnings.append(f"{path.name} has no rotation; using identity rotation.")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    return T, warnings


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return points.reshape(0, 3)
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (np.asarray(T, dtype=np.float64) @ homog.T).T[:, :3]
