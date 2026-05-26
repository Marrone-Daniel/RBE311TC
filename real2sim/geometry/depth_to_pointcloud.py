from __future__ import annotations

import numpy as np

from .camera import CameraIntrinsics


def masked_depth_to_pointcloud(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    max_points: int | None = None,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        raise RuntimeError("Mask/depth produced zero valid 3D points")
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (ys.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    points = np.stack([x, y, z], axis=1)
    if max_points is not None and points.shape[0] > int(max_points):
        idx = np.linspace(0, points.shape[0] - 1, int(max_points)).astype(np.int64)
        points = points[idx]
    return points
