from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObjectEstimate:
    center_camera: np.ndarray
    size_camera: np.ndarray
    T_camera_object: np.ndarray


def estimate_axis_aligned_object(points_camera: np.ndarray, *, scale_factor: float = 1.0) -> ObjectEstimate:
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise RuntimeError(f"Expected Nx3 point cloud, got {points.shape}")
    lo = np.percentile(points, 2.0, axis=0)
    hi = np.percentile(points, 98.0, axis=0)
    center = (lo + hi) * 0.5
    size = np.maximum((hi - lo) * float(scale_factor), 1e-4)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = center
    return ObjectEstimate(center_camera=center, size_camera=size, T_camera_object=T)
