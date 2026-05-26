from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    T_world_camera: np.ndarray

    @property
    def R_world_camera(self) -> np.ndarray:
        return self.T_world_camera[:3, :3]

    @property
    def t_world_camera(self) -> np.ndarray:
        return self.T_world_camera[:3, 3]

    def world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        return (points - self.t_world_camera.reshape(1, 3)) @ self.R_world_camera

    def project_world(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points_camera = self.world_to_camera(points_world)
        z = points_camera[:, 2]
        uv = np.empty((points_camera.shape[0], 2), dtype=np.float64)
        uv[:, 0] = self.fx * points_camera[:, 0] / np.maximum(z, 1e-9) + self.cx
        uv[:, 1] = self.fy * points_camera[:, 1] / np.maximum(z, 1e-9) + self.cy
        valid = (
            (z > 1e-5)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < self.width)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < self.height)
        )
        return uv, valid


def load_astra_camera_config(path: str | Path) -> PinholeCamera:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    intr = cfg.get("intrinsics", {})
    ext = cfg.get("extrinsics", {})
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(ext.get("rotation_matrix", np.eye(3)), dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(ext.get("position", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    return PinholeCamera(
        width=int(intr.get("width", 640)),
        height=int(intr.get("height", 480)),
        fx=float(intr.get("fx", 525.0)),
        fy=float(intr.get("fy", 525.0)),
        cx=float(intr.get("cx", (int(intr.get("width", 640)) - 1) * 0.5)),
        cy=float(intr.get("cy", (int(intr.get("height", 480)) - 1) * 0.5)),
        T_world_camera=T,
    )
