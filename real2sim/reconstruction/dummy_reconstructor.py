from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from real2sim.io_utils import ensure_dir, write_text


@dataclass(frozen=True)
class ReconstructionResult:
    mesh_path: Path | None
    gaussian_path: Path | None


def _box_vertices(size: np.ndarray) -> np.ndarray:
    sx, sy, sz = np.asarray(size, dtype=np.float64) * 0.5
    return np.asarray(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=np.float64,
    )


def write_box_obj(path: str | Path, *, size: np.ndarray) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    verts = _box_vertices(size)
    faces = [(1, 2, 3, 4), (5, 8, 7, 6), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 8, 4), (4, 8, 5, 1)]
    lines = ["# Dummy Real2Sim box mesh"]
    lines += [f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}" for v in verts]
    lines += [f"f {' '.join(str(i) for i in face)}" for face in faces]
    return write_text(path, "\n".join(lines) + "\n")


def write_pointcloud_ply(path: str | Path, points: np.ndarray, *, rgb: np.ndarray | None = None) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    points = np.asarray(points, dtype=np.float64)
    if rgb is None:
        colors = np.full((points.shape[0], 3), 200, dtype=np.uint8)
    else:
        colors = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        if colors.shape[0] != points.shape[0]:
            colors = np.full((points.shape[0], 3), 200, dtype=np.uint8)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    body = [
        f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {int(c[0])} {int(c[1])} {int(c[2])}"
        for p, c in zip(points, colors, strict=False)
    ]
    return write_text(path, "\n".join(header + body) + "\n")


class DummyReconstructor:
    def reconstruct(
        self,
        *,
        visual_dir: str | Path,
        center_camera: np.ndarray,
        size_camera: np.ndarray,
        points_camera: np.ndarray,
        save_visual_mesh: bool,
        save_gaussian: bool,
    ) -> ReconstructionResult:
        visual_dir = ensure_dir(visual_dir)
        mesh_path = write_box_obj(visual_dir / "object_mesh.obj", size=size_camera) if save_visual_mesh else None
        gaussian_path = (
            write_pointcloud_ply(visual_dir / "object_gaussian.ply", points_camera) if save_gaussian else None
        )
        return ReconstructionResult(mesh_path=mesh_path, gaussian_path=gaussian_path)
