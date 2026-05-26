from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

SH_C0 = 0.28209479177387814


def _choose_uv_axes(xyz: np.ndarray) -> tuple[int, int]:
    span = np.ptp(np.asarray(xyz, dtype=np.float64), axis=0)
    axes = np.argsort(span)[-2:]
    return int(axes[0]), int(axes[1])


def _sample_texture(texture_bgr: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = texture_bgr.shape[:2]
    uv = np.mod(uv, 1.0)
    px = np.clip((uv[:, 0] * (w - 1)).astype(np.int64), 0, w - 1)
    py = np.clip(((1.0 - uv[:, 1]) * (h - 1)).astype(np.int64), 0, h - 1)
    bgr = texture_bgr[py, px].astype(np.float32) / 255.0
    return bgr[:, ::-1]


def recolor_gaussian_ply(
    ply_path: str | Path,
    texture_path: str | Path,
    *,
    blend: float = 0.88,
    backup: bool = True,
) -> Path:
    from gaussian_renderer.core.util_gau import load_ply, save_ply

    ply_path = Path(ply_path)
    texture_bgr = cv2.imread(Path(texture_path).as_posix(), cv2.IMREAD_COLOR)
    if texture_bgr is None:
        raise RuntimeError(f"Cannot read texture atlas: {texture_path}")
    gaussian = load_ply(ply_path)
    xyz = np.asarray(gaussian.xyz, dtype=np.float32)
    if xyz.size == 0:
        return ply_path

    axis_u, axis_v = _choose_uv_axes(xyz)
    coords = xyz[:, [axis_u, axis_v]]
    lo = np.min(coords, axis=0)
    hi = np.max(coords, axis=0)
    span = np.maximum(hi - lo, 1e-8)
    uv = (coords - lo.reshape(1, 2)) / span.reshape(1, 2)
    sampled = _sample_texture(texture_bgr, uv)
    old_rgb = np.clip(np.asarray(gaussian.sh, dtype=np.float32) * SH_C0 + 0.5, 0.0, 1.0)
    a = float(np.clip(blend, 0.0, 1.0))
    new_rgb = (1.0 - a) * old_rgb + a * sampled
    gaussian.sh = ((new_rgb - 0.5) / SH_C0).astype(np.float32)

    if backup:
        backup_path = ply_path.with_suffix(".before_appearance.ply")
        if not backup_path.exists():
            shutil.copy2(ply_path, backup_path)
    save_ply(gaussian, ply_path, save_sh_degree=0)
    return ply_path


def recolor_gaussian_dir(
    gs_dir: str | Path,
    texture_path: str | Path,
    *,
    body_names: list[str],
    blend: float,
    backup: bool = True,
) -> list[Path]:
    gs_dir = Path(gs_dir)
    updated: list[Path] = []
    for name in body_names:
        ply = gs_dir / f"{name}.ply"
        if ply.exists():
            updated.append(recolor_gaussian_ply(ply, texture_path, blend=blend, backup=backup))
    return updated
