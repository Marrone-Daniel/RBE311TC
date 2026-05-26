from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from real2sim.io_utils import ensure_dir


def save_mask_overlay(path: str | Path, rgb: np.ndarray, mask: np.ndarray) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    rgb = np.asarray(rgb, dtype=np.uint8)
    overlay = rgb.copy()
    overlay[np.asarray(mask, dtype=bool)] = (0.4 * overlay[np.asarray(mask, dtype=bool)] + np.array([255, 0, 0]) * 0.6).astype(
        np.uint8
    )
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path.as_posix(), bgr):
        raise RuntimeError(f"Failed to write overlay: {path}")
    return path


def save_mask_check(path: str | Path, mask: np.ndarray) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    image = (np.asarray(mask, dtype=np.uint8) * 255)
    if not cv2.imwrite(path.as_posix(), image):
        raise RuntimeError(f"Failed to write mask check: {path}")
    return path
