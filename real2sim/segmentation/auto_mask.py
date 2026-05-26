from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_u8.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    return labels == idx


def auto_grabcut_mask(
    image_bgr: np.ndarray,
    *,
    bbox: tuple[int, int, int, int] | None = None,
    margin_ratio: float = 0.08,
    iterations: int = 5,
) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    h, w = image.shape[:2]
    if bbox is None:
        mx = max(1, int(w * float(margin_ratio)))
        my = max(1, int(h * float(margin_ratio)))
        rect = (mx, my, max(2, w - 2 * mx), max(2, h - 2 * my))
    else:
        x, y, bw, bh = [int(v) for v in bbox]
        rect = (max(0, x), max(0, y), max(2, min(bw, w - x)), max(2, min(bh, h - y)))

    grab = np.zeros((h, w), dtype=np.uint8)
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(image, grab, rect, bgd, fgd, int(iterations), cv2.GC_INIT_WITH_RECT)
    mask = (grab == cv2.GC_FGD) | (grab == cv2.GC_PR_FGD)
    mask = largest_component(mask)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    return largest_component(mask)


def save_mask_debug(path: str | Path, image_bgr: np.ndarray, mask: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = image_bgr.copy()
    m = np.asarray(mask, dtype=bool)
    overlay[m] = (0.45 * overlay[m] + 0.55 * np.array([0, 0, 255], dtype=np.float32)).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    if not cv2.imwrite(path.as_posix(), overlay):
        raise RuntimeError(f"Failed to write mask debug: {path}")
    return path
