from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from real2sim.appearance.projection import ProjectedSamples


@dataclass(frozen=True)
class AtlasResult:
    partial_bgr: np.ndarray
    completed_bgr: np.ndarray
    visibility_mask: np.ndarray
    coverage_ratio: float
    target_tiles: dict[str, list[int]]


def _tile_rect(index: int, count: int, atlas_size: int) -> tuple[int, int, int, int]:
    cols = int(np.ceil(np.sqrt(max(1, count))))
    rows = int(np.ceil(count / cols))
    tile_w = atlas_size // cols
    tile_h = atlas_size // rows
    row, col = divmod(index, cols)
    x0, y0 = col * tile_w, row * tile_h
    x1 = atlas_size if col == cols - 1 else (col + 1) * tile_w
    y1 = atlas_size if row == rows - 1 else (row + 1) * tile_h
    return x0, y0, x1, y1


def build_projected_atlas(
    image_bgr: np.ndarray,
    projected: list[ProjectedSamples],
    *,
    atlas_size: int,
    inpaint_radius: int = 7,
) -> AtlasResult:
    atlas_size = int(atlas_size)
    partial = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    accum = np.zeros((atlas_size, atlas_size, 3), dtype=np.float64)
    counts = np.zeros((atlas_size, atlas_size), dtype=np.float64)
    targets = sorted({item.sample.body_name for item in projected})
    target_to_tile = {name: idx for idx, name in enumerate(targets)}
    target_tiles = {name: list(_tile_rect(idx, len(targets), atlas_size)) for name, idx in target_to_tile.items()}

    h, w = image_bgr.shape[:2]
    for item in projected:
        x0, y0, x1, y1 = _tile_rect(target_to_tile[item.sample.body_name], len(targets), atlas_size)
        uv_atlas = item.sample.uv_local.copy()
        uv_atlas[:, 0] = x0 + uv_atlas[:, 0] * max(1, x1 - x0 - 1)
        uv_atlas[:, 1] = y0 + (1.0 - uv_atlas[:, 1]) * max(1, y1 - y0 - 1)
        atlas_px = np.rint(uv_atlas).astype(np.int64)
        image_px = np.rint(item.uv_image).astype(np.int64)
        valid = item.visible
        for idx in np.nonzero(valid)[0]:
            ax, ay = int(atlas_px[idx, 0]), int(atlas_px[idx, 1])
            ix, iy = int(image_px[idx, 0]), int(image_px[idx, 1])
            if not (0 <= ax < atlas_size and 0 <= ay < atlas_size and 0 <= ix < w and 0 <= iy < h):
                continue
            accum[ay, ax] += image_bgr[iy, ix].astype(np.float64)
            counts[ay, ax] += 1.0

    known = counts > 0
    partial[known] = np.clip(accum[known] / counts[known, None], 0, 255).astype(np.uint8)
    if known.any():
        # Fill tiny holes before global inpainting; this avoids isolated black pixels.
        known_u8 = known.astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        grow = cv2.dilate(known_u8, kernel, iterations=1).astype(bool)
        blurred = cv2.GaussianBlur(partial, (0, 0), 1.0)
        partial[~known & grow] = blurred[~known & grow]
        known = known | grow

    missing = (~known).astype(np.uint8) * 255
    if missing.any():
        completed = cv2.inpaint(partial, missing, float(inpaint_radius), cv2.INPAINT_TELEA)
    else:
        completed = partial.copy()
    completed = cv2.GaussianBlur(completed, (0, 0), 0.35)
    return AtlasResult(
        partial_bgr=partial,
        completed_bgr=completed,
        visibility_mask=(known.astype(np.uint8) * 255),
        coverage_ratio=float(known.mean()),
        target_tiles=target_tiles,
    )
