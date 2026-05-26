from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from real2sim.io_utils import ensure_dir
from real2sim.segmentation.auto_mask import auto_grabcut_mask, save_mask_debug


def read_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(Path(path).as_posix(), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read RGB image: {path}")
    return image


def make_mask(
    image_bgr: np.ndarray,
    *,
    mode: str = "full",
    bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if mode == "full":
        return np.ones((h, w), dtype=bool)
    if mode == "roi":
        if bbox is None:
            raise RuntimeError("--bbox X Y W H is required when --mask-mode roi")
        x, y, bw, bh = [int(v) for v in bbox]
        mask = np.zeros((h, w), dtype=np.uint8)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x0 + max(1, bw)), min(h, y0 + max(1, bh))
        mask[y0:y1, x0:x1] = 1
        return mask.astype(bool)
    if mode == "auto":
        return auto_grabcut_mask(image_bgr, bbox=bbox)
    raise RuntimeError(f"Unsupported mask mode: {mode}")


def crop_to_mask(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise RuntimeError("Appearance mask is empty.")
    pad = max(8, int(0.04 * max(image_bgr.shape[:2])))
    x0, x1 = max(0, int(xs.min()) - pad), min(image_bgr.shape[1], int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(image_bgr.shape[0], int(ys.max()) + pad + 1)
    return image_bgr[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def complete_texture_atlas(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    atlas_size: int = 1024,
    inpaint_radius: int = 7,
) -> np.ndarray:
    crop, crop_mask = crop_to_mask(image_bgr, mask)
    texture = cv2.resize(crop, (int(atlas_size), int(atlas_size)), interpolation=cv2.INTER_CUBIC)
    known = cv2.resize(crop_mask.astype(np.uint8), (int(atlas_size), int(atlas_size)), interpolation=cv2.INTER_NEAREST) > 0

    unknown = (~known).astype(np.uint8) * 255
    if unknown.any():
        texture = cv2.inpaint(texture, unknown, float(inpaint_radius), cv2.INPAINT_TELEA)

    # Smooth only enough to hide hard segmentation seams, while preserving the
    # real captured appearance as the dominant signal.
    soft = cv2.GaussianBlur(texture, (0, 0), 0.55)
    texture = cv2.addWeighted(texture, 0.82, soft, 0.18, 0)
    return texture


def save_texture_outputs(
    *,
    image_path: str | Path,
    output_texture: str | Path,
    debug_overlay: str | Path,
    mask_path: str | Path,
    mask_mode: str,
    bbox: tuple[int, int, int, int] | None,
    atlas_size: int,
) -> dict[str, Path | int]:
    image_bgr = read_bgr(image_path)
    mask = make_mask(image_bgr, mode=mask_mode, bbox=bbox)
    texture = complete_texture_atlas(image_bgr, mask, atlas_size=int(atlas_size))

    output_texture = Path(output_texture)
    ensure_dir(output_texture.parent)
    if not cv2.imwrite(output_texture.as_posix(), texture):
        raise RuntimeError(f"Failed to write texture atlas: {output_texture}")

    mask_path = Path(mask_path)
    ensure_dir(mask_path.parent)
    if not cv2.imwrite(mask_path.as_posix(), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"Failed to write appearance mask: {mask_path}")

    overlay = save_mask_debug(debug_overlay, image_bgr, mask)
    return {
        "texture": output_texture,
        "mask": mask_path,
        "overlay": overlay,
        "mask_pixels": int(mask.sum()),
    }
