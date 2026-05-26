from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    mask = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask image: {path}")
    return mask > 0
