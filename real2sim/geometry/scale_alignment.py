from __future__ import annotations

import numpy as np


def apply_scale(size: np.ndarray, scale_factor: float) -> np.ndarray:
    return np.maximum(np.asarray(size, dtype=np.float64) * float(scale_factor), 1e-4)
