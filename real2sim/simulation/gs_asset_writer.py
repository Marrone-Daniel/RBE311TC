from __future__ import annotations

from pathlib import Path

import numpy as np

from real2sim.io_utils import dump_mapping


def write_gs_asset_config(
    path: str | Path,
    *,
    object_name: str,
    gaussian_path: str | None,
    mesh_path: str | None,
    T_world_object: np.ndarray,
) -> Path:
    data = {
        "object": {
            "name": object_name,
            "gaussian": gaussian_path or "",
            "mesh": mesh_path or "",
            "T_world_object": [[float(v) for v in row] for row in np.asarray(T_world_object)],
        }
    }
    return dump_mapping(path, data)
