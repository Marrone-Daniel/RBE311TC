from __future__ import annotations

from pathlib import Path

import numpy as np

from real2sim.io_utils import write_json
from real2sim.reconstruction.dummy_reconstructor import write_box_obj


def generate_box_collision(
    *,
    collision_dir: str | Path,
    object_name: str,
    center_camera: np.ndarray,
    size_camera: np.ndarray,
    size_world: np.ndarray,
) -> dict:
    collision_dir = Path(collision_dir)
    half_size = np.asarray(size_world, dtype=np.float64) * 0.5
    mesh_path = write_box_obj(collision_dir / "collision_mesh.obj", size=np.asarray(size_world, dtype=np.float64))
    metadata = {
        "object_name": object_name,
        "collision_type": "box",
        "center_camera": [float(v) for v in center_camera],
        "size_camera": [float(v) for v in size_camera],
        "size_world": [float(v) for v in size_world],
        "mujoco_half_size": [float(v) for v in half_size],
        "collision_mesh": mesh_path.name,
    }
    write_json(collision_dir / "collision_metadata.json", metadata)
    return metadata
