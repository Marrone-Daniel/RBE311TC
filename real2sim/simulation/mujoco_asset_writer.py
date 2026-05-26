from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from real2sim.io_utils import write_text


def write_mujoco_box_xml(
    path: str | Path,
    *,
    object_name: str,
    position: np.ndarray,
    quat_wxyz: np.ndarray,
    half_size: np.ndarray,
    mass: float,
    friction: list[float],
    rgba: list[float],
) -> Path:
    pos = " ".join(f"{float(v):.8f}" for v in position)
    quat = " ".join(f"{float(v):.8f}" for v in quat_wxyz)
    size = " ".join(f"{float(v):.8f}" for v in half_size)
    friction_s = " ".join(f"{float(v):.8f}" for v in friction)
    rgba_s = " ".join(f"{float(v):.8f}" for v in rgba)
    name = escape(object_name)
    xml = f"""<body name="{name}" pos="{pos}" quat="{quat}">
  <freejoint/>
  <geom name="{name}_collision"
        type="box"
        size="{size}"
        mass="{float(mass):.8f}"
        friction="{friction_s}"
        rgba="{rgba_s}"/>
  <site name="{name}_site"
        pos="0 0 0"
        size="0.01"
        rgba="0 1 0 1"/>
</body>
"""
    return write_text(path, xml)
