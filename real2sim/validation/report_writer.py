from __future__ import annotations

from pathlib import Path
from typing import Iterable

from real2sim.io_utils import write_text


def write_report(
    path: str | Path,
    *,
    input_files: dict[str, str],
    mask_pixels: int,
    center_camera: list[float],
    size_camera: list[float],
    center_world: list[float],
    size_world: list[float],
    camera_pose_used: str,
    generated_files: Iterable[str],
    warnings: Iterable[str],
) -> Path:
    warning_lines = [f"- {item}" for item in warnings] or ["- None"]
    file_lines = [f"- `{item}`" for item in generated_files]
    text = f"""# Real2Sim Report

## Inputs

- RGB: `{input_files.get('rgb', '')}`
- Depth: `{input_files.get('depth', '')}`
- Mask: `{input_files.get('mask', '')}`
- Intrinsics: `{input_files.get('intrinsics', '')}`
- Camera pose: `{input_files.get('camera_pose', '')}`

## Estimated Object

- Mask pixels: `{mask_pixels}`
- Center camera: `{center_camera}`
- Size camera: `{size_camera}`
- Center world: `{center_world}`
- Size world: `{size_world}`
- Camera/world transform: `{camera_pose_used}`

## Generated Files

{chr(10).join(file_lines)}

## Warnings And Limitations

{chr(10).join(warning_lines)}

- Dummy mode uses a masked depth point cloud and an axis-aligned bounding box.
- Visual reconstruction is a placeholder box mesh unless an external backend is implemented.
- Collision geometry is simplified and should be checked before contact-rich simulation.
"""
    return write_text(path, text)
