from __future__ import annotations

from pathlib import Path
from typing import Any

from real2sim.io_utils import dump_mapping, write_text


def _normalize_asset(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "role": spec.get("role", "static_scene"),
        "mujoco_body": spec.get("mujoco_body", name),
        "mujoco_xml": spec.get("mujoco_xml", ""),
        "mesh_file": spec.get("mesh_file", ""),
        "gaussian_file": spec.get("gaussian_file", ""),
        "binding_mode": spec.get("binding_mode", "mesh_vertices"),
        "T_world_asset": spec.get(
            "T_world_asset",
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ),
        "scale": spec.get("scale", [1.0, 1.0, 1.0]),
        "physics_source": spec.get("physics_source", "existing_mujoco"),
        "visual_source": spec.get("visual_source", "3dgs"),
    }


def write_scene_binding_config(
    path: str | Path,
    *,
    static_assets: dict[str, Any],
) -> Path:
    assets = {}
    for name, spec in static_assets.items():
        if not isinstance(spec, dict):
            raise RuntimeError(f"scene.static_assets.{name} must be a mapping")
        assets[name] = _normalize_asset(name, spec)
    return dump_mapping(
        path,
        {
            "scene_gs_binding": {
                "description": "Existing MuJoCo/mesh assets keep physics; 3DGS assets are bound as visual layers.",
                "assets": assets,
            }
        },
    )


def write_scene_binding_report(
    path: str | Path,
    *,
    static_assets: dict[str, Any],
    warnings: list[str],
) -> Path:
    lines = [
        "# Scene 3DGS Binding Report",
        "",
        "Static FR5/table/base assets are treated as existing simulation geometry.",
        "The generated binding config records which Gaussian visual asset should follow each existing MuJoCo body or mesh.",
        "",
        "## Assets",
        "",
    ]
    if not static_assets:
        lines.append("- No static assets configured.")
    for name, spec in static_assets.items():
        spec = spec if isinstance(spec, dict) else {}
        lines.extend(
            [
                f"- `{name}`",
                f"  - MuJoCo body: `{spec.get('mujoco_body', name)}`",
                f"  - MuJoCo XML: `{spec.get('mujoco_xml', '')}`",
                f"  - mesh: `{spec.get('mesh_file', '')}`",
                f"  - gaussian: `{spec.get('gaussian_file', '')}`",
                f"  - binding: `{spec.get('binding_mode', 'mesh_vertices')}`",
            ]
        )
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    return write_text(path, "\n".join(lines) + "\n")
