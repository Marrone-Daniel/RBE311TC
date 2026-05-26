from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import load_mapping


@dataclass
class ReconstructionConfig:
    backend: str = "dummy"
    save_visual_mesh: bool = True
    save_gaussian: bool = False
    external_command: str | None = None


@dataclass
class AssetConfig:
    existing_mesh_file: str = ""
    gaussian_file: str = ""
    object_pose_file: str = "object_pose_world.yaml"
    mesh_frame: str = "object"
    gaussian_frame: str = "object"
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    binding_mode: str = "mesh_vertices"


@dataclass
class SceneConfig:
    enabled: bool = True
    export_scene_gs_config: bool = True
    static_assets: dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraConfig:
    intrinsics_file: str = "intrinsics.yaml"
    camera_pose_file: str = "camera_pose.yaml"
    default_camera_frame: str = "opencv"
    depth_scale: float | None = None


@dataclass
class ObjectConfig:
    name: str = "target_object"
    mass: float = 0.05
    friction: list[float] = field(default_factory=lambda: [1.0, 0.005, 0.0001])
    collision_type: str = "box"
    scale_factor: float = 1.0
    min_size: list[float] = field(default_factory=lambda: [0.005, 0.005, 0.005])
    rgba: list[float] = field(default_factory=lambda: [0.8, 0.3, 0.2, 1.0])


@dataclass
class SimulationConfig:
    engine: str = "mujoco"
    export_mujoco_xml: bool = True
    export_gs_config: bool = True


@dataclass
class DebugConfig:
    save_overlay: bool = True
    save_pointcloud: bool = True
    save_report: bool = True
    max_debug_points: int = 200000


@dataclass
class Real2SimConfig:
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    asset: AssetConfig = field(default_factory=AssetConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    object: ObjectConfig = field(default_factory=ObjectConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise RuntimeError(f"Config section '{name}' must be a mapping")
    return section


def load_real2sim_config(path: str | Path | None) -> Real2SimConfig:
    data = load_mapping(path) if path else {}
    return Real2SimConfig(
        reconstruction=ReconstructionConfig(**_section(data, "reconstruction")),
        asset=AssetConfig(**_section(data, "asset")),
        scene=SceneConfig(**_section(data, "scene")),
        camera=CameraConfig(**_section(data, "camera")),
        object=ObjectConfig(**_section(data, "object")),
        simulation=SimulationConfig(**_section(data, "simulation")),
        debug=DebugConfig(**_section(data, "debug")),
    )
