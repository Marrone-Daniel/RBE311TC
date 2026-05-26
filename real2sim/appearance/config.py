from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from real2sim.io_utils import load_mapping


@dataclass(frozen=True)
class AppearanceTarget:
    name: str
    role: str = "static"
    enabled: bool = True


@dataclass(frozen=True)
class GaussianAppearanceConfig:
    enabled: bool = True
    points_per_geom: int = 12000
    scale: float = 0.00085
    opacity: float = 0.58
    recolor_blend: float = 0.88
    regenerate: bool = False


@dataclass(frozen=True)
class AppearancePipelineConfig:
    atlas_size: int = 2048
    samples_per_box_face: int = 32
    z_tolerance_m: float = 0.015
    texture_name: str = "fr5_appearance_tex"
    material_name: str = "fr5_appearance_mat"
    texture_file: str = "textures/fr5_static_appearance_atlas.png"
    targets: list[AppearanceTarget] = field(
        default_factory=lambda: [
            AppearanceTarget("grooved_table", "table"),
            AppearanceTarget("fr5_fixed_base", "fixed_base"),
        ]
    )
    robot_adapter_enabled: bool = False
    dynamic_object_adapter_enabled: bool = True
    gaussian: GaussianAppearanceConfig = field(default_factory=GaussianAppearanceConfig)

    @property
    def target_names(self) -> list[str]:
        return [target.name for target in self.targets if target.enabled]


def load_appearance_config(path: str | Path | None = None) -> AppearancePipelineConfig:
    if path is None:
        return AppearancePipelineConfig()
    cfg = load_mapping(path)
    app = cfg.get("appearance", cfg)
    targets_raw = app.get("targets", None)
    targets = AppearancePipelineConfig().targets
    if isinstance(targets_raw, list):
        parsed: list[AppearanceTarget] = []
        for item in targets_raw:
            if isinstance(item, str):
                parsed.append(AppearanceTarget(item))
            elif isinstance(item, dict):
                parsed.append(
                    AppearanceTarget(
                        name=str(item["name"]),
                        role=str(item.get("role", "static")),
                        enabled=bool(item.get("enabled", True)),
                    )
                )
        if parsed:
            targets = parsed
    gs_raw = app.get("gaussian", {})
    gaussian = GaussianAppearanceConfig(
        enabled=bool(gs_raw.get("enabled", True)),
        points_per_geom=int(gs_raw.get("points_per_geom", 12000)),
        scale=float(gs_raw.get("scale", 0.00085)),
        opacity=float(gs_raw.get("opacity", 0.58)),
        recolor_blend=float(gs_raw.get("recolor_blend", 0.88)),
        regenerate=bool(gs_raw.get("regenerate", False)),
    )
    return AppearancePipelineConfig(
        atlas_size=int(app.get("atlas_size", 2048)),
        samples_per_box_face=int(app.get("samples_per_box_face", 32)),
        z_tolerance_m=float(app.get("z_tolerance_m", 0.015)),
        texture_name=str(app.get("texture_name", "fr5_appearance_tex")),
        material_name=str(app.get("material_name", "fr5_appearance_mat")),
        texture_file=str(app.get("texture_file", "textures/fr5_static_appearance_atlas.png")),
        targets=targets,
        robot_adapter_enabled=bool(app.get("robot_adapter_enabled", False)),
        dynamic_object_adapter_enabled=bool(app.get("dynamic_object_adapter_enabled", True)),
        gaussian=gaussian,
    )
