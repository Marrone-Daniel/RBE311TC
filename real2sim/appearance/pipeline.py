from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from real2sim.appearance.atlas import build_projected_atlas
from real2sim.appearance.camera import load_astra_camera_config
from real2sim.appearance.config import AppearancePipelineConfig
from real2sim.appearance.fr5_mujoco_texture import apply_texture_material_to_fr5_xml
from real2sim.appearance.gs_recolor import recolor_gaussian_dir
from real2sim.appearance.mujoco_geometry import sample_target_box_geoms
from real2sim.appearance.projection import compute_visibility, draw_projected_overlay
from real2sim.io_utils import dump_mapping, ensure_dir, write_json, write_text


def _relative_to_model(texture_path: Path, model_xml: Path) -> str:
    try:
        return texture_path.relative_to(model_xml.parent).as_posix()
    except ValueError:
        return texture_path.as_posix()


def _safe_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _get_sample_name(item: Any) -> str:
    """
    The exact projected/sample dataclass in this project is not guaranteed.
    This helper tries common attribute names instead of hard-coding one schema.
    """
    sample = getattr(item, "sample", item)

    for obj in (sample, item):
        for key in (
            "target_name",
            "body_name",
            "geom_name",
            "name",
            "label",
            "source_name",
        ):
            value = getattr(obj, key, None)
            if value:
                return str(value)

    # Last-resort fallback: keep stable but explicit.
    return "unknown_target"


def _matches_target(item: Any, target: str) -> bool:
    name = _get_sample_name(item)
    if name == target:
        return True
    # In many MJCF files a target body contains many geoms whose names include body name.
    return target in name or name in target


def _filter_projected_by_target(projected: list[Any], target: str) -> list[Any]:
    selected = [item for item in projected if _matches_target(item, target)]
    return selected


def _load_mask(mask_dir: Path, target: str, image_shape: tuple[int, int]) -> np.ndarray | None:
    """
    Optional manual mask support.

    Expected names:
      masks/<target>.png
      masks/<target>_mask.png

    White/nonzero = allowed pixels.
    """
    candidates = [
        mask_dir / f"{target}.png",
        mask_dir / f"{target}_mask.png",
        mask_dir / f"{_safe_name(target)}.png",
        mask_dir / f"{_safe_name(target)}_mask.png",
    ]
    for p in candidates:
        if p.exists():
            mask = cv2.imread(p.as_posix(), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read mask: {p}")
            h, w = image_shape
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            return (mask > 0).astype(np.uint8) * 255
    return None


def _apply_soft_mask_to_image(image_bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """
    Current atlas builder samples from the input image directly.
    If we have a manual mask, black out non-target pixels to reduce cross-object color bleeding.
    This is not perfect UV baking, but it is much safer than global mixed projection.
    """
    if mask is None:
        return image_bgr
    out = image_bgr.copy()
    out[mask == 0] = 0
    return out


def _write_target_overlay(
    image_bgr: np.ndarray,
    projected_items: list[Any],
    out_path: Path,
    mask: np.ndarray | None = None,
) -> None:
    overlay = draw_projected_overlay(image_bgr, projected_items)
    if mask is not None:
        # show mask boundary in green
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    ensure_dir(out_path.parent)
    if not cv2.imwrite(out_path.as_posix(), overlay):
        raise RuntimeError(f"Failed to write overlay: {out_path}")


def _count_projected(projected_items: list[Any]) -> tuple[int, int]:
    total = 0
    visible = 0
    for item in projected_items:
        sample = getattr(item, "sample", None)
        points = getattr(sample, "points_world", None)
        if points is not None:
            total += int(points.shape[0])
        visible_arr = getattr(item, "visible", None)
        if visible_arr is not None:
            visible += int(np.asarray(visible_arr).sum())
    return total, visible


def update_fr5_task_config(
    path: str | Path,
    *,
    config: AppearancePipelineConfig,
    texture_path: Path,
) -> Path:
    path = Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg.pop("cube_site", None)
    cfg.setdefault("fr5_3dgs", {})
    cfg["fr5_3dgs"].update(
        {
            "points_per_geom": int(config.gaussian.points_per_geom),
            "scale": float(config.gaussian.scale),
            "opacity": float(config.gaussian.opacity),
            "regenerate_on_missing": True,
            "appearance_transfer": {
                "enabled": True,
                "mode": "mesh_conditioned_projection_safe_per_target",
                "texture_atlas": texture_path.as_posix(),
                "targets": config.target_names,
                "robot_adapter_enabled": bool(config.robot_adapter_enabled),
                "dynamic_object_adapter_enabled": bool(config.dynamic_object_adapter_enabled),
                "recolor_blend": float(config.gaussian.recolor_blend),
                "xml_material": config.material_name,
                "xml_texture": config.texture_name,
                "warning": "This is still projection-based appearance transfer; verify debug overlays before using for training.",
            },
        }
    )
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def regenerate_gaussians(
    *,
    generator_script: Path,
    model_xml: Path,
    gs_dir: Path,
    config: AppearancePipelineConfig,
) -> None:
    cmd = [
        sys.executable,
        generator_script.as_posix(),
        "--model-xml",
        model_xml.as_posix(),
        "--output-dir",
        gs_dir.as_posix(),
        "--points-per-geom",
        str(int(config.gaussian.points_per_geom)),
        "--scale",
        f"{float(config.gaussian.scale):.8g}",
        "--opacity",
        f"{float(config.gaussian.opacity):.8g}",
    ]
    subprocess.run(cmd, cwd=generator_script.parent.as_posix(), check=True)


def run_fr5_appearance_pipeline(
    *,
    image_path: str | Path,
    camera_config: str | Path,
    model_xml: str | Path,
    output_dir: str | Path,
    texture_dir: str | Path,
    config: AppearancePipelineConfig,
    apply_to_fr5: bool = False,
    task_config: str | Path | None = None,
    gs_dir: str | Path | None = None,
    generator_script: str | Path | None = None,
) -> dict:
    """
    Safer replacement for the old global-atlas pipeline.

    Key changes:
    1. still keeps the original CLI/API;
    2. creates a global atlas for backward compatibility;
    3. additionally creates per-target atlas/overlay/visibility outputs;
    4. supports optional manual masks in <output_dir>/../masks or <image_dir>/masks;
    5. prevents automatic GS recolor unless the projection diagnostics are acceptable.

    This does NOT magically solve real UV baking for STL/OBJ meshes.
    It makes the current box-projection version debuggable and prevents one bad atlas
    from silently polluting all objects.
    """
    image_path = Path(image_path)
    camera_config = Path(camera_config)
    model_xml = Path(model_xml)
    output_dir = ensure_dir(output_dir)
    texture_dir = ensure_dir(texture_dir)
    debug_dir = ensure_dir(output_dir / "debug")
    sim_dir = ensure_dir(output_dir / "sim")
    visual_dir = ensure_dir(output_dir / "visual")
    per_target_dir = ensure_dir(output_dir / "per_target")

    image_bgr = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Cannot read input RGB image: {image_path}")

    camera = load_astra_camera_config(camera_config)

    # Current project only exposes box geom sampling. Keep it, but make the result diagnosable.
    samples = sample_target_box_geoms(
        model_xml,
        config.target_names,
        density=int(config.samples_per_box_face),
    )
    projected = compute_visibility(
        camera,
        samples,
        z_tolerance_m=float(config.z_tolerance_m),
    )

    # Backward-compatible global atlas.
    atlas = build_projected_atlas(
        image_bgr,
        projected,
        atlas_size=int(config.atlas_size),
    )

    texture_path = texture_dir / Path(config.texture_file).name
    partial_path = visual_dir / "partial_texture_atlas.png"
    visibility_path = visual_dir / "uv_visibility_mask.png"
    overlay_path = debug_dir / "projected_overlay.png"
    atlas_path = visual_dir / "texture_atlas.png"

    for path, image in (
        (texture_path, atlas.completed_bgr),
        (atlas_path, atlas.completed_bgr),
        (partial_path, atlas.partial_bgr),
        (visibility_path, atlas.visibility_mask),
        (overlay_path, draw_projected_overlay(image_bgr, projected)),
    ):
        ensure_dir(path.parent)
        if not cv2.imwrite(path.as_posix(), image):
            raise RuntimeError(f"Failed to write image: {path}")

    # Optional manual masks. This is the most important quick improvement.
    # Put masks in either:
    #   <image_folder>/masks/<target>.png
    #   <output_dir>/masks/<target>.png
    mask_dirs = [
        image_path.parent / "masks",
        output_dir / "masks",
    ]

    target_reports: dict[str, dict[str, Any]] = {}

    for target in config.target_names:
        safe = _safe_name(target)
        target_dir = ensure_dir(per_target_dir / safe)
        target_projected = _filter_projected_by_target(projected, target)

        mask = None
        mask_path_used = None
        for mask_dir in mask_dirs:
            mask = _load_mask(mask_dir, target, image_bgr.shape[:2])
            if mask is not None:
                mask_path_used = mask_dir.as_posix()
                break

        # If names are not available in the sample dataclass, fallback will be empty.
        # In that case, do not crash; write clear report.
        if not target_projected:
            target_reports[target] = {
                "target": target,
                "status": "no_projected_samples_found_for_target",
                "reason": (
                    "The sample/projection dataclass does not expose a target/body/geom name, "
                    "or sample_target_box_geoms did not generate samples for this target. "
                    "Use the global overlay to inspect alignment."
                ),
                "mask_dir_used": mask_path_used,
            }
            continue

        masked_image = _apply_soft_mask_to_image(image_bgr, mask)
        target_atlas = build_projected_atlas(
            masked_image,
            target_projected,
            atlas_size=int(config.atlas_size),
        )

        target_texture_path = target_dir / "texture_atlas.png"
        target_partial_path = target_dir / "partial_texture_atlas.png"
        target_visibility_path = target_dir / "uv_visibility_mask.png"
        target_overlay_path = target_dir / "projected_overlay.png"
        target_masked_source_path = target_dir / "masked_source.png"

        outputs = [
            (target_texture_path, target_atlas.completed_bgr),
            (target_partial_path, target_atlas.partial_bgr),
            (target_visibility_path, target_atlas.visibility_mask),
            (target_masked_source_path, masked_image),
        ]
        for path, img in outputs:
            if not cv2.imwrite(path.as_posix(), img):
                raise RuntimeError(f"Failed to write target image: {path}")

        _write_target_overlay(
            image_bgr,
            target_projected,
            target_overlay_path,
            mask=mask,
        )

        total_i, visible_i = _count_projected(target_projected)
        target_reports[target] = {
            "target": target,
            "status": "ok",
            "samples_total": total_i,
            "samples_visible": visible_i,
            "visible_ratio": float(visible_i / max(total_i, 1)),
            "uv_coverage_ratio": float(target_atlas.coverage_ratio),
            "mask_used": mask is not None,
            "mask_dir_used": mask_path_used,
            "texture_atlas": target_texture_path.as_posix(),
            "partial_texture_atlas": target_partial_path.as_posix(),
            "visibility_mask": target_visibility_path.as_posix(),
            "projected_overlay": target_overlay_path.as_posix(),
            "masked_source": target_masked_source_path.as_posix(),
        }

        write_text(
            target_dir / "projection_report.md",
            "\n".join(
                [
                    f"# Projection Report: {target}",
                    "",
                    f"- status: {target_reports[target]['status']}",
                    f"- samples_total: {total_i}",
                    f"- samples_visible: {visible_i}",
                    f"- visible_ratio: {visible_i / max(total_i, 1):.4f}",
                    f"- uv_coverage_ratio: {target_atlas.coverage_ratio:.4f}",
                    f"- mask_used: {mask is not None}",
                    f"- mask_dir_used: `{mask_path_used}`",
                    f"- texture_atlas: `{target_texture_path}`",
                    f"- partial_texture_atlas: `{target_partial_path}`",
                    f"- visibility_mask: `{target_visibility_path}`",
                    f"- projected_overlay: `{target_overlay_path}`",
                    "",
                    "Interpretation:",
                    "- If projected_overlay does not align with the real object, fix camera/model pose first.",
                    "- If mask_used=false, colors may bleed from neighboring objects.",
                    "- This is still a projection fallback, not true mesh UV baking.",
                    "",
                ]
            ),
        )

    changed_geoms: list[str] = []
    recolored: list[str] = []
    warnings: list[str] = []

    # Be conservative: applying a bad atlas to XML/3DGS makes the whole scene look worse.
    # Keep XML apply, but block GS recolor by default unless explicitly set through config.gaussian.regenerate.
    if apply_to_fr5:
        xml_info = apply_texture_material_to_fr5_xml(
            model_xml,
            texture_file_relative=_relative_to_model(texture_path, model_xml),
            texture_name=config.texture_name,
            material_name=config.material_name,
            target_bodies=config.target_names,
            remove_target_object=False,
        )
        changed_geoms = [str(v) for v in xml_info["changed_geoms"]]

        if task_config is not None:
            update_fr5_task_config(task_config, config=config, texture_path=texture_path)

        # Only recolor GS when user explicitly asked to regenerate. Otherwise keep it off.
        if config.gaussian.enabled and gs_dir is not None:
            if config.gaussian.regenerate and generator_script is not None:
                gs_dir = Path(gs_dir)
                regenerate_gaussians(
                    generator_script=Path(generator_script),
                    model_xml=model_xml,
                    gs_dir=gs_dir,
                    config=config,
                )
                recolored = [
                    p.as_posix()
                    for p in recolor_gaussian_dir(
                        gs_dir,
                        texture_path,
                        body_names=config.target_names,
                        blend=float(config.gaussian.recolor_blend),
                        backup=True,
                    )
                ]
            else:
                warnings.append(
                    "GS recolor skipped. Set --regenerate-gs if you really want to recolor Gaussian assets. "
                    "First verify per-target projected overlays."
                )

    total_samples = int(sum(item.sample.points_world.shape[0] for item in projected))
    visible_samples = int(sum(item.visible.sum() for item in projected))

    report = {
        "mode": "mesh_conditioned_appearance_transfer_safe_per_target",
        "image": image_path.as_posix(),
        "camera_config": camera_config.as_posix(),
        "model_xml": model_xml.as_posix(),
        "targets": config.target_names,
        "samples_total": total_samples,
        "samples_visible": visible_samples,
        "visible_ratio": float(visible_samples / max(total_samples, 1)),
        "uv_coverage_ratio": float(atlas.coverage_ratio),
        "texture_atlas": texture_path.as_posix(),
        "partial_texture_atlas": partial_path.as_posix(),
        "visibility_mask": visibility_path.as_posix(),
        "projected_overlay": overlay_path.as_posix(),
        "target_tiles": atlas.target_tiles,
        "per_target": target_reports,
        "apply_to_fr5": bool(apply_to_fr5),
        "changed_geoms": changed_geoms,
        "recolored_gaussians": recolored,
        "warnings": warnings,
        "config": asdict(config),
    }

    write_json(output_dir / "appearance_result.json", report)

    dump_mapping(
        sim_dir / "mujoco_appearance_assets.yaml",
        {
            "mujoco_appearance": {
                "texture_name": config.texture_name,
                "material_name": config.material_name,
                "texture_file": _relative_to_model(texture_path, model_xml),
                "targets": config.target_names,
                "note": "Global atlas is backward-compatible only. Prefer checking per_target outputs first.",
            }
        },
    )

    write_text(
        debug_dir / "projection_report.md",
        "\n".join(
            [
                "# Mesh-Conditioned Appearance Projection Report",
                "",
                f"- image: `{image_path}`",
                f"- camera_config: `{camera_config}`",
                f"- model_xml: `{model_xml}`",
                f"- targets: {config.target_names}",
                f"- total_samples: {total_samples}",
                f"- visible_samples: {visible_samples}",
                f"- visible_ratio: {visible_samples / max(total_samples, 1):.4f}",
                f"- uv_coverage_ratio: {atlas.coverage_ratio:.4f}",
                f"- texture_atlas: `{texture_path}`",
                f"- partial_texture_atlas: `{partial_path}`",
                f"- visibility_mask: `{visibility_path}`",
                f"- projected_overlay: `{overlay_path}`",
                f"- apply_to_fr5: {bool(apply_to_fr5)}",
                "",
                "Important notes:",
                "- This is a safer per-target debugging version of the previous global projection pipeline.",
                "- The old version projected all sampled box geometry into one atlas, which can easily cause color bleeding.",
                "- Manual masks are supported via `<image_dir>/masks/<target>.png` or `<output_dir>/masks/<target>.png`.",
                "- If overlay alignment is bad, do not train with this output. Fix camera extrinsic / object pose first.",
                "- True high-quality result requires real mesh UV baking or vertex-color export for mesh geoms.",
                "",
                "Per-target reports:",
                *[
                    f"- {name}: {info}"
                    for name, info in target_reports.items()
                ],
                "",
                "Warnings:",
                *[f"- {w}" for w in warnings],
                "",
            ]
        ),
    )

    return report