from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from real2sim.appearance.fr5_mujoco_texture import DEFAULT_APPEARANCE_BODIES, apply_texture_material_to_fr5_xml
from real2sim.appearance.gs_recolor import recolor_gaussian_dir
from real2sim.appearance.texture_transfer import save_texture_outputs
from real2sim.io_utils import ensure_dir, write_json, write_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = PROJECT_ROOT / "demo" / "fr5_demo"
DEFAULT_MODEL_XML = DEMO_DIR / "assets" / "fr5" / "mjmodel.xml"
DEFAULT_TASK_CONFIG = DEMO_DIR / "configs" / "fr5_table_task.json"
DEFAULT_TEXTURE_DIR = DEMO_DIR / "assets" / "fr5" / "textures"
DEFAULT_GS_DIR = DEMO_DIR / "assets" / "fr5" / "3dgs_mesh"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "real2sim_output" / "appearance_transfer"


def load_task_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_task_config(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def update_task_config(
    path: Path,
    *,
    points_per_geom: int,
    scale: float,
    opacity: float,
    texture_file: Path,
    targets: list[str],
    blend: float,
) -> None:
    cfg = load_task_config(path)
    cfg.pop("cube_site", None)
    cfg.pop("target_site", None)
    cfg["fr5_3dgs"] = {
        "points_per_geom": int(points_per_geom),
        "scale": float(scale),
        "opacity": float(opacity),
        "regenerate_on_missing": True,
        "appearance_transfer": {
            "enabled": True,
            "texture_atlas": texture_file.as_posix(),
            "targets": targets,
            "recolor_blend": float(blend),
            "xml_material": "fr5_appearance_mat",
            "xml_texture": "fr5_appearance_tex",
        },
    }
    save_task_config(path, cfg)


def regenerate_gaussians_if_requested(
    *,
    model_xml: Path,
    gs_dir: Path,
    points_per_geom: int,
    scale: float,
    opacity: float,
    regenerate: bool,
) -> None:
    manifest = gs_dir / "manifest.json"
    if manifest.exists() and not regenerate:
        return
    cmd = [
        sys.executable,
        (DEMO_DIR / "generate_fr5_mesh_gaussians.py").as_posix(),
        "--model-xml",
        model_xml.as_posix(),
        "--output-dir",
        gs_dir.as_posix(),
        "--points-per-geom",
        str(int(points_per_geom)),
        "--scale",
        f"{float(scale):.8g}",
        "--opacity",
        f"{float(opacity):.8g}",
    ]
    subprocess.run(cmd, cwd=DEMO_DIR.as_posix(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply single-RGB appearance transfer to existing FR5 demo assets.")
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--model-xml", default=DEFAULT_MODEL_XML.as_posix())
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG.as_posix())
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--texture-dir", default=DEFAULT_TEXTURE_DIR.as_posix())
    parser.add_argument("--gs-dir", default=DEFAULT_GS_DIR.as_posix())
    parser.add_argument("--mask-mode", choices=["full", "roi", "auto"], default="full")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--atlas-size", type=int, default=1024)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_APPEARANCE_BODIES)
    parser.add_argument("--remove-target-object", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--points-per-geom", type=int, default=12000)
    parser.add_argument("--gs-scale", type=float, default=0.00085)
    parser.add_argument("--gs-opacity", type=float, default=0.58)
    parser.add_argument("--gs-recolor-blend", type=float, default=0.88)
    parser.add_argument("--regenerate-gs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--backup-gs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model_xml = Path(args.model_xml).resolve()
    task_config = Path(args.task_config).resolve()
    output_dir = ensure_dir(args.output_dir)
    texture_dir = ensure_dir(args.texture_dir)
    gs_dir = Path(args.gs_dir).resolve()
    debug_dir = ensure_dir(output_dir / "debug")

    texture_path = texture_dir / "fr5_appearance_atlas.png"
    texture_info = save_texture_outputs(
        image_path=args.image,
        output_texture=texture_path,
        debug_overlay=debug_dir / "appearance_mask_overlay.png",
        mask_path=output_dir / "appearance_mask.png",
        mask_mode=str(args.mask_mode),
        bbox=tuple(args.bbox) if args.bbox else None,
        atlas_size=int(args.atlas_size),
    )

    # Texture file paths in MJCF are relative to the model XML directory.
    rel_texture = texture_path.relative_to(model_xml.parent).as_posix()
    xml_info = apply_texture_material_to_fr5_xml(
        model_xml,
        texture_file_relative=rel_texture,
        target_bodies=list(args.targets),
        remove_target_object=bool(args.remove_target_object),
    )
    regenerate_gaussians_if_requested(
        model_xml=model_xml,
        gs_dir=gs_dir,
        points_per_geom=int(args.points_per_geom),
        scale=float(args.gs_scale),
        opacity=float(args.gs_opacity),
        regenerate=bool(args.regenerate_gs),
    )
    recolored = recolor_gaussian_dir(
        gs_dir,
        texture_path,
        body_names=list(args.targets),
        blend=float(args.gs_recolor_blend),
        backup=bool(args.backup_gs),
    )
    update_task_config(
        task_config,
        points_per_geom=int(args.points_per_geom),
        scale=float(args.gs_scale),
        opacity=float(args.gs_opacity),
        texture_file=texture_path,
        targets=list(args.targets),
        blend=float(args.gs_recolor_blend),
    )

    report = {
        "mode": "fr5_appearance_transfer",
        "image": Path(args.image).resolve().as_posix(),
        "texture_atlas": texture_path.as_posix(),
        "mask": Path(texture_info["mask"]).as_posix(),
        "mask_overlay": Path(texture_info["overlay"]).as_posix(),
        "mask_pixels": int(texture_info["mask_pixels"]),
        "model_xml": model_xml.as_posix(),
        "task_config": task_config.as_posix(),
        "targets": list(args.targets),
        "changed_geoms": xml_info["changed_geoms"],
        "removed_target_object": bool(xml_info["removed_target_object"]),
        "recolored_gaussians": [p.as_posix() for p in recolored],
        "gs_settings": {
            "points_per_geom": int(args.points_per_geom),
            "scale": float(args.gs_scale),
            "opacity": float(args.gs_opacity),
            "recolor_blend": float(args.gs_recolor_blend),
        },
    }
    report_path = write_json(output_dir / "appearance_transfer_result.json", report)
    write_text(
        debug_dir / "appearance_transfer_report.md",
        "\n".join(
            [
                "# FR5 Appearance Transfer Report",
                "",
                f"- image: `{report['image']}`",
                f"- texture_atlas: `{report['texture_atlas']}`",
                f"- mask_pixels: {report['mask_pixels']}",
                f"- model_xml: `{report['model_xml']}`",
                f"- task_config: `{report['task_config']}`",
                f"- removed_target_object: {report['removed_target_object']}",
                f"- recolored_gaussians: {len(recolored)}",
                f"- changed_geoms: {len(report['changed_geoms'])}",
                f"- gs_settings: {report['gs_settings']}",
                "",
            ]
        ),
    )
    print(f"texture_atlas: {texture_path}")
    print(f"model_xml: {model_xml}")
    print(f"task_config: {task_config}")
    print(f"recolored_gaussians: {len(recolored)}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
