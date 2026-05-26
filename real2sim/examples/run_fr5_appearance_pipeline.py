from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from real2sim.appearance.config import load_appearance_config
from real2sim.appearance.pipeline import run_fr5_appearance_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = PROJECT_ROOT / "demo" / "fr5_demo"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mesh-conditioned appearance transfer for the FR5 demo.")
    parser.add_argument("--image", required=True, help="Real RGB photo captured from the calibrated Astra viewpoint.")
    parser.add_argument("--camera-config", default=(DEMO_DIR / "configs" / "astra_camera.json").as_posix())
    parser.add_argument("--model-xml", default=(DEMO_DIR / "assets" / "fr5" / "mjmodel.xml").as_posix())
    parser.add_argument("--task-config", default=(DEMO_DIR / "configs" / "fr5_table_task.json").as_posix())
    parser.add_argument("--config", default="", help="Optional appearance YAML/JSON config.")
    parser.add_argument("--output-dir", default=(PROJECT_ROOT / "real2sim_output" / "appearance").as_posix())
    parser.add_argument("--texture-dir", default=(DEMO_DIR / "assets" / "fr5" / "textures").as_posix())
    parser.add_argument("--gs-dir", default=(DEMO_DIR / "assets" / "fr5" / "3dgs_mesh").as_posix())
    parser.add_argument("--apply-to-fr5", action="store_true", help="Actually modify mjmodel.xml/task config and recolor 3DGS assets.")
    parser.add_argument("--regenerate-gs", action="store_true", help="Regenerate mesh-derived 3DGS assets before recoloring.")
    parser.add_argument("--atlas-size", type=int, default=None)
    parser.add_argument("--samples-per-box-face", type=int, default=None)
    parser.add_argument("--points-per-geom", type=int, default=None)
    parser.add_argument("--gs-scale", type=float, default=None)
    parser.add_argument("--gs-opacity", type=float, default=None)
    parser.add_argument("--gs-recolor-blend", type=float, default=None)
    args = parser.parse_args()

    cfg = load_appearance_config(args.config or None)
    if args.atlas_size is not None:
        cfg = replace(cfg, atlas_size=int(args.atlas_size))
    if args.samples_per_box_face is not None:
        cfg = replace(cfg, samples_per_box_face=int(args.samples_per_box_face))
    gs = cfg.gaussian
    gs_updates = {
        "enabled": gs.enabled,
        "points_per_geom": int(args.points_per_geom if args.points_per_geom is not None else gs.points_per_geom),
        "scale": float(args.gs_scale if args.gs_scale is not None else gs.scale),
        "opacity": float(args.gs_opacity if args.gs_opacity is not None else gs.opacity),
        "recolor_blend": float(args.gs_recolor_blend if args.gs_recolor_blend is not None else gs.recolor_blend),
        "regenerate": bool(args.regenerate_gs or gs.regenerate),
    }
    cfg = replace(cfg, gaussian=gs.__class__(**gs_updates))

    result = run_fr5_appearance_pipeline(
        image_path=args.image,
        camera_config=args.camera_config,
        model_xml=args.model_xml,
        output_dir=args.output_dir,
        texture_dir=args.texture_dir,
        config=cfg,
        apply_to_fr5=bool(args.apply_to_fr5),
        task_config=args.task_config,
        gs_dir=args.gs_dir,
        generator_script=DEMO_DIR / "generate_fr5_mesh_gaussians.py",
    )
    print(f"mode: {result['mode']}")
    print(f"apply_to_fr5: {result['apply_to_fr5']}")
    print(f"texture_atlas: {result['texture_atlas']}")
    print(f"projected_overlay: {result['projected_overlay']}")
    print(f"visibility_mask: {result['visibility_mask']}")
    print(f"uv_coverage_ratio: {result['uv_coverage_ratio']:.4f}")
    print(f"result: {Path(args.output_dir) / 'appearance_result.json'}")


if __name__ == "__main__":
    main()
