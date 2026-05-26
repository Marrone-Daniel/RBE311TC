from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from real2sim.io_utils import dump_mapping, ensure_dir, write_json, write_text
from real2sim.reconstruction.dummy_reconstructor import write_box_obj
from real2sim.reconstruction.sam3dgs_adapter import Sam3DGSAdapter
from real2sim.segmentation.auto_mask import auto_grabcut_mask, save_mask_debug
from real2sim.simulation.mujoco_asset_writer import write_mujoco_box_xml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SH_C0 = 0.28209479177387814


def parse_vec3(values: list[float] | None, default: tuple[float, float, float]) -> np.ndarray:
    if values is None:
        return np.asarray(default, dtype=np.float64)
    if len(values) != 3:
        raise ValueError("Expected exactly three values.")
    return np.asarray(values, dtype=np.float64)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise RuntimeError("Mask is empty.")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def fallback_rect_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if bbox is None:
        bw, bh = int(w * 0.45), int(h * 0.45)
        x, y = (w - bw) // 2, (h - bh) // 2
    else:
        x, y, bw, bh = bbox
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, x0 + int(bw)), min(h, y0 + int(bh))
    mask[y0:y1, x0:x1] = 1
    return mask.astype(bool)


def estimate_size_from_mask(mask: np.ndarray, base_size: np.ndarray) -> np.ndarray:
    x, y, w, h = bbox_from_mask(mask)
    _ = x, y
    ratio = max(float(w) / max(float(h), 1.0), 0.15)
    size = np.asarray(base_size, dtype=np.float64).copy()
    if ratio >= 1.0:
        size[0] = base_size[0]
        size[1] = max(base_size[1] / ratio, base_size[1] * 0.35)
    else:
        size[0] = max(base_size[0] * ratio, base_size[0] * 0.35)
        size[1] = base_size[1]
    return np.clip(size, 0.008, 0.12)


def sample_box_surface(half_size: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    sx, sy, sz = np.asarray(half_size, dtype=np.float64).reshape(3)
    areas = np.asarray([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy], dtype=np.float64)
    probs = areas / max(float(np.sum(areas)), 1e-12)
    faces = rng.choice(6, size=int(count), p=probs)
    points = np.empty((int(count), 3), dtype=np.float32)
    uv = rng.uniform(-1.0, 1.0, size=(int(count), 2))
    for idx, face in enumerate(faces):
        u, v = uv[idx]
        if face == 0:
            points[idx] = [sx, u * sy, v * sz]
        elif face == 1:
            points[idx] = [-sx, u * sy, v * sz]
        elif face == 2:
            points[idx] = [u * sx, sy, v * sz]
        elif face == 3:
            points[idx] = [u * sx, -sy, v * sz]
        elif face == 4:
            points[idx] = [u * sx, v * sy, sz]
        else:
            points[idx] = [u * sx, v * sy, -sz]
    return points


def write_gaussian_box(
    path: Path,
    *,
    half_size: np.ndarray,
    rgb: np.ndarray,
    points: int,
    scale: float,
    opacity: float,
) -> Path:
    from gaussian_renderer.core.gaussiandata import GaussianData
    from gaussian_renderer.core.util_gau import save_ply

    rng = np.random.default_rng(23)
    xyz = sample_box_surface(half_size, int(points), rng)
    rgb01 = np.clip(np.asarray(rgb, dtype=np.float32).reshape(1, 3) / 255.0, 0.0, 1.0)
    sh = np.repeat((rgb01 - 0.5) / SH_C0, xyz.shape[0], axis=0).astype(np.float32)
    rot = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    rot[:, 0] = 1.0
    scales = np.full((xyz.shape[0], 3), float(scale), dtype=np.float32)
    opacities = np.full((xyz.shape[0],), float(opacity), dtype=np.float32)
    ensure_dir(path.parent)
    save_ply(GaussianData(xyz.astype(np.float32), rot, scales, opacities, sh), path, save_sh_degree=0)
    return path


def write_default_input_configs(input_dir: Path, object_name: str) -> None:
    dump_mapping(
        input_dir / "intrinsics.yaml",
        {"camera": {"width": 640, "height": 480, "fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5}},
    )
    dump_mapping(
        input_dir / "camera_pose.yaml",
        {"T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
    )
    dump_mapping(
        input_dir / "config.yaml",
        {
            "reconstruction": {"backend": "sam3dgs_or_dummy", "save_visual_mesh": True, "save_gaussian": True},
            "object": {"name": object_name, "mass": 0.05, "friction": [1.0, 0.005, 0.0001]},
            "simulation": {"engine": "mujoco", "export_mujoco_xml": True, "export_gs_config": True},
        },
    )


def run_apply_to_fr5(output_dir: Path, initial_qpos: np.ndarray) -> None:
    cmd = [
        sys.executable,
        (PROJECT_ROOT / "demo" / "fr5_demo" / "apply_real2sim_scene.py").as_posix(),
        "--real2sim-output",
        output_dir.as_posix(),
        "--initial-qpos",
        *[f"{float(v):.8g}" for v in initial_qpos],
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT.as_posix(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one-image Real2Sim object assets and apply them to FR5 demo XML.")
    parser.add_argument("--image", required=True, type=str, help="Single RGB image used for automatic segmentation and visual asset generation.")
    parser.add_argument("--input-dir", default=(PROJECT_ROOT / "real2sim_input").as_posix())
    parser.add_argument("--output-dir", default=(PROJECT_ROOT / "real2sim_output").as_posix())
    parser.add_argument("--object-name", default="target_object")
    parser.add_argument("--object-pos", type=float, nargs=3, default=[-0.55, 0.0, 0.015])
    parser.add_argument("--object-size", type=float, nargs=3, default=[0.03, 0.03, 0.03], help="Nominal full object size in meters.")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="Optional object ROI for GrabCut.")
    parser.add_argument("--mask-margin-ratio", type=float, default=0.08)
    parser.add_argument("--sam3d-gs-repo", default=(PROJECT_ROOT / "sam3d_gs").as_posix())
    parser.add_argument("--use-sam3dgs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gaussian-points", type=int, default=60000)
    parser.add_argument("--gaussian-scale", type=float, default=0.00085)
    parser.add_argument("--gaussian-opacity", type=float, default=0.58)
    parser.add_argument("--apply-to-fr5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-qpos", type=float, nargs=6, default=[0.0, -1.2, 1.5, 0.9, 1.5617, 0.0])
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    input_dir = ensure_dir(args.input_dir)
    output_dir = ensure_dir(args.output_dir)
    visual_dir = ensure_dir(output_dir / "visual")
    collision_dir = ensure_dir(output_dir / "collision")
    sim_dir = ensure_dir(output_dir / "sim")
    debug_dir = ensure_dir(output_dir / "debug")

    image_bgr = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    warnings: list[str] = []
    try:
        mask = auto_grabcut_mask(
            image_bgr,
            bbox=tuple(args.bbox) if args.bbox else None,
            margin_ratio=float(args.mask_margin_ratio),
            iterations=5,
        )
        if int(mask.sum()) < 100:
            raise RuntimeError("auto mask too small")
    except Exception as exc:
        warnings.append(f"GrabCut mask failed or was too small, using rectangle fallback: {exc}")
        mask = fallback_rect_mask(image_bgr.shape[:2], tuple(args.bbox) if args.bbox else None)

    rgb_out = input_dir / "rgb.png"
    mask_out = input_dir / "mask.png"
    cv2.imwrite(rgb_out.as_posix(), image_bgr)
    cv2.imwrite(mask_out.as_posix(), (mask.astype(np.uint8) * 255))
    mask_debug = save_mask_debug(debug_dir / "auto_mask_overlay.png", image_bgr, mask)
    write_default_input_configs(input_dir, args.object_name)

    full_size = estimate_size_from_mask(mask, parse_vec3(args.object_size, (0.03, 0.03, 0.03)))
    half_size = full_size * 0.5
    position = parse_vec3(args.object_pos, (-0.55, 0.0, 0.015))
    quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    masked_pixels = image_bgr[mask.astype(bool)]
    mean_bgr = np.mean(masked_pixels, axis=0) if masked_pixels.size else np.asarray([50, 80, 200], dtype=np.float64)
    mean_rgb = mean_bgr[::-1]

    mujoco_xml = write_mujoco_box_xml(
        sim_dir / "mujoco_object.xml",
        object_name=args.object_name,
        position=position,
        quat_wxyz=quat,
        half_size=half_size,
        mass=0.05,
        friction=[1.0, 0.005, 0.0001],
        rgba=[float(mean_rgb[0] / 255.0), float(mean_rgb[1] / 255.0), float(mean_rgb[2] / 255.0), 1.0],
    )
    object_pose = dump_mapping(
        sim_dir / "object_pose_world.yaml",
        {"object": {"name": args.object_name, "xyz": position.tolist(), "quat_wxyz": quat.tolist(), "frame": "mujoco_world"}},
    )
    gs_config = dump_mapping(
        sim_dir / "gs_asset_config.yaml",
        {
            "asset": {
                "name": args.object_name,
                "gaussian": (visual_dir / "object_gaussian.ply").as_posix(),
                "mesh": (visual_dir / "object_mesh.obj").as_posix(),
                "mujoco_body": args.object_name,
                "binding": "body_local",
            }
        },
    )
    collision_meta = write_json(
        collision_dir / "collision_metadata.json",
        {
            "object_name": args.object_name,
            "collision_type": "box",
            "position_world": position.tolist(),
            "half_size": half_size.tolist(),
            "full_size": full_size.tolist(),
            "mass": 0.05,
            "friction": [1.0, 0.005, 0.0001],
        },
    )
    mesh_path = write_box_obj(visual_dir / "object_mesh.obj", size=full_size)

    backend_message = "SAM3D-GS disabled."
    gaussian_path: Path | None = None
    if bool(args.use_sam3dgs):
        adapter = Sam3DGSAdapter(args.sam3d_gs_repo)
        result = adapter.run_from_image_and_mask(image_path=rgb_out, mask_path=mask_out, work_dir=output_dir / "sam3dgs_work")
        backend_message = result.message
        if result.gaussian_path is not None and result.gaussian_path.exists():
            gaussian_path = visual_dir / "object_gaussian.ply"
            shutil.copy2(result.gaussian_path, gaussian_path)
        if result.mesh_path is not None and result.mesh_path.exists():
            mesh_path = visual_dir / "object_mesh.obj"
            shutil.copy2(result.mesh_path, mesh_path)

    if gaussian_path is None:
        gaussian_path = write_gaussian_box(
            visual_dir / "object_gaussian.ply",
            half_size=half_size,
            rgb=mean_rgb,
            points=int(args.gaussian_points),
            scale=float(args.gaussian_scale),
            opacity=float(args.gaussian_opacity),
        )
        backend_message = f"{backend_message} Used local Gaussian box fallback."

    report = write_text(
        debug_dir / "one_image_real2sim_report.md",
        "\n".join(
            [
                "# One Image Real2Sim Report",
                "",
                f"- image: `{image_path}`",
                f"- rgb: `{rgb_out}`",
                f"- mask: `{mask_out}`",
                f"- mask_debug: `{mask_debug}`",
                f"- mask_pixels: {int(mask.sum())}",
                f"- object_name: `{args.object_name}`",
                f"- object_position_world: {position.tolist()}",
                f"- object_full_size_m: {full_size.tolist()}",
                f"- gaussian: `{gaussian_path}`",
                f"- mesh: `{mesh_path}`",
                f"- mujoco_xml: `{mujoco_xml}`",
                f"- object_pose: `{object_pose}`",
                f"- gs_config: `{gs_config}`",
                f"- collision_metadata: `{collision_meta}`",
                f"- backend: {backend_message}",
                f"- warnings: {warnings if warnings else 'none'}",
                "",
            ]
        ),
    )
    result_json = write_json(
        output_dir / "real2sim_result.json",
        {
            "image": image_path.as_posix(),
            "rgb": rgb_out.as_posix(),
            "mask": mask_out.as_posix(),
            "gaussian": gaussian_path.as_posix(),
            "mesh": mesh_path.as_posix(),
            "mujoco_xml": mujoco_xml.as_posix(),
            "report": report.as_posix(),
            "backend_message": backend_message,
            "warnings": warnings,
        },
    )

    if bool(args.apply_to_fr5):
        run_apply_to_fr5(output_dir, np.asarray(args.initial_qpos, dtype=np.float64))

    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"mask: {mask_out}")
    print(f"gaussian: {gaussian_path}")
    print(f"mujoco_xml: {mujoco_xml}")
    print(f"result: {result_json}")
    print(f"backend: {backend_message}")


if __name__ == "__main__":
    main()
