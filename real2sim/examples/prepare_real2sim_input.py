from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from real2sim.geometry.coordinate_transform import pose_dict_from_matrix
from real2sim.io_utils import dump_mapping, ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERA_CONFIG = PROJECT_ROOT / "demo" / "fr5_demo" / "configs" / "astra_camera.json"
DEFAULT_CONFIG_TEMPLATE = Path(__file__).resolve().parent / "config_example.yaml"


def _add_orbbec_to_path() -> None:
    lib = PROJECT_ROOT / "third_party" / "pyorbbecsdk" / "install" / "lib"
    if lib.exists() and lib.as_posix() not in sys.path:
        sys.path.insert(0, lib.as_posix())
        os.environ["LD_LIBRARY_PATH"] = f"{lib.as_posix()}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def require_orbbec():
    _add_orbbec_to_path()
    try:
        from pyorbbecsdk import Config, OBAlignMode, OBError, OBFormat, OBSensorType, Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyorbbecsdk is required for RGB-D capture. Build third_party/pyorbbecsdk first, "
            "or export PYTHONPATH=$PWD/third_party/pyorbbecsdk/install/lib:$PYTHONPATH."
        ) from exc
    return Config, OBAlignMode, OBError, OBFormat, OBSensorType, Pipeline


def frame_to_bgr_image(color_frame) -> np.ndarray:
    width = color_frame.get_width()
    height = color_frame.get_height()
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    fmt = color_frame.get_format() if hasattr(color_frame, "get_format") else None
    fmt_name = str(fmt).upper() if fmt is not None else ""
    if "MJPG" in fmt_name or "MJPEG" in fmt_name:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode MJPG frame: bytes={data.size}, frame={width}x{height}")
        return image
    if data.size == width * height * 3:
        rgb = data.reshape((height, width, 3))
        if "BGR" in fmt_name:
            return np.ascontiguousarray(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if data.size == width * height * 2:
        yuv = data.reshape((height, width, 2))
        if "UYVY" in fmt_name:
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_UYVY)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUY2)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    raise RuntimeError(f"Unsupported color buffer: bytes={data.size}, format={fmt}, frame={width}x{height}")


def read_intrinsics_from_pipeline(pipeline, width: int, height: int) -> dict:
    param = pipeline.get_camera_param() if hasattr(pipeline, "get_camera_param") else None
    intr = getattr(param, "rgb_intrinsic", None) if param is not None else None
    fx = float(getattr(intr, "fx", 0.0) or 0.0)
    fy = float(getattr(intr, "fy", 0.0) or 0.0)
    cx = float(getattr(intr, "cx", width * 0.5) or width * 0.5)
    cy = float(getattr(intr, "cy", height * 0.5) or height * 0.5)
    iw = int(getattr(intr, "width", width) or width)
    ih = int(getattr(intr, "height", height) or height)
    if fx <= 1.0 or fy <= 1.0:
        raise RuntimeError("Orbbec SDK returned invalid RGB intrinsics. Capture intrinsics with Orbbec Viewer or astra_camera.json.")
    return {"width": iw, "height": ih, "fx": fx, "fy": fy, "cx": cx, "cy": cy}


def load_pose_from_camera_config(path: Path) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    if not path.exists():
        warnings.append(f"{path} not found; camera_pose.yaml will use identity.")
        return pose_dict_from_matrix(np.eye(4)), warnings
    cfg = json.loads(path.read_text(encoding="utf-8"))
    ext = cfg.get("extrinsics", {})
    if not cfg.get("calibrated", False):
        warnings.append(f"{path} has calibrated=false; camera_pose.yaml may be wrong.")
    pos = np.asarray(ext.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
    rot = np.asarray(ext.get("rotation_matrix", np.eye(3)), dtype=np.float64)
    if pos.shape != (3,) or rot.shape != (3, 3):
        warnings.append(f"{path} extrinsics invalid; camera_pose.yaml will use identity.")
        return pose_dict_from_matrix(np.eye(4)), warnings
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return pose_dict_from_matrix(T), warnings


def depth_frame_to_uint16_mm(depth_frame) -> tuple[np.ndarray, float]:
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = float(depth_frame.get_depth_scale())
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
    depth_m = raw.astype(np.float32) * scale
    depth_mm = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
    return depth_mm, scale


def choose_video_profile(profile_list, width: int, height: int, fmt, fps: int, label: str):
    try:
        return profile_list.get_video_stream_profile(width, height, fmt, fps)
    except Exception as exc:
        raise RuntimeError(
            f"Could not get {label} profile {width}x{height}@{fps}, format={fmt}: {exc}"
        ) from exc


def choose_color_profile(color_profiles, OBFormat, args, warnings: list[str]):
    if args.color_format == "default":
        warnings.append("Using Orbbec default color profile.")
        return color_profiles.get_default_video_stream_profile()
    formats = []
    if args.color_format in ("auto", "mjpg"):
        formats.append(("MJPG", OBFormat.MJPG))
    if args.color_format in ("auto", "rgb"):
        formats.append(("RGB", OBFormat.RGB))
    for label, fmt in formats:
        try:
            profile = color_profiles.get_video_stream_profile(args.width, args.height, fmt, args.fps)
            warnings.append(f"Using color profile {args.width}x{args.height}@{args.fps}, format={label}.")
            return profile
        except Exception:
            continue
    if args.color_format != "auto":
        raise RuntimeError(f"Requested color format {args.color_format} is unavailable.")
    warnings.append("Requested low-bandwidth color profile unavailable; using Orbbec default color profile.")
    return color_profiles.get_default_video_stream_profile()


def choose_depth_profile(depth_profiles, OBFormat, args, warnings: list[str]):
    width = int(args.depth_width or args.width)
    height = int(args.depth_height or args.height)
    fps = int(args.depth_fps or args.fps)
    try:
        profile = depth_profiles.get_video_stream_profile(width, height, OBFormat.Y16, fps)
        warnings.append(f"Using depth profile {width}x{height}@{fps}, format=Y16.")
        return profile
    except Exception:
        warnings.append("Requested depth profile unavailable; using Orbbec default depth profile.")
        return depth_profiles.get_default_video_stream_profile()


def capture_rgbd(args) -> tuple[np.ndarray, np.ndarray, dict, list[str]]:
    Config, OBAlignMode, OBError, OBFormat, OBSensorType, Pipeline = require_orbbec()
    pipeline = Pipeline()
    config = Config()
    warnings: list[str] = []

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = choose_color_profile(color_profiles, OBFormat, args, warnings)
    config.enable_stream(color_profile)

    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    if depth_profiles is None:
        raise RuntimeError("Astra depth profile not available.")
    depth_profile = choose_depth_profile(depth_profiles, OBFormat, args, warnings)
    config.enable_stream(depth_profile)

    if args.align_mode != "none":
        try:
            mode = OBAlignMode.HW_MODE if args.align_mode == "hw" else OBAlignMode.SW_MODE
            config.set_align_mode(mode)
        except Exception as exc:
            warnings.append(f"Could not enable {args.align_mode} depth/color align: {exc}")

    if args.frame_sync:
        try:
            pipeline.enable_frame_sync()
        except Exception as exc:
            warnings.append(f"Frame sync unavailable; continuing without sync: {exc}")

    pipeline.start(config)
    print("Astra RGB-D stream started. Press s to save, q/esc to quit.", flush=True)
    last_rgb = None
    last_depth = None
    intrinsics = None
    frame_idx = 0
    empty_frames = 0
    try:
        while True:
            frames = pipeline.wait_for_frames(int(args.wait_ms))
            if frames is None:
                empty_frames += 1
                if empty_frames >= int(args.max_empty_frames):
                    raise RuntimeError(
                        "Astra RGB-D did not return frames. On Astra Pro Plus over USB2 this is usually USB bandwidth "
                        "or usbfs buffer pressure. Retry with: --fps 10 --align-mode none --color-format mjpg. "
                        "If it still fails, replug the camera into a USB3 port or increase Linux usbfs_memory_mb."
                    )
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                empty_frames += 1
                if empty_frames >= int(args.max_empty_frames):
                    raise RuntimeError(
                        "Astra returned framesets without both color and depth. Retry with lower bandwidth settings: "
                        "--fps 10 --align-mode none --color-format mjpg."
                    )
                continue
            empty_frames = 0
            rgb_bgr = frame_to_bgr_image(color_frame)
            depth_mm, depth_scale = depth_frame_to_uint16_mm(depth_frame)
            if intrinsics is None:
                intrinsics = read_intrinsics_from_pipeline(pipeline, color_frame.get_width(), color_frame.get_height())

            if depth_mm.shape[:2] != rgb_bgr.shape[:2]:
                warnings.append(
                    f"Depth resolution {depth_mm.shape[::-1]} differs from RGB {rgb_bgr.shape[1]}x{rgb_bgr.shape[0]}; "
                    "resizing depth to RGB. Prefer align_mode=sw/hw if supported."
                )
                depth_mm = cv2.resize(depth_mm, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            intrinsics["width"] = int(rgb_bgr.shape[1])
            intrinsics["height"] = int(rgb_bgr.shape[0])
            last_rgb = rgb_bgr
            last_depth = depth_mm

            key = -1
            if not args.no_display:
                depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                preview = np.hstack([rgb_bgr, depth_vis])
                cv2.imshow("Real2Sim RGB | Depth", preview)
                key = cv2.waitKey(1)
            if key == ord("s") or (args.no_display and frame_idx >= max(0, int(args.warmup_frames))):
                break
            if key in (ord("q"), 27):
                raise RuntimeError("Capture cancelled by user.")
            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()

    if last_rgb is None or last_depth is None or intrinsics is None:
        raise RuntimeError("No RGB-D frame captured.")
    return last_rgb, last_depth, intrinsics, warnings


def create_mask(rgb_bgr: np.ndarray, args) -> np.ndarray:
    if args.mask_mode == "existing":
        if not args.mask_file:
            raise RuntimeError("--mask-mode existing requires --mask-file")
        mask = cv2.imread(args.mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask file: {args.mask_file}")
        if mask.shape != rgb_bgr.shape[:2]:
            mask = cv2.resize(mask, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8) * 255
    if args.mask_mode == "full":
        return np.full(rgb_bgr.shape[:2], 255, dtype=np.uint8)
    if args.no_display:
        raise RuntimeError("--mask-mode roi requires display. Use --mask-mode full or --mask-mode existing with --no-display.")

    roi = cv2.selectROI("Select target object ROI, press Enter/Space", rgb_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select target object ROI, press Enter/Space")
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI selection cancelled or empty.")
    mask = np.zeros(rgb_bgr.shape[:2], dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    return mask


def write_config(path: Path, args) -> None:
    template = DEFAULT_CONFIG_TEMPLATE.read_text(encoding="utf-8")
    path.write_text(template, encoding="utf-8")
    if args.object_name == "target_object":
        return
    text = path.read_text(encoding="utf-8")
    text = text.replace("name: target_object", f"name: {args.object_name}")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture files required by the Real2Sim MVP.")
    parser.add_argument("--output-dir", type=str, default="real2sim_input")
    parser.add_argument("--camera-config", type=str, default=DEFAULT_CAMERA_CONFIG.as_posix())
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--depth-width", type=int, default=0)
    parser.add_argument("--depth-height", type=int, default=0)
    parser.add_argument("--depth-fps", type=int, default=0)
    parser.add_argument("--color-format", choices=["auto", "mjpg", "rgb", "default"], default="auto")
    parser.add_argument("--align-mode", choices=["sw", "hw", "none"], default="none")
    parser.add_argument("--frame-sync", action="store_true")
    parser.add_argument("--wait-ms", type=int, default=500)
    parser.add_argument("--max-empty-frames", type=int, default=40)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--mask-mode", choices=["roi", "full", "existing"], default="roi")
    parser.add_argument("--mask-file", type=str, default="")
    parser.add_argument("--object-name", type=str, default="target_object")
    parser.add_argument("--run-real2sim", action="store_true")
    parser.add_argument("--real2sim-output-dir", type=str, default="real2sim_output")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    rgb_bgr, depth_mm, intrinsics, warnings = capture_rgbd(args)
    mask = create_mask(rgb_bgr, args)
    pose, pose_warnings = load_pose_from_camera_config(Path(args.camera_config))
    warnings.extend(pose_warnings)

    cv2.imwrite((output_dir / "rgb.png").as_posix(), rgb_bgr)
    cv2.imwrite((output_dir / "depth.png").as_posix(), depth_mm)
    cv2.imwrite((output_dir / "mask.png").as_posix(), mask)
    dump_mapping(output_dir / "intrinsics.yaml", intrinsics)
    dump_mapping(output_dir / "camera_pose.yaml", pose)
    write_config(output_dir / "config.yaml", args)

    print(f"Saved Real2Sim input files to {output_dir.resolve()}", flush=True)
    for name in ("rgb.png", "depth.png", "mask.png", "intrinsics.yaml", "camera_pose.yaml", "config.yaml"):
        print(f"  {output_dir / name}", flush=True)
    if warnings:
        print("Warnings:", flush=True)
        for warning in warnings:
            print(f"  - {warning}", flush=True)

    if args.run_real2sim:
        from real2sim.pipeline import Real2SimPipeline

        result = Real2SimPipeline.from_config_file(output_dir / "config.yaml").run(
            input_dir=output_dir,
            output_dir=args.real2sim_output_dir,
        )
        print(f"Real2Sim output written to {Path(args.real2sim_output_dir).resolve()}", flush=True)
        print(f"  center_world={result['center_world']}", flush=True)
        print(f"  size_world={result['size_world']}", flush=True)


if __name__ == "__main__":
    main()
