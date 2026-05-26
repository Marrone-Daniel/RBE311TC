from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CAMERA_CONFIG = DEMO_DIR / "configs" / "astra_camera.json"
DEFAULT_CAPTURE_DIR = DEMO_DIR / "data" / "astra_captures"


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install opencv-python or opencv-contrib-python in this environment.") from exc
    return cv2


def require_orbbec():
    local_lib = DEMO_DIR.parents[1] / "third_party" / "pyorbbecsdk" / "install" / "lib"
    if local_lib.exists() and local_lib.as_posix() not in sys.path:
        sys.path.insert(0, local_lib.as_posix())
        os.environ["LD_LIBRARY_PATH"] = f"{local_lib.as_posix()}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    try:
        from pyorbbecsdk import Config, OBError, OBFormat, OBSensorType, Pipeline, VideoStreamProfile
    except ImportError as exc:
        raise RuntimeError(
            "pyorbbecsdk is not importable in this environment. For Astra Pro Plus, build Orbbec's pyorbbecsdk "
            "from the official main/v1.x branch, then export PYTHONPATH=/path/to/pyorbbecsdk/install/lib:$PYTHONPATH. "
            "Do not rely on the current PyPI wheel for this Linux/Python 3.10 environment."
        ) from exc
    return Config, OBError, OBFormat, OBSensorType, Pipeline, VideoStreamProfile


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return DEMO_DIR / path


def frame_to_bgr_image(color_frame):
    cv2 = require_cv2()
    width = color_frame.get_width()
    height = color_frame.get_height()
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    fmt = color_frame.get_format() if hasattr(color_frame, "get_format") else None
    fmt_name = str(fmt).upper() if fmt is not None else ""
    if "MJPG" in fmt_name or "MJPEG" in fmt_name:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not decode MJPG color frame: bytes={data.size}, frame={width}x{height}")
        return bgr
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
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr
    raise RuntimeError(f"Unsupported color frame buffer size: {data.size}, format={fmt}, frame={width}x{height}")


def read_camera_param(pipeline, width: int, height: int) -> dict:
    param = pipeline.get_camera_param() if hasattr(pipeline, "get_camera_param") else None
    intr = getattr(param, "rgb_intrinsic", None) if param is not None else None
    fx = float(getattr(intr, "fx", 0.0) or 0.0)
    fy = float(getattr(intr, "fy", 0.0) or 0.0)
    cx = float(getattr(intr, "cx", 0.0) or 0.0)
    cy = float(getattr(intr, "cy", 0.0) or 0.0)
    iw = int(getattr(intr, "width", width) or width)
    ih = int(getattr(intr, "height", height) or height)
    dist_obj = getattr(param, "rgb_distortion", None) if param is not None else None
    distortion = [
        float(getattr(dist_obj, name, 0.0) or 0.0)
        for name in ("k1", "k2", "p1", "p2", "k3")
    ]
    valid = fx > 1.0 and fy > 1.0 and iw > 0 and ih > 0
    return {
        "width": iw,
        "height": ih,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "distortion": distortion,
        "valid": valid,
    }


def next_capture_index(out_dir: Path) -> int:
    indices: list[int] = []
    for path in out_dir.glob("astra_rgb_*.png"):
        try:
            indices.append(int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def update_camera_config(path: Path, intrinsics: dict, reset_calibration: bool = False) -> None:
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg.setdefault("name", "astra_rgb")
    cfg.setdefault("device", "Orbbec Astra Pro Plus")
    cfg.setdefault("enabled", False)
    if reset_calibration or "calibrated" not in cfg:
        cfg["calibrated"] = False
    cfg["intrinsics"] = {k: v for k, v in intrinsics.items() if k != "valid"}
    cfg.setdefault(
        "extrinsics",
        {
            "frame": "world_from_camera",
            "position": [0.0, 0.0, 1.0],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
    )
    cfg.setdefault("calibration_target", {"type": "aruco_single_marker", "marker_length_m": 0.08, "dictionary": "DICT_4X4_50"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def load_existing_intrinsics(path: Path) -> dict | None:
    if not path.exists():
        return None
    cfg = json.loads(path.read_text(encoding="utf-8"))
    intr = cfg.get("intrinsics", {})
    fx = float(intr.get("fx", 0.0) or 0.0)
    fy = float(intr.get("fy", 0.0) or 0.0)
    if fx <= 1.0 or fy <= 1.0:
        return None
    return intr


def capture_orbbec(cv2, args, cfg_path: Path, out_dir: Path) -> None:
    Config, OBError, OBFormat, OBSensorType, Pipeline, VideoStreamProfile = require_orbbec()

    config = Config()
    pipeline = Pipeline()
    try:
        profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            profile = profiles.get_video_stream_profile(args.width, args.height, OBFormat.RGB, args.fps)
        except OBError:
            profile = profiles.get_default_video_stream_profile()
        config.enable_stream(profile)
    except Exception as exc:
        raise RuntimeError(f"Could not configure Orbbec color stream. Check udev rules and device support: {exc}") from exc

    pipeline.start(config)
    print("Orbbec color stream started. Press s to save, q/esc to exit.", flush=True)
    saved = next_capture_index(out_dir)
    frame_idx = 0
    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            img = frame_to_bgr_image(color_frame)
            intr = read_camera_param(pipeline, color_frame.get_width(), color_frame.get_height())
            if frame_idx == 0:
                update_camera_config(cfg_path, intr, reset_calibration=args.reset_calibration)
                if args.reset_calibration:
                    print(f"Saved RGB intrinsics to {cfg_path} and reset calibrated=false", flush=True)
                else:
                    print(f"Saved RGB intrinsics to {cfg_path}; existing extrinsics/calibrated flag preserved", flush=True)
                if not intr["valid"]:
                    print("Warning: SDK returned invalid intrinsics. Fill fx/fy/cx/cy manually from Orbbec Viewer before calibration.", flush=True)

            key = -1
            if not args.no_display:
                cv2.imshow("Astra RGB", img)
                key = cv2.waitKey(1)
            should_save = key == ord("s") or (args.save_every > 0 and frame_idx % args.save_every == 0)
            if should_save:
                path = out_dir / f"astra_rgb_{saved:05d}.png"
                cv2.imwrite(path.as_posix(), img)
                print(f"Saved {path}", flush=True)
                saved += 1
            frame_idx += 1
            if key in (ord("q"), 27):
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


def capture_opencv(cv2, args, cfg_path: Path, out_dir: Path) -> None:
    cap = cv2.VideoCapture(args.device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open V4L2 camera index {args.device_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    intr = load_existing_intrinsics(cfg_path)
    if intr is None:
        update_camera_config(
            cfg_path,
            {
                "width": args.width,
                "height": args.height,
                "fx": 0.0,
                "fy": 0.0,
                "cx": args.width * 0.5,
                "cy": args.height * 0.5,
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "valid": False,
            },
        )
        print(f"Warning: no valid intrinsics in {cfg_path}. Keep the Orbbec SDK intrinsics before calibration.", flush=True)
    else:
        print(f"Using existing RGB intrinsics from {cfg_path}", flush=True)

    print("OpenCV/V4L2 color stream started. Press s to save, q/esc to exit.", flush=True)
    saved = next_capture_index(out_dir)
    frame_idx = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                continue
            key = -1
            if not args.no_display:
                cv2.imshow("Astra RGB", img)
                key = cv2.waitKey(1)
            should_save = key == ord("s") or (args.save_every > 0 and frame_idx % args.save_every == 0)
            if should_save:
                path = out_dir / f"astra_rgb_{saved:05d}.png"
                cv2.imwrite(path.as_posix(), img)
                print(f"Saved {path}", flush=True)
                saved += 1
            frame_idx += 1
            if key in (ord("q"), 27):
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Astra Pro Plus RGB frames and export RGB intrinsics")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CAPTURE_DIR.as_posix())
    parser.add_argument("--camera-config", type=str, default=DEFAULT_CAMERA_CONFIG.as_posix())
    parser.add_argument("--backend", choices=["orbbec", "opencv"], default="orbbec")
    parser.add_argument("--device-index", type=int, default=0, help="OpenCV/V4L2 camera index used by --backend opencv")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save-every", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q/esc")
    parser.add_argument("--no-display", action="store_true", help="Do not open an OpenCV preview window")
    parser.add_argument(
        "--reset-calibration",
        action="store_true",
        help="Mark camera-config calibrated=false after updating intrinsics. By default existing extrinsics are preserved.",
    )
    args = parser.parse_args()

    cv2 = require_cv2()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = resolve(args.camera_config)

    if args.backend == "opencv":
        capture_opencv(cv2, args, cfg_path, out_dir)
        return

    capture_orbbec(cv2, args, cfg_path, out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
