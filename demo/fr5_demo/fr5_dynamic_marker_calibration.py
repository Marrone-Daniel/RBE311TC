from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from motrixsim import forward_kinematic
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from arm_control import (
    DEFAULT_CONFIG,
    RealRgbSource,
    build_runtime,
    load_config,
    resolve_demo_path,
    set_arm_qpos_and_ctrl,
)
from calibrate_astra_extrinsic import detect_marker, marker_object_points, require_cv2
from fr5_move_to_initial import has_robot_error, wait_until_reached
from fr5_sync_sdk import (
    DEFAULT_ROBOT_IP,
    FairinoArmClient,
    MotionCancelHandler,
    actuator_limits_rad,
    validate_targets_against_limits,
)


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CAMERA_CONFIG = DEMO_DIR / "configs" / "astra_camera.json"
DEFAULT_OUTPUT_ROOT = DEMO_DIR / "data" / "dynamic_marker_calib"
DEFAULT_REPORT = DEMO_DIR / "data" / "dynamic_marker_calib_last_report.json"


DEFAULT_OFFSETS_DEG = np.asarray(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [8.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-8.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, -5.0, 0.0, 0.0, 0.0],
        [0.0, -5.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -10.0, 0.0],
        [0.0, 0.0, 0.0, 8.0, 8.0, 0.0],
        [0.0, 0.0, 0.0, -8.0, -8.0, 0.0],
    ],
    dtype=np.float64,
)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return DEMO_DIR / path


def transform_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    out[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return out


def transform_from_rotvec_translation(rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return transform_from_rt(Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix(), translation)


def transform_to_pose_dict(transform: np.ndarray) -> dict:
    transform = np.asarray(transform, dtype=np.float64)
    return {
        "position": transform[:3, 3].tolist(),
        "rotation_matrix": transform[:3, :3].tolist(),
        "rotation_quat_xyzw": Rotation.from_matrix(transform[:3, :3]).as_quat().tolist(),
    }


def camera_matrix_from_config(camera_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    intr = camera_cfg["intrinsics"]
    camera_matrix = np.array(
        [[float(intr["fx"]), 0.0, float(intr["cx"])], [0.0, float(intr["fy"]), float(intr["cy"])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if camera_matrix[0, 0] <= 1.0 or camera_matrix[1, 1] <= 1.0:
        raise RuntimeError("Invalid Astra intrinsics. Run camera_capture_orbbec.py before dynamic calibration.")
    dist = np.asarray(intr.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)
    return camera_matrix, dist


def link_world_transform(model, data, link_name: str) -> np.ndarray:
    link = model.get_link(link_name)
    if link is None:
        raise RuntimeError(f"Unknown tracking link {link_name!r}. Available links: {model.link_names}")
    forward_kinematic(model, data)
    position = np.asarray(link.get_position(data), dtype=np.float64).reshape(3)
    rotation = np.asarray(link.get_rotation_mat(data), dtype=np.float64).reshape(3, 3)
    return transform_from_rt(rotation, position)


def link_world_transform_from_q(config: dict, model, data, body, qpos_ids: np.ndarray, arm_act_ids: np.ndarray, link_name: str, q_rad: np.ndarray) -> np.ndarray:
    set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, np.asarray(q_rad, dtype=np.float32))
    return link_world_transform(model, data, link_name)


def solve_marker_pose_from_corners(cv2, corners_px: np.ndarray, marker_length: float, camera_matrix: np.ndarray, dist: np.ndarray) -> np.ndarray:
    object_points = marker_object_points(marker_length)
    ok, rvec, tvec = cv2.solvePnP(object_points, np.asarray(corners_px, dtype=np.float64), camera_matrix, dist)
    if not ok:
        raise RuntimeError("cv2.solvePnP failed for dynamic marker sample")
    rotation, _ = cv2.Rodrigues(rvec)
    return transform_from_rt(rotation, tvec.reshape(3))


def detect_marker_corners(cv2, rgb: np.ndarray, *, dictionary: str, marker_id: int) -> np.ndarray:
    corners, detected_id = detect_marker(cv2, rgb, dictionary, int(marker_id))
    if int(detected_id) != int(marker_id):
        raise RuntimeError(f"Detected marker id {detected_id}, expected {marker_id}")
    return corners


def load_offsets(path: str | Path | None) -> np.ndarray:
    if not path:
        return DEFAULT_OFFSETS_DEG.copy()
    data = json.loads(resolve(path).read_text(encoding="utf-8"))
    offsets = np.asarray(data, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[1] != 6:
        raise RuntimeError(f"Pose offsets JSON must be Nx6 degrees, got shape {offsets.shape}")
    return offsets


def save_image(cv2, path: Path, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path.as_posix(), bgr)


def collect_samples(args) -> None:
    cv2 = require_cv2()
    config = load_config(args.config)
    model, data, body, qpos_ids, arm_act_ids, _ = build_runtime(config)
    lows, highs = actuator_limits_rad(model, arm_act_ids)

    session_dir = resolve(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)

    offsets = load_offsets(args.pose_offsets)
    arm = FairinoArmClient(args.robot_ip, speed_percent=float(args.speed_percent))
    canceller = MotionCancelHandler()
    canceller.install()
    canceller.set_arm(arm)
    rgb_source: RealRgbSource | None = None
    samples: list[dict] = []
    try:
        arm.connect()
        if args.reset_errors:
            print("Resetting controller errors with ResetAllError().", flush=True)
            arm.reset_all_error()
            time.sleep(0.5)
        if args.prepare_controller:
            print("Preparing controller: Mode(0 automatic), RobotEnable(1).", flush=True)
            arm.set_mode(0)
            arm.robot_enable(1)
            time.sleep(0.5)
        error_code = arm.get_robot_error_code()
        if has_robot_error(error_code):
            raise RuntimeError(f"Controller reports error code {error_code}; refusing dynamic calibration motion.")
        actual = arm.get_actual_joint_deg()
        if actual is None:
            raise RuntimeError("Could not read current FR5 joint angles.")
        start_deg = np.asarray(actual, dtype=np.float64)
        targets_deg = start_deg[None, :] + offsets
        targets_rad = np.deg2rad(targets_deg)
        validate_targets_against_limits(targets_rad, lows, highs)
        for idx, q_rad in enumerate(targets_rad):
            t_world_tcp = link_world_transform_from_q(config, model, data, body, qpos_ids, arm_act_ids, str(config["tcp_site"]) if False else args.tracking_link, q_rad)
            if float(t_world_tcp[2, 3]) < float(args.min_link_z):
                raise RuntimeError(
                    f"Tracking link z below safety threshold at pose {idx}: z={t_world_tcp[2, 3]:.4f}, min={args.min_link_z:.4f}"
                )

        print(f"Dynamic calibration session: {session_dir}", flush=True)
        print(f"Start joint deg: {start_deg.tolist()}", flush=True)
        print(f"Targets: {len(targets_deg)} poses, tracking_link={args.tracking_link}", flush=True)
        if not args.execute_real:
            print("Dry run only. Add --execute-real to move and capture.", flush=True)
            return

        rgb_source = RealRgbSource(
            source="live",
            image_path=None,
            width=int(args.real_rgb_width),
            height=int(args.real_rgb_height),
            fps=int(args.real_rgb_fps),
            allow_fallback=False,
        )
        rgb_source.start()

        for idx, target_deg in enumerate(targets_deg):
            canceller.check()
            print(f"\nPose {idx + 1}/{len(targets_deg)} target_deg={target_deg.tolist()}", flush=True)
            arm.move_j(
                target_deg,
                vel=float(args.vel),
                acc=float(args.acc),
                ovl=float(args.ovl),
                blend_t=0.0,
                rpc_timeout=float(args.rpc_timeout),
            )
            wait_until_reached(
                arm,
                target_deg,
                tolerance_deg=float(args.tolerance_deg),
                motion_timeout=float(args.motion_timeout),
                no_motion_timeout=float(args.no_motion_timeout),
                progress_epsilon_deg=float(args.progress_epsilon_deg),
                poll_dt=0.5,
            )
            time.sleep(float(args.settle_s))
            rgb = None
            corners = None
            detection_error = None
            for _ in range(max(1, int(args.detect_tries))):
                rgb = rgb_source.read(timeout_ms=250)
                if rgb is None:
                    continue
                try:
                    corners = detect_marker_corners(cv2, rgb, dictionary=args.dictionary, marker_id=int(args.marker_id))
                    break
                except Exception as exc:
                    detection_error = str(exc)
                    time.sleep(0.05)
            if rgb is None:
                raise RuntimeError("Live Astra did not return an RGB frame during dynamic calibration.")
            image_path = session_dir / f"sample_{idx:03d}.png"
            save_image(cv2, image_path, rgb)
            actual_after = arm.get_actual_joint_deg()
            if actual_after is None:
                raise RuntimeError("Lost FR5 joint readback after MoveJ.")
            sample = {
                "index": idx,
                "image": image_path.relative_to(session_dir).as_posix(),
                "timestamp": time.time(),
                "target_joint_deg": target_deg.tolist(),
                "actual_joint_deg": [float(v) for v in actual_after],
                "actual_joint_rad": np.deg2rad(np.asarray(actual_after, dtype=np.float64)).tolist(),
                "tracking_link": args.tracking_link,
                "marker_id": int(args.marker_id),
                "marker_length_m": float(args.marker_length),
                "detected": corners is not None,
                "marker_corners_px": None if corners is None else corners.tolist(),
                "detection_error": detection_error,
            }
            samples.append(sample)
            print(f"  saved={image_path} detected={sample['detected']} error={detection_error}", flush=True)

        metadata = {
            "type": "fr5_dynamic_eye_to_hand_samples",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "robot_ip": args.robot_ip,
            "tracking_link": args.tracking_link,
            "dictionary": args.dictionary,
            "marker_id": int(args.marker_id),
            "marker_length_m": float(args.marker_length),
            "samples": samples,
        }
        (session_dir / "samples.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"\nSaved sample metadata: {session_dir / 'samples.json'}", flush=True)
    except KeyboardInterrupt:
        print("Dynamic calibration collection cancelled. StopMotion was sent best-effort.", flush=True)
    finally:
        if rgb_source is not None:
            rgb_source.stop()
        arm.close()


def latest_samples_file() -> Path:
    candidates = sorted(DEFAULT_OUTPUT_ROOT.glob("*/samples.json"))
    if not candidates:
        raise RuntimeError(f"No dynamic calibration samples found under {DEFAULT_OUTPUT_ROOT}")
    return candidates[-1]


def build_observations(args) -> tuple[dict, dict, list[dict], np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    samples_path = latest_samples_file() if not args.samples else resolve(args.samples)
    session_dir = samples_path.parent
    metadata = json.loads(samples_path.read_text(encoding="utf-8"))
    camera_path = resolve(args.camera_config)
    camera_cfg = json.loads(camera_path.read_text(encoding="utf-8"))
    camera_matrix, dist = camera_matrix_from_config(camera_cfg)
    cv2 = require_cv2()
    config = load_config(args.config)
    model, data, body, qpos_ids, arm_act_ids, _ = build_runtime(config)

    observations = []
    t_world_links = []
    t_camera_markers = []
    marker_length = float(args.marker_length or metadata.get("marker_length_m", 0.05))
    metadata_length = float(metadata.get("marker_length_m", marker_length))
    if abs(marker_length - metadata_length) > 1e-6:
        print(
            f"WARNING: solve marker length {marker_length:.4f}m differs from collected metadata "
            f"{metadata_length:.4f}m. Using solve value.",
            flush=True,
        )
    marker_id = int(args.marker_id if args.marker_id is not None else metadata.get("marker_id", 0))
    dictionary = str(args.dictionary or metadata.get("dictionary", "DICT_4X4_50"))
    excluded = {int(item) for item in getattr(args, "exclude_samples", [])}
    for sample in metadata["samples"]:
        if int(sample.get("index", -1)) in excluded:
            continue
        image_path = session_dir / sample["image"]
        image = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if sample.get("detected") and sample.get("marker_corners_px") is not None:
            corners = np.asarray(sample["marker_corners_px"], dtype=np.float64)
        else:
            try:
                corners, _ = detect_marker(cv2, image, dictionary, marker_id)
            except Exception:
                continue
        q_rad = np.asarray(sample.get("actual_joint_rad") or np.deg2rad(sample["actual_joint_deg"]), dtype=np.float64)
        t_world_link = link_world_transform_from_q(config, model, data, body, qpos_ids, arm_act_ids, args.tracking_link, q_rad)
        t_camera_marker = solve_marker_pose_from_corners(cv2, corners, marker_length, camera_matrix, dist)
        observations.append({**sample, "image_abs": image_path.as_posix(), "marker_corners_px": corners.tolist()})
        t_world_links.append(t_world_link)
        t_camera_markers.append(t_camera_marker)

    if len(observations) < int(args.min_samples):
        raise RuntimeError(f"Need at least {args.min_samples} detected dynamic samples, got {len(observations)}")
    return metadata, camera_cfg, observations, camera_matrix, dist, t_world_links, t_camera_markers


def mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    rotations = Rotation.from_matrix([t[:3, :3] for t in transforms]).mean().as_matrix()
    translations = np.mean([t[:3, 3] for t in transforms], axis=0)
    return transform_from_rt(rotations, translations)


def optimize_dynamic_extrinsic(args):
    metadata, camera_cfg, observations, camera_matrix, dist, t_world_links, t_camera_markers = build_observations(args)
    marker_length = float(args.marker_length or observations[0].get("marker_length_m", 0.05))
    object_points = marker_object_points(marker_length)
    image_points = [np.asarray(obs["marker_corners_px"], dtype=np.float64) for obs in observations]

    initial_link_marker = transform_from_rotvec_translation(
        np.deg2rad(np.asarray(args.initial_marker_rpy_deg, dtype=np.float64)),
        np.asarray(args.initial_marker_pos, dtype=np.float64),
    )
    initial_world_cameras = [t_w_l @ initial_link_marker @ np.linalg.inv(t_c_m) for t_w_l, t_c_m in zip(t_world_links, t_camera_markers)]
    initial_world_camera = mean_transform(initial_world_cameras)
    p0 = np.concatenate(
        [
            Rotation.from_matrix(initial_world_camera[:3, :3]).as_rotvec(),
            initial_world_camera[:3, 3],
            Rotation.from_matrix(initial_link_marker[:3, :3]).as_rotvec(),
            initial_link_marker[:3, 3],
        ]
    )

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t_world_camera = transform_from_rotvec_translation(params[:3], params[3:6])
        t_link_marker = transform_from_rotvec_translation(params[6:9], params[9:12])
        return t_world_camera, t_link_marker

    def residuals(params: np.ndarray) -> np.ndarray:
        t_world_camera, t_link_marker = unpack(params)
        t_camera_world = np.linalg.inv(t_world_camera)
        residual = []
        for t_world_link, corners_px in zip(t_world_links, image_points):
            t_camera_marker = t_camera_world @ t_world_link @ t_link_marker
            rvec = Rotation.from_matrix(t_camera_marker[:3, :3]).as_rotvec()
            tvec = t_camera_marker[:3, 3]
            projected, _ = require_cv2().projectPoints(object_points, rvec, tvec, camera_matrix, dist)
            residual.append((projected.reshape(-1, 2) - corners_px).reshape(-1))
        return np.concatenate(residual)

    result = least_squares(
        residuals,
        p0,
        loss=str(args.loss),
        f_scale=float(args.loss_scale),
        max_nfev=int(args.max_nfev),
        verbose=1 if args.verbose else 0,
    )
    t_world_camera, t_link_marker = unpack(result.x)
    residual_px = residuals(result.x).reshape(-1, 2)
    error_px = np.linalg.norm(residual_px, axis=1)
    per_sample_error = []
    for idx, obs in enumerate(observations):
        sample_errors = error_px[idx * 4 : (idx + 1) * 4]
        per_sample_error.append(
            {
                "index": int(obs["index"]),
                "image": obs["image"],
                "mean_px": float(np.mean(sample_errors)),
                "max_px": float(np.max(sample_errors)),
            }
        )

    camera_cfg["name"] = camera_cfg.get("name", "astra_rgb")
    camera_cfg["enabled"] = True
    camera_cfg["calibrated"] = True
    camera_cfg["extrinsics"] = {
        "frame": "world_from_camera",
        "position": t_world_camera[:3, 3].tolist(),
        "rotation_matrix": t_world_camera[:3, :3].tolist(),
    }
    camera_cfg["calibration_target"] = {
        "type": "fr5_dynamic_marker_eye_to_hand",
        "tracking_link": args.tracking_link,
        "samples": len(observations),
        "excluded_samples": sorted(int(item) for item in getattr(args, "exclude_samples", [])),
        "marker_length_m": marker_length,
        "link_from_marker": transform_to_pose_dict(t_link_marker),
        "reprojection_error_px": {
            "mean_px": float(np.mean(error_px)),
            "max_px": float(np.max(error_px)),
            "per_sample_px": per_sample_error,
            "per_corner_px": error_px.tolist(),
        },
        "least_squares": {
            "success": bool(result.success),
            "cost": float(result.cost),
            "message": str(result.message),
            "nfev": int(result.nfev),
        },
    }
    camera_path = resolve(args.camera_config)
    mean_error = float(np.mean(error_px))
    max_allowed = float(getattr(args, "max_mean_reprojection_px", 2.0))
    if args.write_camera and mean_error > max_allowed:
        raise RuntimeError(
            f"Refusing to write camera config: mean reprojection error {mean_error:.3f}px "
            f"> --max-mean-reprojection-px {max_allowed:.3f}px. "
            "Use the correct --tracking-link, exclude bad samples, or collect cleaner samples."
        )
    if args.write_camera:
        camera_path.write_text(json.dumps(camera_cfg, indent=2), encoding="utf-8")

    report = {
        "camera_config": camera_path.as_posix(),
        "camera_config_written": bool(args.write_camera),
        "samples_metadata": metadata,
        "world_from_camera": transform_to_pose_dict(t_world_camera),
        "link_from_marker": transform_to_pose_dict(t_link_marker),
        "tracking_link": args.tracking_link,
        "excluded_samples": sorted(int(item) for item in getattr(args, "exclude_samples", [])),
        "samples": observations,
        "reprojection_error_px": camera_cfg["calibration_target"]["reprojection_error_px"],
        "least_squares": camera_cfg["calibration_target"]["least_squares"],
    }
    report_path = resolve(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.write_camera:
        print(f"Saved calibrated camera config: {camera_path}", flush=True)
    else:
        print(f"Camera config not written (--no-write-camera): {camera_path}", flush=True)
    print(f"Saved dynamic calibration report: {report_path}", flush=True)
    print(f"Samples used: {len(observations)}", flush=True)
    print(f"world_from_camera position: {camera_cfg['extrinsics']['position']}", flush=True)
    print(f"link_from_marker position: {report['link_from_marker']['position']}", flush=True)
    print(f"Reprojection error: mean={np.mean(error_px):.3f}px max={np.max(error_px):.3f}px", flush=True)
    worst = sorted(per_sample_error, key=lambda item: item["mean_px"], reverse=True)[:5]
    print("Worst samples:", flush=True)
    for item in worst:
        print(f"  sample={item['index']} mean={item['mean_px']:.3f}px max={item['max_px']:.3f}px image={item['image']}", flush=True)
    if mean_error > 2.0:
        print("WARNING: mean reprojection error is high. Use more diverse poses, avoid blur/glare, and keep the marker rigid.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic eye-to-hand calibration with a 5cm marker attached to an FR5 link")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Move the real FR5 through small poses and capture Astra marker samples")
    collect.add_argument("--config", type=str, default=DEFAULT_CONFIG.as_posix())
    collect.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP)
    collect.add_argument("--tracking-link", type=str, default="wrist1_link")
    collect.add_argument("--marker-id", type=int, default=0)
    collect.add_argument("--marker-length", type=float, default=0.05)
    collect.add_argument("--dictionary", type=str, default="DICT_4X4_50")
    collect.add_argument("--pose-offsets", type=str, default="", help="Optional JSON Nx6 joint offsets in degrees")
    collect.add_argument("--output-dir", type=str, default="")
    collect.add_argument("--real-rgb-width", type=int, default=640)
    collect.add_argument("--real-rgb-height", type=int, default=480)
    collect.add_argument("--real-rgb-fps", type=int, default=10)
    collect.add_argument("--detect-tries", type=int, default=10)
    collect.add_argument("--settle-s", type=float, default=0.4)
    collect.add_argument("--speed-percent", type=float, default=5.0)
    collect.add_argument("--vel", type=float, default=5.0)
    collect.add_argument("--acc", type=float, default=5.0)
    collect.add_argument("--ovl", type=float, default=10.0)
    collect.add_argument("--rpc-timeout", type=float, default=5.0)
    collect.add_argument("--motion-timeout", type=float, default=120.0)
    collect.add_argument("--no-motion-timeout", type=float, default=12.0)
    collect.add_argument("--progress-epsilon-deg", type=float, default=0.005)
    collect.add_argument("--tolerance-deg", type=float, default=1.0)
    collect.add_argument("--min-link-z", type=float, default=0.04)
    collect.add_argument("--reset-errors", action="store_true")
    collect.add_argument("--prepare-controller", action=argparse.BooleanOptionalAction, default=True)
    collect.add_argument("--execute-real", action="store_true", help="Actually move the physical FR5")
    collect.set_defaults(func=collect_samples)

    solve = sub.add_parser("solve", help="Solve camera extrinsics from collected dynamic marker samples")
    solve.add_argument("--config", type=str, default=DEFAULT_CONFIG.as_posix())
    solve.add_argument("--camera-config", type=str, default=DEFAULT_CAMERA_CONFIG.as_posix())
    solve.add_argument("--samples", type=str, default="", help="Path to samples.json; defaults to latest session")
    solve.add_argument("--tracking-link", type=str, default="wrist1_link")
    solve.add_argument("--marker-id", type=int, default=None)
    solve.add_argument("--marker-length", type=float, default=0.0, help="Meters. 0 uses the value recorded in samples.json")
    solve.add_argument("--dictionary", type=str, default="")
    solve.add_argument("--min-samples", type=int, default=8)
    solve.add_argument("--initial-marker-pos", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    solve.add_argument("--initial-marker-rpy-deg", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    solve.add_argument("--loss", type=str, default="soft_l1")
    solve.add_argument("--loss-scale", type=float, default=2.0)
    solve.add_argument("--max-nfev", type=int, default=300)
    solve.add_argument("--exclude-samples", type=int, nargs="*", default=[], help="Sample indices to ignore during solve")
    solve.add_argument("--max-mean-reprojection-px", type=float, default=2.0, help="Refuse --write-camera when mean reprojection error is above this threshold")
    solve.add_argument("--report", type=str, default=DEFAULT_REPORT.as_posix())
    solve.add_argument("--write-camera", action=argparse.BooleanOptionalAction, default=True)
    solve.add_argument("--verbose", action="store_true")
    solve.set_defaults(func=optimize_dynamic_extrinsic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
