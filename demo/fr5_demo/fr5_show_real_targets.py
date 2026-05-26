from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from arm_control import RealRgbSource, require_cv2
from real_rgb_table_perception import (
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_TASK_CONFIG,
    CameraProjector,
    attach_world_coordinates,
    default_tape_plane_z,
    detect_tapes_rgb,
    draw_detection_overlay,
    load_json,
    nearest_wall_grasp_point,
    real_myd_part_offset_from_config,
    real_tape_offset_from_config,
    resolve_demo_path,
    task_goal_pos_from_config,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "real_target_checks"


def save_snapshot(
    output_dir: Path,
    *,
    cv2,
    image_bgr: np.ndarray,
    detections: dict,
    config: dict,
    goal_pos: np.ndarray,
    tape_plane_z: float,
    active_color: str,
    myd_part_offset_m: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    png_path = output_dir / f"target_check_{stamp}.png"
    json_path = output_dir / f"target_check_{stamp}.json"
    if not cv2.imwrite(png_path.as_posix(), image_bgr):
        raise RuntimeError(f"Failed to write snapshot: {png_path}")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image": png_path.as_posix(),
        "active_color": active_color,
        "tape_plane_z_m": float(tape_plane_z),
        "myd_part_target_world_m": np.asarray(goal_pos, dtype=float).round(6).tolist(),
        "myd_part_session_offset_m": np.asarray(myd_part_offset_m, dtype=float).round(6).tolist(),
        "myd_part_config_offset_m": real_myd_part_offset_from_config(config).round(6).tolist(),
        "detections": {
            color: {
                "center_px": det.center_px.round(3).tolist(),
                "area_px": float(det.area_px),
                "score": float(det.score),
                "slot_name": str(det.slot_name),
                "bbox_xywh": list(det.bbox_xywh),
                "world_m": None if det.world_m is None else det.world_m.round(6).tolist(),
                "nearest_grasp_world_m": (
                    None if det.world_m is None else nearest_wall_grasp_point(config, det.world_m).round(6).tolist()
                ),
            }
            for color, det in detections.items()
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved target check: {png_path}", flush=True)
    print(f"Saved target data:  {json_path}", flush=True)


def write_myd_part_offset(config_path: Path, config: dict, offset_m: np.ndarray) -> dict:
    updated = dict(config)
    deployment = dict(updated.get("real_deployment", {}))
    deployment["myd_part_manual_offset_m"] = np.asarray(offset_m, dtype=float).round(6).tolist()
    deployment["myd_part_manual_offset_notes"] = (
        "Real-deployment only correction added by fr5_show_real_targets.py. "
        "It is added on top of sim_tape_pick_place.goal_pos_m + goal_offset_m."
    )
    updated["real_deployment"] = deployment
    config_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show live Astra RGB detection of tape coordinates and fixed myd_part target projection."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_TASK_CONFIG.as_posix())
    parser.add_argument("--camera-config", type=str, default=DEFAULT_CAMERA_CONFIG.as_posix())
    parser.add_argument("--real-rgb-source", choices=["live", "latest"], default="live")
    parser.add_argument("--real-rgb-image", type=str, default="")
    parser.add_argument("--real-rgb-width", type=int, default=640)
    parser.add_argument("--real-rgb-height", type=int, default=480)
    parser.add_argument("--real-rgb-fps", type=int, default=15)
    parser.add_argument("--colors", nargs="+", default=["red", "yellow", "white"], choices=["red", "yellow", "white"])
    parser.add_argument("--active-color", choices=["red", "yellow", "white"], default="red")
    parser.add_argument("--tape-plane-z", type=float, default=None, help="World Z plane used for RGB pixel -> table coordinate. Default comes from config target_object.initial_pos_m[2].")
    parser.add_argument("--min-area-px", type=float, default=350.0)
    parser.add_argument("--max-area-px", type=float, default=50000.0)
    parser.add_argument("--slot-roi", action=argparse.BooleanOptionalAction, default=True, help="Restrict tape detection to projected black-frame slots from config.")
    parser.add_argument("--slot-color-prior", action=argparse.BooleanOptionalAction, default=True, help="Use the three-slot prior: red/yellow occupy two slots, white is constrained to the remaining slot.")
    parser.add_argument("--slot-fallback-center", action=argparse.BooleanOptionalAction, default=True, help="If white is not segmented reliably, fall back to the remaining slot center instead of a reflection.")
    parser.add_argument("--slot-padding-m", type=float, default=0.025)
    parser.add_argument("--background-diff", action=argparse.BooleanOptionalAction, default=False, help="Use background subtraction before color/slot filtering. Press b in the window to capture the current empty-table background.")
    parser.add_argument("--background-image", type=str, default="", help="Optional RGB background image captured without tape.")
    parser.add_argument("--background-diff-thresh", type=float, default=32.0)
    parser.add_argument("--white-min-area-px", type=float, default=80.0)
    parser.add_argument("--white-max-area-px", type=float, default=18000.0)
    parser.add_argument("--white-min-circularity", type=float, default=0.12)
    parser.add_argument("--white-min-extent", type=float, default=0.12)
    parser.add_argument("--myd-part-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"), help="Temporary myd_part target offset in world meters.")
    parser.add_argument("--myd-nudge-step-m", type=float, default=0.005, help="Keyboard nudge step for myd_part target projection.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--no-display", action="store_true", help="Process one frame and save it without opening an OpenCV window.")
    args = parser.parse_args()

    cv2 = require_cv2()
    config_path = resolve_demo_path(args.config)
    camera_path = resolve_demo_path(args.camera_config)
    config = load_json(config_path)
    projector = CameraProjector.from_config(camera_path)
    tape_plane_z = default_tape_plane_z(config) if args.tape_plane_z is None else float(args.tape_plane_z)
    session_myd_offset = np.asarray(args.myd_part_offset, dtype=np.float64).reshape(3)
    goal_pos = task_goal_pos_from_config(config, manual_offset_m=session_myd_offset)
    output_dir = resolve_demo_path(args.output_dir)
    background_rgb = None
    if args.background_image:
        bg_bgr = cv2.imread(resolve_demo_path(args.background_image).as_posix(), cv2.IMREAD_COLOR)
        if bg_bgr is None:
            raise RuntimeError(f"Could not read background image: {args.background_image}")
        background_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)

    print("Real target check:", flush=True)
    print(f"  camera_config: {camera_path}", flush=True)
    print(f"  task_config:   {config_path}", flush=True)
    print(f"  tape_plane_z:  {tape_plane_z:.4f} m", flush=True)
    print(f"  config tape offset: {real_tape_offset_from_config(config).round(4).tolist()} m", flush=True)
    print(f"  myd_part goal: {goal_pos.round(4).tolist()} m", flush=True)
    print(f"  config myd offset: {real_myd_part_offset_from_config(config).round(4).tolist()} m", flush=True)
    print("  keys: s save snapshot, q/esc quit", flush=True)
    print("        j/l x-/x+, k/i y-/y+, o/u z-/z+, r reset session offset, w write total offset to config", flush=True)
    print("        b capture empty-table background for reflection suppression", flush=True)

    source = RealRgbSource(
        source=str(args.real_rgb_source),
        image_path=args.real_rgb_image or None,
        width=int(args.real_rgb_width),
        height=int(args.real_rgb_height),
        fps=int(args.real_rgb_fps),
        allow_fallback=True,
    )
    try:
        rgb = source.start()
        while True:
            frame = source.read(timeout_ms=max(20, int(1000 / max(1, int(args.real_rgb_fps)))))
            if frame is not None:
                rgb = frame
            goal_pos = task_goal_pos_from_config(config, manual_offset_m=session_myd_offset)
            detections = detect_tapes_rgb(
                rgb,
                cv2=cv2,
                colors=args.colors,
                min_area_px=float(args.min_area_px),
                max_area_px=float(args.max_area_px),
                config=config,
                projector=projector,
                z_world=tape_plane_z,
                use_slot_roi=bool(args.slot_roi),
                slot_padding_m=float(args.slot_padding_m),
                slot_color_prior=bool(args.slot_color_prior),
                slot_fallback_center=bool(args.slot_fallback_center),
                background_rgb=background_rgb,
                background_diff=bool(args.background_diff or background_rgb is not None),
                background_diff_thresh=float(args.background_diff_thresh),
                white_min_area_px=float(args.white_min_area_px),
                white_max_area_px=float(args.white_max_area_px),
                white_min_circularity=float(args.white_min_circularity),
                white_min_extent=float(args.white_min_extent),
            )
            attach_world_coordinates(detections, projector, cv2=cv2, z_world=tape_plane_z)
            overlay = draw_detection_overlay(
                rgb,
                cv2=cv2,
                detections=detections,
                projector=projector,
                config=config,
                tape_plane_z=tape_plane_z,
                active_color=str(args.active_color),
                draw_slots=bool(args.slot_roi),
                slot_padding_m=float(args.slot_padding_m),
                myd_part_offset_m=session_myd_offset,
            )
            if detections:
                text = " | ".join(
                    f"{name}: px={det.center_px.round(1).tolist()} world={None if det.world_m is None else det.world_m.round(4).tolist()}"
                    for name, det in detections.items()
                )
                print(text, flush=True)
            if args.no_display:
                save_snapshot(
                    output_dir,
                    cv2=cv2,
                    image_bgr=overlay,
                    detections=detections,
                    config=config,
                    goal_pos=goal_pos,
                    tape_plane_z=tape_plane_z,
                    active_color=str(args.active_color),
                    myd_part_offset_m=session_myd_offset,
                )
                break
            cv2.imshow("FR5 real RGB target check", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_snapshot(
                    output_dir,
                    cv2=cv2,
                    image_bgr=overlay,
                    detections=detections,
                    config=config,
                    goal_pos=goal_pos,
                    tape_plane_z=tape_plane_z,
                    active_color=str(args.active_color),
                    myd_part_offset_m=session_myd_offset,
                )
            elif key in (ord("j"), ord("l"), ord("k"), ord("i"), ord("o"), ord("u")):
                step = float(args.myd_nudge_step_m)
                if key == ord("j"):
                    session_myd_offset[0] -= step
                elif key == ord("l"):
                    session_myd_offset[0] += step
                elif key == ord("k"):
                    session_myd_offset[1] -= step
                elif key == ord("i"):
                    session_myd_offset[1] += step
                elif key == ord("o"):
                    session_myd_offset[2] -= step
                elif key == ord("u"):
                    session_myd_offset[2] += step
                goal_now = task_goal_pos_from_config(config, manual_offset_m=session_myd_offset)
                print(
                    f"myd_part session_offset={session_myd_offset.round(4).tolist()} goal={goal_now.round(4).tolist()}",
                    flush=True,
                )
            elif key == ord("r"):
                session_myd_offset[:] = 0.0
                print("myd_part session offset reset to [0, 0, 0].", flush=True)
            elif key == ord("w"):
                total_offset = real_myd_part_offset_from_config(config) + session_myd_offset
                config = write_myd_part_offset(config_path, config, total_offset)
                session_myd_offset[:] = 0.0
                print(f"Wrote real_deployment.myd_part_manual_offset_m={total_offset.round(6).tolist()} to {config_path}", flush=True)
            elif key == ord("b"):
                background_rgb = np.ascontiguousarray(rgb.copy())
                output_dir.mkdir(parents=True, exist_ok=True)
                bg_path = output_dir / f"background_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
                if not cv2.imwrite(bg_path.as_posix(), cv2.cvtColor(background_rgb, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"Failed to write background image: {bg_path}")
                args.background_diff = True
                print(f"Captured reflection background: {bg_path}", flush=True)
    finally:
        source.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
