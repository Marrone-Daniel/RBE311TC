from __future__ import annotations

import argparse
import json
import shutil
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from arm_control import RealRgbSource, build_runtime, require_cv2, set_arm_qpos_and_ctrl, site_position
from fr5_il_dataset import DEFAULT_IL_DEMO_DIR, make_target_feature
from fr5_record_demonstration import (
    EpisodeBuffer,
    connect_gripper_best_effort,
    make_episode_dir,
    move_to_initial_from_config,
    write_episode,
)
from fr5_sim_tape_pick_place import (
    Waypoint as SimWaypoint,
    grasp_wall_offset,
    interpolate_waypoints,
    rotate_gripper_joint6,
    solve_tcp_ik_with_axis,
)
from fr5_sync_sdk import DEFAULT_ROBOT_IP, FairinoArmClient
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


@dataclass(frozen=True)
class Waypoint:
    name: str
    xyz_m: np.ndarray
    gripper: float
    hold_s: float = 0.0


class EpisodeStop(Exception):
    pass


class SessionQuit(Exception):
    pass


class TerminalKeyPoller:
    def __init__(self) -> None:
        self.enabled = False
        self._fd = None
        self._old_settings = None

    def __enter__(self) -> "TerminalKeyPoller":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        self.enabled = False

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None
        return sys.stdin.read(1)


KEY_POLLER: TerminalKeyPoller | None = None


def pose_uses_mm(pose: list[float] | np.ndarray) -> bool:
    xyz = np.asarray(pose[:3], dtype=np.float64)
    return bool(np.max(np.abs(xyz)) > 3.0)


def tcp_xyz_m(pose: list[float] | np.ndarray) -> np.ndarray:
    xyz = np.asarray(pose[:3], dtype=np.float64)
    return xyz / 1000.0 if pose_uses_mm(pose) else xyz


def pose_with_xyz_m(reference_pose: list[float] | np.ndarray, xyz_m: np.ndarray) -> list[float]:
    ref = [float(v) for v in reference_pose[:6]]
    xyz = np.asarray(xyz_m, dtype=np.float64).reshape(3)
    if pose_uses_mm(ref):
        ref[:3] = (xyz * 1000.0).astype(float).tolist()
    else:
        ref[:3] = xyz.astype(float).tolist()
    return ref


def validate_workspace(points: list[np.ndarray], args) -> None:
    x_min, x_max = float(args.workspace_x_min), float(args.workspace_x_max)
    y_min, y_max = float(args.workspace_y_min), float(args.workspace_y_max)
    z_min, z_max = float(args.workspace_z_min), float(args.workspace_z_max)
    for idx, point in enumerate(points):
        x, y, z = np.asarray(point, dtype=np.float64).reshape(3)
        if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
            raise RuntimeError(
                f"Waypoint {idx} {point.round(4).tolist()} is outside workspace: "
                f"x[{x_min},{x_max}] y[{y_min},{y_max}] z[{z_min},{z_max}]"
            )


def planned_waypoints(config: dict, tape_center: np.ndarray, goal_center: np.ndarray, args) -> tuple[list[Waypoint], np.ndarray, np.ndarray]:
    task = config.get("sim_tape_pick_place", {})
    tape_center = np.asarray(tape_center, dtype=np.float64).reshape(3)
    goal_center = np.asarray(goal_center, dtype=np.float64).reshape(3)
    grasp = nearest_wall_grasp_point(config, tape_center)
    wall_offset = grasp[:2] - tape_center[:2]
    place_xy = goal_center[:2] + wall_offset

    if args.grasp_z is not None:
        grasp_z = float(args.grasp_z)
    else:
        grasp_z = max(
            float(tape_center[2]) + float(args.grasp_z_clearance_m),
            float(args.min_grasp_z),
        )
    approach_z_raw = float(args.approach_z) if args.approach_z is not None else float(task.get("approach_height_m", 0.15))
    approach_z = max(approach_z_raw, grasp_z + float(args.approach_clearance_m))
    lift_z_raw = float(args.lift_z) if args.lift_z is not None else float(task.get("lift_height_m", 0.20))
    lift_z = max(lift_z_raw, approach_z)
    place_z = float(args.place_z) if args.place_z is not None else float(task.get("release_height_m", 0.015))
    place_approach_z = float(args.place_approach_z) if args.place_approach_z is not None else max(lift_z, goal_center[2] + 0.10)
    open_g = float(task.get("default_gripper_opening", 0.0))
    closed_g = float(task.get("default_gripper_closed", 1.0))
    release_g = float(task.get("release_gripper_closure", 0.5))

    above_grasp = np.asarray([grasp[0], grasp[1], approach_z], dtype=np.float64)
    grasp_low = np.asarray([grasp[0], grasp[1], grasp_z], dtype=np.float64)
    grasp_lift = np.asarray([grasp[0], grasp[1], lift_z], dtype=np.float64)
    above_place = np.asarray([place_xy[0], place_xy[1], place_approach_z], dtype=np.float64)
    place_low = np.asarray([place_xy[0], place_xy[1], place_z], dtype=np.float64)
    retreat = np.asarray([place_xy[0], place_xy[1], max(place_approach_z, lift_z)], dtype=np.float64)
    waypoints = [
        Waypoint("above_grasp_open", above_grasp, open_g),
        Waypoint("grasp_depth_open", grasp_low, open_g),
        Waypoint("close_gripper", grasp_low, closed_g, hold_s=float(args.gripper_hold_s)),
        Waypoint("lift_tape", grasp_lift, closed_g),
        Waypoint("above_myd_part", above_place, closed_g),
        Waypoint("lower_to_release", place_low, closed_g),
        Waypoint("release_partial_open", place_low, release_g, hold_s=float(args.release_hold_s)),
        Waypoint("retreat", retreat, release_g),
    ]
    validate_workspace([w.xyz_m for w in waypoints], args)
    return waypoints, grasp, np.asarray([place_xy[0], place_xy[1], goal_center[2]], dtype=np.float64)


def planned_sim_ik_waypoints(
    config: dict,
    tape_center: np.ndarray,
    goal_center: np.ndarray,
    args,
) -> tuple[list[SimWaypoint], list[SimWaypoint], np.ndarray, np.ndarray]:
    """Build the same IK waypoint sequence as fr5_sim_tape_pick_place.py for one real detected tape."""
    task = config.get("sim_tape_pick_place", {})
    model, data, body, qpos_ids, arm_act_ids, _ = build_runtime(config, None)
    tcp_site = str(config["tcp_site"])

    q_home = np.asarray(config["initial_qpos"], dtype=np.float32)
    set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q_home)

    tape_center = np.asarray(tape_center, dtype=np.float32).reshape(3)
    goal_center = np.asarray(goal_center, dtype=np.float32).reshape(3)
    wall_offset = grasp_wall_offset(task, tape_center)
    grasp_pos = tape_center + wall_offset

    drop_pos = goal_center.copy()
    drop_pos[2] = tape_center[2]
    drop_pos[0] += float(args.drop_delta_x if args.drop_delta_x is not None else task.get("drop_delta_x_m", 0.0))
    drop_pos[:2] += np.asarray(args.drop_xy_offset, dtype=np.float32)

    approach_h = float(args.approach_z) if args.approach_z is not None else float(task.get("approach_height_m", 0.13))
    grasp_h = float(args.grasp_height) if args.grasp_height is not None else float(task.get("grasp_height_m", 0.035))
    lift_delta = float(args.lift_delta) if args.lift_delta is not None else float(task.get("lift_delta_m", 0.14))
    lift_h = float(args.lift_z) if args.lift_z is not None else float(task.get("lift_height_m", grasp_h + lift_delta))
    if args.release_height is not None:
        release_h = float(args.release_height)
    elif "release_drop_delta_m" in task:
        release_h = max(0.0, lift_h - float(task.get("release_drop_delta_m", 0.03)))
    else:
        release_h = float(task.get("release_height_m", 0.035))

    vertical_descend_segments = max(1, int(task.get("vertical_descend_segments", 6)))
    vertical_descend_frames = max(vertical_descend_segments, int(float(task.get("vertical_descend_duration_s", 1.0)) * args.hz))
    vertical_axis = np.asarray(task.get("gripper_vertical_axis", [0.0, 0.0, -1.0]), dtype=np.float32)
    vertical_axis_index = int(task.get("gripper_vertical_axis_index", 2))
    vertical_axis_weight = float(task.get("gripper_vertical_axis_weight", 0.08))
    vertical_axis_tol = float(task.get("gripper_vertical_axis_tol", 0.035))
    open_g = float(task.get("default_gripper_opening", 0.0))
    closed_g = float(task.get("default_gripper_closed", 1.0))
    release_g = float(task.get("release_gripper_closure", open_g))

    q_above_seed = rotate_gripper_joint6(q_home, task, q_home)
    q_above = solve_tcp_ik_with_axis(
        model,
        data,
        body,
        qpos_ids,
        arm_act_ids,
        tcp_site,
        q_above_seed,
        grasp_pos + [0.0, 0.0, approach_h],
        desired_axis=vertical_axis,
        axis_index=vertical_axis_index,
        axis_weight=vertical_axis_weight,
        axis_tol=vertical_axis_tol,
    )
    descend_qs: list[np.ndarray] = []
    q_desc_seed = q_above
    for z in np.linspace(approach_h, grasp_h, vertical_descend_segments + 1, dtype=np.float32)[1:]:
        q_desc = solve_tcp_ik_with_axis(
            model,
            data,
            body,
            qpos_ids,
            arm_act_ids,
            tcp_site,
            q_desc_seed,
            grasp_pos + [0.0, 0.0, float(z)],
            desired_axis=vertical_axis,
            axis_index=vertical_axis_index,
            axis_weight=vertical_axis_weight,
            axis_tol=vertical_axis_tol,
        )
        descend_qs.append(q_desc)
        q_desc_seed = q_desc
    q_grasp = descend_qs[-1]
    q_lift = solve_tcp_ik_with_axis(
        model,
        data,
        body,
        qpos_ids,
        arm_act_ids,
        tcp_site,
        q_grasp,
        grasp_pos + [0.0, 0.0, grasp_h + lift_delta],
        desired_axis=vertical_axis,
        axis_index=vertical_axis_index,
        axis_weight=vertical_axis_weight,
        axis_tol=vertical_axis_tol,
    )
    q_transfer = solve_tcp_ik_with_axis(
        model,
        data,
        body,
        qpos_ids,
        arm_act_ids,
        tcp_site,
        q_lift,
        drop_pos + wall_offset + [0.0, 0.0, lift_h],
        desired_axis=vertical_axis,
        axis_index=vertical_axis_index,
        axis_weight=vertical_axis_weight,
        axis_tol=vertical_axis_tol,
    )
    q_lower = solve_tcp_ik_with_axis(
        model,
        data,
        body,
        qpos_ids,
        arm_act_ids,
        tcp_site,
        q_transfer,
        drop_pos + wall_offset + [0.0, 0.0, release_h],
        desired_axis=vertical_axis,
        axis_index=vertical_axis_index,
        axis_weight=vertical_axis_weight,
        axis_tol=vertical_axis_tol,
    )
    q_retreat = solve_tcp_ik_with_axis(
        model,
        data,
        body,
        qpos_ids,
        arm_act_ids,
        tcp_site,
        q_lower,
        drop_pos + wall_offset + [0.0, 0.0, approach_h],
        desired_axis=vertical_axis,
        axis_index=vertical_axis_index,
        axis_weight=vertical_axis_weight,                                  
        axis_tol=vertical_axis_tol,
    )

    name = str(args.active_color)
    waypoints: list[SimWaypoint] = [
        SimWaypoint("home", q_home, open_g, int(5 * args.hz), False, False, name),
        SimWaypoint(f"{name}_rotate_gripper_90", q_above, open_g, int(3 * args.hz), False, False, name),
        SimWaypoint(f"{name}_above_grasp", q_above, open_g, int(1.0 * args.hz), False, False, name),
        *[
            SimWaypoint(
                f"{name}_vertical_descend",
                q_desc,
                open_g,
                max(1, int(round(vertical_descend_frames / vertical_descend_segments))),
                False,
                False,
                name,
            )
            for q_desc in descend_qs
        ],
        SimWaypoint(f"{name}_close", q_grasp, closed_g, int(1.2 * args.hz), True, False, name),
        SimWaypoint(f"{name}_lift", q_lift, closed_g, int(1.6 * args.hz), True, False, name),
        SimWaypoint(f"{name}_transfer", q_transfer, closed_g, int(1.2 * args.hz), True, False, name),
        SimWaypoint(f"{name}_lower", q_lower, closed_g, int(0.8 * args.hz), True, False, name),
        SimWaypoint(f"{name}_release", q_lower, release_g, int(0.5 * args.hz), False, True, name),
        SimWaypoint(f"{name}_retreat", q_retreat, release_g, int(0.8 * args.hz), False, False, name),
    ]
    frames = [waypoints[0]] + interpolate_waypoints(waypoints)
    return frames, waypoints, grasp_pos.astype(np.float64), (drop_pos + wall_offset).astype(np.float64)


class FkTableClearanceGuard:
    def __init__(self, config: dict, args) -> None:
        self.enabled = bool(args.fk_table_safety)
        self.table_z = float(args.fk_table_z)
        self.min_clearance = float(args.min_gripper_table_clearance)
        self.gripper_radius = float(args.fk_gripper_radius)
        self.link_names = [str(name) for name in args.fk_safety_links]
        self.tcp_site = str(config["tcp_site"])
        self.model = None
        self.data = None
        self.body = None
        self.qpos_ids = None
        self.arm_act_ids = None
        if self.enabled:
            self.model, self.data, self.body, self.qpos_ids, self.arm_act_ids, _ = build_runtime(config, None)

    def clearance_m(self, q_rad: np.ndarray) -> tuple[float, str, float]:
        if not self.enabled:
            return float("inf"), "disabled", float("inf")
        set_arm_qpos_and_ctrl(self.data, self.model, self.body, self.qpos_ids, self.arm_act_ids, np.asarray(q_rad, dtype=np.float32))
        points: list[tuple[str, float]] = [(self.tcp_site, float(site_position(self.model, self.data, self.tcp_site)[2]))]
        for link_name in self.link_names:
            if link_name not in self.model.link_names:
                continue
            link = self.model.get_link(link_name)
            points.append((link_name, float(np.asarray(link.get_position(self.data), dtype=np.float64).reshape(3)[2])))
        name, center_z = min(points, key=lambda item: item[1])
        lowest_z = float(center_z) - self.gripper_radius
        return lowest_z - self.table_z, name, lowest_z

    def check(self, q_rad: np.ndarray, phase: str) -> None:
        clearance, source_name, lowest_z = self.clearance_m(q_rad)
        if clearance < self.min_clearance:
            raise RuntimeError(
                f"FK table safety stop at phase={phase}: estimated lowest gripper point from {source_name} "
                f"is z={lowest_z:.4f}m, table_z={self.table_z:.4f}m, clearance={clearance:.4f}m "
                f"< required {self.min_clearance:.4f}m. This check uses robot joints + MJCF FK, not camera pixels."
            )


def print_fk_clearance_preflight(frames: list[SimWaypoint], config: dict, args) -> None:
    if not frames or not bool(args.fk_table_safety):
        return
    guard = FkTableClearanceGuard(config, args)
    min_clearance = float("inf")
    min_phase = ""
    for item in frames:
        clearance, _, _ = guard.clearance_m(item.q)
        if clearance < min_clearance:
            min_clearance = clearance
            min_phase = str(item.name)
        guard.check(item.q, str(item.name))
    print(
        f"  FK table safety preflight passed: min_clearance={min_clearance:.4f}m@{min_phase}, "
        f"required={float(args.min_gripper_table_clearance):.4f}m",
        flush=True,
    )


def command_gripper(gripper, closure: float, args) -> None:
    if gripper is None:
        return
    try:
        gripper.command_closure(float(closure), speed=int(args.gripper_speed), force=int(args.gripper_force))
    except Exception as exc:
        print(f"Warning: gripper command failed, continuing with virtual gripper value: {exc}", flush=True)


def record_real_frame(
    buffer: EpisodeBuffer,
    *,
    cv2,
    rgb: np.ndarray,
    arm: FairinoArmClient,
    gripper_closure: float,
) -> None:
    q_deg = arm.get_actual_joint_deg()
    tcp_pose = arm.get_actual_tcp_pose()
    if q_deg is None:
        raise RuntimeError("Could not read FR5 joint angles while recording.")
    if tcp_pose is None:
        raise RuntimeError("Could not read FR5 TCP pose while recording.")
    rel_path = f"rgb/{buffer.frames:06d}.png"
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite((buffer.episode_dir / rel_path).as_posix(), bgr):
        raise RuntimeError(f"Failed to write image: {buffer.episode_dir / rel_path}")
    buffer.timestamps.append(time.monotonic() - buffer.start_time)
    buffer.joint_deg.append(np.asarray(q_deg, dtype=np.float32))
    buffer.tcp_pos.append(tcp_xyz_m(tcp_pose).astype(np.float32))
    buffer.gripper_closure.append(float(gripper_closure))
    buffer.image_files.append(rel_path)


def annotate_status(cv2, image_bgr: np.ndarray, status: str) -> np.ndarray:
    cv2.rectangle(image_bgr, (0, 0), (image_bgr.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image_bgr, status, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return image_bgr


def read_overlay_frame(
    *,
    cv2,
    rgb_source: RealRgbSource,
    last_rgb: np.ndarray,
    projector: CameraProjector,
    config: dict,
    colors: list[str],
    active_color: str,
    tape_plane_z: float,
    args,
    status: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    myd_part_offset = np.asarray(args.myd_part_offset, dtype=np.float64).reshape(3)
    if hasattr(rgb_source, "read_latest"):
        rgb = rgb_source.read_latest(timeout_ms=int(args.real_rgb_timeout_ms), max_drain=int(args.real_rgb_drain_frames))
    else:
        rgb = rgb_source.read(timeout_ms=int(args.real_rgb_timeout_ms))
    if rgb is None:
        rgb = last_rgb
    detections = detect_tapes_rgb(
        rgb,
        cv2=cv2,
        colors=colors,
        min_area_px=float(args.min_area_px),
        max_area_px=float(args.max_area_px),
        config=config,
        projector=projector,
        z_world=tape_plane_z,
        use_slot_roi=bool(args.slot_roi),
        slot_padding_m=float(args.slot_padding_m),
        slot_color_prior=bool(args.slot_color_prior),
        slot_fallback_center=bool(args.slot_fallback_center),
        background_rgb=getattr(args, "_background_rgb", None),
        background_diff=bool(args.background_diff or getattr(args, "_background_rgb", None) is not None),
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
        active_color=active_color,
        draw_slots=bool(args.slot_roi),
        slot_padding_m=float(args.slot_padding_m),
        myd_part_offset_m=myd_part_offset,
    )
    annotate_status(cv2, overlay, status)
    return rgb, overlay, detections


def wait_key_or_none(cv2, delay_ms: int = 1) -> int | None:
    key = cv2.waitKey(int(delay_ms)) & 0xFF
    if key != 255:
        return key
    if KEY_POLLER is not None:
        char = KEY_POLLER.poll()
        if char:
            return ord(char)
    return None


def check_key_for_stop(cv2) -> None:
    key = wait_key_or_none(cv2, 1)
    if key in (ord("q"), 27):
        raise SessionQuit()
    if key == ord("s"):
        raise EpisodeStop()


def move_to_waypoint(
    *,
    cv2,
    arm: FairinoArmClient,
    rgb_source: RealRgbSource,
    last_rgb: np.ndarray,
    projector: CameraProjector,
    config: dict,
    colors: list[str],
    active_color: str,
    tape_plane_z: float,
    args,
    gripper,
    buffer: EpisodeBuffer,
    target_features: list[np.ndarray],
    rewards: list[float],
    target_feature: np.ndarray,
    reference_pose: list[float],
    waypoint: Waypoint,
    current_gripper: float,
    record_period_s: float,
    last_record_time: float,
    idx_start: int,
) -> tuple[np.ndarray, list[float], float, float, int]:
    start_pose = arm.get_actual_tcp_pose()
    if start_pose is None:
        start_pose = reference_pose
    start_xyz = tcp_xyz_m(start_pose)
    distance_m = float(np.linalg.norm(waypoint.xyz_m - start_xyz))
    steps = max(1, int(np.ceil(distance_m / max(1e-4, float(args.cart_step_m)))))
    idx = int(idx_start)
    current_rgb = last_rgb
    for step in range(1, steps + 1):
        alpha = step / float(steps)
        xyz = (1.0 - alpha) * start_xyz + alpha * waypoint.xyz_m
        target_pose = pose_with_xyz_m(reference_pose, xyz)
        arm.servo_cart(target_pose, cmd_t=float(args.cmd_t), idx=idx)
        idx += 1
        current_rgb, overlay, _ = read_overlay_frame(
            cv2=cv2,
            rgb_source=rgb_source,
            last_rgb=current_rgb,
            projector=projector,
            config=config,
            colors=colors,
            active_color=active_color,
            tape_plane_z=tape_plane_z,
            args=args,
            status=f"episode active: {waypoint.name}  keys: s stop/save/return, q quit",
        )
        cv2.imshow("FR5 real keyboard grasp", overlay)
        now = time.monotonic()
        if now - last_record_time >= record_period_s:
            record_real_frame(buffer, cv2=cv2, rgb=current_rgb, arm=arm, gripper_closure=current_gripper)
            target_features.append(target_feature.copy())
            rewards.append(0.0)
            last_record_time = now
        check_key_for_stop(cv2)
        time.sleep(max(0.0, float(args.cmd_t)))
    return current_rgb, pose_with_xyz_m(reference_pose, waypoint.xyz_m), current_gripper, last_record_time, idx


def move_along_sim_ik_frames(
    *,
    cv2,
    arm: FairinoArmClient,
    rgb_source: RealRgbSource,
    last_rgb: np.ndarray,
    projector: CameraProjector,
    config: dict,
    colors: list[str],
    active_color: str,
    tape_plane_z: float,
    args,
    gripper,
    buffer: EpisodeBuffer,
    target_features: list[np.ndarray],
    rewards: list[float],
    target_feature: np.ndarray,
    frames: list[SimWaypoint],
    fk_guard: FkTableClearanceGuard,
    current_gripper: float,
    record_period_s: float,
    last_record_time: float,
) -> tuple[np.ndarray, float, float]:
    if len(frames) < 2:
        return last_rgb, current_gripper, last_record_time
    current_rgb = last_rgb
    real_dt = float(args.real_servo_dt)
    source_dt = 1.0 / max(1e-6, float(args.hz))
    q_src = np.stack([np.asarray(item.q, dtype=np.float64) for item in frames], axis=0)
    g_src = np.asarray([float(item.gripper) for item in frames], dtype=np.float64)
    names = [str(item.name) for item in frames]
    phase_gripper_targets: dict[str, float] = {}
    for item in frames:
        phase_gripper_targets[str(item.name)] = float(item.gripper)
    src_t = np.arange(q_src.shape[0], dtype=np.float64) * source_dt
    duration = float(src_t[-1])
    dst_t = np.arange(0.0, duration + real_dt * 0.5, real_dt, dtype=np.float64)
    q_dst = np.empty((dst_t.shape[0], q_src.shape[1]), dtype=np.float64)
    for joint_idx in range(q_src.shape[1]):
        q_dst[:, joint_idx] = np.interp(dst_t, src_t, q_src[:, joint_idx])
    g_dst = np.interp(dst_t, src_t, g_src)
    step_deg = np.max(np.abs(np.diff(np.rad2deg(q_dst), axis=0)), axis=1) if q_dst.shape[0] > 1 else np.asarray([0.0])
    max_step = float(np.max(step_deg)) if step_deg.size else 0.0
    if max_step > float(args.max_real_step_deg):
        raise RuntimeError(
            f"Refusing real ServoJ: max resampled joint step {max_step:.3f}deg exceeds "
            f"--max-real-step-deg {float(args.max_real_step_deg):.3f}. Lower --real-servo-dt or check IK."
        )
    actual_start = arm.get_actual_joint_deg()
    if actual_start is not None:
        start_err = float(np.max(np.abs(np.asarray(actual_start, dtype=np.float64) - np.rad2deg(q_dst[0]))))
        if start_err > float(args.start_tolerance_deg):
            raise RuntimeError(
                f"Refusing real ServoJ: robot start differs from sim trajectory start by {start_err:.3f}deg "
                f"> --start-tolerance-deg {float(args.start_tolerance_deg):.3f}."
            )
    min_clearance = float("inf")
    min_clearance_phase = ""
    for t_rel, q in zip(dst_t, q_dst):
        phase_index = min(int(float(t_rel) / source_dt), len(names) - 1)
        phase = names[phase_index]
        clearance, _, _ = fk_guard.clearance_m(q)
        if clearance < min_clearance:
            min_clearance = clearance
            min_clearance_phase = phase
        fk_guard.check(q, phase)
    print(
        f"  ServoJ smooth replay: source_frames={len(frames)} source_hz={float(args.hz):.1f} "
        f"real_dt={real_dt:.4f}s real_frames={len(dst_t)} max_step={max_step:.3f}deg "
        f"min_fk_clearance={min_clearance:.4f}m@{min_clearance_phase}",
        flush=True,
    )

    last_display_time = 0.0
    last_phase = ""
    last_gripper_command_time = -1e9
    start_time = time.monotonic()
    for idx, (t_rel, q, g) in enumerate(zip(dst_t, q_dst, g_dst)):
        phase_index = min(int(float(t_rel) / source_dt), len(names) - 1)
        phase = names[phase_index]
        if phase != last_phase:
            last_phase = phase
            if phase.endswith("_close") or phase.endswith("_release") or phase == "home":
                current_gripper = float(phase_gripper_targets.get(phase, g))
                command_gripper(gripper, current_gripper, args)
                last_gripper_command_time = time.monotonic()
        elif (
            abs(float(g) - current_gripper) >= float(args.gripper_command_delta)
            and time.monotonic() - last_gripper_command_time >= float(args.gripper_command_min_interval)
        ):
            current_gripper = float(g)
            command_gripper(gripper, current_gripper, args)
            last_gripper_command_time = time.monotonic()

        try:
            fk_guard.check(q, phase)
            arm.servo_j(np.rad2deg(q), idx=idx, cmd_t=real_dt, vel=float(args.servoj_vel))
        except Exception:
            arm.servo_end_best_effort()
            arm.stop_motion_best_effort()
            raise
        now = time.monotonic()
        if now - last_display_time >= record_period_s:
            current_rgb, overlay, _ = read_overlay_frame(
                cv2=cv2,
                rgb_source=rgb_source,
                last_rgb=current_rgb,
                projector=projector,
                config=config,
                colors=colors,
                active_color=active_color,
                tape_plane_z=tape_plane_z,
                args=args,
                status=f"sim-IK ServoJ smooth: {phase}  keys: s stop/save/return, q quit",
            )
            cv2.imshow("FR5 real keyboard grasp", overlay)
            last_display_time = now
            if now - last_record_time >= record_period_s:
                record_real_frame(buffer, cv2=cv2, rgb=current_rgb, arm=arm, gripper_closure=float(g))
                target_features.append(target_feature.copy())
                rewards.append(0.0)
                last_record_time = now
        check_key_for_stop(cv2)
        target_time = start_time + float(t_rel) + real_dt
        sleep_s = target_time - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
    return current_rgb, current_gripper, last_record_time


def save_episode_or_discard(
    buffer: EpisodeBuffer,
    *,
    args,
    target_features: list[np.ndarray],
    rewards: list[float],
    meta: dict,
) -> bool:
    if buffer.frames < int(args.min_episode_frames):
        print(f"Discarding short episode: {buffer.episode_dir}, frames={buffer.frames}", flush=True)
        if buffer.episode_dir.exists():
            shutil.rmtree(buffer.episode_dir)
        return False
    write_episode(
        buffer.episode_dir,
        timestamps=buffer.timestamps,
        joint_deg=buffer.joint_deg,
        tcp_pos=buffer.tcp_pos,
        gripper_closure=buffer.gripper_closure,
        image_files=buffer.image_files,
        target_features=target_features,
        rewards=rewards,
        meta=meta,
    )
    print(f"Saved real SDK episode: {buffer.episode_dir} frames={buffer.frames}", flush=True)
    return True


def return_to_initial_after_episode(arm: FairinoArmClient, config: dict, args) -> None:
    if not bool(args.return_initial_on_stop):
        return
    print("Returning to initial_qpos after episode stop/save...", flush=True)
    move_to_initial_from_config(
        arm,
        config,
        vel=float(args.return_vel),
        acc=float(args.return_acc),
        ovl=float(args.return_ovl),
        tolerance_deg=float(args.return_tolerance_deg),
        timeout_s=float(args.return_timeout),
    )


def return_to_initial_before_episode(arm: FairinoArmClient, config: dict, args) -> None:
    if not bool(args.return_initial_before_episode):
        return
    print("Returning to initial_qpos before episode to lock a known vertical TCP posture...", flush=True)
    move_to_initial_from_config(
        arm,
        config,
        vel=float(args.return_vel),
        acc=float(args.return_acc),
        ovl=float(args.return_ovl),
        tolerance_deg=float(args.return_tolerance_deg),
        timeout_s=float(args.return_timeout),
    )


def with_safe_entry_waypoints(waypoints: list[Waypoint], current_xyz: np.ndarray, args) -> list[Waypoint]:
    if not bool(args.safe_segment_motion) or not waypoints:
        return waypoints
    current = np.asarray(current_xyz, dtype=np.float64).reshape(3)
    first = waypoints[0]
    safe_z = max(float(current[2]), float(first.xyz_m[2]), float(args.min_travel_z))
    if args.safe_travel_z is not None:
        safe_z = max(safe_z, float(args.safe_travel_z))
    safe_lift = Waypoint("safe_lift_before_xy", np.asarray([current[0], current[1], safe_z], dtype=np.float64), first.gripper)
    safe_xy = Waypoint("safe_xy_above_grasp", np.asarray([first.xyz_m[0], first.xyz_m[1], safe_z], dtype=np.float64), first.gripper)
    rest = list(waypoints)
    if abs(float(first.xyz_m[2]) - safe_z) < 1e-6:
        rest = rest[1:]
    else:
        rest[0] = Waypoint(first.name, np.asarray([first.xyz_m[0], first.xyz_m[1], first.xyz_m[2]], dtype=np.float64), first.gripper, first.hold_s)
    return [safe_lift, safe_xy] + rest


def run_episode(
    *,
    cv2,
    episode_index: int,
    arm: FairinoArmClient,
    gripper,
    rgb_source: RealRgbSource,
    last_rgb: np.ndarray,
    projector: CameraProjector,
    config: dict,
    camera_config_path: Path,
    colors: list[str],
    active_color: str,
    tape_plane_z: float,
    args,
) -> np.ndarray:
    rgb, overlay, detections = read_overlay_frame(
        cv2=cv2,
        rgb_source=rgb_source,
        last_rgb=last_rgb,
        projector=projector,
        config=config,
        colors=colors,
        active_color=active_color,
        tape_plane_z=tape_plane_z,
        args=args,
        status="starting episode: detecting target",
    )
    cv2.imshow("FR5 real keyboard grasp", overlay)
    if active_color not in detections:
        print(f"Cannot start episode: no {active_color!r} tape detected.", flush=True)
        return rgb

    tape_world = detections[active_color].world_m
    if tape_world is None:
        print(f"Cannot start episode: {active_color!r} pixel could not be projected to table plane.", flush=True)
        return rgb
    goal_world = task_goal_pos_from_config(config, manual_offset_m=np.asarray(args.myd_part_offset, dtype=np.float64))
    sim_frames: list[SimWaypoint] = []
    sim_waypoints: list[SimWaypoint] = []
    waypoints: list[Waypoint] = []
    if str(args.motion_mode) == "sim-ik-servoj":
        sim_frames, sim_waypoints, grasp_world, place_center = planned_sim_ik_waypoints(config, tape_world, goal_world, args)
    else:
        waypoints, grasp_world, place_center = planned_waypoints(config, tape_world, goal_world, args)
    print(f"\nEpisode {episode_index} plan:", flush=True)
    print(f"  motion_mode: {args.motion_mode}", flush=True)
    print(f"  active_color: {active_color}", flush=True)
    print(f"  tape_center_world_m: {np.asarray(tape_world).round(4).tolist()}", flush=True)
    print(f"  grasp_world_m:       {grasp_world.round(4).tolist()}", flush=True)
    print(f"  myd_goal_world_m:    {goal_world.round(4).tolist()}", flush=True)
    print(f"  place_center_world_m:{place_center.round(4).tolist()}", flush=True)
    if str(args.motion_mode) == "sim-ik-servoj":
        print(f"  sim sparse waypoints={len(sim_waypoints)} interpolated frames={len(sim_frames)}", flush=True)
        for wp in sim_waypoints:
            print(f"  - {wp.name}: q_deg={np.rad2deg(wp.q).round(2).tolist()} gripper={wp.gripper:.2f} frames={wp.frames}", flush=True)
    else:
        for wp in waypoints:
            print(f"  - {wp.name}: xyz={wp.xyz_m.round(4).tolist()} gripper={wp.gripper:.2f}", flush=True)
    if not args.execute_real:
        if str(args.motion_mode) == "sim-ik-servoj":
            print_fk_clearance_preflight(sim_frames, config, args)
        print("Dry-run only. Add --execute-real to send SDK motion.", flush=True)
        return rgb

    return_to_initial_before_episode(arm, config, args)
    reference_pose = None
    idx = 0
    if str(args.motion_mode) == "legacy-cart":
        pose = arm.get_actual_tcp_pose()
        if pose is None:
            raise RuntimeError("Could not read initial TCP pose before ServoCart.")
        reference_pose = [float(v) for v in pose[:6]]
        print(f"  reference_tcp_pose: {np.asarray(reference_pose).round(3).tolist()} ({'mm' if pose_uses_mm(reference_pose) else 'm'} + deg)", flush=True)
        waypoints = with_safe_entry_waypoints(waypoints, tcp_xyz_m(reference_pose), args)
        print("  final real ServoCart path:", flush=True)
        for wp in waypoints:
            print(f"    - {wp.name}: xyz={wp.xyz_m.round(4).tolist()} gripper={wp.gripper:.2f}", flush=True)
    else:
        print("  final real ServoJ path uses the same interpolated IK frames as fr5_sim_tape_pick_place.py.", flush=True)

    episode_dir = make_episode_dir(args.output_dir, args.episode_name or None, segment_index=episode_index)
    buffer = EpisodeBuffer(episode_dir=episode_dir, start_time=time.monotonic())
    target_feature = make_target_feature(np.asarray(tape_world, dtype=np.float32), np.asarray(goal_world, dtype=np.float32), camera_config=camera_config_path)
    target_features: list[np.ndarray] = []
    rewards: list[float] = []
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "fr5_real_keyboard_grasp",
        "schema": "fr5_il_episode_v2",
        "robot_ip": str(args.robot_ip),
        "config": resolve_demo_path(args.config).as_posix(),
        "camera_config": camera_config_path.as_posix(),
        "active_color": active_color,
        "object_start_pos_m": np.asarray(tape_world, dtype=float).round(6).tolist(),
        "grasp_pos_m": grasp_world.round(6).tolist(),
        "goal_pos_m": goal_world.round(6).tolist(),
        "place_center_pos_m": place_center.round(6).tolist(),
        "control_hz": float(args.hz),
        "motion_mode": str(args.motion_mode),
        "cmd_t": float(args.cmd_t),
        "fk_table_safety": {
            "enabled": bool(args.fk_table_safety),
            "table_z": float(args.fk_table_z),
            "min_gripper_table_clearance": float(args.min_gripper_table_clearance),
            "fk_gripper_radius": float(args.fk_gripper_radius),
            "links": [str(name) for name in args.fk_safety_links],
        },
        "target_feature_mode": "single_rgb_table_plane_projection",
        "keyboard_start_stop": {"start": "a", "stop_episode": "s", "quit": "q/esc"},
        "execute_gripper": bool(args.execute_gripper),
    }

    current_gripper = float(config.get("sim_tape_pick_place", {}).get("default_gripper_opening", 0.0))
    command_gripper(gripper, current_gripper, args)
    if float(args.pre_open_hold_s) > 0.0:
        print(f"Opening gripper before approach; hold {float(args.pre_open_hold_s):.2f}s.", flush=True)
        time.sleep(float(args.pre_open_hold_s) * 0.5)
        command_gripper(gripper, current_gripper, args)
        time.sleep(float(args.pre_open_hold_s) * 0.5)
    arm.servo_start()
    record_period_s = 1.0 / max(1e-6, float(args.hz))
    last_record_time = 0.0
    try:
        if str(args.motion_mode) == "sim-ik-servoj":
            fk_guard = FkTableClearanceGuard(config, args)
            rgb, current_gripper, last_record_time = move_along_sim_ik_frames(
                cv2=cv2,
                arm=arm,
                rgb_source=rgb_source,
                last_rgb=rgb,
                projector=projector,
                config=config,
                colors=colors,
                active_color=active_color,
                tape_plane_z=tape_plane_z,
                args=args,
                gripper=gripper,
                buffer=buffer,
                target_features=target_features,
                rewards=rewards,
                target_feature=target_feature,
                frames=sim_frames,
                fk_guard=fk_guard,
                current_gripper=current_gripper,
                record_period_s=record_period_s,
                last_record_time=last_record_time,
            )
        else:
            for wp in waypoints:
                if abs(float(wp.gripper) - current_gripper) > 1e-5:
                    current_gripper = float(wp.gripper)
                    command_gripper(gripper, current_gripper, args)
                rgb, reference_pose, current_gripper, last_record_time, idx = move_to_waypoint(
                    cv2=cv2,
                    arm=arm,
                    rgb_source=rgb_source,
                    last_rgb=rgb,
                    projector=projector,
                    config=config,
                    colors=colors,
                    active_color=active_color,
                    tape_plane_z=tape_plane_z,
                    args=args,
                    gripper=gripper,
                    buffer=buffer,
                    target_features=target_features,
                    rewards=rewards,
                    target_feature=target_feature,
                    reference_pose=reference_pose,
                    waypoint=wp,
                    current_gripper=current_gripper,
                    record_period_s=record_period_s,
                    last_record_time=last_record_time,
                    idx_start=idx,
                )
                if wp.hold_s > 0.0:
                    end = time.monotonic() + float(wp.hold_s)
                    while time.monotonic() < end:
                        rgb, overlay, _ = read_overlay_frame(
                            cv2=cv2,
                            rgb_source=rgb_source,
                            last_rgb=rgb,
                            projector=projector,
                            config=config,
                            colors=colors,
                            active_color=active_color,
                            tape_plane_z=tape_plane_z,
                            args=args,
                            status=f"episode active: {wp.name} hold  keys: s stop/save/return, q quit",
                        )
                        cv2.imshow("FR5 real keyboard grasp", overlay)
                        now = time.monotonic()
                        if now - last_record_time >= record_period_s:
                            record_real_frame(buffer, cv2=cv2, rgb=rgb, arm=arm, gripper_closure=current_gripper)
                            target_features.append(target_feature.copy())
                            rewards.append(0.0)
                            last_record_time = now
                        check_key_for_stop(cv2)
                        time.sleep(min(0.03, max(0.0, end - time.monotonic())))
        print("Episode motion complete. Press s to save/stop and return to initial_qpos.", flush=True)
        while True:
            rgb, overlay, _ = read_overlay_frame(
                cv2=cv2,
                rgb_source=rgb_source,
                last_rgb=rgb,
                projector=projector,
                config=config,
                colors=colors,
                active_color=active_color,
                tape_plane_z=tape_plane_z,
                args=args,
                status="motion complete: press s to save/return, q quit",
            )
            cv2.imshow("FR5 real keyboard grasp", overlay)
            now = time.monotonic()
            if now - last_record_time >= record_period_s:
                record_real_frame(buffer, cv2=cv2, rgb=rgb, arm=arm, gripper_closure=current_gripper)
                target_features.append(target_feature.copy())
                rewards.append(1.0)
                last_record_time = now
            key = wait_key_or_none(cv2, 1)
            if key == ord("s"):
                break
            if key in (ord("q"), 27):
                raise SessionQuit()
            time.sleep(0.01)
    except EpisodeStop:
        print("Episode stop requested with s. Sending best-effort stop and saving valid frames.", flush=True)
        arm.stop_motion_best_effort()
    finally:
        arm.servo_end_best_effort()

    save_episode_or_discard(buffer, args=args, target_features=target_features, rewards=rewards, meta=meta)
    return_to_initial_after_episode(arm, config, args)
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real FR5 SDK scripted tape grasp with live Astra RGB target detection and keyboard episode boundaries."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_TASK_CONFIG.as_posix())
    parser.add_argument("--camera-config", type=str, default=DEFAULT_CAMERA_CONFIG.as_posix())
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP)
    parser.add_argument("--execute-real", action="store_true", help="Actually send FR5 motion commands. Without this, only detection/planning is shown.")
    parser.add_argument("--prepare-controller", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-errors", action="store_true")
    parser.add_argument("--speed-percent", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--cmd-t", type=float, default=None, help="Servo command period. Default is 1/hz so real playback timing matches simulation.")
    parser.add_argument("--real-servo-dt", type=float, default=0.008, help="High-frequency ServoJ period for sim-IK real replay. Keep <=0.016 for Fairino servo mode.")
    parser.add_argument("--servoj-vel", type=float, default=5.0, help="Fairino ServoJ velocity argument for sim-IK real replay.")
    parser.add_argument("--max-real-step-deg", type=float, default=0.35, help="Abort if any resampled real ServoJ command changes a joint more than this many degrees.")
    parser.add_argument("--start-tolerance-deg", type=float, default=2.0, help="Abort sim-IK replay if current real joints are farther from trajectory start.")
    parser.add_argument("--fk-table-safety", action=argparse.BooleanOptionalAction, default=True, help="Use MJCF forward kinematics from commanded joints to stop before gripper approaches the table.")
    parser.add_argument("--fk-table-z", type=float, default=0.0, help="World Z of the physical table top used by FK safety.")
    parser.add_argument("--min-gripper-table-clearance", type=float, default=0.020, help="Minimum estimated clearance between gripper lowest point and table.")
    parser.add_argument("--fk-gripper-radius", type=float, default=0.0, help="Extra conservative radius subtracted from the FK point Z.")
    parser.add_argument(
        "--fk-safety-links",
        nargs="+",
        default=["left_pad", "right_pad", "left_follower", "right_follower", "robotiq_base_link"],
        help="MJCF links checked by FK table safety in addition to tcp site.",
    )
    parser.add_argument(
        "--motion-mode",
        choices=["sim-ik-servoj", "legacy-cart"],
        default="sim-ik-servoj",
        help="Default reuses fr5_sim_tape_pick_place.py IK waypoints and sends them with ServoJ. legacy-cart keeps the old Cartesian ServoCart path.",
    )
    parser.add_argument("--cart-step-m", type=float, default=0.004, help="Linear Cartesian ServoCart interpolation step in meters.")
    parser.add_argument("--active-color", choices=["red", "yellow", "white"], default="red")
    parser.add_argument("--colors", nargs="+", choices=["red", "yellow", "white"], default=["red", "yellow", "white"])
    parser.add_argument("--tape-plane-z", type=float, default=None)
    parser.add_argument("--min-area-px", type=float, default=350.0)
    parser.add_argument("--max-area-px", type=float, default=50000.0)
    parser.add_argument("--slot-roi", action=argparse.BooleanOptionalAction, default=True, help="Restrict tape detection to projected black-frame slots from config.")
    parser.add_argument("--slot-color-prior", action=argparse.BooleanOptionalAction, default=True, help="Use the three-slot prior: red/yellow occupy two slots, white is constrained to the remaining slot.")
    parser.add_argument("--slot-fallback-center", action=argparse.BooleanOptionalAction, default=True, help="If white is not segmented reliably, fall back to the remaining slot center instead of a reflection.")
    parser.add_argument("--slot-padding-m", type=float, default=0.025)
    parser.add_argument("--background-diff", action=argparse.BooleanOptionalAction, default=False, help="Use background subtraction before color/slot filtering. Press b in idle to capture empty-table background.")
    parser.add_argument("--background-image", type=str, default="", help="Optional RGB background image captured without tape.")
    parser.add_argument("--background-diff-thresh", type=float, default=32.0)
    parser.add_argument("--white-min-area-px", type=float, default=80.0)
    parser.add_argument("--white-max-area-px", type=float, default=18000.0)
    parser.add_argument("--white-min-circularity", type=float, default=0.12)
    parser.add_argument("--white-min-extent", type=float, default=0.12)
    parser.add_argument("--myd-part-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"), help="Temporary myd_part target offset in world meters.")
    parser.add_argument("--approach-z", type=float, default=None)
    parser.add_argument("--approach-clearance-m", type=float, default=0.05, help="Keep the approach waypoint this much above the grasp waypoint.")
    parser.add_argument("--grasp-z", type=float, default=None)
    parser.add_argument("--grasp-height", type=float, default=None, help="Sim-equivalent relative grasp height. Defaults to sim_tape_pick_place.grasp_height_m.")
    parser.add_argument("--grasp-z-clearance-m", type=float, default=0.035, help="Default real grasp z is detected tape z plus this clearance.")
    parser.add_argument("--min-grasp-z", type=float, default=0.045, help="Minimum real grasp z when --grasp-z is not explicitly set.")
    parser.add_argument("--safe-segment-motion", action=argparse.BooleanOptionalAction, default=True, help="Use lift-then-horizontal-then-vertical entry instead of a diagonal approach.")
    parser.add_argument("--min-travel-z", type=float, default=0.18, help="Minimum Z for horizontal travel before descending over the tape.")
    parser.add_argument("--safe-travel-z", type=float, default=None, help="Optional fixed Z for horizontal travel. Overrides only upward.")
    parser.add_argument("--lift-z", type=float, default=None)
    parser.add_argument("--lift-delta", type=float, default=None, help="Sim-equivalent lift delta above grasp height. Defaults to sim_tape_pick_place.lift_delta_m.")
    parser.add_argument("--place-approach-z", type=float, default=None)
    parser.add_argument("--place-z", type=float, default=None)
    parser.add_argument("--release-height", type=float, default=None, help="Sim-equivalent relative release height. Defaults to sim release_drop_delta_m/release_height_m logic.")
    parser.add_argument("--drop-delta-x", type=float, default=None, help="Sim-equivalent target X offset. Defaults to sim_tape_pick_place.drop_delta_x_m.")
    parser.add_argument("--drop-xy-offset", nargs=2, type=float, default=[0.0, 0.0], metavar=("DX", "DY"), help="Extra target XY offset in meters for real sim-IK playback.")
    parser.add_argument("--gripper-hold-s", type=float, default=0.7)
    parser.add_argument("--pre-open-hold-s", type=float, default=1.0, help="Wait after commanding gripper open before moving toward tape.")
    parser.add_argument("--release-hold-s", type=float, default=0.4)
    parser.add_argument("--gripper-command-delta", type=float, default=0.25, help="Minimum closure change before sending another gripper command during sim-IK replay.")
    parser.add_argument("--gripper-command-min-interval", type=float, default=0.25, help="Minimum seconds between gripper Modbus commands during sim-IK replay.")
    parser.add_argument("--workspace-x-min", type=float, default=-0.95)
    parser.add_argument("--workspace-x-max", type=float, default=-0.20)
    parser.add_argument("--workspace-y-min", type=float, default=-0.50)
    parser.add_argument("--workspace-y-max", type=float, default=0.30)
    parser.add_argument("--workspace-z-min", type=float, default=0.005)
    parser.add_argument("--workspace-z-max", type=float, default=0.35)
    parser.add_argument("--return-initial-on-stop", action=argparse.BooleanOptionalAction, default=True, help="After s stops/saves an episode, MoveJ back to config initial_qpos.")
    parser.add_argument("--return-initial-before-episode", action=argparse.BooleanOptionalAction, default=True, help="Before each real episode, MoveJ to config initial_qpos so ServoCart uses the known vertical TCP posture.")
    parser.add_argument("--return-vel", type=float, default=20.0)
    parser.add_argument("--return-acc", type=float, default=20.0)
    parser.add_argument("--return-ovl", type=float, default=20.0)
    parser.add_argument("--return-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--return-timeout", type=float, default=120.0)
    parser.add_argument("--real-rgb-source", choices=["live", "latest"], default="live")
    parser.add_argument("--real-rgb-image", type=str, default="")
    parser.add_argument("--real-rgb-width", type=int, default=640)
    parser.add_argument("--real-rgb-height", type=int, default=480)
    parser.add_argument("--real-rgb-fps", type=int, default=10)
    parser.add_argument("--real-rgb-timeout-ms", type=int, default=1, help="Non-blocking live RGB wait used during real servo replay.")
    parser.add_argument("--real-rgb-drain-frames", type=int, default=12, help="Drain this many queued Astra frames and keep only the newest frame.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_IL_DEMO_DIR.as_posix())
    parser.add_argument("--episode-name", type=str, default="")
    parser.add_argument("--min-episode-frames", type=int, default=3)
    parser.add_argument("--execute-gripper", action="store_true")
    parser.add_argument("--allow-virtual-gripper", action="store_true", help="Allow real arm motion even if --execute-gripper fails to connect. Not recommended for grasping.")
    parser.add_argument("--strict-gripper", action="store_true")
    parser.add_argument("--gripper-backend", choices=["raw", "pyrobotiq"], default="raw")
    parser.add_argument("--gripper-debug", action="store_true")
    parser.add_argument("--gripper-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--gripper-baudrate", type=int, default=115200)
    parser.add_argument("--gripper-slave-id", type=int, default=9)
    parser.add_argument("--gripper-timeout", type=float, default=0.5)
    parser.add_argument("--gripper-retries", type=int, default=2)
    parser.add_argument("--gripper-speed", type=int, default=255)
    parser.add_argument("--gripper-force", type=int, default=150)
    args = parser.parse_args()

    if args.hz <= 0.0:
        raise RuntimeError("--hz must be positive")
    if args.cmd_t is None:
        args.cmd_t = 1.0 / float(args.hz)
    if args.cmd_t <= 0.0:
        raise RuntimeError("--cmd-t must be positive")
    if args.real_servo_dt <= 0.0 or args.real_servo_dt > 0.016:
        raise RuntimeError("--real-servo-dt must be in (0, 0.016] for Fairino ServoJ safety")
    if args.min_gripper_table_clearance < 0.0:
        raise RuntimeError("--min-gripper-table-clearance must be non-negative")
    if args.fk_gripper_radius < 0.0:
        raise RuntimeError("--fk-gripper-radius must be non-negative")
    if args.real_rgb_timeout_ms < 0:
        raise RuntimeError("--real-rgb-timeout-ms must be non-negative")
    if args.real_rgb_drain_frames < 0:
        raise RuntimeError("--real-rgb-drain-frames must be non-negative")

    cv2 = require_cv2()
    config_path = resolve_demo_path(args.config)
    camera_path = resolve_demo_path(args.camera_config)
    config = load_json(config_path)
    projector = CameraProjector.from_config(camera_path)
    tape_plane_z = default_tape_plane_z(config) if args.tape_plane_z is None else float(args.tape_plane_z)
    active_color = str(args.active_color)
    if args.background_image:
        bg_bgr = cv2.imread(resolve_demo_path(args.background_image).as_posix(), cv2.IMREAD_COLOR)
        if bg_bgr is None:
            raise RuntimeError(f"Could not read background image: {args.background_image}")
        setattr(args, "_background_rgb", cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB))
    else:
        setattr(args, "_background_rgb", None)

    print("\nFR5 real SDK grasp collection keys:", flush=True)
    print("  a      start one episode from current detected tape position", flush=True)
    print("  b      capture empty-table background for reflection suppression", flush=True)
    print("  s      stop/save current episode, return to initial_qpos, then wait for next a", flush=True)
    print("  q/esc  quit safely", flush=True)
    print(f"  execute_real={bool(args.execute_real)} robot_ip={args.robot_ip}", flush=True)
    print(f"  camera_config={camera_path}", flush=True)
    print(f"  tape_plane_z={tape_plane_z:.4f}m active_color={active_color}", flush=True)
    print(f"  config tape offset={real_tape_offset_from_config(config).round(4).tolist()}m", flush=True)
    print(f"  config myd offset={real_myd_part_offset_from_config(config).round(4).tolist()}m temporary offset={np.asarray(args.myd_part_offset, dtype=float).round(4).tolist()}m", flush=True)

    arm = None
    gripper = None
    if args.execute_real:
        arm = FairinoArmClient(args.robot_ip, speed_percent=float(args.speed_percent))
        arm.connect()
        if args.prepare_controller:
            arm.set_mode(0)
            arm.robot_enable(1)
        if args.reset_errors:
            arm.reset_all_error()
        err = arm.get_robot_error_code()
        if err is not None and any(int(v) != 0 for v in err):
            raise RuntimeError(f"Controller reports error code {err}; refusing motion. Resolve or retry with --reset-errors if resettable.")
        gripper = connect_gripper_best_effort(args)
        if args.execute_gripper and gripper is None and not bool(args.allow_virtual_gripper):
            raise RuntimeError(
                "Physical gripper is required for this real run, but it did not connect/activate. "
                "Fix USB/Modbus or add --allow-virtual-gripper to move the arm without physical gripper commands."
            )

    rgb_source = RealRgbSource(
        source=str(args.real_rgb_source),
        image_path=args.real_rgb_image or None,
        width=int(args.real_rgb_width),
        height=int(args.real_rgb_height),
        fps=int(args.real_rgb_fps),
        allow_fallback=True,
    )
    episode_index = 0
    global KEY_POLLER
    try:
        last_rgb = rgb_source.start()
        with TerminalKeyPoller() as poller:
            KEY_POLLER = poller
            while True:
                last_rgb, overlay, detections = read_overlay_frame(
                    cv2=cv2,
                    rgb_source=rgb_source,
                    last_rgb=last_rgb,
                    projector=projector,
                    config=config,
                    colors=list(args.colors),
                    active_color=active_color,
                    tape_plane_z=tape_plane_z,
                    args=args,
                    status="idle: press a to start episode, q quit",
                )
                cv2.imshow("FR5 real keyboard grasp", overlay)
                key = wait_key_or_none(cv2, 1)
                if key in (ord("q"), 27):
                    break
                if key == ord("b"):
                    setattr(args, "_background_rgb", np.ascontiguousarray(last_rgb.copy()))
                    out_dir = resolve_demo_path(args.output_dir) / "_backgrounds"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    bg_path = out_dir / f"background_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
                    if not cv2.imwrite(bg_path.as_posix(), cv2.cvtColor(getattr(args, "_background_rgb"), cv2.COLOR_RGB2BGR)):
                        raise RuntimeError(f"Failed to write background image: {bg_path}")
                    args.background_diff = True
                    print(f"Captured reflection background: {bg_path}", flush=True)
                    continue
                if key == ord("a"):
                    if arm is None and args.execute_real:
                        raise RuntimeError("Internal error: --execute-real was set but arm is not connected.")
                    if arm is None:
                        print("Planning only because --execute-real is not set.", flush=True)
                        if active_color not in detections:
                            print(f"No {active_color} tape detected.", flush=True)
                        elif detections[active_color].world_m is None:
                            print(f"{active_color} tape was detected in image, but projection to table plane failed.", flush=True)
                        else:
                            goal = task_goal_pos_from_config(config, manual_offset_m=np.asarray(args.myd_part_offset, dtype=np.float64))
                            waypoints, grasp, place = planned_waypoints(config, detections[active_color].world_m, goal, args)
                            print(f"Dry-run plan for {active_color}: tape={detections[active_color].world_m.round(4).tolist()} grasp={grasp.round(4).tolist()} goal={goal.round(4).tolist()}", flush=True)
                            for wp in waypoints:
                                print(f"  - {wp.name}: xyz={wp.xyz_m.round(4).tolist()} gripper={wp.gripper:.2f}", flush=True)
                    else:
                        try:
                            last_rgb = run_episode(
                                cv2=cv2,
                                episode_index=episode_index,
                                arm=arm,
                                gripper=gripper,
                                rgb_source=rgb_source,
                                last_rgb=last_rgb,
                                projector=projector,
                                config=config,
                                camera_config_path=camera_path,
                                colors=list(args.colors),
                                active_color=active_color,
                                tape_plane_z=tape_plane_z,
                                args=args,
                            )
                            episode_index += 1
                        except SessionQuit:
                            break
                        except Exception:
                            if arm is not None:
                                arm.stop_motion_best_effort()
                                arm.servo_end_best_effort()
                            raise
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping safely.", flush=True)
        if arm is not None:
            arm.stop_motion_best_effort()
            arm.servo_end_best_effort()
    finally:
        KEY_POLLER = None
        rgb_source.stop()
        cv2.destroyAllWindows()
        if gripper is not None:
            gripper.close()
        if arm is not None:
            arm.close()
            print("RPC connection closed.", flush=True)


if __name__ == "__main__":
    main()
