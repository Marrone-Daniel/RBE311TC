from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from motrixsim import forward_kinematic, step as sim_step
from motrixsim.render import RenderApp

from arm_control import (
    DEFAULT_CONFIG,
    DEFAULT_FR5_GS_DIR,
    build_runtime,
    camera_name_from_config,
    camera_resolution_from_config,
    collect_fr5_gaussian_assets,
    ensure_fr5_gaussian_assets,
    find_camera,
    find_camera_id,
    load_config,
    render_fr5_gs_rgb,
    require_cv2,
    resolve_demo_path,
    set_arm_qpos_and_ctrl,
    set_gripper,
    site_position,
)
from fr5_il_dataset import DEFAULT_IL_DEMO_DIR, make_target_feature
from fr5_record_demonstration import write_episode


DEFAULT_CAMERA_CONFIG = Path(__file__).resolve().parent / "configs" / "astra_camera.json"
CHILD_ENV_FLAG = "FR5_SIM_TAPE_PICK_PLACE_CHILD"
COLOR_BACKGROUND = (0x9C, 0x9F, 0xA2)
COLOR_TAPE = (0x9C, 0x9F, 0xA2)
COLOR_TABLE = (0x98, 0x92, 0x89)
COLOR_GROOVE = (0x87, 0x84, 0x7A)
COLOR_FIXED_BASE = (0x7E, 0x77, 0x6F)
JOINT_LIMITS = np.asarray(
    [
        [-3.0543, 3.0543],
        [-4.6251, 1.4835],
        [-2.8274, 2.8274],
        [-4.6251, 1.4835],
        [-3.0543, 3.0543],
        [-3.0543, 3.0543],
    ],
    dtype=np.float32,
)


class GraspFailureRetry(RuntimeError):
    def __init__(self, *, episode_dir: Path, message: str):
        super().__init__(message)
        self.episode_dir = episode_dir


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 10000):
        candidate = path.with_name(f"{path.name}_{idx:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not make unique path for {path}")


def discard_episode_dir(episode_dir: Path, output_dir: str | Path, reason: str) -> Path:
    discard_root = resolve_demo_path(output_dir) / "_discarded_grasp_failures"
    discard_root.mkdir(parents=True, exist_ok=True)
    target = unique_path(discard_root / episode_dir.name)
    reason_path = episode_dir / "discard_reason.txt"
    reason_path.write_text(reason + "\n", encoding="utf-8")
    shutil.move(episode_dir.as_posix(), target.as_posix())
    return target


@dataclass(frozen=True)
class Waypoint:
    name: str
    q: np.ndarray
    gripper: float
    frames: int
    attach: bool
    release: bool = False
    object_name: str = ""


def clamp_q(q: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=np.float32), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def set_object_pose(data, model, qslice: tuple[int, int], pos: np.ndarray) -> None:
    q = np.asarray(data.dof_pos, dtype=np.float32).copy()
    start, end = qslice
    if end - start < 7:
        raise RuntimeError(f"Object qpos slice must contain a freejoint qpos with 7 values, got {qslice}")
    q[start : start + 3] = np.asarray(pos, dtype=np.float32)
    q[start + 3 : start + 7] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    data.set_dof_pos(q, model)
    data.set_dof_vel(np.zeros_like(data.dof_vel))
    forward_kinematic(model, data)


def object_pos_from_site(model, data, site_name: str) -> np.ndarray:
    return np.asarray(site_position(model, data, site_name), dtype=np.float32)


def solve_tcp_ik(
    model,
    data,
    body,
    qpos_ids: np.ndarray,
    arm_act_ids: np.ndarray,
    tcp_site: str,
    q_seed: np.ndarray,
    target_pos: np.ndarray,
    *,
    max_iters: int = 80,
    tol: float = 0.0025,
    step_rad: float = 0.006,
    damping: float = 0.025,
) -> np.ndarray:
    q = clamp_q(q_seed)
    target = np.asarray(target_pos, dtype=np.float32)
    for _ in range(int(max_iters)):
        set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q)
        cur = np.asarray(site_position(model, data, tcp_site), dtype=np.float32)
        err = target - cur
        if float(np.linalg.norm(err)) <= float(tol):
            return q
        jac = np.zeros((3, q.shape[0]), dtype=np.float32)
        for j in range(q.shape[0]):
            qp = q.copy()
            qp[j] += float(step_rad)
            qp = clamp_q(qp)
            set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, qp)
            pp = np.asarray(site_position(model, data, tcp_site), dtype=np.float32)
            qm = q.copy()
            qm[j] -= float(step_rad)
            qm = clamp_q(qm)
            set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, qm)
            pm = np.asarray(site_position(model, data, tcp_site), dtype=np.float32)
            denom = max(float(qp[j] - qm[j]), 1e-6)
            jac[:, j] = (pp - pm) / denom
        lhs = jac @ jac.T + float(damping) ** 2 * np.eye(3, dtype=np.float32)
        dq = jac.T @ np.linalg.solve(lhs, err)
        q = clamp_q(q + np.clip(dq, -0.12, 0.12).astype(np.float32))
    return q


def site_axis_world(model, data, site_name: str, axis_index: int = 2) -> np.ndarray:
    forward_kinematic(model, data)
    site = model.get_site(site_name)
    rot = np.asarray(site.get_rotation_mat(data), dtype=np.float32).reshape(3, 3)
    axis = rot[:, int(axis_index)]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    return (axis / norm).astype(np.float32)


def solve_tcp_ik_with_axis(
    model,
    data,
    body,
    qpos_ids: np.ndarray,
    arm_act_ids: np.ndarray,
    tcp_site: str,
    q_seed: np.ndarray,
    target_pos: np.ndarray,
    *,
    desired_axis: np.ndarray,
    axis_index: int = 2,
    max_iters: int = 120,
    pos_tol: float = 0.0025,
    axis_tol: float = 0.035,
    step_rad: float = 0.006,
    damping: float = 0.025,
    axis_weight: float = 0.08,
) -> np.ndarray:
    q = clamp_q(q_seed)
    target = np.asarray(target_pos, dtype=np.float32)
    desired = np.asarray(desired_axis, dtype=np.float32)
    desired = desired / max(float(np.linalg.norm(desired)), 1e-6)
    for _ in range(int(max_iters)):
        set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q)
        cur_pos = np.asarray(site_position(model, data, tcp_site), dtype=np.float32)
        cur_axis = site_axis_world(model, data, tcp_site, int(axis_index))
        pos_err = target - cur_pos
        axis_err = desired - cur_axis
        if float(np.linalg.norm(pos_err)) <= float(pos_tol) and float(np.linalg.norm(axis_err)) <= float(axis_tol):
            return q
        err = np.concatenate([pos_err, float(axis_weight) * axis_err]).astype(np.float32)
        jac = np.zeros((6, q.shape[0]), dtype=np.float32)
        for j in range(q.shape[0]):
            qp = q.copy()
            qp[j] += float(step_rad)
            qp = clamp_q(qp)
            set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, qp)
            pp = np.asarray(site_position(model, data, tcp_site), dtype=np.float32)
            ap = site_axis_world(model, data, tcp_site, int(axis_index))

            qm = q.copy()
            qm[j] -= float(step_rad)
            qm = clamp_q(qm)
            set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, qm)
            pm = np.asarray(site_position(model, data, tcp_site), dtype=np.float32)
            am = site_axis_world(model, data, tcp_site, int(axis_index))

            denom = max(float(qp[j] - qm[j]), 1e-6)
            jac[:3, j] = (pp - pm) / denom
            jac[3:, j] = float(axis_weight) * ((ap - am) / denom)
        lhs = jac @ jac.T + float(damping) ** 2 * np.eye(6, dtype=np.float32)
        dq = jac.T @ np.linalg.solve(lhs, err)
        q = clamp_q(q + np.clip(dq, -0.12, 0.12).astype(np.float32))
    return q


def interpolate_waypoints(waypoints: list[Waypoint]) -> list[Waypoint]:
    frames: list[Waypoint] = []
    if len(waypoints) < 2:
        return waypoints
    for prev, nxt in zip(waypoints[:-1], waypoints[1:]):
        n = max(1, int(nxt.frames))
        for i in range(n):
            alpha = (i + 1) / float(n)
            q = (1.0 - alpha) * prev.q + alpha * nxt.q
            g = (1.0 - alpha) * prev.gripper + alpha * nxt.gripper
            frames.append(
                Waypoint(
                    nxt.name,
                    q.astype(np.float32),
                    float(g),
                    1,
                    bool(nxt.attach),
                    bool(nxt.release),
                    str(nxt.object_name),
                )
            )
    return frames


def random_drop_pos(start_pos: np.ndarray, delta_x: float, radius: float, rng: np.random.Generator) -> np.ndarray:
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    r = float(radius) * math.sqrt(float(rng.uniform(0.0, 1.0)))
    pos = np.asarray(start_pos, dtype=np.float32).copy()
    pos[0] += float(delta_x) + r * math.cos(theta)
    pos[1] += r * math.sin(theta)
    return pos


def random_xy_pos_around(center_pos: np.ndarray, radius: float, rng: np.random.Generator) -> np.ndarray:
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    r = float(radius) * math.sqrt(float(rng.uniform(0.0, 1.0)))
    pos = np.asarray(center_pos, dtype=np.float32).copy()
    pos[0] += r * math.cos(theta)
    pos[1] += r * math.sin(theta)
    return pos


def random_xy_pos_in_slot(center_pos: np.ndarray, half_extents_xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.asarray(center_pos, dtype=np.float32).copy()
    half = np.asarray(half_extents_xy, dtype=np.float32)
    if half.shape[0] < 2:
        raise RuntimeError(f"slot_half_extents_m must contain x/y half extents, got {half_extents_xy}")
    pos[0] += float(rng.uniform(-float(half[0]), float(half[0])))
    pos[1] += float(rng.uniform(-float(half[1]), float(half[1])))
    return pos


def qpos_slice_from_joint(model, joint_name: str) -> tuple[int, int]:
    joint_id = model.get_joint_index(joint_name)
    start = int(model.joint_dof_pos_indices[joint_id])
    count = int(model.joint_dof_pos_nums[joint_id])
    return start, start + count


def tape_specs_from_task(task: dict) -> list[dict]:
    specs = task.get("tape_objects", [])
    if specs:
        return [dict(item) for item in specs]
    return [
        {
            "name": str(task.get("object_name", "red_tape_roll")),
            "site": str(task.get("object_site", "red_tape_roll_site")),
            "qpos_slice": task.get("object_qpos_slice", [0, 7]),
            "slot_center_m": task.get("start_random_center_m", task.get("start_pos_m", [-0.55, 0.0, 0.02])),
            "slot_radius_m": task.get("start_random_radius_m", 0.04),
        }
    ]


def maybe_randomize_tape_slots(specs: list[dict], task: dict, rng: np.random.Generator) -> list[dict]:
    """Randomly assign configured physical slots to tape colors while preserving tape order.

    The task order still follows the tape object list, e.g. red -> yellow -> white.
    Only the initial black-frame slot occupied by each color changes per episode.
    """
    if not bool(task.get("randomize_tape_slots", False)) or len(specs) <= 1:
        return [dict(spec) for spec in specs]
    slot_keys = ("slot_name", "slot_center_m", "slot_half_extents_m", "slot_radius_m")
    slots: list[dict] = []
    for spec in specs:
        slot = {key: spec[key] for key in slot_keys if key in spec}
        if "slot_center_m" not in slot:
            slot["slot_center_m"] = task.get("start_random_center_m", task.get("start_pos_m", [-0.55, 0.0, 0.02]))
        slots.append(slot)
    order = rng.permutation(len(slots))
    out: list[dict] = []
    for spec, slot_idx in zip(specs, order):
        item = dict(spec)
        for key in slot_keys:
            item.pop(key, None)
        item.update(slots[int(slot_idx)])
        item["assigned_slot_from"] = slots[int(slot_idx)].get("slot_name", f"slot_{int(slot_idx)}")
        out.append(item)
    return out


def tape_qpos_slice(model, spec: dict) -> tuple[int, int]:
    if "qpos_slice" in spec:
        raw = spec["qpos_slice"]
        return int(raw[0]), int(raw[1])
    if "joint" in spec:
        return qpos_slice_from_joint(model, str(spec["joint"]))
    return qpos_slice_from_joint(model, f"{spec['name']}_joint")


def random_tape_start_pos(spec: dict, task: dict, rng: np.random.Generator) -> np.ndarray:
    center = np.asarray(spec.get("slot_center_m", task.get("start_random_center_m", [-0.55, 0.0, 0.02])), dtype=np.float32)
    if not bool(task.get("randomize_start", True)):
        return center.copy()
    if "slot_half_extents_m" in spec:
        return random_xy_pos_in_slot(center, np.asarray(spec["slot_half_extents_m"], dtype=np.float32), rng)
    radius = float(spec.get("slot_radius_m", task.get("start_random_radius_m", 0.04)))
    return random_xy_pos_around(center, radius, rng)


def initialize_tape_objects(
    data, model, task: dict, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], tuple[int, int], dict[str, str]]:
    specs = maybe_randomize_tape_slots(tape_specs_from_task(task), task, rng)
    primary_name = str(task.get("object_name", specs[0].get("name", "red_tape_roll")))
    start_positions: dict[str, np.ndarray] = {}
    assigned_slots: dict[str, str] = {}
    primary_qslice: tuple[int, int] | None = None
    for spec in specs:
        name = str(spec.get("name", ""))
        if not name:
            raise RuntimeError(f"Invalid tape object spec without name: {spec}")
        qslice = tape_qpos_slice(model, spec)
        pos = random_tape_start_pos(spec, task, rng)
        set_object_pose(data, model, qslice, pos)
        start_positions[name] = pos
        assigned_slots[name] = str(spec.get("assigned_slot_from", spec.get("slot_name", "")))
        if name == primary_name:
            primary_qslice = qslice
    if primary_name not in start_positions:
        raise RuntimeError(f"Primary task object {primary_name!r} is not present in sim_tape_pick_place.tape_objects")
    if primary_qslice is None:
        raise RuntimeError(f"Could not resolve qpos slice for primary task object {primary_name!r}")
    return start_positions, primary_qslice, assigned_slots


def tape_spec_by_name(task: dict) -> dict[str, dict]:
    return {str(spec["name"]): dict(spec) for spec in tape_specs_from_task(task)}


def task_goal_pos(model, data, task: dict) -> np.ndarray:
    goal_site = str(task.get("goal_site", "") or "")
    if goal_site:
        pos = np.asarray(site_position(model, data, goal_site), dtype=np.float32)
    else:
        pos = np.asarray(task.get("goal_pos_m", task.get("start_pos_m", [-0.55, 0.0, 0.02])), dtype=np.float32)
    return pos + np.asarray(task.get("goal_offset_m", [0.0, 0.0, 0.0]), dtype=np.float32)


def grasp_wall_offset(task: dict, object_pos: np.ndarray | None = None) -> np.ndarray:
    radius = float(task.get("grasp_wall_center_radius_m", 0.0637))
    if str(task.get("grasp_wall_strategy", "")) == "nearest_to_robot_base" and object_pos is not None:
        base_xy = np.asarray(task.get("robot_base_xy_m", [0.0, 0.0]), dtype=np.float32)
        obj_xy = np.asarray(object_pos, dtype=np.float32)[:2]
        direction = base_xy - obj_xy
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            offset_xy = radius * direction / norm
            return np.asarray([offset_xy[0], offset_xy[1], 0.0], dtype=np.float32)
    angle = float(task.get("grasp_wall_angle_rad", -0.5 * math.pi))
    return np.asarray([radius * math.cos(angle), radius * math.sin(angle), 0.0], dtype=np.float32)


def rotate_gripper_joint6(q: np.ndarray, task: dict, reference_q: np.ndarray | None = None) -> np.ndarray:
    out = np.asarray(q, dtype=np.float32).copy()
    if out.shape[0] >= 6:
        base = float(np.asarray(reference_q, dtype=np.float32)[5]) if reference_q is not None else float(out[5])
        out[5] = base + float(task.get("gripper_rotate_joint6_rad", 0.0))
    return clamp_q(out)


def make_schematic_rgb(
    width: int,
    height: int,
    *,
    object_pos: np.ndarray,
    tcp_pos: np.ndarray,
    start_pos: np.ndarray,
    drop_pos: np.ndarray,
) -> np.ndarray:
    cv2 = require_cv2()
    img = np.full((int(height), int(width), 3), (92, 94, 92), dtype=np.uint8)

    def pix(pos: np.ndarray) -> tuple[int, int]:
        x_min, x_max = -0.75, -0.35
        y_min, y_max = -0.20, 0.20
        u = int((float(pos[0]) - x_min) / (x_max - x_min) * (width - 1))
        v = int((y_max - float(pos[1])) / (y_max - y_min) * (height - 1))
        return int(np.clip(u, 0, width - 1)), int(np.clip(v, 0, height - 1))

    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (75, 78, 75), 2)
    cv2.line(img, pix(start_pos), pix(drop_pos), (40, 40, 230), 2)
    cv2.circle(img, pix(drop_pos), 10, (60, 180, 60), 2)
    cv2.circle(img, pix(object_pos), 16, (20, 20, 220), -1)
    cv2.circle(img, pix(object_pos), 7, (225, 225, 225), -1)
    cv2.circle(img, pix(tcp_pos), 5, (250, 250, 20), -1)
    cv2.putText(img, "red tape pick-place sim", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def validate_camera_config_for_training(camera_config: str | Path) -> dict:
    camera_path = resolve_demo_path(camera_config)
    with camera_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    intr = cfg.get("intrinsics", {})
    ext = cfg.get("extrinsics", {})
    missing = []
    for key in ("width", "height", "fx", "fy", "cx", "cy"):
        if key not in intr:
            missing.append(f"intrinsics.{key}")
    for key in ("position", "rotation_matrix"):
        if key not in ext:
            missing.append(f"extrinsics.{key}")
    if missing:
        raise RuntimeError(
            f"Camera config is incomplete for simulator training RGB: {camera_path}. "
            f"Missing: {missing}. Run Astra capture and dynamic marker calibration first."
        )
    if not bool(cfg.get("calibrated", False)):
        raise RuntimeError(
            f"Camera config is not marked calibrated: {camera_path}. "
            "For training-aligned simulated RGB, run fr5_dynamic_marker_calibration.py solve --write-camera "
            "or set calibrated=true only after confirming the extrinsics are correct."
        )
    return cfg


def load_camera_projection(camera_config: str | Path) -> tuple[dict, np.ndarray, np.ndarray, int, int]:
    cfg = validate_camera_config_for_training(camera_config)
    intr = cfg["intrinsics"]
    ext = cfg["extrinsics"]
    k = np.asarray(
        [
            [float(intr["fx"]), 0.0, float(intr["cx"])],
            [0.0, float(intr["fy"]), float(intr["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    t_world_camera = np.eye(4, dtype=np.float64)
    t_world_camera[:3, :3] = np.asarray(ext["rotation_matrix"], dtype=np.float64)
    t_world_camera[:3, 3] = np.asarray(ext["position"], dtype=np.float64)
    t_camera_world = np.linalg.inv(t_world_camera)
    return cfg, k, t_camera_world, int(intr["width"]), int(intr["height"])


def project_world_points(points_world: np.ndarray, k: np.ndarray, t_camera_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    pts_cam = (t_camera_world @ homog.T).T[:, :3]
    z = pts_cam[:, 2]
    uv = np.empty((pts.shape[0], 2), dtype=np.float64)
    safe_z = np.maximum(z, 1e-9)
    uv[:, 0] = k[0, 0] * pts_cam[:, 0] / safe_z + k[0, 2]
    uv[:, 1] = k[1, 1] * pts_cam[:, 1] / safe_z + k[1, 2]
    return uv, z


def draw_projected_polygon(
    cv2,
    image: np.ndarray,
    points_world: np.ndarray,
    k: np.ndarray,
    t_camera_world: np.ndarray,
    color_rgb: tuple[int, int, int],
) -> None:
    uv, z = project_world_points(points_world, k, t_camera_world)
    if np.any(z <= 1e-5):
        return
    pts = np.rint(uv).astype(np.int32)
    cv2.fillConvexPoly(image, pts, tuple(int(c) for c in color_rgb))


def box_top_corners(center: tuple[float, float, float], half: tuple[float, float, float]) -> np.ndarray:
    cx, cy, cz = center
    hx, hy, hz = half
    z = cz + hz
    return np.asarray(
        [
            [cx - hx, cy - hy, z],
            [cx + hx, cy - hy, z],
            [cx + hx, cy + hy, z],
            [cx - hx, cy + hy, z],
        ],
        dtype=np.float64,
    )


def draw_camera_solid_table(cv2, image: np.ndarray, k: np.ndarray, t_camera_world: np.ndarray) -> None:
    draw_projected_polygon(
        cv2,
        image,
        box_top_corners((-0.485, -0.26, -0.0265), (0.75, 0.5, 0.0235)),
        k,
        t_camera_world,
        COLOR_GROOVE,
    )
    y = -0.7525
    for _ in range(50):
        draw_projected_polygon(
            cv2,
            image,
            box_top_corners((-0.485, y, -0.0015), (0.75, 0.0075, 0.0015)),
            k,
            t_camera_world,
            COLOR_TABLE,
        )
        y += 0.04
        draw_projected_polygon(
            cv2,
            image,
            box_top_corners((-0.485, y - 0.015, -0.0015), (0.75, 0.0075, 0.0015)),
            k,
            t_camera_world,
            COLOR_TABLE,
        )


def draw_camera_solid_tape(
    cv2,
    image: np.ndarray,
    k: np.ndarray,
    t_camera_world: np.ndarray,
    object_pos: np.ndarray,
) -> None:
    center = np.asarray(object_pos, dtype=np.float64)
    outer = float(0.113 * 0.5)
    inner = float(0.077 * 0.5)
    z = center[2] + 0.0215
    angles = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
    outer_pts = np.stack([center[0] + outer * np.cos(angles), center[1] + outer * np.sin(angles), np.full_like(angles, z)], axis=1)
    inner_pts = np.stack([center[0] + inner * np.cos(angles), center[1] + inner * np.sin(angles), np.full_like(angles, z)], axis=1)
    uv_outer, z_outer = project_world_points(outer_pts, k, t_camera_world)
    uv_inner, z_inner = project_world_points(inner_pts, k, t_camera_world)
    if np.any(z_outer <= 1e-5) or np.any(z_inner <= 1e-5):
        return
    outer_px = np.rint(uv_outer).astype(np.int32)
    inner_px = np.rint(uv_inner).astype(np.int32)
    cv2.fillPoly(image, [outer_px], tuple(int(c) for c in COLOR_TAPE))
    cv2.fillPoly(image, [inner_px], tuple(int(c) for c in COLOR_BACKGROUND))
    cv2.polylines(image, [outer_px], isClosed=True, color=(120, 122, 124), thickness=1)
    cv2.polylines(image, [inner_px], isClosed=True, color=(120, 122, 124), thickness=1)


def make_camera_solid_rgb(
    camera_config: str | Path,
    *,
    object_pos: np.ndarray,
) -> np.ndarray:
    cv2 = require_cv2()
    _, k, t_camera_world, width, height = load_camera_projection(camera_config)
    image = np.full((height, width, 3), COLOR_BACKGROUND, dtype=np.uint8)
    draw_camera_solid_table(cv2, image, k, t_camera_world)
    draw_projected_polygon(
        cv2,
        image,
        box_top_corners((0.0, 0.0, 0.01), (0.118, 0.1, 0.01)),
        k,
        t_camera_world,
        COLOR_FIXED_BASE,
    )
    draw_camera_solid_tape(cv2, image, k, t_camera_world, object_pos)
    return np.ascontiguousarray(image)


def attach_assist_ready(
    *,
    tcp: np.ndarray,
    obj: np.ndarray,
    gripper_closure: float,
    gripper_threshold: float,
    xy_distance: float,
    z_distance: float,
) -> bool:
    tcp = np.asarray(tcp, dtype=np.float32)
    obj = np.asarray(obj, dtype=np.float32)
    if float(gripper_closure) < float(gripper_threshold):
        return False
    if float(np.linalg.norm((tcp - obj)[:2])) > float(xy_distance):
        return False
    if abs(float(tcp[2] - obj[2])) > float(z_distance):
        return False
    return True


def dense_task_reward(
    *,
    tcp: np.ndarray,
    obj: np.ndarray,
    goal: np.ndarray,
    gripper: float,
    attached: bool,
    release: bool,
    place_radius: float,
) -> float:
    tcp = np.asarray(tcp, dtype=np.float32)
    obj = np.asarray(obj, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    tcp_obj = float(np.linalg.norm((tcp - obj)[:2]))
    obj_goal = float(np.linalg.norm((obj - goal)[:2]))
    reward = -0.02
    reward += 0.15 * max(0.0, 1.0 - tcp_obj / 0.12)
    reward += 0.20 * float(np.clip(gripper, 0.0, 1.0)) if tcp_obj < 0.08 else 0.0
    if bool(attached):
        reward += 0.35
    reward += 0.25 * max(0.0, 1.0 - obj_goal / 0.25)
    if bool(release) and obj_goal <= float(place_radius):
        reward += 1.0
    return float(np.clip(reward, -1.0, 1.5))


def fill_empty_rgb_background(rgb: np.ndarray, color_rgb: tuple[int, int, int] = COLOR_BACKGROUND) -> np.ndarray:
    out = np.ascontiguousarray(rgb).copy()
    empty = out.max(axis=-1) <= 3
    out[empty] = np.asarray(color_rgb, dtype=np.uint8)
    return out


def phase_keyframes(records: list[dict]) -> list[dict]:
    wanted = [
        ("before_grasp", {"above_grasp", "descend"}),
        ("after_close", {"close"}),
        ("after_lift", {"lift"}),
        ("after_transfer", {"transfer"}),
        ("after_release", {"release", "retreat"}),
    ]
    out: list[dict] = []
    used: set[int] = set()
    for label, phases in wanted:
        candidates = [
            record
            for record in records
            if str(record["phase"]).split("_")[-1] in phases and int(record["frame"]) not in used
        ]
        if not candidates:
            continue
        record = candidates[len(candidates) // 2]
        item = dict(record)
        item["label"] = label
        out.append(item)
        used.add(int(record["frame"]))
    return out


def write_debug_outputs(episode_dir: Path, records: list[dict], meta: dict) -> dict[str, str]:
    cv2 = require_cv2()
    debug_dir = episode_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    keyframes = phase_keyframes(records)
    panels = []
    for item in keyframes:
        rgb_path = episode_dir / str(item["image_file"])
        bgr = cv2.imread(rgb_path.as_posix(), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        panel = cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_AREA)
        text = (
            f"{item['label']} f={int(item['frame'])} "
            f"z={float(item['object_pos'][2]):.3f} g={float(item['gripper']):.2f}"
        )
        cv2.rectangle(panel, (0, 0), (319, 28), (0, 0, 0), -1)
        cv2.putText(panel, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        panels.append(panel)
    outputs: dict[str, str] = {}
    if panels:
        sheet = np.concatenate(panels, axis=1)
        sheet_path = debug_dir / "grasp_sequence.png"
        cv2.imwrite(sheet_path.as_posix(), sheet)
        outputs["grasp_sequence"] = sheet_path.relative_to(episode_dir).as_posix()
    check = {
        "grasp_success": bool(meta["grasp_success"]),
        "place_success": bool(meta["place_success"]),
        "task_success": bool(meta["task_success"]),
        "object_start_pos_m": meta["object_start_pos_m"],
        "object_drop_target_pos_m": meta["object_drop_target_pos_m"],
        "object_final_pos_m": meta["object_final_pos_m"],
        "object_max_z_m": meta["object_max_z_m"],
        "object_close_z_m": meta["object_close_z_m"],
        "object_lift_phase_max_z_m": meta["object_lift_phase_max_z_m"],
        "object_lift_delta_from_close_m": meta["object_lift_delta_from_close_m"],
        "lift_success_delta_z_m": meta["lift_success_delta_z_m"],
        "place_error_xy_m": meta["place_error_xy_m"],
        "place_success_radius_m": meta["place_success_radius_m"],
        "attach_assist": meta["attach_assist"],
        "grasp_mode": meta.get("grasp_mode", ""),
        "keyframes": keyframes,
    }
    check_path = debug_dir / "grasp_check.json"
    check_path.write_text(json.dumps(check, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["grasp_check"] = check_path.relative_to(episode_dir).as_posix()
    report_path = debug_dir / "grasp_check.md"
    report_path.write_text(
        "\n".join(
            [
                "# Red Tape Grasp Check",
                "",
                f"- grasp_success: `{bool(meta['grasp_success'])}`",
                f"- place_success: `{bool(meta['place_success'])}`",
                f"- task_success: `{bool(meta['task_success'])}`",
                f"- grasp_mode: `{meta.get('grasp_mode', '')}`",
                f"- attach_assist: `{bool(meta['attach_assist'])}`",
                f"- object_max_z_m: `{float(meta['object_max_z_m']):.4f}`",
                f"- object_close_z_m: `{float(meta['object_close_z_m']):.4f}`",
                f"- object_lift_phase_max_z_m: `{float(meta['object_lift_phase_max_z_m']):.4f}`",
                f"- object_lift_delta_from_close_m: `{float(meta['object_lift_delta_from_close_m']):.4f}`",
                f"- required_lift_delta_m: `{float(meta['lift_success_delta_z_m']):.4f}`",
                f"- place_error_xy_m: `{float(meta['place_error_xy_m']):.4f}`",
                f"- place_success_radius_m: `{float(meta['place_success_radius_m']):.4f}`",
                "",
                "Check `debug/grasp_sequence.png` first. It shows before grasp, after close, after lift, after transfer, and after release when available.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs["grasp_report"] = report_path.relative_to(episode_dir).as_posix()
    return outputs


def compute_grasp_metrics(records: list[dict], object_initial_z: float, lift_threshold: float) -> dict[str, float | bool]:
    close_records = [r for r in records if str(r.get("phase")).split("_")[-1] == "close"]
    lift_records = [r for r in records if str(r.get("phase")).split("_")[-1] in {"lift", "transfer"}]
    close_z = float(close_records[-1]["object_pos"][2]) if close_records else float(object_initial_z)
    lift_phase_max_z = max((float(r["object_pos"][2]) for r in lift_records), default=float(object_initial_z))
    lift_delta_from_initial = lift_phase_max_z - float(object_initial_z)
    lift_delta_from_close = lift_phase_max_z - close_z
    required_from_close = max(0.006, 0.75 * float(lift_threshold))
    grasp_success = bool(lift_delta_from_initial >= float(lift_threshold) and lift_delta_from_close >= required_from_close)
    return {
        "object_close_z_m": close_z,
        "object_lift_phase_max_z_m": lift_phase_max_z,
        "object_lift_delta_from_initial_m": lift_delta_from_initial,
        "object_lift_delta_from_close_m": lift_delta_from_close,
        "required_lift_delta_from_close_m": required_from_close,
        "grasp_success": grasp_success,
    }


def drain_capture_tasks(capture_tasks: list[tuple[str, object]], capture_dir: Path) -> list[str]:
    saved: list[str] = []
    pending: list[tuple[str, object]] = []
    for name, task in capture_tasks:
        if getattr(task, "state", None) == "pending":
            pending.append((name, task))
            continue
        try:
            img = task.take_image()
            if img is None:
                continue
            capture_dir.mkdir(parents=True, exist_ok=True)
            out = capture_dir / f"{name}.png"
            img.save_to_disk(out.as_posix())
            saved.append(out.name)
        except Exception as exc:
            print(f"Warning: render capture failed for {name}: {exc}", flush=True)
    capture_tasks[:] = pending
    return saved


def save_render_capture(render, rcam, data, out_path: Path, *, timeout_s: float = 2.0) -> None:
    if rcam is None:
        raise RuntimeError("Render camera is unavailable; cannot write visual simulator RGB.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render.sync(data)
    task = rcam.capture()
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        render.sync(data)
        if getattr(task, "state", None) != "pending":
            img = task.take_image()
            if img is None:
                raise RuntimeError(f"Render capture returned no image for {out_path}")
            img.save_to_disk(out_path.as_posix())
            if not out_path.exists() or out_path.stat().st_size <= 0:
                raise RuntimeError(f"Render capture did not create a valid image: {out_path}")
            return
        time.sleep(0.01)
    raise RuntimeError(f"Timed out waiting for RenderApp capture: {out_path}")


def make_episode_dir(output_dir: str | Path, episode_name: str | None, index: int) -> Path:
    root = resolve_demo_path(output_dir)
    name = episode_name or datetime.now().strftime(f"sim_tape_%Y%m%d_%H%M%S_{index:03d}")
    if episode_name and index > 0:
        name = f"{episode_name}_{index:03d}"
    episode_dir = root / name
    if episode_dir.exists():
        raise RuntimeError(f"Episode directory already exists: {episode_dir}")
    (episode_dir / "rgb").mkdir(parents=True)
    return episode_dir


def argv_without_options(argv: list[str], option_names: set[str]) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(argv):
        item = argv[idx]
        key = item.split("=", 1)[0]
        if key in option_names:
            if "=" not in item and idx + 1 < len(argv) and not argv[idx + 1].startswith("-"):
                idx += 2
            else:
                idx += 1
            continue
        out.append(item)
        idx += 1
    return out


def run_window_episodes_in_subprocesses(args) -> None:
    base_argv = argv_without_options(sys.argv[1:], {"--episodes", "--seed", "--episode-name"})
    batch_name = args.episode_name or datetime.now().strftime("visual_batch_%Y%m%d_%H%M%S")
    env = os.environ.copy()
    env[CHILD_ENV_FLAG] = "1"
    for idx in range(int(args.episodes)):
        child_name = f"{batch_name}_{idx:03d}"
        child_seed = int(args.seed) + idx
        cmd = [
            sys.executable,
            Path(__file__).as_posix(),
            *base_argv,
            "--episodes",
            "1",
            "--seed",
            str(child_seed),
            "--episode-name",
            child_name,
        ]
        print(f"\nLaunching visual episode {idx + 1}/{int(args.episodes)} in a fresh process.", flush=True)
        subprocess.run(cmd, check=True, env=env)
    print("Generated visual episodes via subprocess batch.", flush=True)


def maybe_make_gs_renderer(args, config, model):
    if not args.use_3dgs:
        return None, None, None
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "--use-3dgs needs CUDA because the current 3DGS renderer uses CUDA tensors. "
            "Leave it off to use the default visual RenderApp renderer."
        )
    validate_camera_config_for_training(args.camera_config)
    camera_id = find_camera_id(model, camera_name_from_config(args.camera_config))
    if camera_id is None:
        raise RuntimeError("No calibrated camera in model. Provide --camera-config after dynamic calibration.")
    width, height = camera_resolution_from_config(args.camera_config)
    gs_dir = resolve_demo_path(args.fr5_gs_dir)
    ensure_fr5_gaussian_assets(
        config,
        gs_dir,
        regenerate=bool(args.fr5_gs_regenerate),
        points_per_geom=args.fr5_gs_points_per_geom,
    )
    gaussians = collect_fr5_gaussian_assets(model, gs_dir)
    if "red_tape_roll" not in gaussians:
        ensure_fr5_gaussian_assets(
            config,
            gs_dir,
            regenerate=True,
            points_per_geom=args.fr5_gs_points_per_geom,
        )
        gaussians = collect_fr5_gaussian_assets(model, gs_dir)
    from gaussian_renderer import GSRendererMotrixSim

    return GSRendererMotrixSim(gaussians, model), int(camera_id), (int(width), int(height))


def run_one_episode(args, index: int, rng: np.random.Generator) -> Path:
    cv2 = require_cv2()
    config = load_config(args.config)
    task = config.get("sim_tape_pick_place", {})
    if args.render_mode == "schematic":
        print(
            "Warning: --render-mode schematic writes non-camera schematic RGB. "
            "Use default visual mode or --use-3dgs for camera-aligned training data.",
            flush=True,
        )
    else:
        validate_camera_config_for_training(args.camera_config)
    runtime_camera_config = args.camera_config if args.render_mode in {"visual", "gs"} else None
    model, data, body, qpos_ids, arm_act_ids, gripper_act_ids = build_runtime(
        config,
        runtime_camera_config,
    )
    render_camera_name = camera_name_from_config(args.camera_config) if args.render_mode == "visual" else None
    render_camera = find_camera(model, render_camera_name) if args.render_mode == "visual" else None
    if args.render_mode == "visual" and render_camera is None:
        raise RuntimeError(
            f"Could not find calibrated render camera {render_camera_name!r}. "
            "Check --camera-config and astra_camera.json."
        )
    tcp_site = str(config["tcp_site"])
    object_site = str(task.get("object_site", config.get("target_site", "red_tape_roll_site")))
    if args.start_random_radius is not None:
        task = dict(task)
        task["start_random_radius_m"] = float(args.start_random_radius)
        task.pop("tape_objects", None)
    tape_start_positions, qslice, tape_assigned_slots = initialize_tape_objects(data, model, task, rng)
    specs_by_name = tape_spec_by_name(task)
    tape_order = [str(spec["name"]) for spec in tape_specs_from_task(task)]
    primary_name = str(task.get("object_name", tape_order[0]))
    start_pos = np.asarray(tape_start_positions[primary_name], dtype=np.float32)
    start_center = np.asarray(task.get("start_random_center_m", task.get("start_pos_m", start_pos)), dtype=np.float32)
    start_radius = float(task.get("start_random_radius_m", 0.04))
    goal_pos = task_goal_pos(model, data, task)
    object_sites = {name: str(specs_by_name[name].get("site", f"{name}_site")) for name in tape_order}
    object_qslices = {name: tape_qpos_slice(model, specs_by_name[name]) for name in tape_order}
    drop_positions: dict[str, np.ndarray] = {}
    grasp_positions: dict[str, np.ndarray] = {}
    wall_offsets: dict[str, np.ndarray] = {}
    stack_on_goal = bool(task.get("stack_on_goal", len(tape_order) > 1))
    stack_spacing = float(task.get("stack_spacing_m", 0.046))
    shared_drop_xy_offset = np.zeros(2, dtype=np.float32)
    if stack_on_goal:
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        r = float(args.drop_random_radius if args.drop_random_radius is not None else task.get("drop_random_radius_m", 0.0))
        r *= math.sqrt(float(rng.uniform(0.0, 1.0)))
        shared_drop_xy_offset[:] = [r * math.cos(theta), r * math.sin(theta)]
    for seq_idx, name in enumerate(tape_order):
        obj_start = np.asarray(tape_start_positions[name], dtype=np.float32)
        wall_offsets[name] = grasp_wall_offset(task, obj_start)
        grasp_positions[name] = obj_start + wall_offsets[name]
        drop_base = np.asarray(goal_pos, dtype=np.float32).copy()
        drop_base[2] = obj_start[2] + (stack_spacing * seq_idx if stack_on_goal else 0.0)
        if stack_on_goal:
            drop_base[0] += float(args.drop_delta_x if args.drop_delta_x is not None else task.get("drop_delta_x_m", 0.0))
            drop_base[:2] += shared_drop_xy_offset
            drop_positions[name] = drop_base.astype(np.float32)
        else:
            drop_positions[name] = random_drop_pos(
                drop_base,
                float(args.drop_delta_x if args.drop_delta_x is not None else task.get("drop_delta_x_m", 0.0)),
                float(args.drop_random_radius if args.drop_random_radius is not None else task.get("drop_random_radius_m", 0.0)),
                rng,
            )
    q_home = np.asarray(config["initial_qpos"], dtype=np.float32)
    set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q_home)
    set_gripper(data, model, body, gripper_act_ids, float(task.get("default_gripper_opening", 0.0)))
    for _ in range(50):
        sim_step(model, data)

    approach_h = float(task.get("approach_height_m", 0.13))
    grasp_h = float(task.get("grasp_height_m", 0.035))
    lift_delta = float(task.get("lift_delta_m", 0.14))
    lift_h = float(task.get("lift_height_m", float(task.get("grasp_height_m", 0.035)) + lift_delta))
    if "release_drop_delta_m" in task:
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
    waypoints = [Waypoint("home", q_home, open_g, int(0.5 * args.hz), False, False, primary_name)]
    q_seed = q_home
    for seq_idx, name in enumerate(tape_order):
        grasp_pos = grasp_positions[name]
        drop_pos = drop_positions[name]
        wall_offset = wall_offsets[name]
        q_above_seed = rotate_gripper_joint6(q_seed, task, q_home)
        q_above = solve_tcp_ik_with_axis(
            model,
            data,
            body,
            qpos_ids,
            arm_act_ids,
            tcp_site,
            q_above_seed,
            grasp_pos + [0, 0, approach_h],
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
                grasp_pos + [0, 0, float(z)],
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
            grasp_pos + [0, 0, grasp_h + lift_delta],
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
            drop_pos + wall_offset + [0, 0, lift_h],
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
            drop_pos + wall_offset + [0, 0, release_h],
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
            drop_pos + wall_offset + [0, 0, approach_h],
            desired_axis=vertical_axis,
            axis_index=vertical_axis_index,
            axis_weight=vertical_axis_weight,
            axis_tol=vertical_axis_tol,
        )
        prefix = f"{seq_idx + 1}_{name}"
        waypoints.extend(
            [
                Waypoint(f"{prefix}_rotate_gripper_90", q_above, open_g, int(0.5 * args.hz), False, False, name),
                Waypoint(f"{prefix}_above_grasp", q_above, open_g, int(1.0 * args.hz), False, False, name),
                *[
                    Waypoint(
                        f"{prefix}_vertical_descend",
                        q_desc,
                        open_g,
                        max(1, int(round(vertical_descend_frames / vertical_descend_segments))),
                        False,
                        False,
                        name,
                    )
                    for q_desc in descend_qs
                ],
                Waypoint(f"{prefix}_close", q_grasp, closed_g, int(1.2 * args.hz), True, False, name),
                Waypoint(f"{prefix}_lift", q_lift, closed_g, int(1.6 * args.hz), True, False, name),
                Waypoint(f"{prefix}_transfer", q_transfer, closed_g, int(1.2 * args.hz), True, False, name),
                Waypoint(f"{prefix}_lower", q_lower, closed_g, int(0.8 * args.hz), True, False, name),
                Waypoint(f"{prefix}_release", q_lower, release_g, int(0.5 * args.hz), False, True, name),
                Waypoint(f"{prefix}_retreat", q_retreat, release_g, int(0.8 * args.hz), False, False, name),
            ]
        )
        q_seed = q_retreat
    frames = interpolate_waypoints(waypoints)
    if args.max_frames > 0:
        frames = frames[: int(args.max_frames)]
    if len(frames) < 2:
        raise RuntimeError("Generated trajectory is too short.")

    gs_renderer, camera_id, gs_size = maybe_make_gs_renderer(args, config, model)
    episode_dir = make_episode_dir(args.output_dir, args.episode_name or None, index)
    steps_per_frame = max(1, round((1.0 / float(args.hz)) / float(model.options.timestep)))
    object_initial_z = float(start_pos[2])
    object_max_z = object_initial_z
    object_max_z_by_name = {name: float(pos[2]) for name, pos in tape_start_positions.items()}
    attach_assist = bool(task.get("attach_assist_default", False) if args.attach_assist is None else args.attach_assist)
    release_teleport_assist = bool(args.release_teleport_assist)
    attach_xy_distance = float(args.attach_distance if args.attach_distance is not None else task.get("attach_distance_m", 0.08))
    attach_z_distance = float(args.attach_z_distance if args.attach_z_distance is not None else task.get("attach_z_distance_m", 0.055))
    attach_gripper_threshold = float(
        args.attach_gripper_threshold if args.attach_gripper_threshold is not None else task.get("attach_gripper_threshold", 0.85)
    )
    attached_name = ""
    attach_offset = np.zeros(3, dtype=np.float32)
    released_objects: set[str] = set()
    timestamps: list[float] = []
    joint_deg: list[np.ndarray] = []
    tcp_pos: list[np.ndarray] = []
    gripper_closure: list[float] = []
    image_files: list[str] = []
    target_features: list[np.ndarray] = []
    rewards: list[float] = []
    debug_records: list[dict] = []
    start_t = time.monotonic()
    render_capture_dir = episode_dir / "debug" / "render"
    render_captures: list[str] = []
    capture_tasks: list[tuple[str, object]] = []
    render = None
    needs_render = bool(args.visualize or args.render_mode == "visual")
    render_context = RenderApp() if needs_render else None

    if render_context is not None:
        render = render_context.__enter__()
        render.launch(model)
        if args.render_mode == "visual":
            render.set_main_camera(render_camera)
        render.sync(data)
        if args.render_mode == "visual":
            print(f"Visual simulator RGB output locked to camera: {render_camera.name}", flush=True)
        print("Visual simulation window launched. Close the window to stop this episode early.", flush=True)
    rcam = render.get_camera(0) if render is not None else None
    if render is not None and rcam is None:
        print("Warning: RenderApp camera 0 is unavailable; visual screenshots are disabled.", flush=True)

    try:
        for frame_idx, item in enumerate(frames):
            if render is not None and render.is_closed:
                break
            set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, item.q)
            set_gripper(data, model, body, gripper_act_ids, float(item.gripper))
            active_name = str(item.object_name or primary_name)
            active_site = object_sites.get(active_name, object_site)
            active_qslice = object_qslices.get(active_name, qslice)
            active_drop_pos = drop_positions.get(active_name, drop_positions[primary_name])
            tcp = np.asarray(site_position(model, data, tcp_site), dtype=np.float32)
            obj = object_pos_from_site(model, data, active_site)
            if attach_assist and item.attach and active_name not in released_objects:
                can_attach = attached_name == active_name or attach_assist_ready(
                    tcp=tcp,
                    obj=obj,
                    gripper_closure=float(item.gripper),
                    gripper_threshold=attach_gripper_threshold,
                    xy_distance=attach_xy_distance,
                    z_distance=attach_z_distance,
                )
                if can_attach and attached_name != active_name:
                    attach_offset = obj - tcp
                    if float(np.linalg.norm(attach_offset[:2])) > attach_xy_distance:
                        attach_offset = np.asarray([0.0, 0.0, -0.015], dtype=np.float32)
                    attached_name = active_name
                if attached_name == active_name:
                    set_object_pose(data, model, active_qslice, tcp + attach_offset)
            elif item.release:
                released_objects.add(active_name)
                attached_name = ""
                if attach_assist and release_teleport_assist:
                    set_object_pose(data, model, active_qslice, active_drop_pos)
            elif active_name in released_objects and attach_assist and release_teleport_assist:
                set_object_pose(data, model, active_qslice, active_drop_pos)
            for _ in range(steps_per_frame):
                sim_step(model, data)
            obj = object_pos_from_site(model, data, active_site)
            object_max_z = max(object_max_z, float(obj[2]))
            object_max_z_by_name[active_name] = max(object_max_z_by_name.get(active_name, float(obj[2])), float(obj[2]))
            tcp = np.asarray(site_position(model, data, tcp_site), dtype=np.float32)
            rel_path = f"rgb/{frame_idx:06d}.png"
            if args.render_mode == "visual":
                if render is None or rcam is None:
                    raise RuntimeError("Visual render mode requires an active RenderApp camera.")
                render.sync(data)
                save_render_capture(render, rcam, data, episode_dir / rel_path)
            elif gs_renderer is not None:
                width, height = gs_size
                rgb = render_fr5_gs_rgb(gs_renderer, model, data, int(camera_id), width, height)
                rgb = fill_empty_rgb_background(rgb)
                bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
                if not cv2.imwrite((episode_dir / rel_path).as_posix(), bgr):
                    raise RuntimeError(f"Failed to write image: {episode_dir / rel_path}")
            elif args.render_mode == "camera-solid":
                rgb = make_camera_solid_rgb(args.camera_config, object_pos=obj)
                bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
                if not cv2.imwrite((episode_dir / rel_path).as_posix(), bgr):
                    raise RuntimeError(f"Failed to write image: {episode_dir / rel_path}")
            else:
                rgb = make_schematic_rgb(
                    int(args.rgb_width),
                    int(args.rgb_height),
                    object_pos=obj,
                    tcp_pos=tcp,
                    start_pos=start_pos,
                    drop_pos=active_drop_pos,
                )
                bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
                if not cv2.imwrite((episode_dir / rel_path).as_posix(), bgr):
                    raise RuntimeError(f"Failed to write image: {episode_dir / rel_path}")
            timestamps.append(time.monotonic() - start_t)
            joint_deg.append(np.rad2deg(item.q).astype(np.float32))
            tcp_pos.append(tcp.astype(np.float32))
            gripper_closure.append(float(item.gripper))
            image_files.append(rel_path)
            target_features.append(make_target_feature(obj, active_drop_pos, camera_config=args.camera_config))
            rewards.append(
                dense_task_reward(
                    tcp=tcp,
                    obj=obj,
                    goal=active_drop_pos,
                    gripper=float(item.gripper),
                    attached=attached_name == active_name,
                    release=bool(item.release),
                    place_radius=float(task.get("place_success_radius_m", task.get("drop_random_radius_m", 0.01))),
                )
            )
            debug_records.append(
                {
                    "frame": int(frame_idx),
                    "phase": str(item.name),
                    "image_file": rel_path,
                    "object_name": active_name,
                    "tcp_pos": tcp.round(6).tolist(),
                    "object_pos": obj.round(6).tolist(),
                    "gripper": float(item.gripper),
                }
            )
            if render is not None:
                render.sync(data)
                phase_leaf = str(item.name).split("_")[-1]
                if args.render_mode != "visual" and rcam is not None and (
                    bool(args.capture_render_every_frame)
                    or phase_leaf in {"descend", "close", "lift", "transfer", "release", "retreat"}
                    and frame_idx % max(1, int(args.hz // 2)) == 0
                ):
                    capture_tasks.append((f"render_{frame_idx:06d}_{item.name}", rcam.capture()))
                render_captures.extend(drain_capture_tasks(capture_tasks, render_capture_dir))
                time.sleep(1.0 / float(args.hz))
            if frame_idx % max(1, int(args.hz)) == 0:
                print(
                    f"episode={index:03d} frame={frame_idx:04d} phase={item.name} "
                    f"tcp={tcp.round(4).tolist()} object={obj.round(4).tolist()} gripper={float(item.gripper):.2f}",
                    flush=True,
                )
    finally:
        if render is not None:
            for _ in range(10):
                if not capture_tasks:
                    break
                render.sync(data)
                render_captures.extend(drain_capture_tasks(capture_tasks, render_capture_dir))
                time.sleep(0.02)
        if render_context is not None:
            render_context.__exit__(None, None, None)

    lift_threshold = float(task.get("lift_success_delta_z_m", 0.01))
    place_radius = float(task.get("place_success_radius_m", task.get("drop_random_radius_m", 0.01)))
    per_tape_results: dict[str, dict] = {}
    final_positions: dict[str, list[float]] = {}
    for name in tape_order:
        records = [record for record in debug_records if record.get("object_name") == name]
        metrics = compute_grasp_metrics(records, float(tape_start_positions[name][2]), lift_threshold)
        final_pos = object_pos_from_site(model, data, object_sites[name])
        final_positions[name] = final_pos.round(6).tolist()
        target_pos = drop_positions[name]
        place_error_i = float(np.linalg.norm(final_pos[:2] - target_pos[:2]))
        place_error_xyz_i = float(np.linalg.norm(final_pos[:3] - target_pos[:3]))
        per_tape_results[name] = {
            "grasp_success": bool(metrics["grasp_success"]),
            "place_success": bool(place_error_i <= place_radius),
            "place_error_xy_m": place_error_i,
            "place_error_xyz_m": place_error_xyz_i,
            "drop_target_pos_m": target_pos.round(6).tolist(),
            "object_final_pos_m": final_pos.round(6).tolist(),
            "object_max_z_m": float(object_max_z_by_name.get(name, final_pos[2])),
            "grasp_metrics": metrics,
        }
    grasp_metrics = per_tape_results[primary_name]["grasp_metrics"]
    lifted = bool(all(item["grasp_success"] for item in per_tape_results.values()))
    final_obj = object_pos_from_site(model, data, object_site)
    drop_pos = drop_positions[primary_name]
    grasp_pos = grasp_positions[primary_name]
    place_error = float(per_tape_results[primary_name]["place_error_xy_m"])
    place_success = bool(all(item["place_success"] for item in per_tape_results.values()))
    task_success = bool(lifted and place_success)
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "sim_tape_pick_place",
        "control_hz": float(args.hz),
        "config": resolve_demo_path(args.config).as_posix(),
        "camera_config": resolve_demo_path(args.camera_config).as_posix() if args.render_mode != "schematic" else "",
        "rgb_width": int(args.rgb_width if args.render_mode == "schematic" else (gs_size[0] if gs_size else camera_resolution_from_config(args.camera_config)[0])),
        "rgb_height": int(args.rgb_height if args.render_mode == "schematic" else (gs_size[1] if gs_size else camera_resolution_from_config(args.camera_config)[1])),
        "rgb_source": (
            "sim_calibrated_astra_3dgs"
            if args.use_3dgs
            else (
                "sim_calibrated_astra_visual_render"
                if args.render_mode == "visual"
                else ("sim_calibrated_astra_solid_debug" if args.render_mode == "camera-solid" else "sim_schematic_debug_only")
            )
        ),
        "camera_aligned_rgb": bool(args.render_mode != "schematic"),
        "camera_name": camera_name_from_config(args.camera_config) if args.render_mode != "schematic" else "",
        "use_3dgs": bool(args.use_3dgs),
        "action_mode": "joint_delta_rad_plus_gripper_delta",
        "task_name": str(task.get("task_name", "three_tape_rings_on_myd_part1")),
        "task_description": str(task.get("task_description", "Move the red, yellow, and white tape rolls onto the myd_part1 insertion target in order.")),
        "object_name": primary_name,
        "tape_order": tape_order,
        "tape_assigned_slots": tape_assigned_slots,
        "goal_site": str(task.get("goal_site", "")),
        "goal_pos_m": goal_pos.round(6).tolist(),
        "start_random_center_m": start_center.round(6).tolist(),
        "start_random_radius_m": float(start_radius),
        "object_start_pos_m": start_pos.round(6).tolist(),
        "tape_start_positions_m": {name: pos.round(6).tolist() for name, pos in tape_start_positions.items()},
        "tape_final_positions_m": final_positions,
        "per_tape_results": per_tape_results,
        "stack_on_goal": bool(stack_on_goal),
        "stack_spacing_m": float(stack_spacing),
        "grasp_tcp_target_pos_m": grasp_pos.round(6).tolist(),
        "grasp_wall_offset_m": wall_offsets[primary_name].round(6).tolist(),
        "object_drop_target_pos_m": drop_pos.round(6).tolist(),
        "release_drop_delta_m": float(task.get("release_drop_delta_m", lift_h - release_h)),
        "release_tcp_height_m": float(release_h),
        "object_final_pos_m": final_obj.round(6).tolist(),
        "object_max_z_m": object_max_z,
        "object_close_z_m": float(grasp_metrics["object_close_z_m"]),
        "object_lift_phase_max_z_m": float(grasp_metrics["object_lift_phase_max_z_m"]),
        "object_lift_delta_from_initial_m": float(grasp_metrics["object_lift_delta_from_initial_m"]),
        "object_lift_delta_from_close_m": float(grasp_metrics["object_lift_delta_from_close_m"]),
        "required_lift_delta_from_close_m": float(grasp_metrics["required_lift_delta_from_close_m"]),
        "lift_success_delta_z_m": lift_threshold,
        "grasp_success": bool(lifted),
        "place_success": bool(place_success),
        "task_success": bool(task_success),
        "place_error_xy_m": place_error,
        "place_success_radius_m": place_radius,
        "attach_assist": bool(attach_assist),
        "release_teleport_assist": bool(release_teleport_assist),
        "grasp_mode": str(task.get("grasp_mode", "inner_outer_wall")),
        "notes": (
            "Scripted sim ring-on-target task for red tape and myd_part1. Grasp success requires the object to rise during "
            "the lift/transfer phase after the close phase, not just a transient collision bump. "
            "Place success means the tape center is within place_success_radius_m of the myd_part1 goal site in XY. "
            "When attach_assist is enabled, it only carries the active tape while grasped; release uses physics unless "
            "--release-teleport-assist is explicitly enabled for legacy debugging. "
            "rgb/*.png is the training observation. Default visual mode captures the real MotrixSim/RenderApp "
            "view locked to the calibrated Astra simulation camera; debug/grasp_sequence.png is only a keyframe sheet. "
            "--render-mode camera-solid and schematic are debug fallbacks; --use-3dgs enables Gaussian visual rendering."
        ),
    }
    if render_captures:
        meta["render_debug_dir"] = "debug/render"
        meta["render_debug_images"] = sorted(set(render_captures))
    write_episode(
        episode_dir,
        timestamps=timestamps,
        joint_deg=joint_deg,
        tcp_pos=tcp_pos,
        gripper_closure=gripper_closure,
        image_files=image_files,
        meta=meta,
        target_features=target_features,
        rewards=rewards,
    )
    debug_outputs = write_debug_outputs(episode_dir, debug_records, meta) if args.save_debug_images else {}
    if debug_outputs:
        meta["debug_outputs"] = debug_outputs
        (episode_dir / "meta.json").write_text(
            json.dumps({**json.loads((episode_dir / "meta.json").read_text(encoding="utf-8")), "debug_outputs": debug_outputs}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    (episode_dir / "pick_place_report.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    status = "success" if task_success else "failed"
    print(f"Saved {status} tape pick-place sim episode: {episode_dir}")
    print(
        f"  frames={len(frames)} grasp_success={lifted} place_success={place_success} "
        f"max_z={object_max_z:.4f} lift_phase_max_z={float(grasp_metrics['object_lift_phase_max_z_m']):.4f} "
        f"lift_from_close={float(grasp_metrics['object_lift_delta_from_close_m']):.4f} "
        f"place_error_xy={place_error:.4f}m"
    )
    if debug_outputs:
        print(f"  debug={episode_dir / debug_outputs.get('grasp_sequence', 'debug')}")
    if render_captures:
        print(f"  visual_render_debug={render_capture_dir}")
    if args.discard_failed_grasp and not lifted:
        inspect_path = episode_dir / "debug" / "grasp_sequence.png"
        if not inspect_path.exists():
            inspect_path = episode_dir / "pick_place_report.json"
        reason = (
            f"Red tape grasp failed: lift/transfer phase max z "
            f"{float(grasp_metrics['object_lift_phase_max_z_m']):.4f}m, "
            f"close z {float(grasp_metrics['object_close_z_m']):.4f}m, "
            f"lift-from-close {float(grasp_metrics['object_lift_delta_from_close_m']):.4f}m "
            f"< required {float(grasp_metrics['required_lift_delta_from_close_m']):.4f}m."
        )
        discarded = discard_episode_dir(episode_dir, args.output_dir, reason)
        raise GraspFailureRetry(
            episode_dir=discarded,
            message=f"{reason} Discarded failed episode to {discarded}. Inspect {discarded / inspect_path.relative_to(episode_dir)}.",
        )
    if args.fail_on_grasp_failure and not lifted:
        inspect_path = episode_dir / "debug" / "grasp_sequence.png"
        if not inspect_path.exists():
            inspect_path = episode_dir / "pick_place_report.json"
        raise RuntimeError(
            f"Red tape grasp failed: lift/transfer phase max z {float(grasp_metrics['object_lift_phase_max_z_m']):.4f}m, "
            f"close z {float(grasp_metrics['object_close_z_m']):.4f}m, "
            f"lift-from-close {float(grasp_metrics['object_lift_delta_from_close_m']):.4f}m "
            f"< required {float(grasp_metrics['required_lift_delta_from_close_m']):.4f}m. "
            f"Inspect {inspect_path} and RenderApp screenshots if --visualize was used."
        )
    if args.fail_on_task_failure and not task_success:
        raise RuntimeError(
            f"Red tape pick-place task failed: grasp_success={lifted}, place_success={place_success}, "
            f"place_error_xy={place_error:.4f}m. Inspect {episode_dir / 'pick_place_report.json'}."
        )
    return episode_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scripted FR5 simulator pick-place episodes for the red tape roll.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--camera-config", type=str, default=DEFAULT_CAMERA_CONFIG.as_posix())
    parser.add_argument("--output-dir", type=str, default=DEFAULT_IL_DEMO_DIR.as_posix())
    parser.add_argument("--episode-name", type=str, default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--drop-delta-x", type=float, default=None)
    parser.add_argument("--drop-random-radius", type=float, default=None)
    parser.add_argument(
        "--start-random-radius",
        type=float,
        default=None,
        help="Randomize the tape start position within this XY radius around config start_random_center_m; defaults to config.",
    )
    parser.add_argument("--attach-assist", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--attach-distance", type=float, default=None, help="XY distance gate for attach_assist; defaults to config attach_distance_m.")
    parser.add_argument(
        "--attach-z-distance",
        type=float,
        default=None,
        help="Vertical distance gate for attach_assist; prevents high-pass gripper/tape sticking.",
    )
    parser.add_argument(
        "--attach-gripper-threshold",
        type=float,
        default=None,
        help="Minimum gripper closure for attach_assist; defaults to config attach_gripper_threshold.",
    )
    parser.add_argument(
        "--release-teleport-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy debug mode: snap a released tape to its target. Default false keeps release physical.",
    )
    parser.add_argument("--save-debug-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize", action="store_true", help="Open the real MotrixSim RenderApp window during data generation")
    parser.add_argument("--capture-render-every-frame", action="store_true", help="When --visualize is set, save every RenderApp camera frame under debug/render")
    parser.add_argument("--fail-on-grasp-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-task-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--discard-failed-grasp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-grasp-retries", type=int, default=5, help="Retries per requested episode when grasp_success=false")
    parser.add_argument("--use-3dgs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--render-mode",
        choices=("visual", "camera-solid", "schematic", "gs"),
        default="camera-solid",
        help=(
            "visual captures the actual MotrixSim RenderApp view locked to the calibrated Astra camera; "
            "camera-solid is the fast no-window default using calibrated camera projection; "
            "schematic is only for debug; gs requires --use-3dgs."
        ),
    )
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=480)
    parser.add_argument("--fr5-gs-dir", type=str, default=DEFAULT_FR5_GS_DIR.as_posix())
    parser.add_argument("--fr5-gs-regenerate", action="store_true")
    parser.add_argument("--fr5-gs-points-per-geom", type=int, default=None)
    args = parser.parse_args()

    if args.hz <= 0.0:
        raise RuntimeError("--hz must be positive")
    if args.episodes <= 0:
        raise RuntimeError("--episodes must be positive")
    if args.render_mode == "gs":
        args.use_3dgs = True
    if args.use_3dgs:
        args.render_mode = "gs"
    if (args.visualize or args.render_mode == "visual") and int(args.episodes) > 1 and os.environ.get(CHILD_ENV_FLAG) != "1":
        run_window_episodes_in_subprocesses(args)
        return
    rng = np.random.default_rng(int(args.seed))
    saved = []
    for idx in range(int(args.episodes)):
        attempt = 0
        while True:
            try:
                saved.append(run_one_episode(args, idx, rng))
                break
            except GraspFailureRetry as exc:
                attempt += 1
                print(
                    f"episode={idx:03d} grasp failed; discarded and retrying "
                    f"({attempt}/{int(args.max_grasp_retries)}): {exc}",
                    flush=True,
                )
                if attempt >= int(args.max_grasp_retries):
                    raise RuntimeError(
                        f"Episode {idx} failed to grasp after {attempt} attempts. "
                        "Check grasp_wall_center_radius_m / grasp_wall_angle_rad / gripper collision / motor gains."
                    ) from exc
    print("Generated episodes:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
