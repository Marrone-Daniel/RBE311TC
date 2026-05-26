from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch
from motrixsim import step as sim_step
from motrixsim.render import Layout, RenderApp

from arm_control import (
    DEFAULT_CONFIG,
    RealRgbSource,
    build_runtime,
    camera_name_from_config,
    find_camera,
    load_config,
    require_cv2,
    resolve_demo_path,
    set_arm_qpos_and_ctrl,
    set_gripper,
    site_position,
)
from fr5_il_dataset import (
    DEFAULT_IL_DEMO_DIR,
    DEFAULT_POLICY_DIR,
    STATE_DIM,
    denormalize_action,
    load_episode_npz,
    load_policy_checkpoint,
    make_target_feature,
    normalize_joint,
    preprocess_rgb_for_policy,
)
from fr5_sync_sdk import DEFAULT_ROBOT_IP, FairinoArmClient, MotionCancelHandler
from fr5_sync_sdk import Robotiq2F85ModbusRtuClient
from fr5_sim_tape_pick_place import (
    DEFAULT_CAMERA_CONFIG,
    attach_assist_ready,
    grasp_wall_offset,
    initialize_tape_objects,
    make_camera_solid_rgb,
    object_pos_from_site,
    random_drop_pos,
    rotate_gripper_joint6,
    save_render_capture,
    set_object_pose,
    solve_tcp_ik_with_axis,
    tape_qpos_slice,
    tape_spec_by_name,
    tape_specs_from_task,
    task_goal_pos,
)


DEFAULT_POLICY = DEFAULT_POLICY_DIR / "fr5_bc_last.pt"
DEFAULT_ROLLOUT_LOG_DIR = Path(__file__).resolve().parent / "data" / "policy_rollouts"


def episode_step_limit(max_steps: int, frame_count: int) -> int:
    usable = max(0, int(frame_count) - 1)
    if int(max_steps) <= 0:
        return usable
    return min(int(max_steps), usable)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def read_episode_rgb(episode_dir: Path, image_file: str) -> np.ndarray:
    cv2 = require_cv2()
    bgr = cv2.imread((episode_dir / image_file).as_posix(), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read episode RGB image: {episode_dir / image_file}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def infer_training_rgb_source(payload: dict) -> str:
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    for episode in meta.get("episodes", []) or []:
        meta_path = resolve_demo_path(episode) / "meta.json"
        if not meta_path.exists():
            continue
        try:
            episode_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source = str(episode_meta.get("rgb_source", "")).strip()
        if source:
            return source
    return ""


def choose_sim_rgb_source(args, payload: dict) -> tuple[str, str]:
    requested = str(args.sim_rgb_source)
    training_source = infer_training_rgb_source(payload)
    if requested != "auto":
        return requested, training_source
    if "visual_render" in training_source:
        return "visual", training_source
    if "solid" in training_source:
        return "camera-solid", training_source
    return "camera-solid", training_source


def validate_sim_rgb_source(args, sim_rgb_source: str, training_rgb_source: str) -> None:
    if not training_rgb_source:
        return
    training_visual = "visual_render" in training_rgb_source
    training_solid = "solid" in training_rgb_source or "schematic" in training_rgb_source
    mismatch = (training_visual and sim_rgb_source != "visual") or (training_solid and sim_rgb_source == "visual")
    if not mismatch:
        return
    msg = (
        f"Policy RGB source mismatch: policy was trained on {training_rgb_source!r}, "
        f"but rollout requested {sim_rgb_source!r}."
    )
    if not bool(args.allow_rgb_source_mismatch):
        raise RuntimeError(msg + " Use the matching --sim-rgb-source for valid evaluation, or add --allow-rgb-source-mismatch only for debug.")
    print("WARNING: " + msg + " This rollout is debug-only and not valid for thesis metrics.", flush=True)


def read_rgb_file(path: Path) -> np.ndarray:
    cv2 = require_cv2()
    bgr = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read rollout RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def policy_requires_target_feature(stats: dict[str, np.ndarray]) -> bool:
    return int(np.asarray(stats.get("state_mean", np.zeros(STATE_DIM))).shape[0]) > 7


def policy_goal_feature_pos(task: dict, goal_pos: np.ndarray, object_start_pos: np.ndarray) -> np.ndarray:
    """Match the target feature used by fr5_sim_tape_pick_place training data.

    The recorded target_feature uses the drop target at object/table height, not the
    elevated goal site z. Feeding the elevated site position makes near-constant goal
    feature dimensions explode after normalization in older checkpoints.
    """
    out = np.asarray(task.get("object_drop_target_pos_m", goal_pos), dtype=np.float32).copy()
    if out.shape != (3,):
        out = np.asarray(goal_pos, dtype=np.float32).copy()
    if "object_drop_target_pos_m" not in task:
        out[2] = float(np.asarray(object_start_pos, dtype=np.float32)[2])
    return out


def target_feature_from_episode_dir(episode_dir: Path, camera_config: str | Path | None = None) -> np.ndarray:
    report_path = episode_dir / "pick_place_report.json"
    meta_path = episode_dir / "meta.json"
    source_path = report_path if report_path.exists() else meta_path
    if not source_path.exists():
        raise RuntimeError(f"Policy requires target conditioning, but no episode target metadata exists: {episode_dir}")
    meta = json.loads(source_path.read_text(encoding="utf-8"))
    object_pos = np.asarray(meta.get("object_start_pos_m", []), dtype=np.float32)
    goal_pos = np.asarray(meta.get("goal_pos_m", meta.get("object_drop_target_pos_m", [])), dtype=np.float32)
    if object_pos.shape != (3,) or goal_pos.shape != (3,):
        raise RuntimeError(f"Could not read object_start_pos_m/goal_pos_m from {source_path}")
    camera_config = meta.get("camera_config", "") or camera_config
    if not camera_config:
        raise RuntimeError(f"Could not read camera_config from {source_path}")
    return make_target_feature(object_pos, goal_pos, camera_config=camera_config)


@torch.no_grad()
def predict_delta_rad(
    model,
    stats,
    image_size: int,
    device: torch.device,
    rgb: np.ndarray,
    q_rad: np.ndarray,
    gripper_closure: float,
    target_feature: np.ndarray | None = None,
) -> np.ndarray:
    image_t = preprocess_rgb_for_policy(rgb, image_size=int(image_size)).to(device=device, dtype=torch.float32)
    joint_t = normalize_joint(q_rad, stats, gripper_closure=float(gripper_closure), target_feature=target_feature).to(
        device=device, dtype=torch.float32
    )
    action_norm = model(image_t, joint_t)
    return denormalize_action(action_norm, stats).astype(np.float32)


def clamp_arm_delta(delta_rad: np.ndarray, max_action_deg: float) -> np.ndarray:
    limit = np.deg2rad(float(max_action_deg))
    return np.clip(np.asarray(delta_rad, dtype=np.float32), -limit, limit)


def policy_lookahead_frames(args, payload: dict | None = None) -> int:
    requested = int(getattr(args, "lookahead_frames", 0) or 0)
    if requested > 0:
        return requested
    meta = (payload or {}).get("meta", {}) if isinstance(payload, dict) else {}
    return max(1, int(meta.get("lookahead_frames", 1) or 1))


def receding_horizon_step(pred_full: np.ndarray, args, lookahead_frames: int) -> tuple[np.ndarray, float]:
    """Convert an N-frame lookahead action into one frame of closed-loop control."""
    horizon = max(1, int(lookahead_frames))
    pred_full = np.asarray(pred_full, dtype=np.float32)
    pred = clamp_arm_delta(pred_full[:6] / float(horizon), args.max_action_deg)
    pred_grip = float(
        np.clip(
            float(pred_full[6]) / float(horizon),
            -float(args.max_gripper_delta),
            float(args.max_gripper_delta),
        )
    )
    return pred, pred_grip


def phase_guard_mode(args) -> str:
    mode = str(getattr(args, "phase_guard", "off") or "off")
    return "scripted" if mode == "inference" else mode


def phase_guard_active(args) -> bool:
    return phase_guard_mode(args) != "off"


def policy_ablation_mode(args) -> str:
    return str(getattr(args, "policy_ablation", "normal") or "normal")


def phase_guard_action(
    *,
    args,
    task: dict,
    model,
    data,
    body,
    qpos_ids: np.ndarray,
    arm_act_ids: np.ndarray,
    tcp_site: str,
    q_policy: np.ndarray,
    gripper_closure: float,
    vertical_reference_q: np.ndarray,
    phase: str,
    close_count: int,
    release_count: int,
    obj: np.ndarray,
    goal_pos: np.ndarray,
    start_z: float,
    attach_offset: np.ndarray,
    ik_cache: dict | None = None,
) -> tuple[np.ndarray, float, str, int, int, dict]:
    """Closed-loop phase guard for task rollout evaluation.

    This is intentionally an inference/evaluation switch. It supplies the missing
    task phase timing around the learned policy: approach, vertical grasp, close,
    lift, transfer, lower, partial release, and retreat.
    """
    tcp = np.asarray(site_position(model, data, tcp_site), dtype=np.float32)
    obj = np.asarray(obj, dtype=np.float32)
    goal_pos = np.asarray(goal_pos, dtype=np.float32)
    start_z = float(start_z)
    approach_h = float(task.get("approach_height_m", 0.15))
    grasp_h = float(task.get("grasp_height_m", 0.02))
    lift_delta = float(task.get("lift_delta_m", 0.14))
    lift_h = float(task.get("lift_height_m", grasp_h + lift_delta))
    if "release_drop_delta_m" in task:
        release_h = max(0.0, lift_h - float(task.get("release_drop_delta_m", 0.03)))
    else:
        release_h = float(task.get("release_height_m", 0.035))
    open_g = float(task.get("default_gripper_opening", 0.0))
    closed_g = float(task.get("default_gripper_closed", 1.0))
    release_g = float(task.get("release_gripper_closure", open_g))
    xy_tol = float(getattr(args, "phase_guard_xy_tol", 0.025))
    z_tol = float(getattr(args, "phase_guard_z_tol", 0.04))
    close_frames = max(1, int(getattr(args, "phase_guard_close_frames", 8)))
    release_frames = max(1, int(getattr(args, "phase_guard_release_frames", 5)))
    max_close_frames = max(close_frames, int(getattr(args, "phase_guard_max_close_frames", 30)))
    max_release_frames = max(release_frames, int(getattr(args, "phase_guard_max_release_frames", 30)))
    max_ik_iters = max(1, int(getattr(args, "phase_guard_ik_iters", 50)))

    wall_offset = grasp_wall_offset(task, obj)
    if np.linalg.norm(np.asarray(attach_offset, dtype=np.float32)) > 1e-6:
        wall_offset = -np.asarray(attach_offset, dtype=np.float32)
        wall_offset[2] = 0.0
    grasp_xy = obj + wall_offset
    drop_base = goal_pos.copy()
    drop_base[2] = start_z
    drop_xy = drop_base + wall_offset

    target_pos = grasp_xy + np.asarray([0.0, 0.0, approach_h], dtype=np.float32)
    desired_gripper = open_g
    next_phase = phase
    next_close_count = close_count
    next_release_count = release_count

    tcp_grasp_xy = float(np.linalg.norm((tcp - grasp_xy)[:2]))
    tcp_drop_xy = float(np.linalg.norm((tcp - drop_xy)[:2]))
    if phase == "approach" and tcp_grasp_xy <= xy_tol:
        next_phase = "descend"
    elif phase == "descend":
        target_pos = grasp_xy + np.asarray([0.0, 0.0, grasp_h], dtype=np.float32)
        if tcp_grasp_xy <= xy_tol and abs(float(tcp[2] - target_pos[2])) <= z_tol:
            next_phase = "close"
    elif phase == "close":
        target_pos = grasp_xy + np.asarray([0.0, 0.0, grasp_h], dtype=np.float32)
        desired_gripper = closed_g
        next_close_count = close_count + 1
        close_ready = gripper_closure >= float(task.get("attach_gripper_threshold", 0.85))
        if (next_close_count >= close_frames and close_ready) or next_close_count >= max_close_frames:
            next_phase = "lift"
    elif phase == "lift":
        target_pos = grasp_xy + np.asarray([0.0, 0.0, grasp_h + lift_delta], dtype=np.float32)
        desired_gripper = closed_g
        if abs(float(tcp[2] - target_pos[2])) <= z_tol:
            next_phase = "transfer"
    elif phase == "transfer":
        target_pos = drop_xy + np.asarray([0.0, 0.0, lift_h], dtype=np.float32)
        desired_gripper = closed_g
        if tcp_drop_xy <= xy_tol:
            next_phase = "lower"
    elif phase == "lower":
        target_pos = drop_xy + np.asarray([0.0, 0.0, release_h], dtype=np.float32)
        desired_gripper = closed_g
        if tcp_drop_xy <= xy_tol and abs(float(tcp[2] - target_pos[2])) <= z_tol:
            next_phase = "release"
    elif phase == "release":
        target_pos = drop_xy + np.asarray([0.0, 0.0, release_h], dtype=np.float32)
        desired_gripper = release_g
        next_release_count = release_count + 1
        release_ready = gripper_closure <= release_g + 0.03
        if (next_release_count >= release_frames and release_ready) or next_release_count >= max_release_frames:
            next_phase = "retreat"
    elif phase == "retreat":
        target_pos = drop_xy + np.asarray([0.0, 0.0, approach_h], dtype=np.float32)
        desired_gripper = release_g
    else:
        next_phase = "approach"

    if next_phase != phase:
        return phase_guard_action(
            args=args,
            task=task,
            model=model,
            data=data,
            body=body,
            qpos_ids=qpos_ids,
            arm_act_ids=arm_act_ids,
            tcp_site=tcp_site,
            q_policy=q_policy,
            gripper_closure=gripper_closure,
            vertical_reference_q=vertical_reference_q,
            phase=next_phase,
            close_count=next_close_count,
            release_count=next_release_count,
            obj=obj,
            goal_pos=goal_pos,
            start_z=start_z,
            attach_offset=attach_offset,
            ik_cache=ik_cache,
        )

    vertical_axis = np.asarray(task.get("gripper_vertical_axis", [0.0, 0.0, -1.0]), dtype=np.float32)
    axis_index = int(task.get("gripper_vertical_axis_index", 2))
    axis_weight = float(task.get("gripper_vertical_axis_weight", 0.02))
    axis_tol = float(task.get("gripper_vertical_axis_tol", 0.02))
    target_key = (phase, tuple(np.round(np.asarray(target_pos, dtype=np.float32), 4).tolist()))
    if ik_cache is not None and ik_cache.get("key") == target_key and "q_guard" in ik_cache:
        q_guard = np.asarray(ik_cache["q_guard"], dtype=np.float32)
    else:
        cached_seed = np.asarray(ik_cache.get("q_guard"), dtype=np.float32) if ik_cache is not None and "q_guard" in ik_cache else None
        q_seed_source = cached_seed if cached_seed is not None else np.asarray(q_policy, dtype=np.float32)
        q_seed = rotate_gripper_joint6(q_seed_source, task, np.asarray(vertical_reference_q, dtype=np.float32))
        q_guard = solve_tcp_ik_with_axis(
            model,
            data,
            body,
            qpos_ids,
            arm_act_ids,
            tcp_site,
            q_seed,
            target_pos,
            desired_axis=vertical_axis,
            axis_index=axis_index,
            axis_weight=axis_weight,
            axis_tol=axis_tol,
            max_iters=max_ik_iters,
        )
        if ik_cache is not None:
            ik_cache["key"] = target_key
            ik_cache["q_guard"] = np.asarray(q_guard, dtype=np.float32)
    pred = clamp_arm_delta(q_guard - np.asarray(q_policy, dtype=np.float32), float(args.max_action_deg))
    pred_grip = float(
        np.clip(
            desired_gripper - float(gripper_closure),
            -float(args.max_gripper_delta),
            float(args.max_gripper_delta),
        )
    )
    info = {
        "phase": phase,
        "target_pos": np.asarray(target_pos, dtype=np.float32),
        "desired_gripper": float(desired_gripper),
        "tcp_grasp_xy": float(tcp_grasp_xy),
        "tcp_drop_xy": float(tcp_drop_xy),
    }
    return pred, pred_grip, phase, next_close_count, next_release_count, info


def maybe_make_rgb_widget(render, rgb: np.ndarray, enabled: bool, width: int, height: int):
    if not enabled:
        return None
    widget = render.create_image(rgb)
    render.widgets.create_image_widget(widget, layout=Layout(left=10, top=10, width=int(width), height=int(height)))
    return widget


def make_rollout_log_path(args, *, mode: str) -> Path:
    root = resolve_demo_path(args.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    policy_name = Path(args.policy).stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = str(getattr(args, "log_suffix", "") or "").strip()
    suffix = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in suffix)
    name = f"{mode}_{policy_name}_{stamp}"
    if suffix:
        name = f"{name}_{suffix}"
    path = root / name
    if not path.with_suffix(".json").exists() and not path.with_suffix(".npz").exists():
        return path
    for idx in range(1, 10000):
        candidate = root / f"{name}_{idx:03d}"
        if not candidate.with_suffix(".json").exists() and not candidate.with_suffix(".npz").exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique rollout log path under {root}")


def save_rollout_log(base_path: Path, arrays: dict[str, list], meta: dict) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    packed = {}
    for key, values in arrays.items():
        if not values:
            continue
        try:
            packed[key] = np.asarray(values)
        except Exception:
            packed[key] = np.asarray(values, dtype=object)
    np.savez_compressed(base_path.with_suffix(".npz").as_posix(), **packed)
    with base_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved rollout log: {base_path.with_suffix('.npz')}", flush=True)
    print(f"Saved rollout meta: {base_path.with_suffix('.json')}", flush=True)


def new_rollout_arrays() -> dict[str, list]:
    return {
        "frame": [],
        "pred_action_raw": [],
        "pred_action_clipped": [],
        "pred_gripper_raw": [],
        "pred_gripper_clipped": [],
        "q_policy_target": [],
        "q_sim_actual": [],
        "q_error_target_minus_sim": [],
        "gripper_policy": [],
        "tcp_sim": [],
        "action_clipped_mask": [],
        "image_file": [],
    }


def append_common_rollout_record(
    arrays: dict[str, list],
    *,
    frame_idx: int,
    pred_full: np.ndarray,
    pred: np.ndarray,
    pred_grip: float,
    q_target: np.ndarray,
    sim_data,
    qpos_ids: np.ndarray,
    gripper_closure: float,
    tcp: np.ndarray,
    max_action_deg: float,
    image_file: str = "",
) -> None:
    raw_arm = np.asarray(pred_full[:6], dtype=np.float32)
    raw_grip = float(pred_full[6]) if np.asarray(pred_full).shape[0] > 6 else 0.0
    q_sim = np.asarray(sim_data.dof_pos[qpos_ids], dtype=np.float32)
    arrays["frame"].append(int(frame_idx))
    arrays["pred_action_raw"].append(raw_arm)
    arrays["pred_action_clipped"].append(np.asarray(pred, dtype=np.float32))
    arrays["pred_gripper_raw"].append(raw_grip)
    arrays["pred_gripper_clipped"].append(float(pred_grip))
    arrays["q_policy_target"].append(np.asarray(q_target, dtype=np.float32))
    arrays["q_sim_actual"].append(q_sim)
    arrays["q_error_target_minus_sim"].append(np.asarray(q_target, dtype=np.float32) - q_sim)
    arrays["gripper_policy"].append(float(gripper_closure))
    arrays["tcp_sim"].append(np.asarray(tcp, dtype=np.float32))
    limit = np.deg2rad(float(max_action_deg))
    arrays["action_clipped_mask"].append(np.abs(raw_arm) >= limit - 1e-7)
    arrays["image_file"].append(str(image_file))


def print_rollout_summary(arrays: dict[str, list], *, prefix: str = "Rollout") -> dict:
    frames = len(arrays.get("frame", []))
    if frames <= 0:
        print(f"{prefix}: no frames logged.", flush=True)
        return {"frames": 0}
    pred = np.asarray(arrays["pred_action_clipped"], dtype=np.float64)
    raw = np.asarray(arrays["pred_action_raw"], dtype=np.float64)
    qerr = np.asarray(arrays["q_error_target_minus_sim"], dtype=np.float64)
    tcp = np.asarray(arrays["tcp_sim"], dtype=np.float64)
    clipped = np.asarray(arrays["action_clipped_mask"], dtype=bool)
    summary = {
        "frames": int(frames),
        "max_abs_pred_deg": np.rad2deg(np.max(np.abs(pred), axis=0)).round(6).tolist(),
        "max_abs_raw_pred_deg": np.rad2deg(np.max(np.abs(raw), axis=0)).round(6).tolist(),
        "clip_fraction": float(np.mean(clipped)),
        "max_sim_tracking_error_deg": np.rad2deg(np.max(np.abs(qerr), axis=0)).round(6).tolist(),
        "max_tcp_step_m": float(np.max(np.linalg.norm(np.diff(tcp, axis=0), axis=1))) if tcp.shape[0] > 1 else 0.0,
        "min_tcp_z_m": float(np.min(tcp[:, 2])),
        "final_tcp_m": tcp[-1].round(6).tolist(),
    }
    print(f"{prefix} summary:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def print_xml_control_diagnostics(config: dict) -> dict:
    xml_path = resolve_demo_path(config["model_xml"])
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as exc:
        print(f"Warning: could not parse control diagnostics from {xml_path}: {exc}", flush=True)
        return {}
    rows = []
    for actuator in root.findall(".//actuator/position"):
        name = actuator.get("name", "")
        if not name.startswith("j"):
            continue
        kp = float(actuator.get("kp", "nan"))
        kv = float(actuator.get("kv", "nan"))
        forcerange = actuator.get("forcerange", "")
        ctrlrange = actuator.get("ctrlrange", "")
        rows.append({"name": name, "kp": kp, "kv": kv, "forcerange": forcerange, "ctrlrange": ctrlrange})
    print("Sim position-actuator diagnostics:", flush=True)
    for row in rows:
        print(
            f"  {row['name']}: kp={row['kp']:.3g}, kv={row['kv']:.3g}, "
            f"forcerange={row['forcerange']}, ctrlrange={row['ctrlrange']}",
            flush=True,
        )
    print(
        "  If q_policy_target is sane but q_sim_actual collapses or lags badly, suspect actuator gains/force limits/contact.",
        flush=True,
    )
    return {"xml": xml_path.as_posix(), "position_actuators": rows}


def run_episode_rollout(args, model_policy, stats, image_size: int, device: torch.device, payload: dict) -> None:
    episode_dir = resolve_demo_path(args.episode)
    if not (episode_dir / "states.npz").exists():
        episode_dir = resolve_demo_path(DEFAULT_IL_DEMO_DIR / args.episode)
    pack = load_episode_npz(episode_dir)
    q_expert = np.asarray(pack["joint_rad"], dtype=np.float32)
    action_expert = np.asarray(pack["action_joint_delta_rad"], dtype=np.float32)
    grip_expert = (
        np.asarray(pack["gripper_closure"], dtype=np.float32)
        if "gripper_closure" in pack
        else np.zeros(q_expert.shape[0], dtype=np.float32)
    )
    grip_action_expert = (
        np.asarray(pack["action_gripper_delta"], dtype=np.float32)
        if "action_gripper_delta" in pack
        else np.zeros(q_expert.shape[0], dtype=np.float32)
    )
    image_files = [str(item) for item in np.asarray(pack["image_files"]).tolist()]
    config = load_config(args.config)
    runtime = build_runtime(config)
    sim_model, sim_data, body, qpos_ids, arm_act_ids, gripper_act_ids = runtime
    control_diag = print_xml_control_diagnostics(config) if bool(args.diagnose_control) else {}
    target_feature = (
        target_feature_from_episode_dir(episode_dir, args.camera_config) if policy_requires_target_feature(stats) else None
    )
    q_policy = q_expert[0].copy()
    grip_policy = float(grip_expert[0])
    steps_per_frame = max(1, round((1.0 / float(args.hz)) / float(sim_model.options.timestep)))
    lookahead_frames = policy_lookahead_frames(args, payload)
    print(
        f"Episode rollout timing: hz={float(args.hz):.3f}, lookahead_frames={lookahead_frames}, "
        f"execution_lead={lookahead_frames / float(args.hz):.3f}s, receding_horizon_step=1/{lookahead_frames}",
        flush=True,
    )

    losses = []
    arrays = new_rollout_arrays()
    arrays.update(
        {
            "q_expert": [],
            "expert_action": [],
            "expert_gripper": [],
            "expert_gripper_action": [],
            "action_error": [],
            "q_error_policy_minus_expert": [],
        }
    )
    log_base = make_rollout_log_path(args, mode="episode") if bool(args.save_rollout_data) else None
    if args.no_window:
        for idx in range(episode_step_limit(args.max_steps, q_expert.shape[0])):
            rgb = read_episode_rgb(episode_dir, image_files[idx])
            pred_full = predict_delta_rad(
                model_policy, stats, image_size, device, rgb, q_policy, grip_policy, target_feature=target_feature
            )
            pred, pred_grip = receding_horizon_step(pred_full, args, lookahead_frames)
            target_idx = min(idx + lookahead_frames, q_expert.shape[0] - 1)
            target = np.concatenate(
                [
                    (q_expert[target_idx] - q_expert[idx]) / float(lookahead_frames),
                    np.asarray([(float(grip_expert[target_idx]) - float(grip_expert[idx])) / float(lookahead_frames)], dtype=np.float32),
                ]
            )
            pred_eval = np.concatenate([pred, np.asarray([pred_grip], dtype=np.float32)])
            losses.append(float(np.mean((pred_eval - target) ** 2)))
            q_policy = q_policy + pred
            grip_policy = float(np.clip(grip_policy + pred_grip, 0.0, 1.0))
            set_arm_qpos_and_ctrl(sim_data, sim_model, body, qpos_ids, arm_act_ids, q_policy)
            set_gripper(sim_data, sim_model, body, gripper_act_ids, grip_policy)
            for _ in range(steps_per_frame):
                sim_step(sim_model, sim_data)
            tcp = site_position(sim_model, sim_data, config["tcp_site"])
            append_common_rollout_record(
                arrays,
                frame_idx=idx,
                pred_full=pred_full,
                pred=pred,
                pred_grip=pred_grip,
                q_target=q_policy,
                sim_data=sim_data,
                qpos_ids=qpos_ids,
                gripper_closure=grip_policy,
                tcp=tcp,
                max_action_deg=float(args.max_action_deg),
                image_file=image_files[idx],
            )
            arrays["q_expert"].append(q_expert[idx])
            arrays["expert_action"].append(target[:6])
            arrays["expert_gripper"].append(float(grip_expert[idx]))
            arrays["expert_gripper_action"].append(float(target[6]))
            arrays["action_error"].append(pred_eval - target)
            arrays["q_error_policy_minus_expert"].append(q_policy - q_expert[min(idx + 1, q_expert.shape[0] - 1)])
            if idx % max(1, int(args.log_every)) == 0:
                print(
                    f"frame={idx:05d} pred_deg={np.rad2deg(pred).round(3).tolist()} "
                    f"expert_step_deg={np.rad2deg(target[:6]).round(3).tolist()} "
                    f"qerr_next_deg={np.rad2deg(arrays['q_error_policy_minus_expert'][-1]).round(3).tolist()} "
                    f"sim_track_err_deg={np.rad2deg(arrays['q_error_target_minus_sim'][-1]).round(3).tolist()} "
                    f"tcp={np.asarray(tcp).round(4).tolist()}",
                    flush=True,
                )
        print(f"Episode policy check: frames={len(losses)}, mse_rad2={np.mean(losses):.8f}")
        summary = print_rollout_summary(arrays, prefix="Episode rollout")
        if log_base is not None:
            save_rollout_log(
                log_base,
                arrays,
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "mode": "episode",
                    "policy": resolve_demo_path(args.policy).as_posix(),
                    "episode": episode_dir.as_posix(),
                    "config": resolve_demo_path(args.config).as_posix(),
                    "mse_rad2": float(np.mean(losses)) if losses else None,
                    "lookahead_frames": int(lookahead_frames),
                    "execution_lead_s": float(lookahead_frames / float(args.hz)),
                    "receding_horizon": True,
                    "summary": summary,
                    "control_diagnostics": control_diag,
                    "notes": "q_error_target_minus_sim reveals simulator actuator/contact collapse; q_error_policy_minus_expert reveals closed-loop policy drift.",
                },
            )
        return

    with RenderApp() as render:
        render.launch(sim_model)
        first_rgb = read_episode_rgb(episode_dir, image_files[0])
        rgb_widget = maybe_make_rgb_widget(render, first_rgb, bool(args.rgb_widget), args.rgb_widget_width, args.rgb_widget_height)
        frame = 0
        while not render.is_closed and frame < episode_step_limit(args.max_steps, q_expert.shape[0]):
            rgb = read_episode_rgb(episode_dir, image_files[frame])
            pred_full = predict_delta_rad(
                model_policy, stats, image_size, device, rgb, q_policy, grip_policy, target_feature=target_feature
            )
            pred, pred_grip = receding_horizon_step(pred_full, args, lookahead_frames)
            target_idx = min(frame + lookahead_frames, q_expert.shape[0] - 1)
            target = np.concatenate(
                [
                    (q_expert[target_idx] - q_expert[frame]) / float(lookahead_frames),
                    np.asarray([(float(grip_expert[target_idx]) - float(grip_expert[frame])) / float(lookahead_frames)], dtype=np.float32),
                ]
            )
            pred_eval = np.concatenate([pred, np.asarray([pred_grip], dtype=np.float32)])
            losses.append(float(np.mean((pred_eval - target) ** 2)))
            q_policy = q_policy + pred
            grip_policy = float(np.clip(grip_policy + pred_grip, 0.0, 1.0))
            set_arm_qpos_and_ctrl(sim_data, sim_model, body, qpos_ids, arm_act_ids, q_policy)
            set_gripper(sim_data, sim_model, body, gripper_act_ids, grip_policy)
            for _ in range(steps_per_frame):
                sim_step(sim_model, sim_data)
            tcp = site_position(sim_model, sim_data, config["tcp_site"])
            append_common_rollout_record(
                arrays,
                frame_idx=frame,
                pred_full=pred_full,
                pred=pred,
                pred_grip=pred_grip,
                q_target=q_policy,
                sim_data=sim_data,
                qpos_ids=qpos_ids,
                gripper_closure=grip_policy,
                tcp=tcp,
                max_action_deg=float(args.max_action_deg),
                image_file=image_files[frame],
            )
            arrays["q_expert"].append(q_expert[frame])
            arrays["expert_action"].append(target[:6])
            arrays["expert_gripper"].append(float(grip_expert[frame]))
            arrays["expert_gripper_action"].append(float(target[6]))
            arrays["action_error"].append(pred_eval - target)
            arrays["q_error_policy_minus_expert"].append(q_policy - q_expert[min(frame + 1, q_expert.shape[0] - 1)])
            if rgb_widget is not None:
                rgb_widget.pixels = rgb
            if frame % max(1, int(args.log_every)) == 0:
                print(
                    f"frame={frame:05d} pred_deg={np.rad2deg(pred).round(3).tolist()} "
                    f"pred_gripper={pred_grip:.3f} gripper={grip_policy:.3f} "
                    f"expert_step_deg={np.rad2deg(target[:6]).round(3).tolist()} "
                    f"qerr_next_deg={np.rad2deg(arrays['q_error_policy_minus_expert'][-1]).round(3).tolist()} "
                    f"sim_track_err_deg={np.rad2deg(arrays['q_error_target_minus_sim'][-1]).round(3).tolist()} "
                    f"tcp={tcp.round(4).tolist()}",
                    flush=True,
                )
            render.sync(sim_data)
            frame += 1
            time.sleep(1.0 / float(args.hz))
    if losses:
        print(f"Episode policy check: frames={len(losses)}, mse_rad2={np.mean(losses):.8f}")
    summary = print_rollout_summary(arrays, prefix="Episode rollout")
    if log_base is not None:
        save_rollout_log(
            log_base,
            arrays,
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "mode": "episode",
                "policy": resolve_demo_path(args.policy).as_posix(),
                "episode": episode_dir.as_posix(),
                "config": resolve_demo_path(args.config).as_posix(),
                "mse_rad2": float(np.mean(losses)) if losses else None,
                "lookahead_frames": int(lookahead_frames),
                "execution_lead_s": float(lookahead_frames / float(args.hz)),
                "receding_horizon": True,
                "summary": summary,
                "control_diagnostics": control_diag,
                "notes": "q_error_target_minus_sim reveals simulator actuator/contact collapse; q_error_policy_minus_expert reveals closed-loop policy drift.",
            },
        )


def run_live_rollout(args, model_policy, stats, image_size: int, device: torch.device, payload: dict) -> None:
    config = load_config(args.config)
    sim_model, sim_data, body, qpos_ids, arm_act_ids, gripper_act_ids = build_runtime(config)
    control_diag = print_xml_control_diagnostics(config) if bool(args.diagnose_control) else {}
    steps_per_frame = max(1, round((1.0 / float(args.hz)) / float(sim_model.options.timestep)))
    lookahead_frames = policy_lookahead_frames(args, payload)
    print(
        f"Live rollout timing: hz={float(args.hz):.3f}, lookahead_frames={lookahead_frames}, "
        f"execution_lead={lookahead_frames / float(args.hz):.3f}s, receding_horizon_step=1/{lookahead_frames}",
        flush=True,
    )
    rgb_source = RealRgbSource(
        source="live",
        image_path=None,
        width=int(args.real_rgb_width),
        height=int(args.real_rgb_height),
        fps=int(args.real_rgb_fps),
        allow_fallback=bool(args.allow_latest_fallback),
    )
    arm = FairinoArmClient(args.robot_ip, speed_percent=float(args.speed_percent))
    canceller = MotionCancelHandler() if args.execute_real else None
    servo_started = False
    gripper = None
    gripper_closure = float(np.clip(args.initial_gripper_closure, 0.0, 1.0))
    last_gripper_sent = None
    arrays = new_rollout_arrays()
    arrays.update({"q_real_readback": [], "q_real_target_delta": []})
    log_base = make_rollout_log_path(args, mode="live") if bool(args.save_rollout_data) else None

    try:
        arm.connect()
        err = arm.get_robot_error_code()
        if err is not None and any(int(v) != 0 for v in err):
            if args.execute_real:
                raise RuntimeError(f"Controller reports error code {err}; refusing policy execution.")
            print(f"Warning: controller reports error code {err}; live rollout will mirror readback only.", flush=True)
        if args.execute_real:
            if canceller is not None:
                canceller.set_arm(arm)
                canceller.install()
            if args.prepare_controller:
                arm.set_mode(0)
                arm.robot_enable(1)
                time.sleep(0.5)
            arm.servo_start()
            servo_started = True
            if canceller is not None:
                canceller.set_servo_started(True)
        if args.execute_gripper:
            gripper = Robotiq2F85ModbusRtuClient(
                args.gripper_port,
                baudrate=int(args.gripper_baudrate),
                slave_id=int(args.gripper_slave_id),
                timeout=float(args.gripper_timeout),
                retries=int(args.gripper_retries),
            )
            gripper.connect()
            gripper.activate()
            gripper.command_closure(gripper_closure, speed=int(args.gripper_speed), force=int(args.gripper_force))
            last_gripper_sent = gripper_closure
        first_rgb = rgb_source.start()

        def step_once(frame_idx: int, rgb: np.ndarray) -> bool:
            nonlocal gripper_closure, last_gripper_sent
            if policy_requires_target_feature(stats):
                raise RuntimeError(
                    "This policy was trained with target-conditioning. Live rollout needs a detector or tracker to provide "
                    "the current tape target feature; use sim-task rollout or an unconditioned policy for live mode."
                )
            q_deg = arm.get_actual_joint_deg()
            if q_deg is None:
                raise RuntimeError("Could not read current FR5 joint angles during live policy rollout.")
            q_rad = np.deg2rad(np.asarray(q_deg, dtype=np.float32))
            pred_full = predict_delta_rad(model_policy, stats, image_size, device, rgb, q_rad, gripper_closure)
            pred, pred_grip = receding_horizon_step(pred_full, args, lookahead_frames)
            q_target = q_rad + pred
            gripper_closure = float(np.clip(gripper_closure + pred_grip, 0.0, 1.0))
            set_arm_qpos_and_ctrl(sim_data, sim_model, body, qpos_ids, arm_act_ids, q_target)
            set_gripper(sim_data, sim_model, body, gripper_act_ids, gripper_closure)
            for _ in range(steps_per_frame):
                sim_step(sim_model, sim_data)
            tcp = site_position(sim_model, sim_data, config["tcp_site"])
            append_common_rollout_record(
                arrays,
                frame_idx=frame_idx,
                pred_full=pred_full,
                pred=pred,
                pred_grip=pred_grip,
                q_target=q_target,
                sim_data=sim_data,
                qpos_ids=qpos_ids,
                gripper_closure=gripper_closure,
                tcp=tcp,
                max_action_deg=float(args.max_action_deg),
            )
            arrays["q_real_readback"].append(q_rad)
            arrays["q_real_target_delta"].append(q_target - q_rad)
            if float(tcp[2]) < float(args.min_tcp_z):
                raise RuntimeError(f"Policy target failed sim safety: tcp_z={float(tcp[2]):.4f} < {float(args.min_tcp_z):.4f}")
            if args.execute_real:
                arm.servo_j(np.rad2deg(q_target), idx=frame_idx, cmd_t=1.0 / float(args.hz), vel=float(args.servo_vel))
            if gripper is not None and (
                last_gripper_sent is None or abs(gripper_closure - float(last_gripper_sent)) >= float(args.gripper_send_epsilon)
            ):
                gripper.command_closure(gripper_closure, speed=int(args.gripper_speed), force=int(args.gripper_force))
                last_gripper_sent = gripper_closure
            if frame_idx % max(1, int(args.log_every)) == 0:
                print(
                    f"frame={frame_idx:05d} pred_deg={np.rad2deg(pred).round(3).tolist()} "
                    f"pred_gripper={pred_grip:.3f} gripper={gripper_closure:.3f} "
                    f"sim_track_err_deg={np.rad2deg(arrays['q_error_target_minus_sim'][-1]).round(3).tolist()} "
                    f"tcp={tcp.round(4).tolist()} real={'yes' if args.execute_real else 'no'}",
                    flush=True,
                )
            return True

        if args.no_window:
            rgb = first_rgb
            for frame_idx in range(int(args.max_steps)):
                if canceller is not None:
                    canceller.check()
                latest = rgb_source.read(timeout_ms=max(20, int(1000 / max(1, int(args.real_rgb_fps)))))
                if latest is not None:
                    rgb = latest
                step_once(frame_idx, rgb)
                time.sleep(1.0 / float(args.hz))
            return

        with RenderApp() as render:
            render.launch(sim_model)
            rgb_widget = maybe_make_rgb_widget(render, first_rgb, bool(args.rgb_widget), args.rgb_widget_width, args.rgb_widget_height)
            render.sync(sim_data)
            rgb = first_rgb
            for frame_idx in range(int(args.max_steps)):
                if render.is_closed:
                    break
                if canceller is not None:
                    canceller.check()
                latest = rgb_source.read(timeout_ms=max(20, int(1000 / max(1, int(args.real_rgb_fps)))))
                if latest is not None:
                    rgb = latest
                step_once(frame_idx, rgb)
                if rgb_widget is not None:
                    rgb_widget.pixels = rgb
                render.sync(sim_data)
                time.sleep(1.0 / float(args.hz))
    except KeyboardInterrupt:
        print("Policy rollout cancelled.", flush=True)
    finally:
        summary = print_rollout_summary(arrays, prefix="Live rollout")
        if log_base is not None:
            save_rollout_log(
                log_base,
                arrays,
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "mode": "live",
                    "policy": resolve_demo_path(args.policy).as_posix(),
                    "config": resolve_demo_path(args.config).as_posix(),
                    "execute_real": bool(args.execute_real),
                    "execute_gripper": bool(args.execute_gripper),
                    "lookahead_frames": int(lookahead_frames),
                    "execution_lead_s": float(lookahead_frames / float(args.hz)),
                    "receding_horizon": True,
                    "summary": summary,
                    "control_diagnostics": control_diag,
                    "notes": "q_real_target_delta is commanded ServoJ delta; q_error_target_minus_sim isolates simulator tracking/contact collapse.",
                },
            )
        if servo_started:
            arm.servo_end_best_effort()
        if gripper is not None:
            gripper.close()
        arm.close()
        rgb_source.stop()


def run_sim_task_rollout(args, model_policy, stats, image_size: int, device: torch.device, payload: dict) -> None:
    config = load_config(args.config)
    task = config.get("sim_tape_pick_place", {})
    sim_rgb_source, training_rgb_source = choose_sim_rgb_source(args, payload)
    validate_sim_rgb_source(args, sim_rgb_source, training_rgb_source)
    if sim_rgb_source == "visual" and bool(args.no_window):
        if bool(args.allow_rgb_source_mismatch):
            print(
                "Warning: policy was trained on visual RenderApp RGB, but --no-window was requested. "
                "Falling back to camera-solid debug RGB because --allow-rgb-source-mismatch is set.",
                flush=True,
            )
            sim_rgb_source = "camera-solid"
        else:
            raise RuntimeError(
                "This policy appears to have been trained on visual RenderApp RGB "
                f"({training_rgb_source or 'unknown source'}), but sim-task rollout was requested with --no-window. "
                "Use --sim-rgb-source visual without --no-window for a valid closed-loop visual evaluation, "
                "or pass --sim-rgb-source camera-solid --allow-rgb-source-mismatch only for a fast debug run."
            )
    runtime_camera_config = args.camera_config if sim_rgb_source == "visual" else None
    model, data, body, qpos_ids, arm_act_ids, gripper_act_ids = build_runtime(config, runtime_camera_config)
    control_diag = print_xml_control_diagnostics(config) if bool(args.diagnose_control) else {}
    rng = np.random.default_rng(int(args.sim_seed))
    steps = int(args.max_steps) if int(args.max_steps) > 0 else 100
    steps_per_frame = max(1, round((1.0 / float(args.hz)) / float(model.options.timestep)))
    lookahead_frames = policy_lookahead_frames(args, payload)
    log_base = make_rollout_log_path(args, mode="simtask") if bool(args.save_rollout_data) else None
    arrays = new_rollout_arrays()
    arrays.update(
        {
            "object_pos": [],
            "goal_pos": [],
            "object_goal_error_xy": [],
            "tcp_object_dist_xy": [],
            "tcp_goal_dist_xy": [],
            "attached": [],
            "phase_guard_phase": [],
            "phase_guard_target_pos": [],
            "phase_guard_desired_gripper": [],
            "active_object_name": [],
            "active_object_index": [],
            "active_object_pos": [],
            "active_goal_pos": [],
        }
    )

    if args.sim_start_random_radius is not None:
        task = dict(task)
        task["start_random_radius_m"] = float(args.sim_start_random_radius)
        task.pop("tape_objects", None)
    tape_start_positions, _primary_qslice, tape_assigned_slots = initialize_tape_objects(data, model, task, rng)
    specs_by_name = tape_spec_by_name(task)
    tape_order = [str(spec["name"]) for spec in tape_specs_from_task(task)]
    primary_name = str(task.get("object_name", tape_order[0] if tape_order else "red_tape_roll"))
    object_sites = {name: str(specs_by_name[name].get("site", f"{name}_site")) for name in tape_order}
    object_qslices = {name: tape_qpos_slice(model, specs_by_name[name]) for name in tape_order}
    start_pos = np.asarray(tape_start_positions[primary_name], dtype=np.float32)
    goal_pos = task_goal_pos(model, data, task)
    stack_on_goal = bool(task.get("stack_on_goal", len(tape_order) > 1))
    stack_spacing = float(task.get("stack_spacing_m", 0.046))
    shared_drop_xy_offset = np.zeros(2, dtype=np.float32)
    if stack_on_goal:
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        r = float(task.get("drop_random_radius_m", 0.0)) * math.sqrt(float(rng.uniform(0.0, 1.0)))
        shared_drop_xy_offset[:] = [r * math.cos(theta), r * math.sin(theta)]
    drop_positions: dict[str, np.ndarray] = {}
    for seq_idx, name in enumerate(tape_order):
        obj_start = np.asarray(tape_start_positions[name], dtype=np.float32)
        drop_base = np.asarray(goal_pos, dtype=np.float32).copy()
        drop_base[2] = float(obj_start[2]) + (stack_spacing * seq_idx if stack_on_goal else 0.0)
        if stack_on_goal:
            drop_base[0] += float(task.get("drop_delta_x_m", 0.0))
            drop_base[:2] += shared_drop_xy_offset
            drop_positions[name] = drop_base.astype(np.float32)
        else:
            drop_positions[name] = random_drop_pos(
                drop_base,
                float(task.get("drop_delta_x_m", 0.0)),
                float(task.get("drop_random_radius_m", 0.0)),
                rng,
            )
    q_policy = np.asarray(config["initial_qpos"], dtype=np.float32)
    gripper_closure = float(np.clip(args.initial_gripper_closure, 0.0, 1.0))
    set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q_policy)
    set_gripper(data, model, body, gripper_act_ids, gripper_closure)
    for _ in range(50):
        sim_step(model, data)

    attach_assist = bool(task.get("attach_assist_default", False) if args.sim_attach_assist is None else args.sim_attach_assist)
    attach_offset = np.zeros(3, dtype=np.float32)
    attach_xy_distance = float(args.sim_attach_distance if args.sim_attach_distance is not None else task.get("attach_distance_m", 0.08))
    attach_z_distance = float(args.sim_attach_z_distance if args.sim_attach_z_distance is not None else task.get("attach_z_distance_m", 0.055))
    attach_gripper_threshold = float(
        args.sim_attach_gripper_threshold
        if args.sim_attach_gripper_threshold is not None
        else task.get("attach_gripper_threshold", 0.85)
    )
    release_threshold = float(args.sim_release_gripper_threshold)
    if release_threshold < 0.0:
        release_threshold = float(task.get("release_gripper_closure", 0.55)) + 0.03
    active_index = 0
    active_name = tape_order[active_index]
    active_site = object_sites[active_name]
    active_qslice = object_qslices[active_name]
    active_start_pos = np.asarray(tape_start_positions[active_name], dtype=np.float32)
    active_drop_pos = np.asarray(drop_positions[active_name], dtype=np.float32)
    completed_objects: set[str] = set()
    released_objects: set[str] = set()
    attached_name = ""
    guard_phase = "approach"
    guard_close_count = 0
    guard_release_count = 0
    guard_last_phase = ""
    guard_phase_frame_count = 0
    guard_reference_q = np.asarray(config["initial_qpos"], dtype=np.float32)
    guard_mode = phase_guard_mode(args)
    guard_enabled = phase_guard_active(args)
    guard_ik_cache: dict = {}
    guard_object_ref = active_start_pos.copy()
    prev_pred = np.zeros(6, dtype=np.float32)
    prev_pred_grip = 0.0
    ablation_mode = policy_ablation_mode(args)
    ablation_rng = np.random.default_rng(int(args.policy_ablation_seed) + int(args.sim_seed))

    print(
        f"Sim autonomous rollout: active={active_name} start={start_pos.round(4).tolist()} goal={goal_pos.round(4).tolist()} "
        f"steps={steps} attach_assist={attach_assist} sim_rgb_source={sim_rgb_source} "
        f"training_rgb_source={training_rgb_source or 'unknown'} lookahead_frames={lookahead_frames} "
        f"execution_lead={lookahead_frames / float(args.hz):.3f}s receding_horizon_step=1/{lookahead_frames} "
        f"phase_guard={guard_mode} policy_ablation={ablation_mode}",
        flush=True,
    )

    render_context = None
    render = None
    rgb_widget = None
    rcam = None
    render_camera = None
    visual_rgb_dir = None
    try:
        if sim_rgb_source == "visual":
            render_camera_name = camera_name_from_config(args.camera_config)
            render_camera = find_camera(model, render_camera_name)
            if render_camera is None:
                raise RuntimeError(
                    f"Could not find calibrated render camera {render_camera_name!r}. "
                    "Check --camera-config and astra_camera.json."
                )
            visual_rgb_dir = (
                (log_base.parent / f"{log_base.name}_rgb") if log_base is not None else (resolve_demo_path(args.log_dir) / "simtask_visual_rgb")
            )
            visual_rgb_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_window or sim_rgb_source == "visual":
            render_context = RenderApp()
            render = render_context.__enter__()
            render.launch(model)
            if render_camera is not None:
                render.set_main_camera(render_camera)
            rcam = render.get_camera(0)
            if sim_rgb_source == "visual" and rcam is None:
                raise RuntimeError("RenderApp camera 0 is unavailable; cannot capture visual policy RGB.")
            if sim_rgb_source == "visual":
                first_path = visual_rgb_dir / "policy_rgb_000000_preview.png"
                save_render_capture(render, rcam, data, first_path)
                first_rgb = read_rgb_file(first_path)
            else:
                first_rgb = make_camera_solid_rgb(args.camera_config, object_pos=object_pos_from_site(model, data, active_site))
            rgb_widget = maybe_make_rgb_widget(
                render,
                first_rgb,
                bool(args.rgb_widget),
                int(args.rgb_widget_width),
                int(args.rgb_widget_height),
            )
            render.sync(data)
            print("Sim task RenderApp window launched. Close the window to stop rollout early.", flush=True)

        for frame_idx in range(steps):
            if render is not None and render.is_closed:
                break
            obj = object_pos_from_site(model, data, active_site)
            image_file = ""
            if sim_rgb_source == "visual":
                assert visual_rgb_dir is not None
                image_path = visual_rgb_dir / f"policy_rgb_{frame_idx:06d}.png"
                save_render_capture(render, rcam, data, image_path)
                rgb = read_rgb_file(image_path)
                image_file = image_path.as_posix()
            else:
                rgb = make_camera_solid_rgb(args.camera_config, object_pos=obj)
            target_feature = (
                make_target_feature(obj, active_drop_pos, camera_config=args.camera_config) if policy_requires_target_feature(stats) else None
            )
            rgb_policy = rgb
            target_feature_policy = target_feature
            if ablation_mode == "zero_image":
                rgb_policy = np.zeros_like(rgb)
            if ablation_mode == "zero_target" and target_feature_policy is not None:
                target_feature_policy = np.zeros_like(target_feature_policy, dtype=np.float32)
            pred_full = predict_delta_rad(
                model_policy,
                stats,
                image_size,
                device,
                rgb_policy,
                q_policy,
                gripper_closure,
                target_feature=target_feature_policy,
            )
            if ablation_mode == "zero_policy":
                pred_full = np.zeros_like(pred_full, dtype=np.float32)
            elif ablation_mode == "random_policy":
                pred_full = np.concatenate(
                    [
                        ablation_rng.normal(
                            loc=0.0,
                            scale=np.deg2rad(float(args.max_action_deg)) * 0.5,
                            size=6,
                        ).astype(np.float32),
                        np.asarray(
                            [
                                float(
                                    ablation_rng.normal(
                                        loc=0.0,
                                        scale=max(0.01, float(args.max_gripper_delta) * max(1, int(lookahead_frames))),
                                    )
                                )
                            ],
                            dtype=np.float32,
                        ),
                    ]
                ).astype(np.float32)
            pred, pred_grip = receding_horizon_step(pred_full, args, lookahead_frames)
            guard_info = {}
            if guard_enabled:
                guard_pred, guard_pred_grip, guard_phase, guard_close_count, guard_release_count, guard_info = phase_guard_action(
                    args=args,
                    task=task,
                    model=model,
                    data=data,
                    body=body,
                    qpos_ids=qpos_ids,
                    arm_act_ids=arm_act_ids,
                    tcp_site=config["tcp_site"],
                    q_policy=q_policy,
                    gripper_closure=gripper_closure,
                    vertical_reference_q=guard_reference_q,
                    phase=guard_phase,
                    close_count=guard_close_count,
                    release_count=guard_release_count,
                    obj=guard_object_ref if attached_name == active_name else obj,
                    goal_pos=active_drop_pos,
                    start_z=float(active_start_pos[2]),
                    attach_offset=attach_offset if attached_name == active_name else np.zeros(3, dtype=np.float32),
                    ik_cache=guard_ik_cache,
                )
                phase_name_now = str(guard_info.get("phase", ""))
                if phase_name_now == guard_last_phase:
                    guard_phase_frame_count += 1
                else:
                    guard_last_phase = phase_name_now
                    guard_phase_frame_count = 1
                max_phase_frames = int(args.phase_guard_max_phase_frames)
                if max_phase_frames > 0 and guard_phase_frame_count >= max_phase_frames:
                    next_phase_by_stall = {
                        "approach": "descend",
                        "descend": "close",
                        "lift": "transfer",
                        "transfer": "lower",
                        "lower": "release",
                        "retreat": "approach",
                    }
                    if phase_name_now in next_phase_by_stall:
                        guard_phase = next_phase_by_stall[phase_name_now]
                        guard_last_phase = guard_phase
                        guard_phase_frame_count = 0
                        guard_ik_cache.clear()
                        print(f"Phase guard advanced after stall: {phase_name_now} -> {guard_phase}", flush=True)
                if guard_mode == "scripted":
                    residual_scale = float(args.phase_guard_residual_scale)
                    if residual_scale > 0.0:
                        residual = clamp_arm_delta(pred, float(args.phase_guard_residual_max_deg)) * residual_scale
                        guard_pred = clamp_arm_delta(guard_pred + residual, float(args.max_action_deg))
                        guard_pred_grip = float(
                            np.clip(
                                guard_pred_grip + residual_scale * float(pred_grip),
                                -float(args.max_gripper_delta),
                                float(args.max_gripper_delta),
                            )
                        )
                    pred, pred_grip = guard_pred, guard_pred_grip
                elif guard_mode == "soft":
                    soft_correction = clamp_arm_delta(guard_pred, float(args.phase_guard_soft_max_deg))
                    pred = clamp_arm_delta(pred + float(args.phase_guard_soft_gain) * soft_correction, float(args.max_action_deg))
                    phase_name = str(guard_info.get("phase", ""))
                    if phase_name in {"approach", "descend"}:
                        pred_grip = min(float(pred_grip), 0.0)
                    elif phase_name in {"close", "lift", "transfer", "lower"}:
                        pred_grip = max(float(pred_grip), float(guard_pred_grip))
                    elif phase_name in {"release", "retreat"}:
                        pred_grip = min(float(pred_grip), float(guard_pred_grip))
                    pred_grip = float(
                        np.clip(
                            pred_grip,
                            -float(args.max_gripper_delta),
                            float(args.max_gripper_delta),
                        )
                    )
            action_ema = float(args.action_ema)
            if action_ema > 0.0:
                action_ema = float(np.clip(action_ema, 0.0, 0.98))
                pred = ((action_ema * prev_pred) + ((1.0 - action_ema) * np.asarray(pred, dtype=np.float32))).astype(np.float32)
            max_action_step_deg = float(args.max_action_step_deg)
            if max_action_step_deg > 0.0:
                step_limit = np.deg2rad(max_action_step_deg)
                pred = (prev_pred + np.clip(np.asarray(pred, dtype=np.float32) - prev_pred, -step_limit, step_limit)).astype(np.float32)
            gripper_ema = float(args.gripper_ema)
            if gripper_ema > 0.0:
                gripper_ema = float(np.clip(gripper_ema, 0.0, 0.98))
                pred_grip = float(gripper_ema * prev_pred_grip + (1.0 - gripper_ema) * float(pred_grip))
            prev_pred = np.asarray(pred, dtype=np.float32).copy()
            prev_pred_grip = float(pred_grip)
            q_policy = q_policy + pred
            gripper_closure = float(np.clip(gripper_closure + pred_grip, 0.0, 1.0))
            set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q_policy)
            set_gripper(data, model, body, gripper_act_ids, gripper_closure)
            tcp = np.asarray(site_position(model, data, config["tcp_site"]), dtype=np.float32)
            obj = object_pos_from_site(model, data, active_site)
            if attach_assist and active_name not in released_objects:
                if attached_name != active_name and attach_assist_ready(
                    tcp=tcp,
                    obj=obj,
                    gripper_closure=gripper_closure,
                    gripper_threshold=attach_gripper_threshold,
                    xy_distance=attach_xy_distance,
                    z_distance=attach_z_distance,
                ):
                    attach_offset = obj - tcp
                    if float(np.linalg.norm(attach_offset[:2])) > attach_xy_distance:
                        attach_offset = np.asarray([0.0, 0.0, -0.015], dtype=np.float32)
                    attached_name = active_name
                    guard_object_ref = obj.copy()
                if attached_name == active_name:
                    if gripper_closure <= release_threshold:
                        released_objects.add(active_name)
                        attached_name = ""
                    else:
                        set_object_pose(data, model, active_qslice, tcp + attach_offset)
            for _ in range(steps_per_frame):
                sim_step(model, data)
            tcp = np.asarray(site_position(model, data, config["tcp_site"]), dtype=np.float32)
            obj = object_pos_from_site(model, data, active_site)
            active_place_error = float(np.linalg.norm(obj[:2] - active_drop_pos[:2]))
            active_done = active_name in released_objects and active_place_error <= float(task.get("place_success_radius_m", 0.025))
            append_common_rollout_record(
                arrays,
                frame_idx=frame_idx,
                pred_full=pred_full,
                pred=pred,
                pred_grip=pred_grip,
                q_target=q_policy,
                sim_data=data,
                qpos_ids=qpos_ids,
                gripper_closure=gripper_closure,
                tcp=tcp,
                max_action_deg=float(args.max_action_deg),
                image_file=image_file,
            )
            arrays["object_pos"].append(obj)
            arrays["goal_pos"].append(active_drop_pos)
            arrays["object_goal_error_xy"].append(active_place_error)
            arrays["tcp_object_dist_xy"].append(float(np.linalg.norm((tcp - obj)[:2])))
            arrays["tcp_goal_dist_xy"].append(float(np.linalg.norm(tcp[:2] - active_drop_pos[:2])))
            arrays["attached"].append(bool(attached_name == active_name))
            arrays["phase_guard_phase"].append(str(guard_info.get("phase", "")) if guard_enabled else "")
            arrays["phase_guard_target_pos"].append(
                np.asarray(guard_info.get("target_pos", [np.nan, np.nan, np.nan]), dtype=np.float32)
            )
            arrays["phase_guard_desired_gripper"].append(float(guard_info.get("desired_gripper", np.nan)))
            arrays["active_object_name"].append(str(active_name))
            arrays["active_object_index"].append(int(active_index))
            arrays["active_object_pos"].append(np.asarray(obj, dtype=np.float32))
            arrays["active_goal_pos"].append(np.asarray(active_drop_pos, dtype=np.float32))
            if rgb_widget is not None:
                rgb_widget.pixels = rgb
            if frame_idx % max(1, int(args.log_every)) == 0:
                print(
                    f"frame={frame_idx:05d} pred_deg={np.rad2deg(pred).round(3).tolist()} "
                    f"gripper={gripper_closure:.3f} tcp={tcp.round(4).tolist()} "
                    f"active={active_name} object={obj.round(4).tolist()} goal_err_xy={arrays['object_goal_error_xy'][-1]:.4f} "
                    f"tcp_obj_xy={arrays['tcp_object_dist_xy'][-1]:.4f} attached={attached_name == active_name} "
                    f"phase={arrays['phase_guard_phase'][-1] if guard_enabled else 'policy'}",
                    flush=True,
                )
            if active_done:
                completed_objects.add(active_name)
                if active_index + 1 < len(tape_order):
                    active_index += 1
                    active_name = tape_order[active_index]
                    active_site = object_sites[active_name]
                    active_qslice = object_qslices[active_name]
                    active_start_pos = np.asarray(tape_start_positions[active_name], dtype=np.float32)
                    active_drop_pos = np.asarray(drop_positions[active_name], dtype=np.float32)
                    guard_phase = "approach"
                    guard_close_count = 0
                    guard_release_count = 0
                    guard_last_phase = ""
                    guard_phase_frame_count = 0
                    guard_ik_cache.clear()
                    guard_object_ref = object_pos_from_site(model, data, active_site).copy()
                    attach_offset = np.zeros(3, dtype=np.float32)
                    prev_pred[:] = 0.0
                    prev_pred_grip = 0.0
                    print(f"Advanced to next tape: {active_name}", flush=True)
                else:
                    break
            if render is not None:
                render.sync(data)
                time.sleep(1.0 / float(args.hz))
    finally:
        if render_context is not None:
            render_context.__exit__(None, None, None)

    place_radius = float(task.get("place_success_radius_m", 0.025))
    final_positions: dict[str, list[float]] = {}
    per_tape_results: dict[str, dict] = {}
    place_errors = []
    moved_distances = []
    for name in tape_order:
        final_pos = object_pos_from_site(model, data, object_sites[name])
        target_pos = np.asarray(drop_positions[name], dtype=np.float32)
        start_i = np.asarray(tape_start_positions[name], dtype=np.float32)
        place_error_i = float(np.linalg.norm(final_pos[:2] - target_pos[:2]))
        moved_i = float(np.linalg.norm(final_pos[:2] - start_i[:2]))
        final_positions[name] = final_pos.round(6).tolist()
        place_errors.append(place_error_i)
        moved_distances.append(moved_i)
        per_tape_results[name] = {
            "place_success": bool(place_error_i <= place_radius),
            "place_error_xy_m": place_error_i,
            "drop_target_pos_m": target_pos.round(6).tolist(),
            "object_start_pos_m": start_i.round(6).tolist(),
            "object_final_pos_m": final_pos.round(6).tolist(),
            "object_moved_xy_m": moved_i,
            "completed_by_rollout_state": bool(name in completed_objects),
            "released_by_rollout_state": bool(name in released_objects),
        }
    final_obj = object_pos_from_site(model, data, object_sites[primary_name])
    place_error = float(max(place_errors)) if place_errors else float("nan")
    task_success = bool(place_errors and all(error <= place_radius for error in place_errors))
    summary = print_rollout_summary(arrays, prefix="Sim autonomous rollout")
    attached_arr = np.asarray(arrays["attached"], dtype=bool) if arrays["attached"] else np.zeros(0, dtype=bool)
    tcp_object_dist = np.asarray(arrays["tcp_object_dist_xy"], dtype=np.float64) if arrays["tcp_object_dist_xy"] else np.zeros(0)
    tcp_goal_dist = np.asarray(arrays["tcp_goal_dist_xy"], dtype=np.float64) if arrays["tcp_goal_dist_xy"] else np.zeros(0)
    object_goal_error = np.asarray(arrays["object_goal_error_xy"], dtype=np.float64) if arrays["object_goal_error_xy"] else np.zeros(0)
    phase_counts: dict[str, int] = {}
    if guard_enabled:
        for phase_name in arrays.get("phase_guard_phase", []):
            key = str(phase_name)
            phase_counts[key] = int(phase_counts.get(key, 0) + 1)
    summary.update(
        {
            "object_start_pos_m": start_pos.round(6).tolist(),
            "tape_start_positions_m": {name: pos.round(6).tolist() for name, pos in tape_start_positions.items()},
            "goal_pos_m": drop_positions[primary_name].round(6).tolist(),
            "tape_drop_targets_m": {name: pos.round(6).tolist() for name, pos in drop_positions.items()},
            "tape_final_positions_m": final_positions,
            "per_tape_results": per_tape_results,
            "tape_order": tape_order,
            "tape_assigned_slots": tape_assigned_slots,
            "completed_objects": sorted(completed_objects),
            "released_objects": sorted(released_objects),
            "object_final_pos_m": final_obj.round(6).tolist(),
            "place_error_xy_m": place_error,
            "place_success_radius_m": place_radius,
            "task_success": task_success,
            "attach_assist": attach_assist,
            "sim_rgb_source": sim_rgb_source,
            "training_rgb_source": training_rgb_source,
            "attach_ever": bool(attached_arr.any()),
            "attach_first_frame": int(np.argmax(attached_arr)) if attached_arr.any() else -1,
            "min_tcp_object_dist_xy_m": float(np.min(tcp_object_dist)) if tcp_object_dist.size else float("nan"),
            "final_tcp_object_dist_xy_m": float(tcp_object_dist[-1]) if tcp_object_dist.size else float("nan"),
            "min_tcp_goal_dist_xy_m": float(np.min(tcp_goal_dist)) if tcp_goal_dist.size else float("nan"),
            "min_object_goal_error_xy_m": float(np.min(object_goal_error)) if object_goal_error.size else float("nan"),
            "object_moved_xy_m": float(max(moved_distances)) if moved_distances else float("nan"),
            "lookahead_frames": int(lookahead_frames),
            "execution_lead_s": float(lookahead_frames / float(args.hz)),
            "receding_horizon": True,
            "phase_guard": guard_mode,
            "phase_guard_active": bool(guard_enabled),
            "phase_guard_final_phase": str(arrays["phase_guard_phase"][-1]) if guard_enabled and arrays["phase_guard_phase"] else "",
            "phase_guard_phase_counts": phase_counts,
            "phase_guard_residual_scale": float(args.phase_guard_residual_scale),
            "phase_guard_soft_gain": float(args.phase_guard_soft_gain),
            "phase_guard_soft_max_deg": float(args.phase_guard_soft_max_deg),
            "action_ema": float(args.action_ema),
            "gripper_ema": float(args.gripper_ema),
            "max_action_step_deg": float(args.max_action_step_deg),
            "policy_ablation": ablation_mode,
            "policy_ablation_seed": int(args.policy_ablation_seed),
        }
    )
    print(f"Sim task success={task_success} max_place_error_xy={place_error:.4f}m radius={place_radius:.4f}m", flush=True)
    if log_base is not None:
        save_rollout_log(
            log_base,
            arrays,
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "mode": "simtask",
                "policy": resolve_demo_path(args.policy).as_posix(),
                "config": resolve_demo_path(args.config).as_posix(),
                "camera_config": resolve_demo_path(args.camera_config).as_posix(),
                "summary": summary,
                "control_diagnostics": control_diag,
                "notes": (
                    "Autonomous closed-loop test in a randomized simulated task environment, not tied to a recorded episode. "
                    "Ablation modes: normal uses the trained policy, zero_image removes visual input, zero_target removes target conditioning, "
                    "zero_policy removes the learned action, and random_policy replaces the learned action with reproducible noise."
                ),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a trained FR5 behavior-cloning policy in sim, or guarded live mode.")
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY.as_posix())
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--camera-config", type=str, default=DEFAULT_CAMERA_CONFIG.as_posix())
    parser.add_argument("--episode", type=str, default="", help="Evaluate policy on a recorded episode. If omitted, use live Astra + SDK readback.")
    parser.add_argument("--sim-task-rollout", action="store_true", help="Evaluate policy in a randomized simulated tape-on-part1 task, not a recorded episode.")
    parser.add_argument("--sim-seed", type=int, default=20260513)
    parser.add_argument("--sim-start-random-radius", type=float, default=None)
    parser.add_argument("--sim-attach-assist", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sim-attach-distance", type=float, default=None, help="XY gate for sim attach assist; defaults to config attach_distance_m.")
    parser.add_argument(
        "--sim-attach-z-distance",
        type=float,
        default=None,
        help="Vertical gate for sim attach assist; prevents no-grasp high-pass sticking.",
    )
    parser.add_argument(
        "--sim-attach-gripper-threshold",
        type=float,
        default=None,
        help="Gripper closure gate for sim attach assist; defaults to config attach_gripper_threshold.",
    )
    parser.add_argument("--sim-release-gripper-threshold", type=float, default=-1.0, help="Negative uses config release_gripper_closure + 0.03")
    parser.add_argument(
        "--sim-rgb-source",
        choices=("auto", "visual", "camera-solid"),
        default="auto",
        help="RGB source used by --sim-task-rollout. auto matches the policy training RGB when metadata is available.",
    )
    parser.add_argument(
        "--allow-rgb-source-mismatch",
        action="store_true",
        help="Allow fast debug rollout with an RGB source different from the policy training data.",
    )
    parser.add_argument(
        "--phase-guard",
        choices=("off", "soft", "scripted", "inference"),
        default="off",
        help="Validation tier for sim-task rollout: off=pure BC, soft=BC with phase prior, scripted=IK phase controller. inference is a legacy alias for scripted.",
    )
    parser.add_argument(
        "--phase-guard-residual-scale",
        type=float,
        default=0.0,
        help="scripted mode only: scale for adding the policy's clipped action as a residual on top of the IK phase controller.",
    )
    parser.add_argument("--phase-guard-residual-max-deg", type=float, default=2.0)
    parser.add_argument("--phase-guard-soft-gain", type=float, default=0.35)
    parser.add_argument("--phase-guard-soft-max-deg", type=float, default=6.0)
    parser.add_argument("--phase-guard-xy-tol", type=float, default=0.025)
    parser.add_argument("--phase-guard-z-tol", type=float, default=0.04)
    parser.add_argument("--phase-guard-close-frames", type=int, default=8)
    parser.add_argument("--phase-guard-release-frames", type=int, default=5)
    parser.add_argument("--phase-guard-max-close-frames", type=int, default=30)
    parser.add_argument("--phase-guard-max-release-frames", type=int, default=30)
    parser.add_argument(
        "--phase-guard-max-phase-frames",
        type=int,
        default=80,
        help="Advance non-terminal phase after this many frames to avoid validation deadlock. 0 disables.",
    )
    parser.add_argument("--phase-guard-ik-iters", type=int, default=50)
    parser.add_argument(
        "--action-ema",
        type=float,
        default=0.0,
        help="Closed-loop action smoothing for validation. 0 disables; 0.2-0.5 can reduce BC jitter without scripted planning.",
    )
    parser.add_argument("--gripper-ema", type=float, default=0.0)
    parser.add_argument(
        "--max-action-step-deg",
        type=float,
        default=0.0,
        help="Limit change of predicted joint delta between frames in deg/frame. 0 disables.",
    )
    parser.add_argument(
        "--policy-ablation",
        choices=("normal", "zero_image", "zero_target", "zero_policy", "random_policy"),
        default="normal",
        help="Validation ablation for measuring learned policy, image, and target-conditioning contribution.",
    )
    parser.add_argument("--policy-ablation-seed", type=int, default=91017)
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP)
    parser.add_argument("--execute-real", action="store_true", help="Actually stream ServoJ targets to the physical FR5")
    parser.add_argument("--prepare-controller", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speed-percent", type=float, default=5.0)
    parser.add_argument("--servo-vel", type=float, default=5.0)
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="Policy control frequency. Keep at 10Hz for data generated by fr5_sim_tape_pick_place.py/fr5_record_demonstration.py defaults.",
    )
    parser.add_argument(
        "--lookahead-frames",
        type=int,
        default=0,
        help="0 uses the value stored in the policy checkpoint. N>1 executes the predicted N-frame action as 1/N per frame and replans every frame.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Max rollout frames. In --episode mode, 0 means the full episode; in live mode, 0 uses 100 guarded steps.")
    parser.add_argument(
        "--max-action-deg",
        type=float,
        default=15.0,
        help="Clamp each predicted joint delta. The tape-on-part1 demos rotate j6 by about 12.9deg/frame, so 1deg will cripple rollout.",
    )
    parser.add_argument("--max-gripper-delta", type=float, default=0.05)
    parser.add_argument("--min-tcp-z", type=float, default=0.04)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--save-rollout-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-dir", type=str, default=DEFAULT_ROLLOUT_LOG_DIR.as_posix())
    parser.add_argument("--log-suffix", type=str, default="", help="Optional safe suffix appended to rollout log filenames")
    parser.add_argument("--log-every", type=int, default=1, help="Print rollout diagnostics every N frames")
    parser.add_argument("--diagnose-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgb-widget", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgb-widget-width", type=int, default=320)
    parser.add_argument("--rgb-widget-height", type=int, default=240)
    parser.add_argument("--real-rgb-width", type=int, default=640)
    parser.add_argument("--real-rgb-height", type=int, default=480)
    parser.add_argument("--real-rgb-fps", type=int, default=10)
    parser.add_argument("--allow-latest-fallback", action="store_true")
    parser.add_argument("--initial-gripper-closure", type=float, default=0.0)
    parser.add_argument("--execute-gripper", action="store_true", help="Actually send USB Robotiq commands during live rollout")
    parser.add_argument("--gripper-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--gripper-baudrate", type=int, default=115200)
    parser.add_argument("--gripper-slave-id", type=int, default=9)
    parser.add_argument("--gripper-timeout", type=float, default=0.5)
    parser.add_argument("--gripper-retries", type=int, default=2)
    parser.add_argument("--gripper-speed", type=int, default=255)
    parser.add_argument("--gripper-force", type=int, default=150)
    parser.add_argument("--gripper-send-epsilon", type=float, default=0.01)
    args = parser.parse_args()

    if args.hz <= 0.0:
        raise RuntimeError("--hz must be positive")
    if int(args.lookahead_frames) < 0:
        raise RuntimeError("--lookahead-frames must be >= 0")
    if not args.episode and not args.sim_task_rollout and int(args.max_steps) <= 0:
        args.max_steps = 100
    if args.sim_task_rollout and int(args.max_steps) <= 0:
        args.max_steps = 100
    device = choose_device(args.device)
    model_policy, stats, image_size, payload = load_policy_checkpoint(args.policy, device=device)
    print(f"Loaded policy: {resolve_demo_path(args.policy)}")
    print(
        f"  model_type={payload.get('model_type', payload.get('model_class'))}, "
        f"device={device}, image_size={image_size}, action_mode={payload.get('action_mode')}"
    )
    print(f"  checkpoint_lookahead_frames={policy_lookahead_frames(argparse.Namespace(lookahead_frames=0), payload)}")
    if args.sim_task_rollout:
        run_sim_task_rollout(args, model_policy, stats, image_size, device, payload)
    elif args.episode:
        run_episode_rollout(args, model_policy, stats, image_size, device, payload)
    else:
        run_live_rollout(args, model_policy, stats, image_size, device, payload)


if __name__ == "__main__":
    main()
