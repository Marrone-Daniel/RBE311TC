from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from motrixsim import step as sim_step

from arm_control import (
    DEFAULT_CONFIG,
    DEFAULT_FR5_GS_DIR,
    build_runtime,
    camera_name_from_config,
    camera_resolution_from_config,
    collect_fr5_gaussian_assets,
    ensure_fr5_gaussian_assets,
    find_camera_id,
    load_config,
    load_replay_qpos,
    render_fr5_gs_rgb,
    require_cv2,
    resolve_demo_path,
    set_arm_qpos_and_ctrl,
    set_gripper,
    site_position,
)
from fr5_il_dataset import DEFAULT_IL_DEMO_DIR
from fr5_record_demonstration import write_episode
from fr5_sync_sdk import normalize_arm_trajectory, resample_trajectory


DEFAULT_CAMERA_CONFIG = Path(__file__).resolve().parent / "configs" / "astra_camera.json"


def make_sim_episode_dir(output_dir: str | Path, episode_name: str | None) -> Path:
    output_dir = resolve_demo_path(output_dir)
    name = episode_name or datetime.now().strftime("sim_episode_%Y%m%d_%H%M%S")
    episode_dir = output_dir / name
    if episode_dir.exists():
        raise RuntimeError(f"Episode directory already exists: {episode_dir}")
    (episode_dir / "rgb").mkdir(parents=True)
    return episode_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate simulator-rendered FR5 3DGS episodes with the same schema as real IL recordings."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--camera-config", type=str, default=DEFAULT_CAMERA_CONFIG.as_posix())
    parser.add_argument("--output-dir", type=str, default=DEFAULT_IL_DEMO_DIR.as_posix())
    parser.add_argument("--episode-name", type=str, default="")
    parser.add_argument("--replay-qpos", type=str, default="", help="Optional .npz/.npy trajectory; otherwise uses config demo target")
    parser.add_argument("--source-dt", type=float, default=0.04)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fr5-gs-dir", type=str, default=DEFAULT_FR5_GS_DIR.as_posix())
    parser.add_argument("--fr5-gs-regenerate", action="store_true")
    parser.add_argument("--fr5-gs-points-per-geom", type=int, default=None, help="Override config fr5_3dgs.points_per_geom")
    parser.add_argument("--gripper-closure", type=float, default=None, help="0=open, 1=closed; defaults to config gripper_opening")
    args = parser.parse_args()

    if args.hz <= 0.0:
        raise RuntimeError("--hz must be positive")
    cv2 = require_cv2()
    config = load_config(args.config)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "fr5_generate_sim_demonstration.py requires a CUDA GPU because the current 3DGS renderer "
            "moves Gaussian data to CUDA. Run this on the GPU environment, or collect real episodes instead."
        )

    model, data, body, qpos_ids, arm_act_ids, gripper_act_ids = build_runtime(config, args.camera_config)
    camera_id = find_camera_id(model, camera_name_from_config(args.camera_config))
    if camera_id is None:
        raise RuntimeError("No calibrated model camera available. Run dynamic calibration before generating sim episodes.")
    width, height = camera_resolution_from_config(args.camera_config)

    replay = load_replay_qpos(args.replay_qpos) if args.replay_qpos else None
    trajectory = normalize_arm_trajectory(replay, config, qpos_ids)
    trajectory = resample_trajectory(trajectory, float(args.source_dt), 1.0 / float(args.hz)).astype(np.float32)
    if int(args.max_frames) > 0:
        trajectory = trajectory[: int(args.max_frames)]
    if trajectory.shape[0] < 2:
        raise RuntimeError("Need at least 2 trajectory frames to generate an IL episode.")

    gs_dir = resolve_demo_path(args.fr5_gs_dir)
    ensure_fr5_gaussian_assets(
        config,
        gs_dir,
        regenerate=bool(args.fr5_gs_regenerate),
        points_per_geom=args.fr5_gs_points_per_geom,
    )
    gaussians = collect_fr5_gaussian_assets(model, gs_dir)
    if not gaussians:
        raise RuntimeError(f"No FR5 Gaussian PLY assets found in {gs_dir}")
    from gaussian_renderer import GSRendererMotrixSim

    gs_renderer = GSRendererMotrixSim(gaussians, model)
    episode_dir = make_sim_episode_dir(args.output_dir, args.episode_name or None)
    steps_per_frame = max(1, round((1.0 / float(args.hz)) / float(model.options.timestep)))

    timestamps: list[float] = []
    joint_deg: list[np.ndarray] = []
    tcp_pos: list[np.ndarray] = []
    gripper_closure: list[float] = []
    image_files: list[str] = []
    closure = float(config.get("gripper_opening", 0.0) if args.gripper_closure is None else args.gripper_closure)
    closure = float(np.clip(closure, 0.0, 1.0))
    start = time.monotonic()
    for idx, q_rad in enumerate(trajectory):
        set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q_rad)
        set_gripper(data, model, body, gripper_act_ids, closure)
        for _ in range(steps_per_frame):
            sim_step(model, data)
        rgb = render_fr5_gs_rgb(gs_renderer, model, data, int(camera_id), width, height)
        rel_path = f"rgb/{idx:06d}.png"
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite((episode_dir / rel_path).as_posix(), bgr):
            raise RuntimeError(f"Failed to write image: {episode_dir / rel_path}")
        tcp = site_position(model, data, config["tcp_site"])
        timestamps.append(time.monotonic() - start)
        joint_deg.append(np.rad2deg(q_rad).astype(np.float32))
        tcp_pos.append(tcp.astype(np.float32))
        gripper_closure.append(closure)
        image_files.append(rel_path)
        if idx % max(1, int(args.hz)) == 0:
            print(f"frame={idx:05d} tcp={tcp.round(4).tolist()}", flush=True)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "sim_3dgs",
        "control_hz": float(args.hz),
        "config": resolve_demo_path(args.config).as_posix(),
        "camera_config": resolve_demo_path(args.camera_config).as_posix(),
        "rgb_width": int(width),
        "rgb_height": int(height),
        "action_mode": "joint_delta_rad_plus_gripper_delta",
        "gripper_closure": closure,
        "gaussian_assets": sorted(gaussians.keys()),
        "notes": "Simulator-rendered FR5 3DGS RGB plus simulated joint states. Schema matches real recorder output.",
    }
    write_episode(
        episode_dir,
        timestamps=timestamps,
        joint_deg=joint_deg,
        tcp_pos=tcp_pos,
        gripper_closure=gripper_closure,
        image_files=image_files,
        meta=meta,
    )
    with (episode_dir / "trajectory_source.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "replay_qpos": str(args.replay_qpos),
                "source_dt": float(args.source_dt),
                "hz": float(args.hz),
                "frames": int(trajectory.shape[0]),
            },
            f,
            indent=2,
        )
    print(f"Saved simulator episode: {episode_dir}")
    print(f"  frames={trajectory.shape[0]}")
    print(f"  states={episode_dir / 'states.npz'}")


if __name__ == "__main__":
    main()
