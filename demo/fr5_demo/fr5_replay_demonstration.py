from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from motrixsim import step as sim_step
from motrixsim.render import Layout, RenderApp

from arm_control import (
    DEFAULT_CONFIG,
    DEFAULT_FR5_GS_DIR,
    SIM_GS_TEXTURE_NAME,
    build_runtime,
    camera_name_from_config,
    camera_resolution_from_config,
    collect_fr5_gaussian_assets,
    ensure_fr5_gaussian_assets,
    find_camera_id,
    load_config,
    print_model_summary,
    render_fr5_gs_rgb,
    require_cv2,
    resolve_demo_path,
    set_arm_qpos_and_ctrl,
    set_gripper,
    site_position,
)
from fr5_il_dataset import DEFAULT_IL_DEMO_DIR, load_episode_meta, load_episode_npz


DEFAULT_CAMERA_CONFIG = Path(__file__).resolve().parent / "configs" / "astra_camera.json"


def load_episode_rgb(episode_dir: Path, image_file: str) -> np.ndarray:
    cv2 = require_cv2()
    bgr = cv2.imread((episode_dir / image_file).as_posix(), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read episode RGB image: {episode_dir / image_file}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def print_episode_summary(episode_dir: Path, q_rad: np.ndarray, timestamps: np.ndarray, meta: dict) -> None:
    duration = float(timestamps[-1] - timestamps[0]) if timestamps.shape[0] > 1 else 0.0
    diffs = np.diff(q_rad, axis=0)
    max_step_deg = float(np.rad2deg(np.max(np.abs(diffs)))) if diffs.size else 0.0
    print(f"Episode: {episode_dir}")
    print(f"  frames={q_rad.shape[0]}, duration={duration:.3f}s, max_joint_step={max_step_deg:.3f}deg")
    if meta:
        print(f"  schema={meta.get('schema', 'unknown')}, action_mode={meta.get('action_mode', 'unknown')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded FR5 IL episode in MotrixSim for visual/safety inspection.")
    parser.add_argument("--episode", type=str, required=True, help="Episode directory under data/il_demos, or absolute path")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--camera-config", type=str, default=DEFAULT_CAMERA_CONFIG.as_posix())
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--rgb-widget", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgb-widget-width", type=int, default=320)
    parser.add_argument("--rgb-widget-height", type=int, default=240)
    parser.add_argument("--sim-gs-screen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sim-gs-fps", type=int, default=10)
    parser.add_argument("--sim-gs-distance", type=float, default=0.75)
    parser.add_argument("--fr5-gs-dir", type=str, default=DEFAULT_FR5_GS_DIR.as_posix())
    parser.add_argument("--fr5-gs-regenerate", action="store_true")
    parser.add_argument("--fr5-gs-points-per-geom", type=int, default=None, help="Override config fr5_3dgs.points_per_geom")
    args = parser.parse_args()

    episode_dir = resolve_demo_path(args.episode)
    if not (episode_dir / "states.npz").exists():
        episode_dir = resolve_demo_path(DEFAULT_IL_DEMO_DIR / args.episode)
    pack = load_episode_npz(episode_dir)
    meta = load_episode_meta(episode_dir)
    q_rad = np.asarray(pack["joint_rad"], dtype=np.float32)
    gripper = (
        np.asarray(pack["gripper_closure"], dtype=np.float32)
        if "gripper_closure" in pack
        else np.zeros(q_rad.shape[0], dtype=np.float32)
    )
    timestamps = np.asarray(pack.get("timestamp", np.arange(q_rad.shape[0], dtype=np.float64)), dtype=np.float64)
    image_files = [str(item) for item in np.asarray(pack["image_files"]).tolist()]
    print_episode_summary(episode_dir, q_rad, timestamps, meta)
    print(f"  gripper range={float(np.min(gripper)):.3f}..{float(np.max(gripper)):.3f}")

    config = load_config(args.config)
    model, data, body, qpos_ids, arm_act_ids, gripper_act_ids = build_runtime(
        config,
        args.camera_config if args.sim_gs_screen else None,
        sim_gs_screen=bool(args.sim_gs_screen),
        overlay_distance=float(args.sim_gs_distance),
    )
    print_model_summary(model, config)
    if args.check_only:
        return

    control_dt = 1.0 / max(1.0, float(args.fps))
    steps_per_frame = max(1, round(control_dt / float(model.options.timestep)))
    gs_renderer = None
    sim_gs_texture = None
    attached_camera_id = None
    gs_width = gs_height = 0
    if args.sim_gs_screen:
        attached_camera_id = find_camera_id(model, camera_name_from_config(args.camera_config))
        if attached_camera_id is None:
            raise RuntimeError("No calibrated model camera available for simulated 3DGS screen.")
        gs_width, gs_height = camera_resolution_from_config(args.camera_config)
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

    with RenderApp() as render:
        render.launch(model)
        rgb_widget = None
        if args.rgb_widget:
            first_rgb = load_episode_rgb(episode_dir, image_files[0])
            rgb_widget = render.create_image(first_rgb)
            render.widgets.create_image_widget(
                rgb_widget,
                layout=Layout(left=10, top=10, width=int(args.rgb_widget_width), height=int(args.rgb_widget_height)),
            )
        if args.sim_gs_screen:
            first_sim_rgb = render_fr5_gs_rgb(
                gs_renderer, model, data, int(attached_camera_id), gs_width, gs_height, system_camera=render.system_camera
            )
            sim_gs_texture = render.get_texture_image(SIM_GS_TEXTURE_NAME)
            sim_gs_texture.pixels = np.ascontiguousarray(first_sim_rgb)

        frame = 0
        sim_gs_update_period = max(1, round(float(args.fps) / max(1, int(args.sim_gs_fps))))
        try:
            while not render.is_closed:
                idx = frame % q_rad.shape[0]
                set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q_rad[idx])
                set_gripper(data, model, body, gripper_act_ids, float(gripper[idx]))
                for _ in range(steps_per_frame):
                    sim_step(model, data)

                if rgb_widget is not None:
                    rgb_widget.pixels = load_episode_rgb(episode_dir, image_files[idx])
                if args.sim_gs_screen and frame % sim_gs_update_period == 0:
                    sim_rgb = render_fr5_gs_rgb(
                        gs_renderer,
                        model,
                        data,
                        int(attached_camera_id),
                        gs_width,
                        gs_height,
                        system_camera=render.system_camera,
                    )
                    sim_gs_texture.pixels = np.ascontiguousarray(sim_rgb)
                if frame % max(1, int(args.fps)) == 0:
                    tcp = site_position(model, data, config["tcp_site"])
                    print(f"frame={frame:05d} src={idx:05d} tcp={tcp.round(4).tolist()}", flush=True)
                render.sync(data)
                frame += 1
                if not args.loop and frame >= q_rad.shape[0]:
                    break
                time.sleep(control_dt)
        except KeyboardInterrupt:
            print("Replay stopped.", flush=True)


if __name__ == "__main__":
    main()
