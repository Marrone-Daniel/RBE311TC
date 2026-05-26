from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from motrixsim import step as sim_step
from motrixsim.render import Layout, RenderApp

from arm_control import (
    DEFAULT_CONFIG,
    DEFAULT_FR5_GS_DIR,
    SIM_GS_TEXTURE_NAME,
    RealRgbSource,
    build_runtime,
    camera_name_from_config,
    camera_resolution_from_config,
    collect_fr5_gaussian_assets,
    ensure_fr5_gaussian_assets,
    find_camera_id,
    load_config,
    load_replay_qpos,
    print_model_summary,
    render_fr5_gs_rgb,
    resolve_demo_path,
    set_arm_qpos_and_ctrl,
    set_gripper,
    site_position,
)
from fr5_sync_sdk import (
    DEFAULT_ROBOT_IP,
    FairinoArmClient,
    MotionCancelHandler,
    normalize_arm_trajectory,
    resample_trajectory,
    run_sim_safety_check,
    write_report,
)


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CAMERA_CONFIG = DEMO_DIR / "configs" / "astra_camera.json"


def realtime_sleep(next_time: float) -> float:
    now = time.monotonic()
    if next_time > now:
        time.sleep(next_time - now)
    return next_time


def main() -> None:
    parser = argparse.ArgumentParser(description="FR5 physical/sim synchronized motion with simulator-rendered 3DGS camera screen")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--camera-config", type=str, default=DEFAULT_CAMERA_CONFIG.as_posix())
    parser.add_argument("--replay-qpos", type=str, default="", help="Optional .npz/.npy trajectory; otherwise uses config demo target")
    parser.add_argument("--source-dt", type=float, default=0.04)
    parser.add_argument("--sync-dt", type=float, default=0.008, help="Simulation/real sync period")
    parser.add_argument("--max-loops", type=int, default=0, help="0 means run one trajectory; >0 repeats trajectory this many times")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N sync frames; 0 means full requested loops")
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP)
    parser.add_argument("--execute-real", action="store_true", help="Actually send ServoJ commands to the physical FR5")
    parser.add_argument("--prepare-controller", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speed-percent", type=float, default=10.0)
    parser.add_argument("--servo-vel", type=float, default=20.0)
    parser.add_argument("--start-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--min-tcp-z", type=float, default=0.04)
    parser.add_argument("--max-step-rad", type=float, default=0.012)
    parser.add_argument("--max-real-step-deg", type=float, default=0.75)
    parser.add_argument("--max-tcp-speed", type=float, default=0.35)
    parser.add_argument("--sim-gs-screen", action=argparse.BooleanOptionalAction, default=True, help="Show simulator-rendered FR5 3DGS camera screen in the 3D scene")
    parser.add_argument("--sim-gs-fps", type=int, default=15)
    parser.add_argument("--sim-gs-alpha", type=float, default=1.0)
    parser.add_argument("--sim-gs-distance", type=float, default=0.75)
    parser.add_argument("--fr5-gs-dir", type=str, default=DEFAULT_FR5_GS_DIR.as_posix())
    parser.add_argument("--fr5-gs-regenerate", action="store_true")
    parser.add_argument("--fr5-gs-points-per-geom", type=int, default=None, help="Override config fr5_3dgs.points_per_geom")
    parser.add_argument("--real-rgb-source", choices=["live", "latest"], default="live")
    parser.add_argument("--real-rgb-width", type=int, default=640)
    parser.add_argument("--real-rgb-height", type=int, default=480)
    parser.add_argument("--real-rgb-fps", type=int, default=15)
    parser.add_argument("--real-rgb-alpha", type=float, default=1.0)
    parser.add_argument("--real-rgb-distance", type=float, default=0.75)
    parser.add_argument("--allow-latest-fallback", action="store_true", help="Allow static latest capture if live Astra fails")
    parser.add_argument("--real-rgb-widget", action=argparse.BooleanOptionalAction, default=False, help="Optional raw Astra debug widget; not the simulator 3DGS screen")
    parser.add_argument("--real-rgb-widget-width", type=int, default=320)
    parser.add_argument("--real-rgb-widget-height", type=int, default=240)
    parser.add_argument("--report", type=str, default="data/fr5_live_sync_last_report.json")
    parser.add_argument("--preflight-only", action="store_true", help="Run safety checks without opening RenderApp or robot connection")
    parser.add_argument(
        "--render-backend",
        choices=["x11", "wayland", "auto"],
        default=os.environ.get("WINIT_UNIX_BACKEND", "x11"),
    )
    args = parser.parse_args()

    if args.render_backend == "auto":
        os.environ.pop("WINIT_UNIX_BACKEND", None)
    else:
        os.environ["WINIT_UNIX_BACKEND"] = args.render_backend
    if args.sync_dt <= 0.0:
        raise RuntimeError("--sync-dt must be positive")
    if args.execute_real and not (0.001 <= args.sync_dt <= 0.016):
        raise RuntimeError("--sync-dt must be between 0.001 and 0.016 seconds when --execute-real is used")

    config = load_config(args.config)
    base_model, _, _, qpos_ids0, _, _ = build_runtime(config)
    replay = load_replay_qpos(args.replay_qpos) if args.replay_qpos else None
    trajectory = normalize_arm_trajectory(replay, config, qpos_ids0)
    trajectory = resample_trajectory(trajectory, args.source_dt, args.sync_dt)

    report, _ = run_sim_safety_check(
        config,
        trajectory,
        min_tcp_z=float(args.min_tcp_z),
        max_step_rad=float(args.max_step_rad),
        max_tcp_speed=float(args.max_tcp_speed),
        control_dt=float(args.sync_dt),
    )
    real_step_deg = float(np.max(np.abs(np.diff(np.rad2deg(trajectory), axis=0)))) if trajectory.shape[0] > 1 else 0.0
    if real_step_deg > float(args.max_real_step_deg):
        raise RuntimeError(
            f"Real ServoJ step too large: {real_step_deg:.3f}deg, limit={args.max_real_step_deg:.3f}deg"
        )
    report_path = resolve_demo_path(args.report)
    write_report(report_path, report, trajectory, bool(args.execute_real))
    print("Simulation safety check passed.")
    print(
        f"  frames={report.frames}, duration={report.duration_s:.3f}s, "
        f"max_step={report.max_step_deg:.3f}deg, min_tcp_z={report.min_tcp_z:.4f}m, "
        f"max_tcp_speed={report.max_tcp_speed_mps:.4f}m/s"
    )
    print(f"  report={report_path}")

    if args.preflight_only:
        return

    model, data, body, qpos_ids, arm_act_ids, gripper_act_ids = build_runtime(
        config,
        args.camera_config,
        sim_gs_screen=bool(args.sim_gs_screen),
        overlay_distance=float(args.sim_gs_distance),
        overlay_alpha=float(args.sim_gs_alpha),
    )
    print_model_summary(model, config)
    attached_camera_name = camera_name_from_config(args.camera_config)
    attached_camera_id = find_camera_id(model, attached_camera_name)
    if attached_camera_id is None:
        raise RuntimeError("No calibrated model camera available. Re-run dynamic calibration before live sync.")
    gs_renderer = None
    gs_width, gs_height = camera_resolution_from_config(args.camera_config)
    if args.sim_gs_screen:
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
        print(f"Simulator 3DGS screen ready with {len(gaussians)} bound assets.", flush=True)
    steps_per_sync = max(1, round(float(args.sync_dt) / float(model.options.timestep)))

    canceller = MotionCancelHandler()
    canceller.install()
    arm = None
    if args.execute_real:
        arm = FairinoArmClient(args.robot_ip, speed_percent=float(args.speed_percent))
        canceller.set_arm(arm)
        arm.connect()
        if args.prepare_controller:
            print("Preparing controller: Mode(0 automatic), RobotEnable(1).")
            arm.set_mode(0)
            arm.robot_enable(1)
            time.sleep(0.5)
        actual = arm.get_actual_joint_deg()
        if actual is None:
            arm.close()
            raise RuntimeError("Could not read physical FR5 joints. Refusing synchronized real motion.")
        delta0 = float(np.max(np.abs(np.asarray(actual, dtype=np.float64) - np.rad2deg(trajectory[0]))))
        if delta0 > float(args.start_tolerance_deg):
            arm.close()
            raise RuntimeError(
                f"Physical FR5 is not at trajectory start: max_delta={delta0:.2f}deg, "
                f"limit={args.start_tolerance_deg:.2f}deg. Run fr5_move_to_initial.py first."
            )

    rgb_source = None
    servo_started = False
    loops_done = 0
    frame_count = 0
    try:
        print("Constructing RenderApp...")
        with RenderApp() as render:
            render.launch(model)
            print("Main render view left as free system camera.")
            sim_gs_texture = None
            if args.sim_gs_screen:
                first_sim_rgb = render_fr5_gs_rgb(
                    gs_renderer,
                    model,
                    data,
                    int(attached_camera_id),
                    gs_width,
                    gs_height,
                    system_camera=render.system_camera,
                )
                sim_gs_texture = render.get_texture_image(SIM_GS_TEXTURE_NAME)
                sim_gs_texture.pixels = np.ascontiguousarray(first_sim_rgb)
            first_rgb = None
            rgb_widget_img = None
            if args.real_rgb_widget:
                rgb_source = RealRgbSource(
                    source=str(args.real_rgb_source),
                    image_path=None,
                    width=int(args.real_rgb_width),
                    height=int(args.real_rgb_height),
                    fps=int(args.real_rgb_fps),
                    allow_fallback=bool(args.allow_latest_fallback),
                )
                first_rgb = rgb_source.start()
                rgb_widget_img = render.create_image(first_rgb)
                render.widgets.create_image_widget(
                    rgb_widget_img,
                    layout=Layout(
                        left=10,
                        top=10,
                        width=int(args.real_rgb_widget_width),
                        height=int(args.real_rgb_widget_height),
                    ),
                )
            render.sync(data)
            print(
                "Live sync window launched. "
                + ("Physical FR5 execution enabled." if args.execute_real else "Simulation-only mode; real FR5 is not moving.")
            )

            if arm is not None:
                arm.servo_start()
                servo_started = True
                canceller.set_servo_started(True)

            next_time = time.monotonic()
            sim_gs_update_period = max(1, round(1.0 / (float(args.sync_dt) * max(1, int(args.sim_gs_fps)))))
            rgb_update_period = max(1, round(1.0 / (float(args.sync_dt) * max(1, int(args.real_rgb_fps)))))
            rgb_timeout_ms = max(20, int(round(1000.0 / max(1, int(args.real_rgb_fps)))))
            while not render.is_closed:
                for idx, q in enumerate(trajectory):
                    canceller.check()
                    if render.is_closed:
                        break
                    set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q.astype(np.float32))
                    set_gripper(data, model, body, gripper_act_ids, float(config.get("gripper_opening", 0.0)))
                    for _ in range(steps_per_sync):
                        sim_step(model, data)

                    if args.sim_gs_screen and sim_gs_texture is not None and frame_count % sim_gs_update_period == 0:
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

                    if frame_count % rgb_update_period == 0 and rgb_source is not None:
                        rgb = rgb_source.read(timeout_ms=rgb_timeout_ms)
                        if rgb is not None:
                            rgb = np.ascontiguousarray(rgb)
                            if rgb_widget_img is not None:
                                rgb_widget_img.pixels = rgb

                    if arm is not None:
                        arm.servo_j(np.rad2deg(q), idx=frame_count, cmd_t=float(args.sync_dt), vel=float(args.servo_vel))

                    if frame_count % 50 == 0:
                        tcp = site_position(model, data, config["tcp_site"])
                        print(
                            f"frame={frame_count:05d} q={data.dof_pos[qpos_ids].round(4).tolist()} "
                            f"tcp={tcp.round(4).tolist()}",
                            flush=True,
                        )

                    render.sync(data)
                    frame_count += 1
                    if int(args.max_frames) > 0 and frame_count >= int(args.max_frames):
                        return
                    next_time += float(args.sync_dt)
                    realtime_sleep(next_time)

                loops_done += 1
                if int(args.max_loops) == 0:
                    break
                if loops_done >= int(args.max_loops):
                    break
    except KeyboardInterrupt:
        print("Live sync cancelled. StopMotion was sent best-effort if real execution was active.", flush=True)
    finally:
        if arm is not None:
            if servo_started:
                arm.servo_end_best_effort()
                canceller.set_servo_started(False)
            arm.close()
        if rgb_source is not None:
            rgb_source.stop()


if __name__ == "__main__":
    main()
