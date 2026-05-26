from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from motrixsim import step as sim_step

from arm_control import (
    DEFAULT_CONFIG,
    RealRgbSource,
    build_runtime,
    load_config,
    require_cv2,
    resolve_demo_path,
    set_arm_qpos_and_ctrl,
    set_gripper,
    site_position,
)
from fr5_il_dataset import DEFAULT_IL_DEMO_DIR
from fr5_sync_sdk import (
    DEFAULT_ROBOT_IP,
    FairinoArmClient,
    MotionCancelHandler,
    Robotiq2F85ModbusRtuClient,
    Robotiq2F85PyrobotiqClient,
)


@dataclass
class EpisodeBuffer:
    episode_dir: Path
    start_time: float
    timestamps: list[float] = field(default_factory=list)
    joint_deg: list[np.ndarray] = field(default_factory=list)
    tcp_pos: list[np.ndarray] = field(default_factory=list)
    gripper_closure: list[float] = field(default_factory=list)
    image_files: list[str] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return len(self.joint_deg)


class AsyncGripperCommander:
    def __init__(
        self,
        gripper,
        *,
        speed: int,
        force: int,
        min_period_s: float,
        epsilon: float,
        debug: bool,
    ) -> None:
        self.gripper = gripper
        self.speed = int(speed)
        self.force = int(force)
        self.min_period_s = max(0.0, float(min_period_s))
        self.epsilon = max(0.0, float(epsilon))
        self.debug = bool(debug)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = threading.Event()
        self._target: float | None = None
        self._last_sent: float | None = None
        self._last_send_time = 0.0
        self._thread = threading.Thread(target=self._run, name="fr5-gripper-commander", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def command(self, closure: float) -> None:
        with self._lock:
            self._target = float(np.clip(closure, 0.0, 1.0))
        self._event.set()

    def stop(self) -> None:
        self._stop.set()
        self._event.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._event.wait(timeout=0.1)
            self._event.clear()
            if self._stop.is_set():
                return
            while not self._stop.is_set():
                with self._lock:
                    target = self._target
                if target is None:
                    break
                if self._last_sent is not None and abs(target - self._last_sent) < self.epsilon:
                    break
                now = time.monotonic()
                wait_s = self.min_period_s - (now - self._last_send_time)
                if wait_s > 0.0:
                    if self._stop.wait(wait_s):
                        return
                try:
                    self.gripper.command_closure(target, speed=self.speed, force=self.force)
                    self._last_sent = target
                    self._last_send_time = time.monotonic()
                except Exception as exc:
                    print(f"Warning: async Robotiq command failed: {exc}", flush=True)
                    self._last_send_time = time.monotonic()
                    if self._stop.wait(max(0.05, self.min_period_s)):
                        return
                with self._lock:
                    if self._target is None or abs(float(self._target) - target) < self.epsilon:
                        break


def make_episode_dir(output_dir: str | Path, episode_name: str | None, *, segment_index: int | None = None) -> Path:
    output_dir = resolve_demo_path(output_dir)
    if episode_name:
        name = episode_name if segment_index is None else f"{episode_name}_{segment_index:03d}"
    else:
        base = datetime.now().strftime("episode_%Y%m%d_%H%M%S")
        name = base if segment_index is None else f"{base}_{segment_index:03d}"
    episode_dir = output_dir / name
    suffix = 1
    while episode_dir.exists():
        episode_dir = output_dir / f"{name}_{suffix:02d}"
        suffix += 1
    (episode_dir / "rgb").mkdir(parents=True)
    return episode_dir


def print_xbox_keymap() -> None:
    print(
        "\nXbox collection keys:\n"
        "  Left stick       TCP X/Y\n"
        "  Right stick Y    TCP Z\n"
        "  LB / RB          TCP yaw\n"
        "  LT / RT          TCP pitch\n"
        "  D-pad up/down    TCP roll\n"
        "  X                close gripper while held\n"
        "  Y                open gripper while held\n"
        "  A                print current gripper target\n"
        "  B                end current episode, return FR5 to initial_qpos, then wait for next input\n"
        "  Ctrl+C           stop safely and save any valid current episode\n",
        flush=True,
    )


def write_episode(
    episode_dir: Path,
    *,
    timestamps: list[float],
    joint_deg: list[np.ndarray],
    tcp_pos: list[np.ndarray],
    gripper_closure: list[float] | None,
    image_files: list[str],
    meta: dict,
    target_features: list[np.ndarray] | None = None,
    rewards: list[float] | None = None,
) -> None:
    q_deg = np.vstack(joint_deg).astype(np.float32)
    q_rad = np.deg2rad(q_deg).astype(np.float32)
    next_q_rad = np.vstack([q_rad[1:], q_rad[-1:]]).astype(np.float32)
    action = (next_q_rad - q_rad).astype(np.float32)
    if gripper_closure is None:
        gripper = np.zeros(q_rad.shape[0], dtype=np.float32)
    else:
        gripper = np.asarray(gripper_closure, dtype=np.float32)
    if gripper.shape[0] != q_rad.shape[0]:
        raise RuntimeError(f"gripper_closure length mismatch: {gripper.shape[0]} vs {q_rad.shape[0]}")
    next_gripper = np.concatenate([gripper[1:], gripper[-1:]]).astype(np.float32)
    gripper_action = (next_gripper - gripper).astype(np.float32)
    tcp = np.vstack(tcp_pos).astype(np.float32) if tcp_pos else np.zeros((q_rad.shape[0], 3), dtype=np.float32)
    arrays = {
        "timestamp": np.asarray(timestamps, dtype=np.float64),
        "joint_deg": q_deg,
        "joint_rad": q_rad,
        "next_joint_rad": next_q_rad,
        "action_joint_delta_rad": action,
        "gripper_closure": gripper,
        "next_gripper_closure": next_gripper,
        "action_gripper_delta": gripper_action,
        "tcp_pos": tcp,
        "image_files": np.asarray(image_files, dtype=object),
    }
    if target_features is not None:
        target = np.asarray(target_features, dtype=np.float32)
        if target.shape[0] != q_rad.shape[0]:
            raise RuntimeError(f"target_features length mismatch: {target.shape[0]} vs {q_rad.shape[0]}")
        arrays["target_feature"] = target
    if rewards is not None:
        reward = np.asarray(rewards, dtype=np.float32)
        if reward.shape[0] != q_rad.shape[0]:
            raise RuntimeError(f"rewards length mismatch: {reward.shape[0]} vs {q_rad.shape[0]}")
        arrays["reward"] = reward
    np.savez_compressed((episode_dir / "states.npz").as_posix(), **arrays)
    np.savez_compressed((episode_dir / "replay_qpos.npz").as_posix(), arm_qpos=q_rad)
    meta = dict(meta)
    meta.update(
        {
            "frames": int(q_rad.shape[0]),
            "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
            "state_file": "states.npz",
            "replay_qpos_file": "replay_qpos.npz",
            "schema": "fr5_il_episode_v2",
        }
    )
    with (episode_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def build_episode_meta(args, *, source: str, segment_index: int, gripper_physical_enabled: bool) -> dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "robot_ip": str(args.robot_ip),
        "control_hz": float(args.hz),
        "config": resolve_demo_path(args.config).as_posix(),
        "rgb_width": int(args.real_rgb_width),
        "rgb_height": int(args.real_rgb_height),
        "rgb_source": str(args.real_rgb_source),
        "action_mode": "joint_delta_rad_plus_gripper_delta",
        "xbox_teleop": bool(args.xbox_teleop),
        "execute_gripper": bool(args.execute_gripper),
        "gripper_physical_enabled": bool(gripper_physical_enabled),
        "segment_index": int(segment_index),
        "source": source,
    }


def save_if_valid(buffer: EpisodeBuffer | None, args, *, source: str, segment_index: int, gripper_physical_enabled: bool) -> bool:
    if buffer is None:
        return False
    if buffer.frames < int(args.min_episode_frames):
        print(f"Discarding short episode: {buffer.episode_dir}, frames={buffer.frames}", flush=True)
        if not bool(getattr(args, "keep_discarded_episodes", False)) and buffer.episode_dir.exists():
            shutil.rmtree(buffer.episode_dir)
        return False
    write_episode(
        buffer.episode_dir,
        timestamps=buffer.timestamps,
        joint_deg=buffer.joint_deg,
        tcp_pos=buffer.tcp_pos,
        gripper_closure=buffer.gripper_closure,
        image_files=buffer.image_files,
        meta=build_episode_meta(
            args,
            source=source,
            segment_index=segment_index,
            gripper_physical_enabled=gripper_physical_enabled,
        ),
    )
    print(f"Saved episode: {buffer.episode_dir} frames={buffer.frames}", flush=True)
    return True


def record_one_sample(
    buffer: EpisodeBuffer,
    *,
    cv2,
    rgb_source: RealRgbSource,
    last_rgb: np.ndarray,
    arm: FairinoArmClient,
    current_gripper: float,
    model,
    data,
    body,
    qpos_ids: np.ndarray,
    arm_act_ids: np.ndarray,
    gripper_act_ids: np.ndarray,
    config: dict,
    steps_per_frame: int,
    real_rgb_fps: int,
) -> np.ndarray:
    q_deg = arm.get_actual_joint_deg()
    if q_deg is None:
        raise RuntimeError("Could not read FR5 joint angles from SDK during recording.")
    rgb = rgb_source.read(timeout_ms=max(20, int(1000 / max(1, int(real_rgb_fps)))))
    if rgb is None:
        rgb = last_rgb
    if rgb is None:
        raise RuntimeError("Astra RGB did not return a frame during recording.")

    q_deg_arr = np.asarray(q_deg, dtype=np.float32)
    q_rad_arr = np.deg2rad(q_deg_arr).astype(np.float32)
    set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q_rad_arr)
    set_gripper(data, model, body, gripper_act_ids, current_gripper)
    for _ in range(steps_per_frame):
        sim_step(model, data)
    tcp = site_position(model, data, config["tcp_site"])

    rel_path = f"rgb/{buffer.frames:06d}.png"
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite((buffer.episode_dir / rel_path).as_posix(), bgr):
        raise RuntimeError(f"Failed to write image: {buffer.episode_dir / rel_path}")

    buffer.timestamps.append(time.monotonic() - buffer.start_time)
    buffer.joint_deg.append(q_deg_arr)
    buffer.tcp_pos.append(tcp.astype(np.float32))
    buffer.gripper_closure.append(float(current_gripper))
    buffer.image_files.append(rel_path)
    if buffer.frames == 1 or buffer.frames % 10 == 0:
        print(
            f"  episode_frame={buffer.frames:05d} q_deg={q_deg_arr.round(3).tolist()} "
            f"gripper={current_gripper:.3f} tcp={tcp.round(4).tolist()}",
            flush=True,
        )
    return rgb


def move_to_initial_from_config(
    arm: FairinoArmClient,
    config: dict,
    *,
    vel: float,
    acc: float,
    ovl: float,
    tolerance_deg: float,
    timeout_s: float,
) -> None:
    target_rad = np.asarray(config["initial_qpos"], dtype=np.float64)
    target_deg = np.rad2deg(target_rad)
    print("Returning FR5 to initial_qpos...", flush=True)
    arm.stop_motion_best_effort()
    time.sleep(0.2)
    arm.set_mode(0)
    arm.robot_enable(1)
    time.sleep(0.3)
    err = arm.get_robot_error_code()
    if err is not None and any(int(v) != 0 for v in err):
        raise RuntimeError(f"Controller reports error code {err}; cannot return to initial_qpos.")
    arm.move_j(target_deg, vel=float(vel), acc=float(acc), ovl=float(ovl), blend_t=0.0, rpc_timeout=5.0)
    start = time.monotonic()
    while True:
        actual = arm.get_actual_joint_deg()
        if actual is None:
            raise RuntimeError("Lost FR5 joint readback while returning to initial_qpos.")
        max_err = float(np.max(np.abs(np.asarray(actual, dtype=np.float64) - target_deg)))
        if max_err <= float(tolerance_deg):
            print(f"Initial pose reached: max_err={max_err:.3f}deg", flush=True)
            return
        if time.monotonic() - start > float(timeout_s):
            arm.stop_motion()
            raise RuntimeError(f"Return-to-initial timed out: max_err={max_err:.3f}deg")
        time.sleep(0.25)


def connect_gripper_best_effort(args):
    if not args.execute_gripper:
        return None
    if args.gripper_backend == "pyrobotiq":
        gripper = Robotiq2F85PyrobotiqClient(
            args.gripper_port,
            slave_id=int(args.gripper_slave_id),
            debug=bool(args.gripper_debug),
        )
    else:
        gripper = Robotiq2F85ModbusRtuClient(
            args.gripper_port,
            baudrate=int(args.gripper_baudrate),
            slave_id=int(args.gripper_slave_id),
            timeout=float(args.gripper_timeout),
            retries=int(args.gripper_retries),
        )
    try:
        gripper.connect()
        gripper.activate()
        print(f"Robotiq gripper connected on {args.gripper_port} with backend={args.gripper_backend}.", flush=True)
        return gripper
    except Exception as exc:
        gripper.close()
        message = (
            f"Warning: Robotiq gripper did not respond on {args.gripper_port}: {exc}. "
            "Continuing with virtual gripper target recording only."
        )
        if args.strict_gripper:
            raise RuntimeError(message) from exc
        print(message, flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record real FR5 + Astra + gripper demonstration episodes for behavior cloning. "
            "With --xbox-teleop, recording starts on first gamepad input and B ends the segment."
        )
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--output-dir", type=str, default=DEFAULT_IL_DEMO_DIR.as_posix())
    parser.add_argument("--episode-name", type=str, default="")
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP)
    parser.add_argument("--speed-percent", type=float, default=5.0)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run the collector; 0 records until Ctrl+C")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames per episode; 0 means unlimited")
    parser.add_argument("--min-episode-frames", type=int, default=3)
    parser.add_argument(
        "--keep-discarded-episodes",
        action="store_true",
        help="Keep directories for episodes shorter than --min-episode-frames",
    )
    parser.add_argument("--real-rgb-source", choices=["live", "latest"], default="live")
    parser.add_argument("--real-rgb-width", type=int, default=640)
    parser.add_argument("--real-rgb-height", type=int, default=480)
    parser.add_argument("--real-rgb-fps", type=int, default=15)
    parser.add_argument("--allow-latest-fallback", action="store_true")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--xbox-teleop", action="store_true", help="Use an Xbox/gamepad to teleoperate the real FR5 while recording")
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--button-a", type=int, default=0)
    parser.add_argument("--button-b", type=int, default=1)
    parser.add_argument("--button-x", type=int, default=2)
    parser.add_argument("--button-y", type=int, default=3)
    parser.add_argument("--button-lb", type=int, default=4)
    parser.add_argument("--button-rb", type=int, default=5)
    parser.add_argument("--gamepad-debug", action="store_true", help="Print recognized X/Y/B/A events during collection")
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--teleop-dt", type=float, default=0.01)
    parser.add_argument("--teleop-alpha", type=float, default=0.25)
    parser.add_argument("--start-motion-threshold", type=float, default=0.08, help="Filtered gamepad magnitude needed to start a new episode")
    parser.add_argument("--motion-command-threshold", type=float, default=0.03, help="Filtered gamepad magnitude needed to send ServoCart")
    parser.add_argument("--scale-xy", type=float, default=3.0, help="FR5_xbox-style ServoCart X/Y increment scale")
    parser.add_argument("--scale-z", type=float, default=2.5)
    parser.add_argument("--scale-rot", type=float, default=1.0)
    parser.add_argument("--scale-pitch", type=float, default=1.2)
    parser.add_argument("--scale-euler", type=float, default=1.0)
    parser.add_argument("--return-vel", type=float, default=35.0)
    parser.add_argument("--return-acc", type=float, default=35.0)
    parser.add_argument("--return-ovl", type=float, default=60.0)
    parser.add_argument("--return-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--return-timeout", type=float, default=120.0)
    parser.add_argument("--initial-gripper-closure", type=float, default=0.0, help="0=open, 1=closed")
    parser.add_argument("--execute-gripper", action="store_true", help="Send USB Robotiq commands while recording")
    parser.add_argument("--strict-gripper", action="store_true", help="Abort if the USB Robotiq gripper cannot be activated")
    parser.add_argument("--gripper-backend", choices=["raw", "pyrobotiq"], default="raw")
    parser.add_argument("--gripper-debug", action="store_true")
    parser.add_argument("--gripper-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--gripper-baudrate", type=int, default=115200)
    parser.add_argument("--gripper-slave-id", type=int, default=9)
    parser.add_argument("--gripper-timeout", type=float, default=0.5)
    parser.add_argument("--gripper-retries", type=int, default=2)
    parser.add_argument("--gripper-speed", type=int, default=255)
    parser.add_argument("--gripper-force", type=int, default=150)
    parser.add_argument("--gripper-step", type=float, default=0.06)
    parser.add_argument("--gripper-send-period", type=float, default=0.03)
    parser.add_argument("--gripper-send-epsilon", type=float, default=0.004)
    args = parser.parse_args()

    if args.hz <= 0.0:
        raise RuntimeError("--hz must be positive")
    if args.xbox_teleop and not (0.001 <= args.teleop_dt <= 0.05):
        raise RuntimeError("--teleop-dt must be between 0.001 and 0.05 seconds")

    cv2 = require_cv2()
    config = load_config(args.config)
    model, data, body, qpos_ids, arm_act_ids, gripper_act_ids = build_runtime(config)
    steps_per_frame = max(1, round((1.0 / float(args.hz)) / float(model.options.timestep)))

    arm = FairinoArmClient(args.robot_ip, speed_percent=float(args.speed_percent))
    rgb_source = RealRgbSource(
        source=str(args.real_rgb_source),
        image_path=None,
        width=int(args.real_rgb_width),
        height=int(args.real_rgb_height),
        fps=int(args.real_rgb_fps),
        allow_fallback=bool(args.allow_latest_fallback),
    )
    gamepad = None
    gripper = None
    gripper_commander: AsyncGripperCommander | None = None
    canceller = MotionCancelHandler() if args.xbox_teleop else None
    servo_started = False
    active_buffer: EpisodeBuffer | None = None
    saved_count = 0
    segment_index = 0
    current_gripper = float(np.clip(args.initial_gripper_closure, 0.0, 1.0))
    next_gripper_send = 0.0
    last_rgb = None

    try:
        arm.connect()
        if canceller is not None:
            canceller.set_arm(arm)
            canceller.install()
        err = arm.get_robot_error_code()
        if err is not None and any(int(v) != 0 for v in err):
            raise RuntimeError(f"Controller reports error code {err}; refusing collection.")

        if args.xbox_teleop:
            from fr5_gamepad import XboxGamepad

            gamepad = XboxGamepad(
                joystick_index=int(args.joystick_index),
                deadzone=float(args.deadzone),
                alpha=float(args.teleop_alpha),
                button_a=int(args.button_a),
                button_b=int(args.button_b),
                button_x=int(args.button_x),
                button_y=int(args.button_y),
                button_lb=int(args.button_lb),
                button_rb=int(args.button_rb),
            )
            print_xbox_keymap()
            arm.set_mode(0)
            arm.robot_enable(1)
            time.sleep(0.5)
            arm.servo_start()
            servo_started = True
            canceller.set_servo_started(True)

        gripper = connect_gripper_best_effort(args)
        if gripper is not None:
            gripper_commander = AsyncGripperCommander(
                gripper,
                speed=int(args.gripper_speed),
                force=int(args.gripper_force),
                min_period_s=float(args.gripper_send_period),
                epsilon=float(args.gripper_send_epsilon),
                debug=bool(args.gripper_debug),
            )
            gripper_commander.start()
            gripper_commander.command(current_gripper)

        last_rgb = rgb_source.start()
        for _ in range(max(0, int(args.warmup_frames))):
            frame = rgb_source.read(timeout_ms=max(20, int(1000 / max(1, int(args.real_rgb_fps)))))
            if frame is not None:
                last_rgb = frame

        start_time = time.monotonic()
        next_loop_time = start_time
        next_record_time = start_time
        next_rgb_drain = start_time
        if args.xbox_teleop:
            print("Collector is idle. Move the gamepad to start episode 0.", flush=True)
        else:
            active_buffer = EpisodeBuffer(
                make_episode_dir(args.output_dir, args.episode_name or None, segment_index=None),
                start_time=time.monotonic(),
            )
            print(f"Recording read-only episode to {active_buffer.episode_dir}", flush=True)

        while True:
            now = time.monotonic()
            if args.duration > 0.0 and now - start_time >= float(args.duration):
                break
            if now >= next_rgb_drain:
                frame = rgb_source.read(timeout_ms=5)
                if frame is not None:
                    last_rgb = frame
                next_rgb_drain = now + 0.1
            if args.xbox_teleop and gamepad is not None:
                canceller.check()
                cmd = gamepad.read()
                command_has_motion = cmd.arm_motion_level >= float(args.motion_command_threshold)
                requested_start = cmd.arm_motion_level >= float(args.start_motion_threshold) or cmd.close_gripper or cmd.open_gripper

                if cmd.return_home:
                    print("[gamepad] B detected: finish current episode and return to initial_qpos.", flush=True)
                    if active_buffer is not None:
                        saved = save_if_valid(
                            active_buffer,
                            args,
                            source="real_xbox",
                            segment_index=segment_index,
                            gripper_physical_enabled=gripper is not None,
                        )
                        saved_count += int(saved)
                        active_buffer = None
                        segment_index += 1
                    if servo_started:
                        arm.servo_end_best_effort()
                        canceller.set_servo_started(False)
                        servo_started = False
                    move_to_initial_from_config(
                        arm,
                        config,
                        vel=float(args.return_vel),
                        acc=float(args.return_acc),
                        ovl=float(args.return_ovl),
                        tolerance_deg=float(args.return_tolerance_deg),
                        timeout_s=float(args.return_timeout),
                    )
                    arm.servo_start()
                    servo_started = True
                    canceller.set_servo_started(True)
                    next_record_time = time.monotonic()
                    print(f"Collector is idle. Move the gamepad to start episode {segment_index}.", flush=True)
                    time.sleep(0.3)
                    continue

                if requested_start and active_buffer is None:
                    active_buffer = EpisodeBuffer(
                        make_episode_dir(args.output_dir, args.episode_name or None, segment_index=segment_index),
                        start_time=time.monotonic(),
                    )
                    next_record_time = time.monotonic()
                    print(f"Started episode {segment_index}: {active_buffer.episode_dir}", flush=True)

                if cmd.close_gripper:
                    current_gripper = min(1.0, current_gripper + float(args.gripper_step))
                    if args.gamepad_debug:
                        print(f"[gamepad] X detected: close gripper target={current_gripper:.3f}", flush=True)
                elif cmd.open_gripper:
                    current_gripper = max(0.0, current_gripper - float(args.gripper_step))
                    if args.gamepad_debug:
                        print(f"[gamepad] Y detected: open gripper target={current_gripper:.3f}", flush=True)
                if gripper_commander is not None and (cmd.close_gripper or cmd.open_gripper) and now >= next_gripper_send:
                    gripper_commander.command(current_gripper)
                    next_gripper_send = now + float(args.gripper_send_period)
                elif gripper is None and (cmd.close_gripper or cmd.open_gripper) and args.gamepad_debug:
                    print("[gamepad] gripper is virtual only; physical Robotiq is not connected.", flush=True)
                if cmd.status:
                    print(
                        f"[status] episode={'none' if active_buffer is None else segment_index} "
                        f"gripper={current_gripper:.3f}",
                        flush=True,
                    )
                if command_has_motion:
                    pose = arm.get_actual_tcp_pose()
                    if pose is None:
                        raise RuntimeError("Could not read TCP pose for Xbox ServoCart teleop.")
                    target = list(pose)
                    target[0] += cmd.vx * float(args.scale_xy)
                    target[1] += cmd.vy * float(args.scale_xy)
                    target[2] += cmd.vz * float(args.scale_z)
                    target[3] += cmd.vroll * float(args.scale_euler)
                    target[4] += cmd.vpitch * float(args.scale_pitch)
                    target[5] += cmd.vyaw * float(args.scale_rot)
                    arm.servo_cart(target, cmd_t=float(args.teleop_dt), idx=int(now * 1000) % 1000000)

            if active_buffer is not None and time.monotonic() >= next_record_time:
                last_rgb = record_one_sample(
                    active_buffer,
                    cv2=cv2,
                    rgb_source=rgb_source,
                    last_rgb=last_rgb,
                    arm=arm,
                    current_gripper=current_gripper,
                    model=model,
                    data=data,
                    body=body,
                    qpos_ids=qpos_ids,
                    arm_act_ids=arm_act_ids,
                    gripper_act_ids=gripper_act_ids,
                    config=config,
                    steps_per_frame=steps_per_frame,
                    real_rgb_fps=int(args.real_rgb_fps),
                )
                next_record_time += 1.0 / float(args.hz)
                if int(args.max_frames) > 0 and active_buffer.frames >= int(args.max_frames):
                    saved = save_if_valid(
                        active_buffer,
                        args,
                        source="real_xbox" if args.xbox_teleop else "real_readback",
                        segment_index=segment_index,
                        gripper_physical_enabled=gripper is not None,
                    )
                    saved_count += int(saved)
                    active_buffer = None
                    segment_index += 1
                    if args.xbox_teleop:
                        print(f"Max frames reached. Move the gamepad to start episode {segment_index}.", flush=True)
                    else:
                        break

            if args.xbox_teleop:
                next_loop_time += float(args.teleop_dt)
                sleep_s = next_loop_time - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
            else:
                sleep_s = next_record_time - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("Recording stopped by Ctrl+C.", flush=True)
    finally:
        if active_buffer is not None:
            saved = save_if_valid(
                active_buffer,
                args,
                source="real_xbox" if args.xbox_teleop else "real_readback",
                segment_index=segment_index,
                gripper_physical_enabled=gripper is not None,
            )
            saved_count += int(saved)
        if servo_started:
            arm.servo_end_best_effort()
            if canceller is not None:
                canceller.set_servo_started(False)
        if gamepad is not None:
            gamepad.close()
        if gripper_commander is not None:
            gripper_commander.stop()
        if gripper is not None:
            gripper.close()
        rgb_source.stop()
        arm.close()

    print(f"Collection finished. saved_episodes={saved_count}", flush=True)


if __name__ == "__main__":
    main()
