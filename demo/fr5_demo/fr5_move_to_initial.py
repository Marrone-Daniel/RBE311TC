from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from arm_control import DEFAULT_CONFIG, build_runtime, load_config, resolve_demo_path
from fr5_sync_sdk import (
    DEFAULT_ROBOT_IP,
    FairinoArmClient,
    MotionCancelHandler,
    actuator_limits_rad,
    diagnose_fairino_network,
    format_fairino_diagnosis,
    validate_targets_against_limits,
)


def has_robot_error(error_code: list[int] | None) -> bool:
    return bool(error_code) and any(int(v) != 0 for v in error_code)


def format_values(values: np.ndarray, *, unit: str) -> str:
    return "[" + ", ".join(f"{float(v):.3f}{unit}" for v in values) + "]"


def wait_until_reached(
    arm: FairinoArmClient,
    target_deg: np.ndarray,
    *,
    tolerance_deg: float,
    motion_timeout: float,
    no_motion_timeout: float,
    progress_epsilon_deg: float,
    poll_dt: float,
) -> None:
    start = time.monotonic()
    last = None
    last_progress = start
    best_error = float("inf")
    while True:
        actual = arm.get_actual_joint_deg()
        now = time.monotonic()
        if actual is None:
            raise RuntimeError("Lost FR5 joint readback while monitoring MoveJ")
        actual_deg = np.asarray(actual, dtype=np.float64)
        err = target_deg - actual_deg
        max_err = float(np.max(np.abs(err)))
        error_improvement = best_error - max_err
        best_error = min(best_error, max_err)
        motion_done = arm.get_motion_done()
        error_code = arm.get_robot_error_code()
        print(
            f"  monitor t={now - start:5.1f}s max_err={max_err:7.3f}deg "
            f"motion_done={motion_done} err_code={error_code}",
            flush=True,
        )
        if has_robot_error(error_code) and motion_done == 1 and max_err > max(float(tolerance_deg), 2.0):
            arm.stop_motion()
            raise RuntimeError(
                f"Robot reports error code {error_code} and did not start moving. "
                "Clear the controller fault on the teach pendant or run with --reset-errors, then retry at low speed."
            )
        if max_err <= float(tolerance_deg):
            print("Initial pose reached within tolerance.")
            return
        if motion_done == 1 and max_err <= max(float(tolerance_deg), 2.0):
            print("Motion done and target is close enough.")
            return
        joint_step = float(np.max(np.abs(actual_deg - last))) if last is not None else 0.0
        if joint_step > float(progress_epsilon_deg) or error_improvement > float(progress_epsilon_deg):
            last_progress = now
        last = actual_deg
        if now - start > float(motion_timeout):
            arm.stop_motion()
            raise RuntimeError(f"MoveJ monitor timed out. best_error={best_error:.3f}deg")
        if now - last_progress > float(no_motion_timeout):
            arm.stop_motion()
            raise RuntimeError(
                "MoveJ command was accepted but joints did not move. "
                "Check robot enable state, automatic mode, safety stop, teach pendant prompts, and controller error code."
            )
        time.sleep(float(poll_dt))


def main() -> None:
    parser = argparse.ArgumentParser(description="Move the physical FR5 slowly to the configured initial pose")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP)
    parser.add_argument("--speed-percent", type=float, default=20.0, help="Global Fairino speed percentage")
    parser.add_argument("--vel", type=float, default=35.0, help="MoveJ velocity percentage")
    parser.add_argument("--acc", type=float, default=35.0, help="MoveJ acceleration percentage")
    parser.add_argument("--ovl", type=float, default=60.0, help="MoveJ override percentage")
    parser.add_argument("--blend-t", type=float, default=0.0, help="MoveJ blendT; 0 is non-blocking, -1 blocks until reached")
    parser.add_argument("--rpc-timeout", type=float, default=5.0, help="Timeout for the MoveJ XML-RPC request")
    parser.add_argument("--motion-timeout", type=float, default=240.0, help="Max seconds to wait for the target pose")
    parser.add_argument("--no-motion-timeout", type=float, default=15.0, help="Abort if joints do not make measurable progress")
    parser.add_argument("--progress-epsilon-deg", type=float, default=0.005, help="Minimum joint/progress change counted as movement")
    parser.add_argument("--tolerance-deg", type=float, default=1.0)
    parser.add_argument("--max-delta-deg", type=float, default=90.0, help="Abort if any joint is farther than this")
    parser.add_argument("--allow-no-readback", action="store_true", help="Allow motion if current joint readback is unavailable")
    parser.add_argument("--diagnose-only", action="store_true", help="Only test Fairino network ports; do not move")
    parser.add_argument("--reset-errors", action="store_true", help="Call ResetAllError before enabling and moving")
    parser.add_argument(
        "--prepare-controller",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set automatic mode and enable the robot before MoveJ",
    )
    parser.add_argument("--execute-real", action="store_true", help="Actually send MoveJ to the physical FR5")
    args = parser.parse_args()

    if args.speed_percent <= 0.0 or args.speed_percent > 100.0:
        raise RuntimeError("--speed-percent must be in (0, 100] for this slow initial-pose script")
    if args.vel <= 0.0 or args.vel > 100.0:
        raise RuntimeError("--vel must be in (0, 100] for this slow initial-pose script")
    if args.acc <= 0.0 or args.acc > 100.0:
        raise RuntimeError("--acc must be in (0, 100] for this slow initial-pose script")
    if args.ovl <= 0.0 or args.ovl > 100.0:
        raise RuntimeError("--ovl must be in (0, 100] for this slow initial-pose script")

    config = load_config(args.config)
    model, _, _, _, arm_act_ids, _ = build_runtime(config)
    target_rad = np.asarray(config["initial_qpos"], dtype=np.float64)
    if target_rad.shape != (6,):
        raise RuntimeError(f"initial_qpos must contain 6 joints, got shape {target_rad.shape}")

    lows, highs = actuator_limits_rad(model, arm_act_ids)
    validate_targets_against_limits(target_rad.reshape(1, -1), lows, highs)

    target_deg = np.rad2deg(target_rad)
    print("FR5 initial target from config:")
    print(f"  config: {resolve_demo_path(args.config)}")
    print(f"  target rad: {format_values(target_rad, unit='rad')}")
    print(f"  target deg: {format_values(target_deg, unit='deg')}")
    print(
        f"  speed_percent={args.speed_percent:.1f}, vel={args.vel:.1f}, "
        f"acc={args.acc:.1f}, ovl={args.ovl:.1f}, blendT={args.blend_t:.1f}"
    )

    if args.diagnose_only:
        diagnosis = diagnose_fairino_network(args.robot_ip)
        print(format_fairino_diagnosis(args.robot_ip, diagnosis))
        return

    if not args.execute_real:
        print("Dry run only. Add --execute-real to move the physical FR5.")
        return

    canceller = MotionCancelHandler()
    canceller.install()
    arm = FairinoArmClient(args.robot_ip, speed_percent=float(args.speed_percent))
    canceller.set_arm(arm)
    try:
        arm.connect()
        if args.reset_errors:
            print("Resetting controller errors with ResetAllError().")
            arm.reset_all_error()
            time.sleep(0.5)
        if args.prepare_controller:
            print("Preparing controller: Mode(0 automatic), RobotEnable(1).")
            arm.set_mode(0)
            arm.robot_enable(1)
            time.sleep(0.5)
        error_code = arm.get_robot_error_code()
        if has_robot_error(error_code):
            raise RuntimeError(
                f"Controller still reports error code {error_code}; refusing MoveJ. "
                "Resolve the fault on the teach pendant, or try --reset-errors if it is a resettable fault."
            )
        actual = arm.get_actual_joint_deg()
        if actual is None:
            if not args.allow_no_readback:
                raise RuntimeError(
                    "Could not read current FR5 joint angles. Refusing to move. "
                    "Use --allow-no-readback only if you have independently verified the robot pose."
                )
            print("Warning: current joint readback unavailable; proceeding because --allow-no-readback was set.")
        else:
            actual_deg = np.asarray(actual, dtype=np.float64)
            delta_deg = target_deg - actual_deg
            max_delta = float(np.max(np.abs(delta_deg)))
            print(f"  current deg: {format_values(actual_deg, unit='deg')}")
            print(f"  delta deg: {format_values(delta_deg, unit='deg')}")
            if max_delta > float(args.max_delta_deg):
                raise RuntimeError(
                    f"Target is too far from current pose: max_delta={max_delta:.2f}deg, "
                    f"limit={args.max_delta_deg:.2f}deg"
                )

        print("Sending non-blocking MoveJ and monitoring real joint readback.")
        arm.move_j(
            target_deg,
            vel=float(args.vel),
            acc=float(args.acc),
            ovl=float(args.ovl),
            blend_t=float(args.blend_t),
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
    except KeyboardInterrupt:
        print("Move-to-initial cancelled. StopMotion was sent best-effort.", flush=True)
    finally:
        arm.close()


if __name__ == "__main__":
    main()
