from __future__ import annotations

import argparse
import time

from fr5_sync_sdk import DEFAULT_ROBOT_IP, FairinoArmClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Immediately send best-effort StopMotion to the physical FR5")
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP)
    parser.add_argument("--speed-percent", type=float, default=5.0)
    parser.add_argument("--servo-end", action="store_true", help="Also send ServoMoveEnd before StopMotion")
    parser.add_argument("--reset-errors", action="store_true", help="Call ResetAllError after StopMotion")
    args = parser.parse_args()

    arm = FairinoArmClient(args.robot_ip, speed_percent=float(args.speed_percent))
    try:
        arm.connect()
        if args.servo_end:
            arm.servo_end_best_effort()
        arm.stop_motion_best_effort()
        time.sleep(0.2)
        print(f"motion_done={arm.get_motion_done()} err_code={arm.get_robot_error_code()}", flush=True)
        if args.reset_errors:
            arm.reset_all_error()
            time.sleep(0.2)
            print(f"after reset err_code={arm.get_robot_error_code()}", flush=True)
    finally:
        arm.close()


if __name__ == "__main__":
    main()
