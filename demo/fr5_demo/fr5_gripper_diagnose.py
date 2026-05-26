from __future__ import annotations

import argparse
import time

from fr5_sync_sdk import Robotiq2F85ModbusRtuClient, Robotiq2F85PyrobotiqClient


def try_command(args, closure: float) -> bool:
    if args.backend == "pyrobotiq":
        gripper = Robotiq2F85PyrobotiqClient(args.port, slave_id=int(args.slave_id), debug=bool(args.debug))
    else:
        gripper = Robotiq2F85ModbusRtuClient(
            args.port,
            baudrate=int(args.baudrate),
            slave_id=int(args.slave_id),
            timeout=float(args.timeout),
            retries=int(args.retries),
        )
    try:
        print(
            f"Opening {args.port}: backend={args.backend}, baudrate={args.baudrate}, slave_id={args.slave_id}, "
            f"timeout={args.timeout}, retries={args.retries}",
            flush=True,
        )
        gripper.connect()
        if args.activate:
            print("Sending Robotiq activate sequence...", flush=True)
            gripper.activate()
        print(f"Sending closure={closure:.3f}...", flush=True)
        gripper.command_closure(float(closure), speed=int(args.speed), force=int(args.force))
        print("Robotiq command write acknowledged.", flush=True)
        return True
    except Exception as exc:
        print(f"Robotiq command failed: {exc}", flush=True)
        return False
    finally:
        gripper.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose USB Robotiq 2F-85 Modbus RTU communication.")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--backend", choices=["raw", "pyrobotiq"], default="raw")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--slave-id", type=int, default=9)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--speed", type=int, default=255)
    parser.add_argument("--force", type=int, default=150)
    parser.add_argument("--closure", type=float, default=0.0)
    parser.add_argument("--activate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep", action="store_true", help="Try open, half, close, open")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        ok = True
        for closure in (0.0, 0.5, 1.0, 0.0):
            ok = try_command(args, closure) and ok
            time.sleep(0.8)
        if not ok:
            raise SystemExit(1)
    else:
        if not try_command(args, float(args.closure)):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
