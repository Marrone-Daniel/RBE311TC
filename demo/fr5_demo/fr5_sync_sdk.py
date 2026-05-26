from __future__ import annotations

import argparse
import json
import math
import os
import time
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from motrixsim import forward_kinematic, step as sim_step

from arm_control import (
    DEFAULT_CONFIG,
    build_runtime,
    load_config,
    load_replay_qpos,
    resolve_demo_path,
    set_arm_qpos_and_ctrl,
    set_gripper,
    site_position,
)


DEFAULT_ROBOT_IP = "192.168.58.2"
ROBOT_DOF = 6
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_FAIRINO_SDK_DIR = PROJECT_ROOT / "fairino-python-sdk-master" / "linux"
FAIRINO_XMLRPC_PORT = 20003
FAIRINO_STATE_PORT = 20004


class FairinoConnectionError(RuntimeError):
    pass


@dataclass
class SafetyReport:
    frames: int
    duration_s: float
    max_step_rad: float
    max_step_deg: float
    min_tcp_z: float
    max_tcp_speed_mps: float


def normalize_arm_trajectory(replay: np.ndarray | None, config: dict, qpos_ids: np.ndarray) -> np.ndarray:
    if replay is None:
        initial = np.asarray(config["initial_qpos"], dtype=np.float64)
        target = np.asarray(config.get("demo_target_qpos", config["initial_qpos"]), dtype=np.float64)
        outbound = np.linspace(initial, target, 120, dtype=np.float64)
        inbound = np.linspace(target, initial, 120, dtype=np.float64)[1:]
        return np.vstack([outbound, inbound])

    replay = np.asarray(replay, dtype=np.float64)
    if replay.ndim != 2:
        raise ValueError(f"Replay qpos must be 2D, got shape {replay.shape}")
    if replay.shape[1] == ROBOT_DOF:
        return replay
    if replay.shape[1] > int(np.max(qpos_ids)):
        return replay[:, qpos_ids]
    raise ValueError(f"Replay has {replay.shape[1]} columns; expected 6 arm joints or full dof_pos.")


def resample_trajectory(q: np.ndarray, source_dt: float, target_dt: float) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape[0] < 2:
        return q.copy()
    source_dt = float(source_dt)
    target_dt = float(target_dt)
    if source_dt <= 0.0 or target_dt <= 0.0:
        raise ValueError("source_dt and target_dt must be positive")
    duration = source_dt * (q.shape[0] - 1)
    count = max(2, int(math.ceil(duration / target_dt)) + 1)
    src_t = np.linspace(0.0, duration, q.shape[0])
    dst_t = np.linspace(0.0, duration, count)
    out = np.empty((count, q.shape[1]), dtype=np.float64)
    for joint in range(q.shape[1]):
        out[:, joint] = np.interp(dst_t, src_t, q[:, joint])
    return out


def actuator_limits_rad(model, arm_act_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    limits = np.asarray(model.actuator_ctrl_limits, dtype=np.float64)
    return limits[0, arm_act_ids], limits[1, arm_act_ids]


def validate_targets_against_limits(q: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> None:
    below = q < lows[None, :]
    above = q > highs[None, :]
    if below.any() or above.any():
        bad = np.argwhere(below | above)[0]
        frame, joint = int(bad[0]), int(bad[1])
        raise RuntimeError(
            "Trajectory violates simulated actuator limits: "
            f"frame={frame}, joint={joint + 1}, value={q[frame, joint]:.6f}, "
            f"limit=[{lows[joint]:.6f}, {highs[joint]:.6f}] rad"
        )


def run_sim_safety_check(
    config: dict,
    trajectory: np.ndarray,
    *,
    min_tcp_z: float,
    max_step_rad: float,
    max_tcp_speed: float,
    control_dt: float | None = None,
) -> tuple[SafetyReport, tuple]:
    runtime = build_runtime(config)
    model, data, body, qpos_ids, arm_act_ids, gripper_act_ids = runtime
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.shape[1] != ROBOT_DOF:
        raise ValueError(f"Arm trajectory must have 6 columns, got shape {trajectory.shape}")

    lows, highs = actuator_limits_rad(model, arm_act_ids)
    validate_targets_against_limits(trajectory, lows, highs)

    diffs = np.diff(trajectory, axis=0)
    max_step = float(np.max(np.abs(diffs))) if diffs.size else 0.0
    if max_step > float(max_step_rad):
        raise RuntimeError(
            f"Trajectory step too large for preflight: {max_step:.6f} rad "
            f"({math.degrees(max_step):.3f} deg), limit={max_step_rad:.6f} rad"
        )

    sim_dt = float(config.get("control_dt", 0.02) if control_dt is None else control_dt)
    steps_per_ctrl = max(1, round(sim_dt / float(model.options.timestep)))
    tcp_positions = []
    for q in trajectory:
        set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q.astype(np.float32))
        set_gripper(data, model, body, gripper_act_ids, float(config.get("gripper_opening", 0.0)))
        for _ in range(steps_per_ctrl):
            sim_step(model, data)
        tcp_positions.append(site_position(model, data, config["tcp_site"]).astype(np.float64))

    tcp = np.vstack(tcp_positions)
    min_z = float(np.min(tcp[:, 2]))
    if min_z < float(min_tcp_z):
        raise RuntimeError(f"TCP z safety check failed: min_z={min_z:.4f}m, required >= {min_tcp_z:.4f}m")

    tcp_d = np.diff(tcp, axis=0)
    tcp_speed = np.linalg.norm(tcp_d, axis=1) / sim_dt if tcp_d.size else np.asarray([0.0])
    max_speed = float(np.max(tcp_speed))
    if max_speed > float(max_tcp_speed):
        raise RuntimeError(
            f"TCP speed safety check failed: max={max_speed:.4f}m/s, required <= {max_tcp_speed:.4f}m/s"
        )

    report = SafetyReport(
        frames=int(trajectory.shape[0]),
        duration_s=float(sim_dt * max(0, trajectory.shape[0] - 1)),
        max_step_rad=max_step,
        max_step_deg=math.degrees(max_step),
        min_tcp_z=min_z,
        max_tcp_speed_mps=max_speed,
    )
    return report, runtime


class FairinoArmClient:
    def __init__(self, ip: str, *, speed_percent: float):
        self.ip = ip
        self.speed_percent = float(speed_percent)
        self.robot = None

    def connect(self) -> None:
        add_local_fairino_sdk_to_path()
        diagnosis = diagnose_fairino_network(self.ip)
        if not diagnosis["xmlrpc_open"]:
            raise FairinoConnectionError(format_fairino_diagnosis(self.ip, diagnosis))
        try:
            from fairino import Robot
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import Fairino Python SDK. Put the official SDK at "
                f"{LOCAL_FAIRINO_SDK_DIR} or install it into this uv environment before using --execute-real."
            ) from exc
        last_rpc = None
        for attempt in range(3):
            if hasattr(Robot.RPC, "is_conect"):
                Robot.RPC.is_conect = True
            self.robot = Robot.RPC(self.ip)
            last_rpc = self.robot
            if getattr(self.robot, "is_conect", True) is not False:
                break
            print(
                f"Fairino SDK RPC init timed out; retrying ({attempt + 1}/3). "
                "If this repeats, check controller realtime port 20004 and RPC service.",
                flush=True,
            )
            try:
                if hasattr(self.robot, "CloseRPC"):
                    self.robot.CloseRPC()
            except Exception:
                pass
            time.sleep(1.0)
        if self.robot is None:
            self.robot = last_rpc
        if getattr(self.robot, "is_conect", True) is False:
            raise FairinoConnectionError(format_fairino_diagnosis(self.ip, diagnosis, sdk_rpc_failed=True))
        self._check_return(self.robot.SetSpeed(self.speed_percent), "SetSpeed")

    def close(self) -> None:
        if self.robot is not None and hasattr(self.robot, "CloseRPC"):
            self.robot.CloseRPC()
        self.robot = None

    def get_actual_joint_deg(self) -> list[float] | None:
        if self.robot is None:
            return None
        for args in ((0,), tuple()):
            try:
                ret = self.robot.GetActualJointPosDegree(*args)
            except TypeError:
                continue
            if isinstance(ret, tuple) and len(ret) >= 2 and int(ret[0]) == 0:
                return [float(v) for v in ret[1][:ROBOT_DOF]]
        return None

    def get_actual_tcp_pose(self) -> list[float] | None:
        if self.robot is None:
            return None
        for args in ((0,), tuple()):
            try:
                ret = self.robot.GetActualTCPPose(*args)
            except TypeError:
                continue
            if isinstance(ret, tuple) and len(ret) >= 2 and int(ret[0]) == 0:
                return [float(v) for v in ret[1][:6]]
        return None

    def set_mode(self, state: int) -> None:
        if self.robot is None:
            raise RuntimeError("Fairino arm is not connected")
        self._check_return(self.robot.Mode(int(state)), "Mode")

    def robot_enable(self, state: int) -> None:
        if self.robot is None:
            raise RuntimeError("Fairino arm is not connected")
        self._check_return(self.robot.RobotEnable(int(state)), "RobotEnable")

    def reset_all_error(self) -> None:
        if self.robot is None:
            raise RuntimeError("Fairino arm is not connected")
        self._check_return(self.robot.ResetAllError(), "ResetAllError")

    def stop_motion(self) -> None:
        if self.robot is None:
            return
        self._check_return(self.robot.StopMotion(), "StopMotion")

    def stop_motion_best_effort(self) -> None:
        try:
            self.stop_motion()
            print("Sent StopMotion().", flush=True)
        except Exception as exc:
            print(f"Warning: StopMotion() failed during cancel: {exc}", flush=True)

    def get_robot_error_code(self) -> list[int] | None:
        if self.robot is None:
            return None
        ret = self.robot.GetRobotErrorCode()
        if isinstance(ret, tuple) and len(ret) >= 2 and int(ret[0]) == 0:
            return [int(v) for v in ret[1]]
        return None

    def get_motion_done(self) -> int | None:
        if self.robot is None:
            return None
        ret = self.robot.GetRobotMotionDone()
        if isinstance(ret, tuple) and len(ret) >= 2 and int(ret[0]) == 0:
            return int(ret[1])
        return None

    def move_j(
        self,
        target_deg: Iterable[float],
        *,
        vel: float,
        acc: float,
        ovl: float,
        blend_t: float = -1.0,
        rpc_timeout: float | None = None,
    ) -> None:
        if self.robot is None:
            raise RuntimeError("Fairino arm is not connected")
        q = [float(v) for v in target_deg]
        old_timeout = socket.getdefaulttimeout()
        if rpc_timeout is not None:
            socket.setdefaulttimeout(float(rpc_timeout))
        try:
            try:
                ret = self.robot.MoveJ(
                    joint_pos=q,
                    tool=0,
                    user=0,
                    vel=float(vel),
                    acc=float(acc),
                    ovl=float(ovl),
                    exaxis_pos=[0.0, 0.0, 0.0, 0.0],
                    blendT=float(blend_t),
                    offset_flag=0,
                    offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                )
            except TypeError:
                ret = self.robot.MoveJ(q, 0, 0, vel=float(vel), acc=float(acc), ovl=float(ovl), blendT=float(blend_t))
        finally:
            if rpc_timeout is not None:
                socket.setdefaulttimeout(old_timeout)
        self._check_return(ret, "MoveJ")

    def servo_trajectory(self, trajectory_deg: np.ndarray, *, cmd_t: float, vel: float) -> None:
        if self.robot is None:
            raise RuntimeError("Fairino arm is not connected")
        self.servo_start()
        try:
            for idx, q in enumerate(np.asarray(trajectory_deg, dtype=np.float64)):
                self.servo_j(q, idx=idx, cmd_t=cmd_t, vel=vel)
                time.sleep(float(cmd_t))
        finally:
            self.servo_end()

    def servo_start(self) -> None:
        if self.robot is None:
            raise RuntimeError("Fairino arm is not connected")
        self._check_return(self.robot.ServoMoveStart(), "ServoMoveStart")

    def servo_j(self, target_deg: Iterable[float], *, idx: int, cmd_t: float, vel: float) -> None:
        if self.robot is None:
            raise RuntimeError("Fairino arm is not connected")
        target = [float(v) for v in target_deg]
        try:
            ret = self.robot.ServoJ(
                joint_pos=target,
                axisPos=[0.0, 0.0, 0.0, 0.0],
                acc=0.0,
                vel=float(vel),
                cmdT=float(cmd_t),
                filterT=0.0,
                gain=0.0,
                id=int(idx + 1),
            )
        except TypeError:
            ret = self.robot.ServoJ(
                target,
                acc=0.0,
                vel=float(vel),
                cmdT=float(cmd_t),
                filterT=0.0,
                gain=0.0,
            )
        self._check_return(ret, f"ServoJ[{idx}]")


    
    def servo_cart(self, target_pose: Iterable[float], *, cmd_t: float, idx: int = 0) -> None:
        if self.robot is None:
            raise RuntimeError("Fairino arm is not connected")

        target = [float(v) for v in target_pose]
        if len(target) != 6:
            raise ValueError(f"ServoCart target_pose must have 6 values, got {len(target)}")

        # The Python wrapper in this Fairino SDK forwards 9 values to XML-RPC,
        # while the controller endpoint used by the validated FR5_xbox script
        # accepts the raw 8-value form below.
        if hasattr(self.robot, "robot") and hasattr(self.robot.robot, "ServoCart"):
            ret = self.robot.robot.ServoCart(
                0,
                target,
                [0.0, 0.0, 0.0, 0.0],
                0.0,
                0.0,
                float(cmd_t),
                0.0,
                0.0,
            )
        else:
            ret = self.robot.ServoCart(
                0,
                target,
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                0.0,
                0.0,
                float(cmd_t),
                0.0,
            )

        self._check_return(ret, f"ServoCart[{idx}]")

    def servo_end(self) -> None:
        if self.robot is None:
            raise RuntimeError("Fairino arm is not connected")
        self._check_return(self.robot.ServoMoveEnd(), "ServoMoveEnd")

    def servo_end_best_effort(self) -> None:
        try:
            self.servo_end()
            print("Sent ServoMoveEnd().", flush=True)
        except Exception as exc:
            print(f"Warning: ServoMoveEnd() failed during cancel: {exc}", flush=True)

    @staticmethod
    def _check_return(ret, name: str) -> None:
        if ret is None:
            return
        if isinstance(ret, tuple):
            code = int(ret[0])
        else:
            code = int(ret)
        if code != 0:
            if code == -4:
                raise FairinoConnectionError(
                    f"Fairino {name} returned -4 / ERR_RPC_ERROR. "
                    "The robot controller XML-RPC service is not responding reliably."
                )
            raise RuntimeError(f"Fairino {name} returned error code {code}")


def add_local_fairino_sdk_to_path() -> Path | None:
    candidates = []
    env_dir = os.environ.get("FAIRINO_SDK_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend(
        [
            LOCAL_FAIRINO_SDK_DIR,
            Path.home() / "fairino-python-sdk-master" / "linux",
            Path.home() / "下载" / "fairino-python-sdk-master" / "linux",
        ]
    )
    sdk_dir = next((path for path in candidates if (path / "fairino" / "Robot.py").exists()), None)
    if sdk_dir is None:
        return None
    sdk_path = sdk_dir.as_posix()
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    return sdk_dir


class MotionCancelHandler:
    def __init__(self) -> None:
        self.arm: FairinoArmClient | None = None
        self.servo_started = False
        self.cancel_requested = False

    def set_arm(self, arm: FairinoArmClient | None) -> None:
        self.arm = arm

    def set_servo_started(self, started: bool) -> None:
        self.servo_started = bool(started)

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def check(self) -> None:
        if self.cancel_requested:
            raise KeyboardInterrupt("motion cancel requested")

    def _handle_signal(self, signum, frame) -> None:
        self.cancel_requested = True
        print(f"\nCancel requested by signal {signum}. Sending best-effort robot stop...", flush=True)
        if self.arm is not None:
            if self.servo_started:
                self.arm.servo_end_best_effort()
            self.arm.stop_motion_best_effort()
        raise KeyboardInterrupt("motion cancel requested")


def probe_tcp_port(host: str, port: int, *, timeout: float = 1.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout)):
            return True, "open"
    except socket.timeout:
        return False, "timeout"
    except OSError as exc:
        return False, str(exc)


def diagnose_fairino_network(host: str, *, timeout: float = 1.0) -> dict:
    xmlrpc_open, xmlrpc_detail = probe_tcp_port(host, FAIRINO_XMLRPC_PORT, timeout=timeout)
    state_open, state_detail = probe_tcp_port(host, FAIRINO_STATE_PORT, timeout=timeout)
    return {
        "xmlrpc_open": xmlrpc_open,
        "xmlrpc_detail": xmlrpc_detail,
        "state_open": state_open,
        "state_detail": state_detail,
    }


def format_fairino_diagnosis(host: str, diagnosis: dict, *, sdk_rpc_failed: bool = False) -> str:
    ports_open = bool(diagnosis["xmlrpc_open"] and diagnosis["state_open"])
    if ports_open and not sdk_rpc_failed:
        lines = [f"Fairino TCP ports are reachable at {host}."]
    else:
        lines = [f"Cannot establish a usable Fairino RPC connection to {host}."]
    lines.extend(
        [
            f"  TCP {FAIRINO_XMLRPC_PORT} XML-RPC command port: {diagnosis['xmlrpc_detail']}",
            f"  TCP {FAIRINO_STATE_PORT} realtime state port: {diagnosis['state_detail']}",
        ]
    )
    if sdk_rpc_failed:
        lines.append(
            "  TCP connect succeeded, but Fairino SDK GetControllerIP() timed out during Robot.RPC initialization."
        )
    if ports_open and not sdk_rpc_failed:
        lines.append("Network port diagnosis passed. Real motion still requires SDK readback checks before MoveJ.")
        return "\n".join(lines)
    lines.extend(
        [
            "Check these before moving the real robot:",
            f"  1. PC Ethernet/Wi-Fi interface is on the same subnet as {host}.",
            f"  2. The controller IP is really {host}.",
            "  3. Robot controller is powered on and its RPC service is enabled.",
            "  4. No firewall/VPN is blocking ports 20003 and 20004.",
            "  5. If connected by direct Ethernet, set the PC address to something like 192.168.58.100/24.",
        ]
    )
    return "\n".join(lines)


class Robotiq2F85ModbusRtuClient:
    def __init__(self, port: str, *, baudrate: int = 115200, slave_id: int = 9, timeout: float = 0.5, retries: int = 1):
        self.port = port
        self.baudrate = int(baudrate)
        self.slave_id = int(slave_id)
        self.timeout = float(timeout)
        self.retries = max(1, int(retries))
        self.serial = None

    def connect(self) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for USB Robotiq control. Install pyserial into the uv environment "
                "or omit --execute-gripper."
            ) from exc
        self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()
        self.serial = None

    def activate(self) -> None:
        self._write_registers(0x03E8, [0x0000, 0x0000, 0x0000])
        time.sleep(0.2)
        self._write_registers(0x03E8, [0x0100, 0x0000, 0x0000])
        time.sleep(1.0)

    def command_closure(self, closure: float, *, speed: int = 255, force: int = 150) -> None:
        closure = float(np.clip(closure, 0.0, 1.0))
        position = int(round(255.0 * closure))
        speed = int(np.clip(speed, 0, 255))
        force = int(np.clip(force, 0, 255))
        self._write_registers(0x03E8, [0x0900, (position << 8) | speed, force << 8])

    def _write_registers(self, address: int, values: list[int]) -> None:
        if self.serial is None:
            raise RuntimeError("Robotiq gripper is not connected")
        payload = bytearray(
            [
                self.slave_id,
                0x10,
                (address >> 8) & 0xFF,
                address & 0xFF,
                0x00,
                len(values),
                len(values) * 2,
            ]
        )
        for value in values:
            payload.extend([(value >> 8) & 0xFF, value & 0xFF])
        crc = self._crc16(payload)
        payload.extend([crc & 0xFF, (crc >> 8) & 0xFF])
        last_response = b""
        for _ in range(self.retries):
            self.serial.reset_input_buffer()
            self.serial.write(bytes(payload))
            self.serial.flush()
            response = self.serial.read(8)
            last_response = response
            if len(response) >= 8:
                return
            time.sleep(0.05)
        raise RuntimeError(
            "Robotiq Modbus RTU write timed out "
            f"(port={self.port}, baudrate={self.baudrate}, slave_id={self.slave_id}, "
            f"timeout={self.timeout}s, response={last_response.hex() or 'empty'})"
        )

    @staticmethod
    def _crc16(data: bytes | bytearray) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc & 0xFFFF


class Robotiq2F85PyrobotiqClient:
    def __init__(self, port: str, *, slave_id: int = 9, debug: bool = False):
        self.port = port
        self.slave_id = int(slave_id)
        self.debug = bool(debug)
        self._module = None
        self.gripper = None

    def connect(self) -> None:
        try:
            import pyrobotiqgripper as rq
        except ImportError as exc:
            raise RuntimeError(
                "pyrobotiqgripper is required for --gripper-backend pyrobotiq. "
                "Install it into the uv environment or use --gripper-backend raw."
            ) from exc
        self._module = rq
        self.gripper = rq.RobotiqGripper(
            com_port=self.port,
            device_id=self.slave_id,
            gripper_type="2F85",
            connection_type=rq.GRIPPER_MODE_RTU,
            debug=self.debug,
        )
        self.gripper.connect()

    def close(self) -> None:
        if self.gripper is not None:
            try:
                self.gripper.disconnect()
            except AttributeError:
                pass
        self.gripper = None

    def activate(self) -> None:
        if self.gripper is None:
            raise RuntimeError("Robotiq pyrobotiq gripper is not connected")
        self.gripper.activate()

    def command_closure(self, closure: float, *, speed: int = 255, force: int = 150) -> None:
        if self.gripper is None:
            raise RuntimeError("Robotiq pyrobotiq gripper is not connected")
        closure = float(np.clip(closure, 0.0, 1.0))
        self.gripper.move(
            int(round(255.0 * closure)),
            speed=int(np.clip(speed, 0, 255)),
            force=int(np.clip(force, 0, 255)),
            wait=False,
            readStatus=False,
            refreshStatus=False,
        )


def write_report(path: Path, report: SafetyReport, trajectory: np.ndarray, execute_real: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "safety_report": report.__dict__,
        "trajectory": {
            "frames": int(trajectory.shape[0]),
            "joints": int(trajectory.shape[1]),
            "first_q_rad": trajectory[0].tolist(),
            "last_q_rad": trajectory[-1].tolist(),
        },
        "execute_real": bool(execute_real),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe FR5 sim-first synchronization SDK entrypoint")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--replay-qpos", type=str, default="", help="Optional .npz/.npy trajectory")
    parser.add_argument("--source-dt", type=float, default=0.04, help="Replay/default trajectory timestep before real-control resampling")
    parser.add_argument("--real-dt", type=float, default=0.008, help="Fairino ServoJ command period")
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP)
    parser.add_argument("--min-tcp-z", type=float, default=0.04)
    parser.add_argument("--max-step-rad", type=float, default=0.012, help="Preflight max joint step in rad")
    parser.add_argument("--max-real-step-deg", type=float, default=0.75, help="Fairino ServoJ max per-command joint step")
    parser.add_argument("--max-tcp-speed", type=float, default=0.35)
    parser.add_argument("--execute-real", action="store_true", help="Actually send commands to the physical FR5 arm")
    parser.add_argument("--real-mode", choices=["servo", "movej"], default="servo")
    parser.add_argument("--speed-percent", type=float, default=10.0)
    parser.add_argument("--movej-vel", type=float, default=10.0)
    parser.add_argument("--movej-acc", type=float, default=10.0)
    parser.add_argument("--movej-ovl", type=float, default=20.0)
    parser.add_argument("--servo-vel", type=float, default=20.0)
    parser.add_argument("--execute-gripper", action="store_true", help="Actually send a USB Robotiq command")
    parser.add_argument("--gripper-port", type=str, default="", help="Example: /dev/ttyUSB0")
    parser.add_argument("--gripper-closure", type=float, default=None, help="0=open, 1=closed; defaults to config gripper_opening")
    parser.add_argument("--gripper-baudrate", type=int, default=115200)
    parser.add_argument("--gripper-slave-id", type=int, default=9)
    parser.add_argument("--gripper-timeout", type=float, default=0.5)
    parser.add_argument("--gripper-retries", type=int, default=2)
    parser.add_argument("--report", type=str, default="data/fr5_sync_last_report.json")
    args = parser.parse_args()

    if args.real_dt < 0.001 or args.real_dt > 0.016:
        raise RuntimeError("--real-dt must be between 0.001 and 0.016 seconds for Fairino ServoJ")
    if args.execute_gripper and not args.gripper_port:
        raise RuntimeError("--execute-gripper requires --gripper-port, e.g. /dev/ttyUSB0")

    config = load_config(args.config)
    base_runtime = build_runtime(config)
    _, _, _, qpos_ids, _, _ = base_runtime
    replay = load_replay_qpos(args.replay_qpos) if args.replay_qpos else None
    trajectory = normalize_arm_trajectory(replay, config, qpos_ids)
    trajectory = resample_trajectory(trajectory, args.source_dt, args.real_dt)

    report, runtime = run_sim_safety_check(
        config,
        trajectory,
        min_tcp_z=float(args.min_tcp_z),
        max_step_rad=float(args.max_step_rad),
        max_tcp_speed=float(args.max_tcp_speed),
        control_dt=float(args.real_dt),
    )
    print("Simulation safety check passed.")
    print(
        f"  frames={report.frames}, duration={report.duration_s:.3f}s, "
        f"max_step={report.max_step_rad:.6f}rad/{report.max_step_deg:.3f}deg, "
        f"min_tcp_z={report.min_tcp_z:.4f}m, max_tcp_speed={report.max_tcp_speed_mps:.4f}m/s"
    )

    real_step_deg = float(np.max(np.abs(np.diff(np.rad2deg(trajectory), axis=0)))) if trajectory.shape[0] > 1 else 0.0
    if real_step_deg > float(args.max_real_step_deg):
        raise RuntimeError(
            f"Real ServoJ step too large: {real_step_deg:.3f}deg, limit={args.max_real_step_deg:.3f}deg"
        )

    report_path = resolve_demo_path(args.report)
    write_report(report_path, report, trajectory, bool(args.execute_real))
    print(f"Wrote safety report: {report_path}")

    if not args.execute_real:
        print("Dry run only. Add --execute-real to send the checked trajectory to the FR5 controller.")
        return

    model, data, body, qpos_ids, arm_act_ids, gripper_act_ids = runtime
    canceller = MotionCancelHandler()
    canceller.install()
    arm = FairinoArmClient(args.robot_ip, speed_percent=float(args.speed_percent))
    canceller.set_arm(arm)
    gripper = None
    try:
        arm.connect()
        actual = arm.get_actual_joint_deg()
        if actual is not None:
            delta0 = np.max(np.abs(np.asarray(actual, dtype=np.float64) - np.rad2deg(trajectory[0])))
            if delta0 > 3.0:
                raise RuntimeError(
                    f"Robot current joints differ from trajectory start by {delta0:.2f}deg. "
                    "Move to the start pose manually or use a safer transition first."
                )
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

        if args.real_mode == "movej":
            for q in trajectory:
                set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q.astype(np.float32))
                forward_kinematic(model, data)
                arm.move_j(np.rad2deg(q), vel=args.movej_vel, acc=args.movej_acc, ovl=args.movej_ovl)
        else:
            steps_per_cmd = max(1, round(float(args.real_dt) / float(model.options.timestep)))
            arm.servo_start()
            canceller.set_servo_started(True)
            try:
                for idx, q in enumerate(trajectory):
                    canceller.check()
                    set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q.astype(np.float32))
                    for _ in range(steps_per_cmd):
                        sim_step(model, data)
                    arm.servo_j(np.rad2deg(q), idx=idx, cmd_t=args.real_dt, vel=args.servo_vel)
                    time.sleep(float(args.real_dt))
            finally:
                arm.servo_end_best_effort()
                canceller.set_servo_started(False)

        if gripper is not None:
            closure = float(config.get("gripper_opening", 0.0) if args.gripper_closure is None else args.gripper_closure)
            set_gripper(data, model, body, gripper_act_ids, closure)
            gripper.command_closure(closure)
    except KeyboardInterrupt:
        print("FR5 sync cancelled. StopMotion was sent best-effort.", flush=True)
    finally:
        if gripper is not None:
            gripper.close()
        arm.close()


if __name__ == "__main__":
    main()
