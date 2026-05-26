from __future__ import annotations

from dataclasses import dataclass


@dataclass
class XboxTeleopCommand:
    vx: float
    vy: float
    vz: float
    vroll: float
    vpitch: float
    vyaw: float
    close_gripper: bool
    open_gripper: bool
    return_home: bool
    status: bool
    any_arm_motion: bool
    arm_motion_level: float


class XboxGamepad:
    """Small pygame-backed Xbox controller reader matching the existing FR5_xbox mapping."""

    def __init__(
        self,
        *,
        joystick_index: int = 0,
        deadzone: float = 0.12,
        alpha: float = 0.25,
        button_a: int = 0,
        button_b: int = 1,
        button_x: int = 2,
        button_y: int = 3,
        button_lb: int = 4,
        button_rb: int = 5,
    ):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for Xbox/gamepad teleop. Install it into the uv environment, "
                "for example: uv add pygame"
            ) from exc
        self.pygame = pygame
        self.deadzone = float(deadzone)
        self.alpha = float(alpha)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.vroll = 0.0
        self._last_a = False
        self._last_b = False
        self.vpitch = 0.0
        self.vyaw = 0.0
        self.button_a = int(button_a)
        self.button_b = int(button_b)
        self.button_x = int(button_x)
        self.button_y = int(button_y)
        self.button_lb = int(button_lb)
        self.button_rb = int(button_rb)
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected by pygame.")
        if int(joystick_index) >= pygame.joystick.get_count():
            raise RuntimeError(
                f"Gamepad index {joystick_index} is unavailable; detected {pygame.joystick.get_count()} device(s)."
            )
        self.js = pygame.joystick.Joystick(int(joystick_index))
        self.js.init()
        print(f"Gamepad connected: {self.js.get_name()}", flush=True)
        print(
            "Gamepad mapping: "
            f"A={self.button_a}, B={self.button_b}, X={self.button_x}, Y={self.button_y}, "
            f"LB={self.button_lb}, RB={self.button_rb}; "
            f"axes={self.js.get_numaxes()}, buttons={self.js.get_numbuttons()}, hats={self.js.get_numhats()}",
            flush=True,
        )

    def close(self) -> None:
        self.pygame.quit()

    def _axis(self, idx: int, default: float = 0.0) -> float:
        if self.js.get_numaxes() <= idx:
            return float(default)
        return float(self.js.get_axis(idx))

    def _button(self, idx: int) -> bool:
        return bool(self.js.get_numbuttons() > idx and self.js.get_button(idx))

    def _hat(self) -> tuple[int, int]:
        if self.js.get_numhats() <= 0:
            return 0, 0
        return self.js.get_hat(0)

    def _dz(self, value: float) -> float:
        return 0.0 if abs(float(value)) < self.deadzone else float(value)

    def read(self) -> XboxTeleopCommand:
        self.pygame.event.pump()

        lx = self._dz(self._axis(0))
        ly = self._dz(self._axis(1))
        rz_axis = self._dz(self._axis(4)) if self.js.get_numaxes() > 4 else 0.0

        lb = self._button(self.button_lb)
        rb = self._button(self.button_rb)
        yaw_cmd = -1.0 if lb and not rb else 1.0 if rb and not lb else 0.0

        lt_raw = self._axis(2, -1.0)
        rt_raw = self._axis(5, -1.0)
        lt_val = (lt_raw + 1.0) / 2.0
        rt_val = (rt_raw + 1.0) / 2.0
        pitch_cmd = rt_val - lt_val

        _, hat_y = self._hat()
        roll_cmd = 1.0 if hat_y == 1 else -1.0 if hat_y == -1 else 0.0

        a_btn = self._button(self.button_a)
        b_btn = self._button(self.button_b)
        x_btn = self._button(self.button_x)
        y_btn = self._button(self.button_y)

        alpha = self.alpha
        self.vx = -alpha * lx + (1.0 - alpha) * self.vx
        self.vy = alpha * ly + (1.0 - alpha) * self.vy
        self.vz = alpha * (-rz_axis) + (1.0 - alpha) * self.vz
        self.vyaw = alpha * yaw_cmd + (1.0 - alpha) * self.vyaw
        self.vpitch = alpha * pitch_cmd + (1.0 - alpha) * self.vpitch
        self.vroll = alpha * roll_cmd + (1.0 - alpha) * self.vroll

        arm_motion_level = max(abs(value) for value in (self.vx, self.vy, self.vz, self.vroll, self.vpitch, self.vyaw))
        return_home_pressed = b_btn and not self._last_b
        status_pressed = a_btn and not self._last_a
        self._last_a = a_btn
        self._last_b = b_btn
        return XboxTeleopCommand(
            vx=self.vx,
            vy=self.vy,
            vz=self.vz,
            vroll=self.vroll,
            vpitch=self.vpitch,
            vyaw=self.vyaw,
            close_gripper=x_btn and not y_btn,
            open_gripper=y_btn and not x_btn,
            return_home=return_home_pressed,
            status=status_pressed,
            any_arm_motion=arm_motion_level >= 0.03,
            arm_motion_level=arm_motion_level,
        )
