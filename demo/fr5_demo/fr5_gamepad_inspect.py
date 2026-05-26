from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Print raw pygame joystick axes/buttons/hats for mapping Xbox controls.")
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--only-changes", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is required. Install it with: uv add pygame") from exc

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick/gamepad detected by pygame.")
    js = pygame.joystick.Joystick(int(args.joystick_index))
    js.init()
    print(f"Gamepad: {js.get_name()}")
    print(f"axes={js.get_numaxes()} buttons={js.get_numbuttons()} hats={js.get_numhats()}")
    print("Press X/Y/B/A/LB/RB and move sticks. Ctrl+C to exit.")

    last = None
    try:
        while True:
            pygame.event.pump()
            axes = [round(float(js.get_axis(i)), 3) for i in range(js.get_numaxes())]
            buttons = [int(js.get_button(i)) for i in range(js.get_numbuttons())]
            hats = [js.get_hat(i) for i in range(js.get_numhats())]
            current = (axes, buttons, hats)
            if not args.only_changes or current != last:
                pressed = [idx for idx, value in enumerate(buttons) if value]
                moved = {idx: value for idx, value in enumerate(axes) if abs(value) > 0.08}
                print(f"buttons_pressed={pressed} axes_moved={moved} hats={hats}", flush=True)
                print(f"  raw buttons={buttons}", flush=True)
                print(f"  raw axes={axes}", flush=True)
            last = current
            time.sleep(float(args.dt))
    except KeyboardInterrupt:
        print("Gamepad inspect stopped.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
