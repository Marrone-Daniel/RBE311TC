from __future__ import annotations

import argparse
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an ArUco marker image for Astra extrinsic calibration")
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--dictionary", type=str, default="DICT_4X4_50")
    parser.add_argument("--pixels", type=int, default=1000)
    parser.add_argument("--border", type=int, default=200, help="White quiet-zone border in pixels")
    parser.add_argument("--output", type=str, default=(DEMO_DIR / "data" / "aruco_marker_0.png").as_posix())
    args = parser.parse_args()

    import cv2

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is missing. Use opencv-contrib-python.")
    if not hasattr(cv2.aruco, args.dictionary):
        raise ValueError(f"Unknown ArUco dictionary: {args.dictionary}")

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    marker = cv2.aruco.generateImageMarker(dictionary, int(args.id), int(args.pixels))
    if args.border > 0:
        marker = cv2.copyMakeBorder(
            marker,
            int(args.border),
            int(args.border),
            int(args.border),
            int(args.border),
            cv2.BORDER_CONSTANT,
            value=255,
        )
    out = Path(args.output)
    if not out.is_absolute():
        out = DEMO_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out.as_posix(), marker)
    print(f"Saved marker image: {out}")
    print("Print it without scaling, then measure the black marker side length, excluding the white border, in meters.")


if __name__ == "__main__":
    main()
