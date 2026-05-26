from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CAMERA_CONFIG = DEMO_DIR / "configs" / "astra_camera.json"


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install opencv-contrib-python for cv2.aruco support.") from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is missing. Install opencv-contrib-python, not plain opencv-python.")
    return cv2


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return DEMO_DIR / path


def euler_xyz_to_matrix(rpy: list[float]) -> np.ndarray:
    rx, ry, rz = [float(v) for v in rpy]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rzm @ rym @ rxm


def matrix_to_euler_xyz(r: np.ndarray) -> list[float]:
    r = np.asarray(r, dtype=np.float64)
    sy = float(-r[2, 0])
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1e-9:
        rx = math.atan2(float(r[2, 1]), float(r[2, 2]))
        rz = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        rx = 0.0
        rz = math.atan2(float(-r[0, 1]), float(r[1, 1]))
    return [float(rx), float(ry), float(rz)]


def normalize(v: np.ndarray, name: str) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        raise RuntimeError(f"Cannot normalize near-zero vector: {name}")
    return v / norm


def infer_shared_marker_rotation(markers: list[dict], normal: list[float] | np.ndarray) -> np.ndarray:
    if len(markers) < 2:
        raise RuntimeError("Need at least two marker centers to infer marker_world_rpy")
    p0 = np.asarray(markers[0]["marker_world_pos"], dtype=np.float64)
    p1 = np.asarray(markers[1]["marker_world_pos"], dtype=np.float64)
    z_axis = normalize(np.asarray(normal, dtype=np.float64), "marker_world_normal")
    x_raw = p1 - p0
    x_axis = x_raw - float(np.dot(x_raw, z_axis)) * z_axis
    x_axis = normalize(x_axis, "marker 0 -> marker 1 projected onto marker plane")
    y_axis = normalize(np.cross(z_axis, x_axis), "inferred marker y axis")
    return np.column_stack([x_axis, y_axis, z_axis])


def make_transform(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return out


def aruco_dictionary(cv2, name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def aruco_detector_params(cv2):
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 80
    params.cornerRefinementMinAccuracy = 0.005
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.03
    params.minCornerDistanceRate = 0.03
    params.minDistanceToBorder = 3
    return params


def detect_markers(cv2, image, dictionary_name: str):
    dictionary = aruco_dictionary(cv2, dictionary_name)
    params = aruco_detector_params(cv2)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
    if ids is None or len(ids) == 0:
        raise RuntimeError("No ArUco marker detected in image")
    return {int(marker_id): corner.reshape(4, 2).astype(np.float64) for corner, marker_id in zip(corners, ids.reshape(-1))}


def detect_marker(cv2, image, dictionary_name: str, marker_id: int | None):
    markers = detect_markers(cv2, image, dictionary_name)
    detected_ids = sorted(markers)
    if marker_id is not None:
        if int(marker_id) not in markers:
            raise RuntimeError(f"Marker id {marker_id} not detected. Detected ids: {detected_ids}")
        return markers[int(marker_id)], int(marker_id)
    first_id = detected_ids[0]
    return markers[first_id], first_id


def marker_object_points(marker_length: float) -> np.ndarray:
    half = float(marker_length) * 0.5
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def solve_marker_pose(cv2, corners_px: np.ndarray, marker_length: float, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    object_points = marker_object_points(marker_length)
    ok, rvec, tvec = cv2.solvePnP(object_points, corners_px.astype(np.float64), camera_matrix, dist_coeffs)
    if not ok:
        raise RuntimeError("cv2.solvePnP failed")
    r_cam_marker, _ = cv2.Rodrigues(rvec)
    return make_transform(r_cam_marker, tvec.reshape(3))


def marker_center_from_entry(entry: dict, marker_length: float, r_world_marker: np.ndarray) -> np.ndarray:
    if "marker_world_pos" in entry and entry["marker_world_pos"] is not None:
        return np.asarray(entry["marker_world_pos"], dtype=np.float64)
    if "marker_world_top_left" in entry and entry["marker_world_top_left"] is not None:
        top_left = np.asarray(entry["marker_world_top_left"], dtype=np.float64)
        top_left_in_marker = np.asarray([-float(marker_length) * 0.5, float(marker_length) * 0.5, 0.0], dtype=np.float64)
        return top_left - r_world_marker @ top_left_in_marker
    raise RuntimeError(f"Marker entry {entry} needs marker_world_pos or marker_world_top_left")


def load_marker_spec(path: str | Path) -> dict:
    spec_path = resolve(path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("markers"), list):
        raise RuntimeError(f"Marker spec must be a JSON object with a markers list: {spec_path}")
    if len(spec["markers"]) < 2:
        raise RuntimeError("Marker spec should contain at least two markers for multi-marker calibration.")
    return spec


def solve_multimarker_extrinsic(cv2, image, spec: dict, camera_matrix: np.ndarray, dist: np.ndarray):
    dictionary = str(spec.get("dictionary", "DICT_4X4_50"))
    default_length = float(spec.get("marker_length_m", 0.20))
    detected = detect_markers(cv2, image, dictionary)
    auto_rpy = bool(spec.get("auto_marker_world_rpy", False))
    inferred_r_world_marker = None
    inferred_rpy = None
    if auto_rpy:
        normal = spec.get("marker_world_normal", [0.0, 0.0, 1.0])
        inferred_r_world_marker = infer_shared_marker_rotation(spec["markers"], normal)
        inferred_rpy = matrix_to_euler_xyz(inferred_r_world_marker)
    object_points = []
    image_points = []
    used = []
    for entry in spec["markers"]:
        marker_id = int(entry["id"])
        if marker_id not in detected:
            continue
        marker_length = float(entry.get("marker_length_m", default_length))
        if auto_rpy and "marker_world_rpy" not in entry:
            rpy = inferred_rpy
            r_world_marker = inferred_r_world_marker
        else:
            rpy = entry.get("marker_world_rpy", spec.get("marker_world_rpy", inferred_rpy or [0.0, 0.0, 0.0]))
            if rpy == "auto":
                if inferred_r_world_marker is None:
                    normal = spec.get("marker_world_normal", [0.0, 0.0, 1.0])
                    inferred_r_world_marker = infer_shared_marker_rotation(spec["markers"], normal)
                    inferred_rpy = matrix_to_euler_xyz(inferred_r_world_marker)
                rpy = inferred_rpy
                r_world_marker = inferred_r_world_marker
            else:
                r_world_marker = euler_xyz_to_matrix(rpy)
        marker_center = marker_center_from_entry(entry, marker_length, r_world_marker)
        t_world_marker = make_transform(r_world_marker, marker_center)
        local_corners = marker_object_points(marker_length)
        world_corners = (t_world_marker[:3, :3] @ local_corners.T).T + t_world_marker[:3, 3]
        object_points.append(world_corners)
        image_points.append(detected[marker_id])
        used.append(
            {
                "id": marker_id,
                "marker_length_m": marker_length,
                "marker_world_pos": marker_center.tolist(),
                "marker_world_rpy": [float(v) for v in rpy],
            }
        )
    if len(used) < 2:
        raise RuntimeError(f"Need at least two specified markers detected. Detected ids: {sorted(detected)}; used ids: {[m['id'] for m in used]}")

    obj = np.vstack(object_points).astype(np.float64)
    img = np.vstack(image_points).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("cv2.solvePnP failed for multi-marker calibration")
    r_cam_world, _ = cv2.Rodrigues(rvec)
    t_cam_world = make_transform(r_cam_world, tvec.reshape(3))
    t_world_camera = np.linalg.inv(t_cam_world)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - img, axis=1)
    return t_world_camera, used, detected, {
        "mean_px": float(np.mean(errors)),
        "max_px": float(np.max(errors)),
        "per_corner_px": errors.tolist(),
        "auto_marker_world_rpy": inferred_rpy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Astra RGB camera extrinsic against FR5/world using one or more ArUco markers")
    parser.add_argument("--image", type=str, required=True, help="RGB image containing a visible ArUco marker")
    parser.add_argument("--camera-config", type=str, default=DEFAULT_CAMERA_CONFIG.as_posix())
    parser.add_argument("--marker-spec", type=str, default="", help="JSON spec for two or more ArUco markers in world coordinates")
    parser.add_argument("--marker-length", type=float, default=0.08, help="Marker side length in meters")
    parser.add_argument("--marker-id", type=int, default=None)
    parser.add_argument("--dictionary", type=str, default="DICT_4X4_50")
    parser.add_argument("--marker-world-pos", type=float, nargs=3, default=None, help="Marker center position in FR5/world frame, meters")
    parser.add_argument("--marker-world-top-left", type=float, nargs=3, default=None, help="Marker top-left corner position in FR5/world frame, meters")
    parser.add_argument("--marker-world-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Marker xyz Euler rotation in world, radians")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    cv2 = require_cv2()
    cfg_path = resolve(args.camera_config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    intr = cfg["intrinsics"]
    camera_matrix = np.array(
        [[float(intr["fx"]), 0.0, float(intr["cx"])], [0.0, float(intr["fy"]), float(intr["cy"])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if camera_matrix[0, 0] <= 1.0 or camera_matrix[1, 1] <= 1.0:
        raise RuntimeError("Invalid fx/fy in camera config. Run capture script or fill intrinsics manually from Orbbec Viewer.")
    dist = np.asarray(intr.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)

    image_path = resolve(args.image)
    image = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    if args.marker_spec:
        spec = load_marker_spec(args.marker_spec)
        t_world_camera, used_markers, detected, reprojection = solve_multimarker_extrinsic(cv2, image, spec, camera_matrix, dist)
        calibration_target = {
            "type": "aruco_multi_marker",
            "dictionary": str(spec.get("dictionary", args.dictionary)),
            "markers": used_markers,
            "detected_ids": sorted(detected),
            "reprojection_error_px": reprojection,
        }
        detected_id = None
    else:
        corners_px, detected_id = detect_marker(cv2, image, args.dictionary, args.marker_id)
        if args.marker_world_pos is None and args.marker_world_top_left is None:
            raise RuntimeError("Provide either --marker-world-pos for marker center or --marker-world-top-left for marker top-left corner.")
        if args.marker_world_pos is not None and args.marker_world_top_left is not None:
            raise RuntimeError("Provide only one of --marker-world-pos or --marker-world-top-left.")

        t_cam_marker = solve_marker_pose(cv2, corners_px, args.marker_length, camera_matrix, dist)
        r_world_marker = euler_xyz_to_matrix(args.marker_world_rpy)
        if args.marker_world_top_left is not None:
            half = float(args.marker_length) * 0.5
            top_left_in_marker = np.asarray([-half, half, 0.0], dtype=np.float64)
            marker_center = np.asarray(args.marker_world_top_left, dtype=np.float64) - r_world_marker @ top_left_in_marker
        else:
            marker_center = np.asarray(args.marker_world_pos, dtype=np.float64)
        t_world_marker = make_transform(r_world_marker, marker_center)
        t_world_camera = t_world_marker @ np.linalg.inv(t_cam_marker)
        calibration_target = {
            "type": "aruco_single_marker",
            "marker_id": detected_id,
            "marker_length_m": float(args.marker_length),
            "dictionary": args.dictionary,
            "marker_world_pos": marker_center.tolist(),
            "marker_world_top_left": list(args.marker_world_top_left) if args.marker_world_top_left is not None else None,
            "marker_world_rpy": list(args.marker_world_rpy),
        }

    cfg["name"] = cfg.get("name", "astra_rgb")
    cfg["enabled"] = True
    cfg["calibrated"] = True
    cfg["extrinsics"] = {
        "frame": "world_from_camera",
        "position": t_world_camera[:3, 3].tolist(),
        "rotation_matrix": t_world_camera[:3, :3].tolist(),
    }
    cfg["calibration_target"] = calibration_target
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if args.marker_spec:
        print(f"Detected marker ids: {calibration_target['detected_ids']}")
        print(f"Used marker ids: {[m['id'] for m in calibration_target['markers']]}")
        print(
            "Reprojection error: "
            f"mean={calibration_target['reprojection_error_px']['mean_px']:.3f}px, "
            f"max={calibration_target['reprojection_error_px']['max_px']:.3f}px"
        )
    else:
        print(f"Detected marker id: {detected_id}")
    print(f"Saved calibrated camera config: {cfg_path}")
    print(f"world_from_camera position: {cfg['extrinsics']['position']}")

    if args.show:
        if args.marker_spec:
            show_corners = []
            show_ids = []
            for marker_id in [m["id"] for m in calibration_target["markers"]]:
                show_corners.append(detected[int(marker_id)].reshape(1, 4, 2).astype(np.float32))
                show_ids.append([int(marker_id)])
            cv2.aruco.drawDetectedMarkers(image, show_corners, np.asarray(show_ids, dtype=np.int32))
        else:
            cv2.aruco.drawDetectedMarkers(image, [corners_px.reshape(1, 4, 2).astype(np.float32)], np.array([[detected_id]], dtype=np.int32))
        cv2.imshow("Detected ArUco", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
