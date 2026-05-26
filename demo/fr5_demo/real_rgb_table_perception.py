from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CAMERA_CONFIG = DEMO_DIR / "configs" / "astra_camera.json"
DEFAULT_TASK_CONFIG = DEMO_DIR / "configs" / "fr5_table_task.json"


def resolve_demo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    project_path = DEMO_DIR.parents[1] / path
    if project_path.exists() or path.parts[:2] == ("demo", "fr5_demo"):
        return project_path
    return DEMO_DIR / path


def load_json(path: str | Path) -> dict:
    resolved = resolve_demo_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {resolved}")
    return data


def rotation_matrix_from_rpy_deg(rpy_deg: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(list(rpy_deg), dtype=np.float64).reshape(3))
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def apply_manual_extrinsic_correction(pos: np.ndarray, rot: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    correction = config.get("manual_extrinsic_correction", {})
    if not correction:
        return pos, rot
    frame = str(correction.get("frame", "camera_local_opencv"))
    delta_pos = np.asarray(correction.get("position_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    delta_rot = rotation_matrix_from_rpy_deg(correction.get("rotation_rpy_deg", [0.0, 0.0, 0.0]))
    if frame == "camera_local_opencv":
        return pos + rot @ delta_pos, rot @ delta_rot
    if frame == "world":
        return pos + delta_pos, delta_rot @ rot
    raise ValueError(f"manual_extrinsic_correction.frame must be camera_local_opencv or world, got {frame!r}")


@dataclass(frozen=True)
class CameraProjector:
    k: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    camera_pos_world: np.ndarray
    r_world_camera: np.ndarray

    @classmethod
    def from_config(cls, camera_config: str | Path) -> "CameraProjector":
        cfg = load_json(camera_config)
        if not cfg.get("calibrated", False):
            raise ValueError(f"Camera config is not calibrated: {resolve_demo_path(camera_config)}")
        intr = cfg.get("intrinsics", {})
        ext = cfg.get("extrinsics", {})
        width = int(intr["width"])
        height = int(intr["height"])
        k = np.asarray(
            [
                [float(intr["fx"]), 0.0, float(intr["cx"])],
                [0.0, float(intr["fy"]), float(intr["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist = np.asarray(intr.get("distortion", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
        pos = np.asarray(ext["position"], dtype=np.float64).reshape(3)
        rot = np.asarray(ext["rotation_matrix"], dtype=np.float64).reshape(3, 3)
        pos, rot = apply_manual_extrinsic_correction(pos, rot, cfg)
        return cls(k=k, dist=dist, width=width, height=height, camera_pos_world=pos, r_world_camera=rot)

    @property
    def r_camera_world(self) -> np.ndarray:
        return self.r_world_camera.T

    @property
    def t_camera_world(self) -> np.ndarray:
        return -self.r_camera_world @ self.camera_pos_world

    def world_to_camera(self, point_world: Iterable[float]) -> np.ndarray:
        point = np.asarray(point_world, dtype=np.float64).reshape(3)
        return self.r_camera_world @ point + self.t_camera_world

    def project_world(self, point_world: Iterable[float], cv2) -> tuple[np.ndarray, float]:
        point = np.asarray(point_world, dtype=np.float64).reshape(1, 1, 3)
        rvec, _ = cv2.Rodrigues(self.r_camera_world)
        tvec = self.t_camera_world.reshape(3, 1)
        pts, _ = cv2.projectPoints(point, rvec, tvec, self.k, self.dist)
        cam = self.world_to_camera(point.reshape(3))
        return pts.reshape(2).astype(np.float64), float(cam[2])

    def pixel_to_world_on_z(self, pixel_xy: Iterable[float], z_world: float, cv2) -> np.ndarray:
        uv = np.asarray(pixel_xy, dtype=np.float64).reshape(1, 1, 2)
        norm = cv2.undistortPoints(uv, self.k, self.dist).reshape(2)
        ray_camera = np.asarray([norm[0], norm[1], 1.0], dtype=np.float64)
        ray_world = self.r_world_camera @ ray_camera
        denom = float(ray_world[2])
        if abs(denom) < 1e-9:
            raise RuntimeError("Camera ray is nearly parallel to the requested world-z plane.")
        scale = (float(z_world) - float(self.camera_pos_world[2])) / denom
        return (self.camera_pos_world + scale * ray_world).astype(np.float64)

    def in_frame(self, pixel_xy: Iterable[float], margin: float = 0.0) -> bool:
        u, v = np.asarray(pixel_xy, dtype=np.float64).reshape(2)
        return -margin <= u < self.width + margin and -margin <= v < self.height + margin


@dataclass
class TapeDetection:
    color: str
    center_px: np.ndarray
    area_px: float
    bbox_xywh: tuple[int, int, int, int]
    score: float = 0.0
    slot_name: str = ""
    world_m: np.ndarray | None = None
    contour: np.ndarray | None = None


COLOR_RANGES_HSV: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 70, 45), (12, 255, 255)), ((168, 70, 45), (179, 255, 255))],
    "yellow": [((18, 60, 60), (42, 255, 255))],
    "white": [((0, 0, 145), (179, 70, 255))],
}

DRAW_COLORS_BGR = {
    "red": (40, 40, 255),
    "yellow": (0, 220, 255),
    "white": (240, 240, 240),
    "part": (255, 0, 255),
    "grasp": (255, 180, 0),
}


def tape_slot_specs(config: dict) -> list[dict]:
    task = config.get("sim_tape_pick_place", {})
    specs = []
    for raw in task.get("tape_objects", []):
        if "slot_center_m" not in raw or "slot_half_extents_m" not in raw:
            continue
        specs.append(
            {
                "name": str(raw.get("slot_name", raw.get("name", f"slot_{len(specs)}"))),
                "center_m": np.asarray(raw["slot_center_m"], dtype=np.float64).reshape(3),
                "half_extents_m": np.asarray(raw["slot_half_extents_m"], dtype=np.float64).reshape(2),
            }
        )
    return specs


def projected_slot_polygons(
    config: dict,
    projector: CameraProjector,
    *,
    cv2,
    z_world: float,
    padding_m: float = 0.02,
) -> list[dict]:
    polygons = []
    for spec in tape_slot_specs(config):
        center = np.asarray(spec["center_m"], dtype=np.float64).reshape(3).copy()
        center[2] = float(z_world)
        half = np.asarray(spec["half_extents_m"], dtype=np.float64).reshape(2) + float(padding_m)
        corners_world = [
            [center[0] - half[0], center[1] - half[1], center[2]],
            [center[0] + half[0], center[1] - half[1], center[2]],
            [center[0] + half[0], center[1] + half[1], center[2]],
            [center[0] - half[0], center[1] + half[1], center[2]],
        ]
        corners = np.asarray([projector.project_world(point, cv2)[0] for point in corners_world], dtype=np.float32)
        polygons.append({"name": str(spec["name"]), "points": corners})
    return polygons


def slot_for_pixel(pixel_xy: Iterable[float], slot_polygons: list[dict], *, cv2) -> str:
    point = tuple(float(v) for v in np.asarray(pixel_xy, dtype=np.float64).reshape(2))
    for slot in slot_polygons:
        if cv2.pointPolygonTest(np.asarray(slot["points"], dtype=np.float32), point, False) >= 0.0:
            return str(slot["name"])
    return ""


def slot_mask_for_image(shape_hw: tuple[int, int], slot_polygons: list[dict], *, cv2) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    for slot in slot_polygons:
        pts = np.asarray(slot["points"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def slot_center_px(slot: dict) -> np.ndarray:
    return np.mean(np.asarray(slot["points"], dtype=np.float64).reshape(-1, 2), axis=0)


def real_myd_part_offset_from_config(config: dict) -> np.ndarray:
    deployment = config.get("real_deployment", {})
    if isinstance(deployment, dict) and "myd_part_manual_offset_m" in deployment:
        return np.asarray(deployment["myd_part_manual_offset_m"], dtype=np.float64).reshape(3)
    task = config.get("sim_tape_pick_place", {})
    if "myd_part_manual_offset_m" in task:
        return np.asarray(task["myd_part_manual_offset_m"], dtype=np.float64).reshape(3)
    return np.zeros(3, dtype=np.float64)


def real_tape_offset_from_config(config: dict) -> np.ndarray:
    deployment = config.get("real_deployment", {})
    if isinstance(deployment, dict) and "tape_manual_offset_m" in deployment:
        return np.asarray(deployment["tape_manual_offset_m"], dtype=np.float64).reshape(3)
    task = config.get("sim_tape_pick_place", {})
    if "tape_manual_offset_m" in task:
        return np.asarray(task["tape_manual_offset_m"], dtype=np.float64).reshape(3)
    return np.zeros(3, dtype=np.float64)


def task_goal_pos_from_config(
    config: dict,
    *,
    manual_offset_m: Iterable[float] | None = None,
    include_real_offset: bool = True,
) -> np.ndarray:
    task = config.get("sim_tape_pick_place", {})
    base = np.asarray(task.get("goal_pos_m", [-0.485, -0.26, 0.0]), dtype=np.float64).reshape(3)
    offset = np.asarray(task.get("goal_offset_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    out = base + offset
    if include_real_offset:
        out = out + real_myd_part_offset_from_config(config)
    if manual_offset_m is not None:
        out = out + np.asarray(manual_offset_m, dtype=np.float64).reshape(3)
    return out


def default_tape_plane_z(config: dict, *, include_real_offset: bool = True) -> float:
    target = config.get("target_object", {})
    if "initial_pos_m" in target:
        z = float(np.asarray(target["initial_pos_m"], dtype=np.float64).reshape(3)[2])
        return z + (float(real_tape_offset_from_config(config)[2]) if include_real_offset else 0.0)
    task = config.get("sim_tape_pick_place", {})
    for spec in task.get("tape_objects", []):
        if "slot_center_m" in spec:
            z = float(np.asarray(spec["slot_center_m"], dtype=np.float64).reshape(3)[2])
            return z + (float(real_tape_offset_from_config(config)[2]) if include_real_offset else 0.0)
    return 0.02 + (float(real_tape_offset_from_config(config)[2]) if include_real_offset else 0.0)


def nearest_wall_grasp_point(config: dict, tape_center_world: np.ndarray) -> np.ndarray:
    task = config.get("sim_tape_pick_place", {})
    radius = float(task.get("grasp_wall_center_radius_m", 0.0475))
    base_xy = np.asarray(task.get("robot_base_xy_m", [0.0, 0.0]), dtype=np.float64).reshape(2)
    center = np.asarray(tape_center_world, dtype=np.float64).reshape(3)
    vec = base_xy - center[:2]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        vec = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        vec = vec / norm
    out = center.copy()
    out[:2] = center[:2] + vec * radius
    return out


def detect_tapes_rgb(
    rgb: np.ndarray,
    *,
    cv2,
    colors: Iterable[str] = ("red", "yellow", "white"),
    min_area_px: float = 350.0,
    max_area_px: float = 50000.0,
    config: dict | None = None,
    projector: CameraProjector | None = None,
    z_world: float | None = None,
    use_slot_roi: bool = True,
    slot_padding_m: float = 0.025,
    white_min_area_px: float = 80.0,
    white_max_area_px: float = 18000.0,
    white_min_circularity: float = 0.12,
    white_min_extent: float = 0.12,
    slot_color_prior: bool = True,
    slot_fallback_center: bool = True,
    background_rgb: np.ndarray | None = None,
    background_diff: bool = False,
    background_diff_thresh: float = 32.0,
) -> dict[str, TapeDetection]:
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    kernel = np.ones((5, 5), dtype=np.uint8)
    diff_mask = None
    if background_diff and background_rgb is not None:
        bg = np.ascontiguousarray(background_rgb)
        if bg.shape[:2] != rgb.shape[:2]:
            bg = cv2.resize(bg, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        diff = cv2.absdiff(np.ascontiguousarray(rgb), bg)
        diff_gray = np.max(diff, axis=2).astype(np.uint8)
        _, diff_mask = cv2.threshold(diff_gray, float(background_diff_thresh), 255, cv2.THRESH_BINARY)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
    slot_polygons: list[dict] = []
    slot_mask = None
    if use_slot_roi and config is not None and projector is not None and z_world is not None:
        slot_polygons = projected_slot_polygons(config, projector, cv2=cv2, z_world=float(z_world), padding_m=float(slot_padding_m))
        if slot_polygons:
            slot_mask = slot_mask_for_image(hsv.shape[:2], slot_polygons, cv2=cv2)
    candidates: dict[str, list[TapeDetection]] = {}
    for color in colors:
        color = str(color).lower()
        ranges = COLOR_RANGES_HSV.get(color)
        if not ranges:
            continue
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.asarray(lo), np.asarray(hi)))
        if diff_mask is not None:
            if color == "white":
                mask = cv2.bitwise_and(mask, diff_mask)
            else:
                mask = cv2.bitwise_and(mask, cv2.dilate(diff_mask, kernel, iterations=1))
        if slot_mask is not None:
            mask = cv2.bitwise_and(mask, slot_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        best_area = 0.0
        best_slot = ""
        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_min = float(white_min_area_px if color == "white" else min_area_px)
            area_max = float(white_max_area_px if color == "white" else max_area_px)
            if area < area_min or area > area_max:
                continue
            moments = cv2.moments(contour)
            if abs(float(moments["m00"])) < 1e-9:
                continue
            center = np.asarray([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float64)
            slot_name = slot_for_pixel(center, slot_polygons, cv2=cv2) if slot_polygons else ""
            if use_slot_roi and slot_polygons and not slot_name:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = float(w) / max(1.0, float(h))
            extent = area / max(1.0, float(w * h))
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 0.0 if perimeter <= 1e-9 else float(4.0 * np.pi * area / (perimeter * perimeter))
            if color == "white":
                if not (0.35 <= aspect <= 2.8):
                    continue
                if extent < float(white_min_extent):
                    continue
                if circularity < float(white_min_circularity):
                    continue
                score = area * (0.4 + circularity + 0.3 * extent)
            else:
                score = area
            if score > best_score:
                best = contour
                best_score = score
                best_area = area
                best_slot = slot_name
            candidates.setdefault(color, []).append(
                TapeDetection(
                    color=color,
                    center_px=center,
                    area_px=area,
                    bbox_xywh=(int(x), int(y), int(w), int(h)),
                    score=float(score),
                    slot_name=slot_name,
                    contour=contour,
                )
            )

    if not slot_color_prior or not slot_polygons:
        detections: dict[str, TapeDetection] = {}
        for color, items in candidates.items():
            if items:
                detections[color] = max(items, key=lambda item: item.score)
        return detections

    detections = {}
    used_slots: set[str] = set()
    slot_names = [str(slot["name"]) for slot in slot_polygons]

    def pick_best(color_name: str, *, avoid_used: bool = True, allowed_slots: set[str] | None = None) -> TapeDetection | None:
        items = list(candidates.get(color_name, []))
        if allowed_slots is not None:
            items = [item for item in items if item.slot_name in allowed_slots]
        if avoid_used:
            unused_items = [item for item in items if item.slot_name and item.slot_name not in used_slots]
            if unused_items:
                items = unused_items
        if not items:
            return None
        return max(items, key=lambda item: item.score)

    for color_name in ("red", "yellow"):
        if color_name not in [str(c).lower() for c in colors]:
            continue
        det = pick_best(color_name, avoid_used=True)
        if det is not None:
            detections[color_name] = det
            if det.slot_name:
                used_slots.add(det.slot_name)

    if "white" in [str(c).lower() for c in colors]:
        remaining_slots = set(slot_names) - used_slots
        white_det = None
        if remaining_slots:
            white_det = pick_best("white", avoid_used=False, allowed_slots=remaining_slots)
        if white_det is None:
            white_det = pick_best("white", avoid_used=True)
        if white_det is not None:
            detections["white"] = white_det
            if white_det.slot_name:
                used_slots.add(white_det.slot_name)
        elif slot_fallback_center and remaining_slots:
            slot_name = sorted(remaining_slots)[0]
            slot = next(item for item in slot_polygons if str(item["name"]) == slot_name)
            center = slot_center_px(slot)
            detections["white"] = TapeDetection(
                color="white",
                center_px=center,
                area_px=0.0,
                bbox_xywh=(int(round(center[0] - 10)), int(round(center[1] - 10)), 20, 20),
                score=0.0,
                slot_name=slot_name,
                contour=None,
            )

    return detections


def attach_world_coordinates(
    detections: dict[str, TapeDetection],
    projector: CameraProjector,
    *,
    cv2,
    z_world: float,
) -> dict[str, TapeDetection]:
    for det in detections.values():
        det.world_m = projector.pixel_to_world_on_z(det.center_px, z_world, cv2)
    return detections


def draw_cross(img_bgr: np.ndarray, center_xy: Iterable[float], color: tuple[int, int, int], *, size: int = 8, thickness: int = 2) -> None:
    u, v = np.asarray(center_xy, dtype=np.float64).reshape(2)
    p = (int(round(u)), int(round(v)))
    cv2 = __import__("cv2")
    cv2.line(img_bgr, (p[0] - size, p[1]), (p[0] + size, p[1]), color, thickness)
    cv2.line(img_bgr, (p[0], p[1] - size), (p[0], p[1] + size), color, thickness)


def draw_detection_overlay(
    rgb: np.ndarray,
    *,
    cv2,
    detections: dict[str, TapeDetection],
    projector: CameraProjector,
    config: dict,
    tape_plane_z: float,
    active_color: str | None = None,
    draw_slots: bool = True,
    slot_padding_m: float = 0.025,
    myd_part_offset_m: Iterable[float] | None = None,
) -> np.ndarray:
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    if draw_slots:
        for slot in projected_slot_polygons(config, projector, cv2=cv2, z_world=tape_plane_z, padding_m=slot_padding_m):
            pts = np.asarray(slot["points"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(bgr, [pts], True, (40, 40, 40), 2, cv2.LINE_AA)
            center = np.mean(np.asarray(slot["points"], dtype=np.float64), axis=0)
            cv2.putText(
                bgr,
                str(slot["name"]),
                (int(round(center[0])) - 25, int(round(center[1]))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )
    for det in detections.values():
        color = DRAW_COLORS_BGR.get(det.color, (0, 255, 0))
        x, y, w, h = det.bbox_xywh
        if det.contour is not None:
            cv2.drawContours(bgr, [det.contour], -1, color, 2)
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 1)
        draw_cross(bgr, det.center_px, color)
        suffix = ""
        if det.world_m is not None:
            suffix = f" xyz=({det.world_m[0]:.3f},{det.world_m[1]:.3f},{det.world_m[2]:.3f})"
        slot = f" slot={det.slot_name}" if det.slot_name else ""
        label = f"{det.color}{slot} px=({det.center_px[0]:.0f},{det.center_px[1]:.0f}){suffix}"
        cv2.putText(bgr, label, (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    goal = task_goal_pos_from_config(config, manual_offset_m=myd_part_offset_m)
    goal_px, goal_z_cam = projector.project_world(goal, cv2)
    if projector.in_frame(goal_px, margin=50):
        draw_cross(bgr, goal_px, DRAW_COLORS_BGR["part"], size=12, thickness=2)
        cv2.putText(
            bgr,
            f"myd_part target px=({goal_px[0]:.0f},{goal_px[1]:.0f}) xyz=({goal[0]:.3f},{goal[1]:.3f},{goal[2]:.3f}) zcam={goal_z_cam:.3f}",
            (int(round(goal_px[0])) + 10, int(round(goal_px[1])) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            DRAW_COLORS_BGR["part"],
            1,
            cv2.LINE_AA,
        )

    if active_color and active_color in detections and detections[active_color].world_m is not None:
        grasp = nearest_wall_grasp_point(config, detections[active_color].world_m)
        grasp_px, _ = projector.project_world(grasp, cv2)
        if projector.in_frame(grasp_px, margin=30):
            draw_cross(bgr, grasp_px, DRAW_COLORS_BGR["grasp"], size=10, thickness=2)
            cv2.putText(
                bgr,
                f"nearest grasp {active_color}",
                (int(round(grasp_px[0])) + 8, int(round(grasp_px[1])) + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                DRAW_COLORS_BGR["grasp"],
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        bgr,
        f"plane_z={float(tape_plane_z):.3f}m  keys: s save, q quit",
        (12, bgr.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return bgr
