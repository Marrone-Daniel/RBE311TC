from __future__ import annotations

import json
from math import atan, degrees
from pathlib import Path

import numpy as np
from motrixsim import msd


def load_camera_config(path: str | Path) -> dict:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Camera config must be a JSON object: {cfg_path}")
    return cfg


def camera_fovy_deg(height: int, fy: float) -> float:
    if fy <= 0.0:
        raise ValueError(f"fy must be positive, got {fy}")
    return degrees(2.0 * atan(float(height) / (2.0 * float(fy))))


def opencv_rotation_to_mujoco_xyaxes(r_world_camera_cv: np.ndarray) -> str:
    r = np.asarray(r_world_camera_cv, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"rotation_matrix must be 3x3, got {r.shape}")
    cam_x_world = r[:, 0]
    cam_y_world = -r[:, 1]
    vals = np.concatenate([cam_x_world, cam_y_world])
    return " ".join(f"{v:.10g}" for v in vals)


def matrix_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    return quat


def v3_text(vals: np.ndarray) -> str:
    return " ".join(f"{float(v):.10g}" for v in np.asarray(vals, dtype=np.float64).reshape(3))


def rotation_matrix_from_rpy_deg(rpy_deg: list[float] | tuple[float, float, float] | np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
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
    delta_rot = rotation_matrix_from_rpy_deg(correction.get("rotation_rpy_deg", [10.0, 0.0, 0.0]))
    if frame == "camera_local_opencv":
        return pos + rot @ delta_pos, rot @ delta_rot
    if frame == "world":
        return pos + delta_pos, delta_rot @ rot
    raise ValueError(f"manual_extrinsic_correction.frame must be 'camera_local_opencv' or 'world', got {frame!r}")


def camera_config_to_mjcf(config: dict) -> str:
    if not config.get("calibrated", False):
        raise ValueError("Camera config is not calibrated. Set calibrated=true after extrinsic calibration.")

    intr = config["intrinsics"]
    ext = config["extrinsics"]
    width = int(intr["width"])
    height = int(intr["height"])
    fy = float(intr["fy"])
    pos = np.asarray(ext["position"], dtype=np.float64)
    rot = np.asarray(ext["rotation_matrix"], dtype=np.float64)
    if pos.shape != (3,):
        raise ValueError(f"extrinsics.position must have 3 values, got {pos.shape}")
    pos, rot = apply_manual_extrinsic_correction(pos, rot, config)

    name = str(config.get("name", "astra_rgb"))
    fovy = camera_fovy_deg(height, fy)
    xyaxes = opencv_rotation_to_mujoco_xyaxes(rot)
    pos_text = " ".join(f"{v:.10g}" for v in pos)
    return f"""<mujoco model="{name}_camera">
  <worldbody>
    <camera name="{name}" pos="{pos_text}" xyaxes="{xyaxes}" fovy="{fovy:.10g}" resolution="{width} {height}"/>
  </worldbody>
</mujoco>"""


def camera_overlay_plane_to_mjcf(
    config: dict,
    *,
    texture_name: str = "astra_real_tex",
    material_name: str = "astra_real_mat",
    distance: float = 0.75,
    alpha: float = 0.45,
    thickness: float = 0.002,
) -> str:
    if not config.get("calibrated", False):
        raise ValueError("Camera config is not calibrated. Set calibrated=true after extrinsic calibration.")

    intr = config["intrinsics"]
    ext = config["extrinsics"]
    width = int(intr["width"])
    height = int(intr["height"])
    fy = float(intr["fy"])
    pos = np.asarray(ext["position"], dtype=np.float64)
    rot = np.asarray(ext["rotation_matrix"], dtype=np.float64)
    pos, rot = apply_manual_extrinsic_correction(pos, rot, config)
    fovy = np.deg2rad(camera_fovy_deg(height, fy))
    aspect = float(width) / float(height)
    dist = float(distance)
    half_h = dist * np.tan(fovy * 0.5)
    half_w = half_h * aspect

    cam_x = rot[:, 0]
    cam_y = -rot[:, 1]
    cam_fwd = rot[:, 2]
    center = pos + cam_fwd * dist
    quat = matrix_to_quat_wxyz(np.column_stack([-cam_x, cam_y, cam_fwd]))
    quat_text = " ".join(f"{float(v):.10g}" for v in quat)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    return f"""<mujoco model="{texture_name}_overlay">
  <asset>
    <texture name="{texture_name}" type="2d" builtin="dynamic" width="{width}" height="{height}" _perinstance="true"/>
    <material name="{material_name}" texture="{texture_name}" rgba="1 1 1 {alpha:.6g}" emission="1 1 1 1" castshadow="false"/>
  </asset>
  <worldbody>
    <geom name="{texture_name}_plane" type="box" size="{half_w:.10g} {half_h:.10g} {float(thickness):.10g}" pos="{v3_text(center)}" quat="{quat_text}" material="{material_name}" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>"""


def camera_screen_to_mjcf(
    config: dict,
    *,
    texture_name: str = "astra_real_tex",
    material_name: str = "astra_real_mat",
    edge_material_name: str = "astra_screen_edge_mat",
    distance: float = 0.75,
    alpha: float = 1.0,
    thickness: float = 0.004,
    edge_radius: float = 0.002,
) -> str:
    if not config.get("calibrated", False):
        raise ValueError("Camera config is not calibrated. Set calibrated=true after extrinsic calibration.")

    intr = config["intrinsics"]
    ext = config["extrinsics"]
    width = int(intr["width"])
    height = int(intr["height"])
    fy = float(intr["fy"])
    pos = np.asarray(ext["position"], dtype=np.float64)
    rot = np.asarray(ext["rotation_matrix"], dtype=np.float64)
    pos, rot = apply_manual_extrinsic_correction(pos, rot, config)
    fovy = np.deg2rad(camera_fovy_deg(height, fy))
    aspect = float(width) / float(height)
    dist = float(distance)
    half_h = dist * np.tan(fovy * 0.5)
    half_w = half_h * aspect

    cam_x = rot[:, 0]
    cam_y = -rot[:, 1]
    cam_fwd = rot[:, 2]
    center = pos + cam_fwd * dist
    c0 = center + (-cam_x * half_w) + (cam_y * half_h)
    c1 = center + (cam_x * half_w) + (cam_y * half_h)
    c2 = center + (cam_x * half_w) + (-cam_y * half_h)
    c3 = center + (-cam_x * half_w) + (-cam_y * half_h)
    quat = matrix_to_quat_wxyz(np.column_stack([-cam_x, cam_y, cam_fwd]))
    quat_text = " ".join(f"{float(v):.10g}" for v in quat)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    return f"""<mujoco model="{texture_name}_screen">
  <asset>
    <texture name="{texture_name}" type="2d" builtin="dynamic" width="{width}" height="{height}" _perinstance="true"/>
    <material name="{material_name}" texture="{texture_name}" rgba="1 1 1 {alpha:.6g}" emission="1 1 1 1" castshadow="false"/>
    <material name="{edge_material_name}" rgba="1 1 1 1" emission="0.6 0.6 0.6 1" castshadow="false"/>
  </asset>
  <worldbody>
    <geom name="{texture_name}_screen" type="box" size="{half_w:.10g} {half_h:.10g} {float(thickness):.10g}" pos="{v3_text(center)}" quat="{quat_text}" material="{material_name}" contype="0" conaffinity="0"/>
    <geom name="{texture_name}_frustum_0" type="capsule" size="{float(edge_radius):.10g}" fromto="{v3_text(pos)} {v3_text(c0)}" material="{edge_material_name}" contype="0" conaffinity="0"/>
    <geom name="{texture_name}_frustum_1" type="capsule" size="{float(edge_radius):.10g}" fromto="{v3_text(pos)} {v3_text(c1)}" material="{edge_material_name}" contype="0" conaffinity="0"/>
    <geom name="{texture_name}_frustum_2" type="capsule" size="{float(edge_radius):.10g}" fromto="{v3_text(pos)} {v3_text(c2)}" material="{edge_material_name}" contype="0" conaffinity="0"/>
    <geom name="{texture_name}_frustum_3" type="capsule" size="{float(edge_radius):.10g}" fromto="{v3_text(pos)} {v3_text(c3)}" material="{edge_material_name}" contype="0" conaffinity="0"/>
    <geom name="{texture_name}_edge_0" type="capsule" size="{float(edge_radius):.10g}" fromto="{v3_text(c0)} {v3_text(c1)}" material="{edge_material_name}" contype="0" conaffinity="0"/>
    <geom name="{texture_name}_edge_1" type="capsule" size="{float(edge_radius):.10g}" fromto="{v3_text(c1)} {v3_text(c2)}" material="{edge_material_name}" contype="0" conaffinity="0"/>
    <geom name="{texture_name}_edge_2" type="capsule" size="{float(edge_radius):.10g}" fromto="{v3_text(c2)} {v3_text(c3)}" material="{edge_material_name}" contype="0" conaffinity="0"/>
    <geom name="{texture_name}_edge_3" type="capsule" size="{float(edge_radius):.10g}" fromto="{v3_text(c3)} {v3_text(c0)}" material="{edge_material_name}" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>"""


def attach_camera_from_config(scene, config_path: str | Path):
    cfg = load_camera_config(config_path)
    camera_xml = camera_config_to_mjcf(cfg)
    scene.attach(msd.from_str(camera_xml))
    return scene


def attach_camera_overlay_from_config(
    scene,
    config_path: str | Path,
    *,
    distance: float = 0.75,
    alpha: float = 0.45,
    texture_name: str = "astra_real_tex",
    material_name: str = "astra_real_mat",
):
    cfg = load_camera_config(config_path)
    overlay_xml = camera_overlay_plane_to_mjcf(
        cfg,
        texture_name=texture_name,
        material_name=material_name,
        distance=distance,
        alpha=alpha,
    )
    scene.attach(msd.from_str(overlay_xml))
    return scene


def attach_camera_screen_from_config(
    scene,
    config_path: str | Path,
    *,
    distance: float = 0.75,
    alpha: float = 1.0,
    texture_name: str = "astra_real_tex",
    material_name: str = "astra_real_mat",
    edge_material_name: str = "astra_screen_edge_mat",
):
    cfg = load_camera_config(config_path)
    screen_xml = camera_screen_to_mjcf(
        cfg,
        texture_name=texture_name,
        material_name=material_name,
        edge_material_name=edge_material_name,
        distance=distance,
        alpha=alpha,
    )
    scene.attach(msd.from_str(screen_xml))
    return scene
