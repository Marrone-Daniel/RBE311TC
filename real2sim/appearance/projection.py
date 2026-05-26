from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from real2sim.appearance.camera import PinholeCamera
from real2sim.appearance.mujoco_geometry import SurfaceSamples


@dataclass(frozen=True)
class ProjectedSamples:
    sample: SurfaceSamples
    uv_image: np.ndarray
    depth_camera: np.ndarray
    in_frame: np.ndarray
    visible: np.ndarray


def compute_visibility(
    camera: PinholeCamera,
    samples: list[SurfaceSamples],
    *,
    z_tolerance_m: float,
) -> list[ProjectedSamples]:
    if not samples:
        return []
    all_points = np.concatenate([sample.points_world for sample in samples], axis=0)
    all_uv, all_in_frame = camera.project_world(all_points)
    all_depth = camera.world_to_camera(all_points)[:, 2]

    zbuf = np.full((camera.height, camera.width), np.inf, dtype=np.float64)
    rounded = np.rint(all_uv).astype(np.int64)
    for idx, valid in enumerate(all_in_frame):
        if not valid:
            continue
        x, y = int(rounded[idx, 0]), int(rounded[idx, 1])
        if 0 <= x < camera.width and 0 <= y < camera.height:
            zbuf[y, x] = min(zbuf[y, x], float(all_depth[idx]))

    out: list[ProjectedSamples] = []
    offset = 0
    for sample in samples:
        n = sample.points_world.shape[0]
        uv = all_uv[offset : offset + n]
        depth = all_depth[offset : offset + n]
        in_frame = all_in_frame[offset : offset + n]
        rounded_sample = np.rint(uv).astype(np.int64)
        visible = np.zeros(n, dtype=bool)
        for i, valid in enumerate(in_frame):
            if not valid:
                continue
            x, y = int(rounded_sample[i, 0]), int(rounded_sample[i, 1])
            if 0 <= x < camera.width and 0 <= y < camera.height:
                visible[i] = depth[i] <= zbuf[y, x] + float(z_tolerance_m)
        out.append(ProjectedSamples(sample, uv, depth, in_frame, visible))
        offset += n
    return out


def draw_projected_overlay(image_bgr: np.ndarray, projected: list[ProjectedSamples]) -> np.ndarray:
    overlay = image_bgr.copy()
    palette = [
        (40, 220, 255),
        (255, 120, 40),
        (80, 255, 120),
        (255, 80, 220),
        (180, 180, 255),
    ]
    for idx, item in enumerate(projected):
        color = palette[idx % len(palette)]
        uv = np.rint(item.uv_image[item.visible]).astype(np.int64)
        for x, y in uv[:: max(1, len(uv) // 2500)]:
            cv2.circle(overlay, (int(x), int(y)), 1, color, -1, lineType=cv2.LINE_AA)
    return overlay
