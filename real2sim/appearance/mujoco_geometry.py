from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SurfaceSamples:
    body_name: str
    geom_name: str
    points_world: np.ndarray
    uv_local: np.ndarray
    face_id: np.ndarray


def parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_matrix(quat)
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ T[:3, :3].T + T[:3, 3].reshape(1, 3)


def box_surface_samples(half_size: np.ndarray, density: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx, sy, sz = np.asarray(half_size, dtype=np.float64)
    density = max(2, int(density))
    lin = np.linspace(-1.0, 1.0, density)
    uu, vv = np.meshgrid(lin, lin, indexing="xy")
    grid = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)
    points: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    face_defs = [
        (np.column_stack([np.full(len(grid), sx), grid[:, 0] * sy, grid[:, 1] * sz]), 0),
        (np.column_stack([np.full(len(grid), -sx), grid[:, 0] * sy, grid[:, 1] * sz]), 1),
        (np.column_stack([grid[:, 0] * sx, np.full(len(grid), sy), grid[:, 1] * sz]), 2),
        (np.column_stack([grid[:, 0] * sx, np.full(len(grid), -sy), grid[:, 1] * sz]), 3),
        (np.column_stack([grid[:, 0] * sx, grid[:, 1] * sy, np.full(len(grid), sz)]), 4),
        (np.column_stack([grid[:, 0] * sx, grid[:, 1] * sy, np.full(len(grid), -sz)]), 5),
    ]
    base_uv = (grid + 1.0) * 0.5
    for pts, face_id in face_defs:
        col = face_id % 3
        row = face_id // 3
        uv = np.empty_like(base_uv)
        uv[:, 0] = (col + base_uv[:, 0]) / 3.0
        uv[:, 1] = (row + base_uv[:, 1]) / 2.0
        points.append(pts)
        uvs.append(uv)
        faces.append(np.full(pts.shape[0], face_id, dtype=np.int32))
    return np.concatenate(points, axis=0), np.concatenate(uvs, axis=0), np.concatenate(faces, axis=0)


def _is_visual_box_geom(geom: ET.Element) -> bool:
    if geom.get("type", "sphere") != "box":
        return False
    if geom.get("group") == "3":
        return False
    cls = geom.get("class", "")
    name = geom.get("name", "")
    return "collision" not in cls and "collision" not in name


def sample_target_box_geoms(model_xml: str | Path, body_names: list[str], *, density: int) -> list[SurfaceSamples]:
    tree = ET.parse(model_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No <worldbody> in {model_xml}")
    wanted = set(body_names)
    samples: list[SurfaceSamples] = []

    def visit_body(body: ET.Element, parent_T: np.ndarray) -> None:
        body_name = body.get("name", "")
        body_T = parent_T @ make_transform(parse_vec(body.get("pos"), (0.0, 0.0, 0.0)), parse_vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0)))
        if body_name in wanted:
            for idx, geom in enumerate(body.findall("geom")):
                if not _is_visual_box_geom(geom):
                    continue
                geom_T = body_T @ make_transform(parse_vec(geom.get("pos"), (0.0, 0.0, 0.0)), parse_vec(geom.get("quat"), (1.0, 0.0, 0.0, 0.0)))
                local, uv, faces = box_surface_samples(parse_vec(geom.get("size"), (0.01, 0.01, 0.01)), density)
                samples.append(
                    SurfaceSamples(
                        body_name=body_name,
                        geom_name=geom.get("name") or f"{body_name}_geom_{idx}",
                        points_world=transform_points(geom_T, local),
                        uv_local=uv,
                        face_id=faces,
                    )
                )
        for child in body.findall("body"):
            visit_body(child, body_T)

    for body in worldbody.findall("body"):
        visit_body(body, np.eye(4, dtype=np.float64))
    return samples
