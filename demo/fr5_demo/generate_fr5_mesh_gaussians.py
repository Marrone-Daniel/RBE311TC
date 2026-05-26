from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from gaussian_renderer.core.gaussiandata import GaussianData
from gaussian_renderer.core.util_gau import save_ply

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_XML = DEMO_DIR / "assets" / "fr5" / "mjmodel.xml"
DEFAULT_OUTPUT_DIR = DEMO_DIR / "assets" / "fr5" / "3dgs_mesh"
SH_C0 = 0.28209479177387814


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
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    rot = quat_wxyz_to_matrix(quat_wxyz)
    return points @ rot.T + pos.reshape(1, 3)


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    return (rgb - 0.5) / SH_C0


def material_rgba(root: ET.Element) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for material in root.findall(".//material"):
        name = material.get("name")
        if name:
            out[name] = parse_vec(material.get("rgba"), (0.75, 0.75, 0.75, 1.0))
    return out


def mesh_assets(root: ET.Element, model_dir: Path) -> dict[str, tuple[Path, np.ndarray]]:
    out: dict[str, tuple[Path, np.ndarray]] = {}
    for mesh in root.findall(".//asset/mesh"):
        name = mesh.get("name")
        file_name = mesh.get("file")
        if not name or not file_name:
            continue
        scale = parse_vec(mesh.get("scale"), (1.0, 1.0, 1.0))
        out[name] = (model_dir / file_name, scale)
    return out


def geom_color(geom: ET.Element, materials: dict[str, np.ndarray]) -> np.ndarray:
    if geom.get("rgba"):
        rgba = parse_vec(geom.get("rgba"), (0.75, 0.75, 0.75, 1.0))
    elif geom.get("material") in materials:
        rgba = materials[str(geom.get("material"))]
    elif geom.get("class") == "visual_gripper" and "black" in materials:
        rgba = materials["black"]
    else:
        rgba = np.asarray((0.75, 0.75, 0.75, 1.0), dtype=np.float64)
    return np.clip(rgba[:3], 0.0, 1.0).astype(np.float32)


def load_mesh_points(mesh_path: Path, scale: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    loaded = trimesh.load(mesh_path.as_posix(), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.dump()))
    mesh = loaded.copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale.reshape(1, 3)
    if len(mesh.faces) > 0 and mesh.area > 1e-12:
        seed = int(rng.integers(0, 2**31 - 1))
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            points, _ = trimesh.sample.sample_surface(mesh, int(count))
        finally:
            np.random.set_state(state)
        return np.asarray(points, dtype=np.float32)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    ids = rng.choice(vertices.shape[0], size=int(count), replace=vertices.shape[0] < int(count))
    return vertices[ids]


def direct_visual_geoms(body: ET.Element) -> list[ET.Element]:
    geoms = []
    for child in list(body):
        if child.tag != "geom":
            continue
        if child.get("mesh") or child.get("type") == "box" or child.get("group") == "2" or child.get("class") == "visual_gripper":
            geoms.append(child)
    return geoms


def sample_box_surface(size: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    sx, sy, sz = np.asarray(size, dtype=np.float64).reshape(3)
    areas = np.asarray([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy], dtype=np.float64)
    probs = areas / np.sum(areas)
    faces = rng.choice(6, size=int(count), p=probs)
    points = np.empty((int(count), 3), dtype=np.float64)
    for idx, face in enumerate(faces):
        u = rng.uniform(-1.0, 1.0)
        v = rng.uniform(-1.0, 1.0)
        if face == 0:
            points[idx] = [sx, u * sy, v * sz]
        elif face == 1:
            points[idx] = [-sx, u * sy, v * sz]
        elif face == 2:
            points[idx] = [u * sx, sy, v * sz]
        elif face == 3:
            points[idx] = [u * sx, -sy, v * sz]
        elif face == 4:
            points[idx] = [u * sx, v * sy, sz]
        else:
            points[idx] = [u * sx, v * sy, -sz]
    return points.astype(np.float32)


def iter_bodies(worldbody: ET.Element):
    for body in worldbody.findall("body"):
        yield from iter_body_tree(body)


def iter_body_tree(body: ET.Element):
    yield body
    for child in body.findall("body"):
        yield from iter_body_tree(child)


def build_body_gaussians(
    body: ET.Element,
    assets: dict[str, tuple[Path, np.ndarray]],
    materials: dict[str, np.ndarray],
    *,
    points_per_geom: int,
    scale: float,
    opacity: float,
    rng: np.random.Generator,
) -> GaussianData | None:
    xyz_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    for geom in direct_visual_geoms(body):
        mesh_name = geom.get("mesh")
        geom_type = geom.get("type", "sphere")
        if mesh_name:
            if mesh_name not in assets:
                continue
            mesh_path, mesh_scale = assets[mesh_name]
            if not mesh_path.exists():
                print(f"Warning: missing mesh asset: {mesh_path}")
                continue
            points = load_mesh_points(mesh_path, mesh_scale, points_per_geom, rng)
        elif geom_type == "box":
            size = parse_vec(geom.get("size"), (0.01, 0.01, 0.01))
            points = sample_box_surface(size, points_per_geom, rng)
        else:
            continue
        if points.shape[0] == 0:
            continue
        pos = parse_vec(geom.get("pos"), (0.0, 0.0, 0.0))
        quat = parse_vec(geom.get("quat"), (1.0, 0.0, 0.0, 0.0))
        points = transform_points(points, pos, quat).astype(np.float32)
        color = geom_color(geom, materials)
        xyz_parts.append(points)
        color_parts.append(np.repeat(color.reshape(1, 3), points.shape[0], axis=0))

    if not xyz_parts:
        return None

    xyz = np.concatenate(xyz_parts, axis=0).astype(np.float32)
    colors = np.concatenate(color_parts, axis=0).astype(np.float32)
    rot = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    rot[:, 0] = 1.0
    scales = np.full((xyz.shape[0], 3), float(scale), dtype=np.float32)
    opacities = np.full((xyz.shape[0],), float(opacity), dtype=np.float32)
    sh = rgb_to_sh(colors).astype(np.float32)
    return GaussianData(xyz, rot, scales, opacities, sh)


def generate_fr5_mesh_gaussians(
    model_xml: str | Path = DEFAULT_MODEL_XML,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    points_per_geom: int = 10000,
    scale: float = 0.0011,
    opacity: float = 0.58,
    seed: int = 7,
) -> dict[str, str]:
    model_xml = Path(model_xml)
    output_dir = Path(output_dir)
    tree = ET.parse(model_xml)
    root = tree.getroot()
    assets = mesh_assets(root, model_xml.parent)
    materials = material_rgba(root)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No <worldbody> in {model_xml}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    generated: dict[str, str] = {}
    manifest: dict[str, dict[str, object]] = {}

    for body in iter_bodies(worldbody):
        name = body.get("name")
        if not name:
            continue
        gaussian = build_body_gaussians(
            body,
            assets,
            materials,
            points_per_geom=int(points_per_geom),
            scale=float(scale),
            opacity=float(opacity),
            rng=rng,
        )
        if gaussian is None:
            continue
        out_path = output_dir / f"{name}.ply"
        save_ply(gaussian, out_path, save_sh_degree=0)
        generated[name] = out_path.as_posix()
        manifest[name] = {"path": out_path.as_posix(), "points": len(gaussian)}
        print(f"generated {name}: {len(gaussian)} points -> {out_path}", flush=True)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}", flush=True)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mesh-derived 3DGS PLY assets for the FR5 MotrixSim model")
    parser.add_argument("--model-xml", type=str, default=DEFAULT_MODEL_XML.as_posix())
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--points-per-geom", type=int, default=10000)
    parser.add_argument("--scale", type=float, default=0.0011)
    parser.add_argument("--opacity", type=float, default=0.58)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    generate_fr5_mesh_gaussians(
        args.model_xml,
        args.output_dir,
        points_per_geom=args.points_per_geom,
        scale=args.scale,
        opacity=args.opacity,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
