from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parents[1]
DEFAULT_MODEL_XML = DEMO_DIR / "assets" / "fr5" / "mjmodel.xml"
DEFAULT_CONFIG = DEMO_DIR / "configs" / "fr5_table_task.json"
DEFAULT_GS_DIR = DEMO_DIR / "assets" / "fr5" / "3dgs_mesh"
DEFAULT_REAL2SIM_OUTPUT = PROJECT_ROOT / "real2sim_output"
SH_C0 = 0.28209479177387814

if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from real2sim.io_utils import dump_mapping, ensure_dir  # noqa: E402


def parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def fmt_vec(values: np.ndarray | list[float]) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def remove_child_by_name(parent: ET.Element, tag: str, name: str) -> bool:
    removed = False
    for child in list(parent):
        if child.tag == tag and child.get("name") == name:
            parent.remove(child)
            removed = True
    return removed


def remove_body_recursive(parent: ET.Element, name: str) -> bool:
    removed = remove_child_by_name(parent, "body", name)
    for child in list(parent):
        if child.tag == "body":
            removed = remove_body_recursive(child, name) or removed
    return removed


def target_body_from_real2sim(path: Path) -> ET.Element | None:
    if not path.exists():
        return None
    body = ET.parse(path).getroot()
    if body.tag != "body":
        raise RuntimeError(f"Expected root <body> in {path}")
    body.set("name", body.get("name") or "target_object")
    if body.find("site") is None:
        ET.SubElement(body, "site", {"name": f"{body.get('name')}_site", "pos": "0 0 0", "size": "0.01", "rgba": "0 1 0 1"})
    return body


def placeholder_target_body() -> ET.Element:
    body = ET.Element("body", {"name": "target_object", "pos": "-0.55 0 0.015"})
    ET.SubElement(body, "freejoint", {"name": "target_object"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": "target_object_collision",
            "type": "box",
            "size": "0.015 0.015 0.015",
            "mass": "0.05",
            "friction": "1 0.005 0.0001",
            "rgba": "0.8 0.3 0.2 1",
        },
    )
    ET.SubElement(body, "site", {"name": "target_object_site", "pos": "0 0 0", "size": "0.006", "rgba": "0 1 0 1"})
    return body


def sample_box_surface(size: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    sx, sy, sz = np.asarray(size, dtype=np.float64).reshape(3)
    areas = np.asarray([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy], dtype=np.float64)
    probs = areas / max(float(np.sum(areas)), 1e-12)
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


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    return (rgb - 0.5) / SH_C0


def write_target_gaussian(body: ET.Element, gs_dir: Path, *, points: int, scale: float, opacity: float) -> Path:
    from gaussian_renderer.core.gaussiandata import GaussianData
    from gaussian_renderer.core.util_gau import save_ply

    name = body.get("name") or "target_object"
    geom = body.find("geom")
    if geom is None or geom.get("type", "box") != "box":
        raise RuntimeError("Target Gaussian generation currently requires a box geom.")
    size = parse_vec(geom.get("size"), (0.015, 0.015, 0.015))
    pos = parse_vec(geom.get("pos"), (0.0, 0.0, 0.0))
    rgba = parse_vec(geom.get("rgba"), (0.8, 0.3, 0.2, 1.0))
    rng = np.random.default_rng(11)
    xyz = sample_box_surface(size, int(points), rng) + pos.reshape(1, 3).astype(np.float32)
    colors = np.repeat(np.clip(rgba[:3], 0.0, 1.0).reshape(1, 3).astype(np.float32), xyz.shape[0], axis=0)
    rot = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    rot[:, 0] = 1.0
    scales = np.full((xyz.shape[0], 3), float(scale), dtype=np.float32)
    opacities = np.full((xyz.shape[0],), float(opacity), dtype=np.float32)
    sh = rgb_to_sh(colors).astype(np.float32)
    ensure_dir(gs_dir)
    out = gs_dir / f"{name}.ply"
    save_ply(GaussianData(xyz.astype(np.float32), rot, scales, opacities, sh), out, save_sh_degree=0)
    return out


def copy_real2sim_target_gaussian(real2sim_output: Path, gs_dir: Path, target_name: str) -> Path | None:
    visual_dir = real2sim_output / "visual"
    candidates = [visual_dir / "object_gaussian.ply"]
    candidates.extend(sorted(visual_dir.glob("*.ply")))
    for src in candidates:
        if not src.exists() or src.name.startswith("pointcloud"):
            continue
        ensure_dir(gs_dir)
        dst = gs_dir / f"{target_name}.ply"
        shutil.copy2(src, dst)
        return dst
    return None


def update_task_config(path: Path, target_site: str, initial_qpos: list[float]) -> None:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg.pop("cube_site", None)
    cfg["target_site"] = target_site
    cfg["initial_qpos"] = initial_qpos
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def body_names(root: ET.Element) -> set[str]:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return set()
    return {body.get("name") for body in worldbody.iter("body") if body.get("name")}


def write_binding(root: ET.Element, gs_dir: Path, output_dir: Path, model_xml: Path) -> Path:
    names = body_names(root)
    static_assets = {}
    for ply in sorted(gs_dir.glob("*.ply")):
        name = ply.stem
        if name not in names or name == "cube":
            continue
        role = "dynamic_target" if name == "target_object" else "static_scene"
        static_assets[name] = {
            "role": role,
            "mujoco_body": name,
            "mujoco_xml": model_xml.as_posix(),
            "mesh_file": "",
            "gaussian_file": ply.as_posix(),
            "binding_mode": "mesh_vertices",
            "physics_source": "existing_mujoco",
            "visual_source": "3dgs",
        }
    path = output_dir / "sim" / "scene_gs_binding.yaml"
    return dump_mapping(
        path,
        {
            "scene_gs_binding": {
                "description": "FR5 demo scene binding generated from existing MuJoCo bodies and 3DGS PLY assets.",
                "assets": static_assets,
            }
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Real2Sim assets to the FR5 demo scene.")
    parser.add_argument("--model-xml", type=str, default=DEFAULT_MODEL_XML.as_posix())
    parser.add_argument("--task-config", type=str, default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--real2sim-output", type=str, default=DEFAULT_REAL2SIM_OUTPUT.as_posix())
    parser.add_argument("--gs-dir", type=str, default=DEFAULT_GS_DIR.as_posix())
    parser.add_argument("--target-points", type=int, default=30000)
    parser.add_argument("--target-scale", type=float, default=0.0011)
    parser.add_argument("--target-opacity", type=float, default=0.62)
    parser.add_argument("--no-placeholder-target", action="store_true")
    parser.add_argument("--initial-qpos", type=float, nargs=6, default=[0.0, -1.2, 1.5, 0.9, 1.5617, 0.0])
    args = parser.parse_args()

    model_xml = Path(args.model_xml)
    task_config = Path(args.task_config)
    real2sim_output = Path(args.real2sim_output)
    gs_dir = Path(args.gs_dir)

    tree = ET.parse(model_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No <worldbody> in {model_xml}")

    removed_cube = remove_body_recursive(worldbody, "cube")
    remove_body_recursive(worldbody, "target_object")

    target_body = target_body_from_real2sim(real2sim_output / "sim" / "mujoco_object.xml")
    source = "real2sim_output/sim/mujoco_object.xml"
    if target_body is None:
        if args.no_placeholder_target:
            source = "none"
        else:
            target_body = placeholder_target_body()
            source = "placeholder target_object"
    if target_body is not None:
        worldbody.insert(1, target_body)
        target_name = target_body.get("name") or "target_object"
        target_site = f"{target_name}_site"
        target_ply = copy_real2sim_target_gaussian(real2sim_output, gs_dir, target_name)
        target_gaussian_source = "real2sim_output/visual/object_gaussian.ply" if target_ply is not None else "generated box gaussian"
        if target_ply is None:
            target_ply = write_target_gaussian(
                target_body,
                gs_dir,
                points=int(args.target_points),
                scale=float(args.target_scale),
                opacity=float(args.target_opacity),
            )
    else:
        target_site = ""
        target_ply = None
        target_gaussian_source = "none"

    ET.indent(tree, space="  ")
    tree.write(model_xml, encoding="unicode")
    update_task_config(task_config, target_site, [float(v) for v in args.initial_qpos])
    binding_path = write_binding(root, gs_dir, real2sim_output, model_xml)

    print(f"model_xml: {model_xml}")
    print(f"removed_cube: {removed_cube}")
    print(f"target_source: {source}")
    if target_ply is not None:
        print(f"target_gaussian: {target_ply}")
        print(f"target_gaussian_source: {target_gaussian_source}")
    print(f"task_config: {task_config}")
    print(f"scene_binding: {binding_path}")


if __name__ == "__main__":
    main()
