from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Real2SimConfig, load_real2sim_config
from .geometry.camera import load_camera_pose, load_intrinsics, transform_points
from .geometry.coordinate_transform import pose_dict_from_matrix
from .geometry.depth_to_pointcloud import masked_depth_to_pointcloud
from .geometry.pose_estimation import estimate_axis_aligned_object
from .io_utils import dump_mapping, ensure_dir, read_depth_meters, read_rgb, write_json, write_text
from .reconstruction.dummy_reconstructor import DummyReconstructor, write_pointcloud_ply
from .reconstruction.recgen_adapter import RecGenAdapter
from .reconstruction.sam3dgs_adapter import Sam3DGSAdapter
from .segmentation.mask_loader import load_mask
from .simulation.collision_generator import generate_box_collision
from .simulation.gs_asset_writer import write_gs_asset_config
from .simulation.mujoco_asset_writer import write_mujoco_box_xml
from .simulation.scene_binding_writer import write_scene_binding_config, write_scene_binding_report
from .validation.projection_check import save_mask_check, save_mask_overlay
from .validation.report_writer import write_report


class Real2SimPipeline:
    def __init__(self, config: Real2SimConfig) -> None:
        self.config = config

    @classmethod
    def from_config_file(cls, path: str | Path | None) -> "Real2SimPipeline":
        return cls(load_real2sim_config(path))

    def _reconstructor(self):
        backend = self.config.reconstruction.backend.lower()
        if backend == "dummy":
            return DummyReconstructor()
        if backend == "recgen":
            return RecGenAdapter()
        if backend == "sam3dgs":
            return Sam3DGSAdapter()
        raise RuntimeError(f"Unknown reconstruction backend: {self.config.reconstruction.backend}")

    def _write_scene_bindings(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        sim_dir: Path,
        debug_dir: Path,
        warnings: list[str],
        generated: list[Path],
    ) -> dict:
        if not self.config.scene.enabled or not self.config.scene.static_assets:
            return {"enabled": False, "assets": 0, "config": ""}
        scene_warnings: list[str] = []
        for name, spec in self.config.scene.static_assets.items():
            if not isinstance(spec, dict):
                scene_warnings.append(f"scene.static_assets.{name} is not a mapping; it will fail validation.")
                continue
            for key in ("mujoco_xml", "mesh_file", "gaussian_file"):
                value = str(spec.get(key, "") or "")
                if not value:
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = input_dir / path if (input_dir / path).exists() else Path.cwd() / path
                if not path.exists():
                    scene_warnings.append(f"scene asset `{name}` references missing {key}: {value}")
        warnings.extend(scene_warnings)
        if self.config.scene.export_scene_gs_config:
            generated.append(
                write_scene_binding_config(
                    sim_dir / "scene_gs_binding.yaml",
                    static_assets=self.config.scene.static_assets,
                )
            )
        if self.config.debug.save_report:
            generated.append(
                write_scene_binding_report(
                    debug_dir / "scene_binding_report.md",
                    static_assets=self.config.scene.static_assets,
                    warnings=scene_warnings,
                )
            )
        return {
            "enabled": True,
            "assets": len(self.config.scene.static_assets),
            "config": "sim/scene_gs_binding.yaml" if self.config.scene.export_scene_gs_config else "",
        }

    def run(self, *, input_dir: str | Path, output_dir: str | Path) -> dict:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        visual_dir = ensure_dir(output_dir / "visual")
        collision_dir = ensure_dir(output_dir / "collision")
        sim_dir = ensure_dir(output_dir / "sim")
        debug_dir = ensure_dir(output_dir / "debug")
        warnings: list[str] = []
        generated: list[Path] = []

        rgb_path = input_dir / "rgb.png"
        depth_path = input_dir / "depth.png"
        mask_path = input_dir / "mask.png"
        intrinsics_path = input_dir / self.config.camera.intrinsics_file
        camera_pose_path = input_dir / self.config.camera.camera_pose_file
        scene_binding = self._write_scene_bindings(
            input_dir=input_dir,
            output_dir=output_dir,
            sim_dir=sim_dir,
            debug_dir=debug_dir,
            warnings=warnings,
            generated=generated,
        )
        has_target_input = rgb_path.exists() and depth_path.exists() and mask_path.exists()
        if not has_target_input:
            if scene_binding["enabled"]:
                result_path = output_dir / "real2sim_result.json"
                result = {
                    "mode": "scene_mesh_3dgs_binding",
                    "target_object_enabled": False,
                    "scene_binding": scene_binding,
                    "warnings": warnings,
                    "generated_files": [str(path.relative_to(output_dir)) for path in generated] + ["real2sim_result.json"],
                }
                write_json(result_path, result)
                return result
            missing = [str(path) for path in (rgb_path, depth_path, mask_path) if not path.exists()]
            raise RuntimeError(f"Missing target RGB-D input files: {missing}")

        rgb = read_rgb(rgb_path)
        depth = read_depth_meters(depth_path, depth_scale=self.config.camera.depth_scale)
        mask = load_mask(mask_path)
        if rgb.shape[:2] != depth.shape[:2] or rgb.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"RGB/depth/mask resolution mismatch: rgb={rgb.shape}, depth={depth.shape}, mask={mask.shape}")

        intrinsics = load_intrinsics(intrinsics_path)
        T_world_camera, pose_warnings = load_camera_pose(camera_pose_path)
        warnings.extend(pose_warnings)
        points_camera = masked_depth_to_pointcloud(
            depth,
            mask,
            intrinsics,
            max_points=int(self.config.debug.max_debug_points),
        )
        estimate = estimate_axis_aligned_object(points_camera, scale_factor=float(self.config.object.scale_factor))
        if self.config.object.collision_type.lower() != "box":
            raise RuntimeError("MVP only supports object.collision_type: box")
        min_size = np.asarray(self.config.object.min_size, dtype=np.float64)
        if min_size.shape != (3,):
            raise RuntimeError("object.min_size must contain three values")
        size_camera = np.maximum(estimate.size_camera, min_size)
        T_world_object = T_world_camera @ estimate.T_camera_object
        points_world = transform_points(T_world_camera, points_camera)
        size_world = size_camera.copy()

        if self.config.debug.save_pointcloud:
            generated.append(write_pointcloud_ply(debug_dir / "pointcloud_debug.ply", points_camera))
        if self.config.debug.save_overlay:
            generated.append(save_mask_overlay(debug_dir / "projected_overlay.png", rgb, mask))
            generated.append(save_mask_check(debug_dir / "mask_check.png", mask))

        recon = self._reconstructor().reconstruct(
            visual_dir=visual_dir,
            center_camera=estimate.center_camera,
            size_camera=size_camera,
            points_camera=points_camera,
            save_visual_mesh=bool(self.config.reconstruction.save_visual_mesh),
            save_gaussian=bool(self.config.reconstruction.save_gaussian),
        )
        if recon.mesh_path is not None:
            generated.append(recon.mesh_path)
        if recon.gaussian_path is not None:
            generated.append(recon.gaussian_path)

        collision_meta = generate_box_collision(
            collision_dir=collision_dir,
            object_name=self.config.object.name,
            center_camera=estimate.center_camera,
            size_camera=size_camera,
            size_world=size_world,
        )
        generated.extend([collision_dir / "collision_metadata.json", collision_dir / "collision_mesh.obj"])
        half_size = np.asarray(collision_meta["mujoco_half_size"], dtype=np.float64)
        pose = pose_dict_from_matrix(T_world_object)

        collision_box_xml = write_text(
            collision_dir / "collision_box.xml",
            f'<geom name="{self.config.object.name}_collision" type="box" size="'
            + " ".join(f"{float(v):.8f}" for v in half_size)
            + '"/>\n',
        )
        generated.append(collision_box_xml)

        if self.config.simulation.export_mujoco_xml:
            generated.append(
                write_mujoco_box_xml(
                    sim_dir / "mujoco_object.xml",
                    object_name=self.config.object.name,
                    position=np.asarray(pose["position"], dtype=np.float64),
                    quat_wxyz=np.asarray(pose["quaternion_wxyz"], dtype=np.float64),
                    half_size=half_size,
                    mass=float(self.config.object.mass),
                    friction=list(self.config.object.friction),
                    rgba=list(self.config.object.rgba),
                )
            )
        if self.config.simulation.export_gs_config:
            generated.append(
                write_gs_asset_config(
                    sim_dir / "gs_asset_config.yaml",
                    object_name=self.config.object.name,
                    gaussian_path=str(recon.gaussian_path.relative_to(output_dir)) if recon.gaussian_path else None,
                    mesh_path=str(recon.mesh_path.relative_to(output_dir)) if recon.mesh_path else None,
                    T_world_object=T_world_object,
                )
            )
        generated.append(dump_mapping(sim_dir / "object_pose_world.yaml", pose))

        result_path = output_dir / "real2sim_result.json"
        report_path = debug_dir / "real2sim_report.md"
        generated_for_summary = list(generated)
        generated_for_summary.append(result_path)
        if self.config.debug.save_report:
            generated_for_summary.append(report_path)

        result = {
            "mask_pixels": int(mask.sum()),
            "center_camera": [float(v) for v in estimate.center_camera],
            "size_camera": [float(v) for v in size_camera],
            "center_world": [float(v) for v in T_world_object[:3, 3]],
            "size_world": [float(v) for v in size_world],
            "T_world_object": [[float(v) for v in row] for row in T_world_object],
            "points_camera": int(points_camera.shape[0]),
            "points_world": int(points_world.shape[0]),
            "scene_binding": scene_binding,
            "warnings": warnings,
            "generated_files": [str(path.relative_to(output_dir)) for path in generated_for_summary],
        }
        if self.config.debug.save_report:
            write_report(
                report_path,
                input_files={
                    "rgb": str(rgb_path),
                    "depth": str(depth_path),
                    "mask": str(mask_path),
                    "intrinsics": str(intrinsics_path),
                    "camera_pose": str(camera_pose_path) if camera_pose_path.exists() else "identity",
                },
                mask_pixels=int(mask.sum()),
                center_camera=result["center_camera"],
                size_camera=result["size_camera"],
                center_world=result["center_world"],
                size_world=result["size_world"],
                camera_pose_used="file" if camera_pose_path.exists() else "identity",
                generated_files=[str(path.relative_to(output_dir)) for path in generated_for_summary],
                warnings=warnings,
            )
        write_json(result_path, result)
        return result
