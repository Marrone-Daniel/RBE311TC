from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Sam3DGSRunResult:
    gaussian_path: Path | None
    mesh_path: Path | None
    used_backend: bool
    message: str


class Sam3DGSAdapter:
    """Thin adapter for the local, script-style `sam3d_gs` checkout.

    The cloned project is not a normal importable Python package. This wrapper
    prepares the folder layout expected by its scripts and calls
    `pipeline/process.py` when the required submodule files are present.
    """

    def __init__(self, repo_root: str | Path = "sam3d_gs") -> None:
        self.repo_root = Path(repo_root).resolve()
        self.project_root = self.repo_root / "Sam-3d-objects"
        self.process_script = self.repo_root / "pipeline" / "process.py"
        self.python = self.repo_root / ".venv" / "bin" / "python"

    def availability_error(self) -> str | None:
        if not self.repo_root.exists():
            return f"sam3d_gs repo not found: {self.repo_root}"
        if not self.process_script.exists():
            return f"sam3d_gs process script not found: {self.process_script}"
        if (not self.project_root.exists()) or (not any(self.project_root.iterdir())):
            return (
                f"{self.project_root} is empty. Run `git submodule update --init --recursive` "
                "inside sam3d_gs and install its SAM-3D-Objects dependencies."
            )
        checkpoint = self.project_root / "checkpoints" / "hf" / "pipeline.yaml"
        if not checkpoint.exists():
            return f"SAM-3D-Objects checkpoint config missing: {checkpoint}"
        return None

    def run_from_image_and_mask(
        self,
        *,
        image_path: str | Path,
        mask_path: str | Path,
        work_dir: str | Path,
        seed: int = 42,
    ) -> Sam3DGSRunResult:
        err = self.availability_error()
        if err is not None:
            return Sam3DGSRunResult(None, None, False, err)

        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        image_dst = work_dir / "input_image.png"
        mask_dir = work_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_dst = mask_dir / "target_object.png"
        shutil.copy2(image_path, image_dst)
        shutil.copy2(mask_path, mask_dst)
        image = cv2.imread(image_dst.as_posix(), cv2.IMREAD_COLOR)
        if image is None:
            return Sam3DGSRunResult(None, None, False, f"Cannot read copied image: {image_dst}")
        h, w = image.shape[:2]
        np.save(work_dir / "extrinsic.npy", np.eye(4, dtype=np.float32))
        np.save(work_dir / "intrinsic.npy", np.asarray([[525.0, 0.0, (w - 1) * 0.5], [0.0, 525.0, (h - 1) * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32))
        np.save(work_dir / "depth.npy", np.ones((h, w), dtype=np.float32))
        np.save(work_dir / "scale.npy", np.asarray(1.0, dtype=np.float32))

        py = self.python if self.python.exists() else Path(sys.executable)
        save_dir = self.project_root / "outputs" / "torch_save_pt"
        save_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            py.as_posix(),
            self.process_script.as_posix(),
            "--project-root",
            self.project_root.as_posix(),
            "--save-dir",
            save_dir.as_posix(),
            "--image-path",
            image_dst.as_posix(),
            "--seed",
            str(int(seed)),
        ]
        try:
            subprocess.run(cmd, cwd=self.repo_root.as_posix(), check=True)
        except subprocess.CalledProcessError as exc:
            return Sam3DGSRunResult(None, None, False, f"SAM3D-GS command failed: {exc}")

        assets = work_dir / "3d_assets"
        gaussian_candidates = sorted(assets.glob("*_resize.ply")) + sorted(assets.glob("*_gs_final.ply"))
        mesh_candidates = sorted(assets.glob("*_resize.obj")) + sorted(assets.glob("*_mesh_final.obj"))
        return Sam3DGSRunResult(
            gaussian_candidates[0] if gaussian_candidates else None,
            mesh_candidates[0] if mesh_candidates else None,
            True,
            "SAM3D-GS completed.",
        )
