from __future__ import annotations

from pathlib import Path


class SceneInserter:
    """Project-specific adapter placeholder for binding assets into a scene."""

    def insert(self, *, mujoco_xml: str | Path, gs_config: str | Path) -> None:
        raise NotImplementedError(
            "Scene insertion is project-specific. Include the generated MuJoCo XML body "
            "and GS asset config in your GS-Playground/MotrixSim scene loader."
        )
