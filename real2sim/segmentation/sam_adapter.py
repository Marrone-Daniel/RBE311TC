from __future__ import annotations


class SamAdapter:
    """Placeholder for prompt/SAM-based mask generation."""

    def __init__(self, *_, **__) -> None:
        raise RuntimeError(
            "SAM mask generation is not bundled with this MVP. Provide mask.png "
            "directly, or implement SamAdapter with your chosen SAM/SAM3D-GS stack."
        )
