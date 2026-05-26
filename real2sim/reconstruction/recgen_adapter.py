from __future__ import annotations


class RecGenAdapter:
    def __init__(self, *_, **__) -> None:
        raise RuntimeError(
            "RecGen backend is not installed or configured. This MVP only ships the wrapper interface. "
            "Use reconstruction.backend: dummy, or provide a project-specific RecGenAdapter implementation."
        )
