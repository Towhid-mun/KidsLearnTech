"""Video generation backends."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .animatediff import AnimateDiffGenerator


class VideoGenerator:
    """Facade that selects a concrete video backend."""

    def __init__(self, media_dir: Path, backend: str | None = None) -> None:
        backend = (backend or "slides").lower()
        self.media_dir = media_dir
        self.backend_name = backend
        self._backend = None
        if backend == "animatediff":
            self._backend = AnimateDiffGenerator(media_dir=self.media_dir)

    def startup(self) -> None:
        if self._backend:
            self._backend.startup()

    def shutdown(self) -> None:
        if self._backend:
            self._backend.shutdown()

    @property
    def enabled(self) -> bool:
        return self._backend is not None and self._backend.available

    def generate(self, prompts: List[str], identifier: str):
        if not self.enabled:
            raise RuntimeError("Video backend is not enabled or failed to load.")
        return self._backend.generate(prompts, identifier)
