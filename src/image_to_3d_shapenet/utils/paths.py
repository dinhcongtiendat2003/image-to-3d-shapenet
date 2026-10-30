"""Project-relative path management without machine-specific defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from image_to_3d_shapenet.config import PathsConfig


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    cache: Path
    outputs: Path

    @classmethod
    def from_config(cls, root: str | Path, config: PathsConfig) -> "ProjectPaths":
        project_root = Path(root).expanduser().resolve()

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()

        return cls(
            root=project_root,
            data=resolve(config.data_dir),
            cache=resolve(config.cache_dir),
            outputs=resolve(config.output_dir),
        )

    def create(self) -> "ProjectPaths":
        for path in (self.data, self.cache, self.outputs):
            path.mkdir(parents=True, exist_ok=True)
        return self

