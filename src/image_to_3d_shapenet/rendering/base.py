"""Backend-independent renderer contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import trimesh

from .camera import CameraSpec


@dataclass(frozen=True)
class RenderResult:
    rgba: np.ndarray
    depth: np.ndarray
    normals: np.ndarray
    mask: np.ndarray
    backend: str
    normal_space: str
    warnings: tuple[str, ...] = ()

    def validate(self, width: int, height: int) -> None:
        if self.rgba.shape != (height, width, 4):
            raise ValueError(f"invalid RGBA shape: {self.rgba.shape}")
        if self.depth.shape != (height, width):
            raise ValueError(f"invalid depth shape: {self.depth.shape}")
        if self.normals.shape != (height, width, 3):
            raise ValueError(f"invalid normal shape: {self.normals.shape}")
        if self.mask.shape != (height, width):
            raise ValueError(f"invalid mask shape: {self.mask.shape}")
        if not np.isfinite(self.depth).all() or not np.isfinite(self.normals).all():
            raise ValueError("render contains NaN or Inf")
        if not np.any(self.mask):
            raise ValueError("render mask is empty")


class RendererBackend(ABC):
    name: str

    @abstractmethod
    def render(
        self,
        mesh: trimesh.Trimesh,
        camera: CameraSpec,
        *,
        background_rgb: tuple[int, int, int],
        transparent_background: bool,
        ambient_light: float,
        directional_light: float,
    ) -> RenderResult:
        """Render RGBA, metric depth, normals and silhouette."""

    def close(self) -> None:
        """Release renderer resources."""

