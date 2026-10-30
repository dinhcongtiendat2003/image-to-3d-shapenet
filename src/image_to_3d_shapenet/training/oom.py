"""Ordered Colab-safe OOM degradation policy."""

from __future__ import annotations

from dataclasses import dataclass
import gc

import torch

from image_to_3d_shapenet.config import TrainingConfig


@dataclass
class ResourceState:
    surface_samples: int
    max_vertices: int
    image_resolution: int


def clear_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


class OOMFallbackManager:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.state = ResourceState(
            surface_samples=config.surface_samples,
            max_vertices=config.max_vertices,
            image_resolution=config.image_resolution,
        )
        self.actions: list[str] = []
        self._cache_cleared = False

    def reduce(self) -> str:
        if not self._cache_cleared:
            clear_device_cache()
            self._cache_cleared = True
            action = "cleared CUDA/Python cache"
        elif self.state.surface_samples > self.config.minimum_surface_samples:
            self.state.surface_samples = max(
                self.config.minimum_surface_samples, self.state.surface_samples // 2
            )
            action = f"reduced surface samples to {self.state.surface_samples}"
        elif self.state.max_vertices > self.config.minimum_vertices:
            self.state.max_vertices = max(
                self.config.minimum_vertices, self.state.max_vertices // 2
            )
            action = f"reduced max vertices to {self.state.max_vertices}"
        elif self.state.image_resolution > self.config.minimum_image_resolution:
            self.state.image_resolution = max(
                self.config.minimum_image_resolution, self.state.image_resolution // 2
            )
            action = f"reduced image resolution to {self.state.image_resolution}"
        else:
            raise RuntimeError(
                "OOM persists after cache clear, surface-sample, vertex, and image-resolution fallbacks"
            )
        self.actions.append(action)
        return action
