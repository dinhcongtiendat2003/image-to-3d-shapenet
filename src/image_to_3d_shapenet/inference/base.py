"""Unified production interface for pretrained single-image-to-3D backends."""

from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from image_to_3d_shapenet.config import InferenceConfig
from image_to_3d_shapenet.utils import peak_vram_gb, reset_peak_vram

from .preprocess import PreprocessedImage, preprocess_image


class BackendUnavailableError(RuntimeError):
    pass


class BackendGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferenceOptions:
    geometry_only: bool = True
    seed: int = 42


@dataclass(frozen=True)
class GenerationResult:
    mesh: trimesh.Trimesh | trimesh.Scene
    backend_name: str
    inference_time_seconds: float
    peak_vram_gb: float
    vertex_count: int
    face_count: int
    texture_status: str
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.vertex_count <= 0 or self.face_count <= 0:
            raise BackendGenerationError("backend returned an empty mesh")
        geometries = (
            list(self.mesh.geometry.values())
            if isinstance(self.mesh, trimesh.Scene)
            else [self.mesh]
        )
        if not all(np.isfinite(np.asarray(item.vertices)).all() for item in geometries):
            raise BackendGenerationError("backend returned NaN or Inf vertices")


def coerce_trimesh(value: Any) -> trimesh.Trimesh | trimesh.Scene:
    if isinstance(value, (trimesh.Trimesh, trimesh.Scene)):
        return value
    if isinstance(value, (list, tuple)) and value:
        return coerce_trimesh(value[0])
    vertices = getattr(value, "vertices", None)
    faces = getattr(value, "faces", None)
    if vertices is None:
        vertices = getattr(value, "v_pos", None)
    if faces is None:
        faces = getattr(value, "t_pos_idx", None)
    if vertices is None or faces is None:
        raise BackendGenerationError(f"unsupported mesh result: {type(value).__name__}")
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def strip_texture(mesh: trimesh.Trimesh | trimesh.Scene) -> None:
    geometries = list(mesh.geometry.values()) if isinstance(mesh, trimesh.Scene) else [mesh]
    for geometry in geometries:
        geometry.visual = trimesh.visual.ColorVisuals(mesh=geometry)


def mesh_counts(mesh: trimesh.Trimesh | trimesh.Scene) -> tuple[int, int]:
    geometries = list(mesh.geometry.values()) if isinstance(mesh, trimesh.Scene) else [mesh]
    return (
        sum(len(item.vertices) for item in geometries),
        sum(len(item.faces) for item in geometries),
    )


class BaseImageTo3DBackend(ABC):
    name: str

    def __init__(self, config: InferenceConfig, device: str) -> None:
        self.config = config
        self.device = device

    @property
    def is_loaded(self) -> bool:
        return hasattr(self, "_model") and self._model is not None

    @abstractmethod
    def load(self) -> None:
        pass

    def preprocess(self, image: str | Path | Image.Image) -> PreprocessedImage:
        return preprocess_image(
            image,
            size=self.config.input_size,
            foreground_ratio=self.config.foreground_ratio,
            remove_background=self.config.remove_background,
        )

    @abstractmethod
    def generate(
        self,
        image: PreprocessedImage | Image.Image,
        options: InferenceOptions,
    ) -> GenerationResult:
        pass

    def unload(self) -> None:
        if hasattr(self, "_model"):
            model = self._model
            del self._model
            del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, OSError):
            pass

    def get_memory_usage(self) -> dict[str, float]:
        return {"peak_vram_gb": peak_vram_gb()}

    def get_metadata(self) -> dict[str, Any]:
        return {"name": self.name, "device": self.device, "loaded": self.is_loaded}

    def _start_timer(self) -> float:
        reset_peak_vram()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (ImportError, OSError):
            pass
        return time.perf_counter()

    def _finish_result(
        self,
        started: float,
        mesh_value: Any,
        *,
        options: InferenceOptions,
        texture_status: str,
        warnings: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (ImportError, OSError):
            pass
        mesh = coerce_trimesh(mesh_value)
        if options.geometry_only:
            strip_texture(mesh)
            texture_status = "none"
        vertices, faces = mesh_counts(mesh)
        result = GenerationResult(
            mesh=mesh,
            backend_name=self.name,
            inference_time_seconds=time.perf_counter() - started,
            peak_vram_gb=peak_vram_gb(),
            vertex_count=vertices,
            face_count=faces,
            texture_status=texture_status,
            warnings=warnings,
            metadata=metadata or {},
        )
        result.validate()
        return result

