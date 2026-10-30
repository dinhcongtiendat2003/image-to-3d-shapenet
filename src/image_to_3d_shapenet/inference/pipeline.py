"""Production inference orchestration and OBJ/GLB export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import trimesh

from image_to_3d_shapenet.config import InferenceConfig

from .auto import AutoBackend
from .base import GenerationResult, InferenceOptions


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _export_mesh(mesh: trimesh.Trimesh | trimesh.Scene, output: Path) -> tuple[Path, Path]:
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    obj_path = output / "base_mesh.obj"
    glb_path = output / "base_mesh.glb"
    obj_payload = mesh.export(file_type="obj")
    if isinstance(obj_payload, bytes):
        _atomic_bytes(obj_path, obj_payload)
    else:
        _atomic_text(obj_path, str(obj_payload))
    glb_payload = mesh.export(file_type="glb")
    if not isinstance(glb_payload, bytes):
        raise RuntimeError("GLB exporter did not return binary data")
    _atomic_bytes(glb_path, glb_payload)
    return obj_path, glb_path


def run_inference(
    image: str | Path,
    output_dir: str | Path,
    config: InferenceConfig,
    *,
    requested_backend: str | None = None,
    backend: AutoBackend | None = None,
) -> dict[str, Any]:
    """Run exactly one real pretrained backend at a time and export base geometry."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    auto = backend or AutoBackend(config, requested=requested_backend)
    try:
        prepared = auto.preprocess(image)
        prepared_path = output / "input_preprocessed.png"
        prepared.image.save(prepared_path, format="PNG")
        auto.load()
        result: GenerationResult = auto.generate(
            prepared,
            InferenceOptions(geometry_only=config.geometry_only, seed=config.seed),
        )
        obj_path, glb_path = _export_mesh(result.mesh, output)
        payload = {
            "backend": result.backend_name,
            "inference_time_seconds": result.inference_time_seconds,
            "peak_vram_gb": result.peak_vram_gb,
            "vertex_count": result.vertex_count,
            "face_count": result.face_count,
            "texture_status": result.texture_status,
            "geometry_only": config.geometry_only,
            "warnings": list(result.warnings),
            "backend_metadata": result.metadata,
            "preprocessing": {
                "original_size": list(prepared.original_size),
                "foreground_bbox": list(prepared.foreground_bbox),
                "scale": prepared.scale,
                "background_removed": prepared.background_removed,
            },
            "files": {
                "obj": obj_path.name,
                "glb": glb_path.name,
                "preprocessed_image": prepared_path.name,
            },
        }
        _atomic_text(output / "metadata.json", json.dumps(payload, indent=2))
        return payload
    finally:
        auto.unload()

