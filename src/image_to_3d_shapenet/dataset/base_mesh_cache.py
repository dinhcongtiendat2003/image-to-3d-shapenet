"""Resumable base-mesh prediction cache; foundation models run only on misses."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from image_to_3d_shapenet.config import ProjectConfig
from image_to_3d_shapenet.dataset.meshes import load_mesh
from image_to_3d_shapenet.inference import run_inference
from image_to_3d_shapenet.utils import ProjectPaths


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _configured_model_version(config: ProjectConfig, backend: str) -> str:
    versions = {
        "triposr": config.inference.triposr_model_id,
        "stable-fast-3d": config.inference.stable_fast_3d_model_id,
        "hunyuan3d-2mini": f"{config.inference.hunyuan_model_id}/{config.inference.hunyuan_subfolder}",
        "auto": "auto-selection",
    }
    return versions[backend]


def _synthetic_base_mesh(source: Path, destination: Path, sample_id: str) -> None:
    mesh = load_mesh(source).copy()
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    scales = 0.88 + np.asarray(list(digest[:3]), dtype=np.float64) / 2550.0
    mesh.vertices = np.asarray(mesh.vertices) * scales
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(destination, file_type="obj")


def generate_base_mesh_cache(
    config: ProjectConfig,
    project_root: str | Path,
    *,
    backend: str | None = None,
    maximum: int | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths = ProjectPaths.from_config(root, config.paths).create()
    # Quick mode creates/repairs its tiny rendered dataset before caching.
    if config.runtime.quick_test:
        from image_to_3d_shapenet.rendering import render_dataset

        render_dataset(config, root)
    render_root = paths.outputs / config.render.output_subdir
    render_manifest = _read(render_root / "manifest.json", {})
    samples = sorted(render_manifest.get("samples", {}).items())
    if maximum is not None:
        samples = samples[:maximum]
    if not samples:
        raise RuntimeError("render manifest has no samples; run render_dataset first")

    requested = backend or config.inference.backend
    cache_root = paths.outputs / config.training.base_mesh_cache_subdir
    manifest_path = cache_root / "manifest.json"
    manifest = _read(
        manifest_path,
        {
            "version": 1,
            "requested_backend": requested,
            "configured_model_version": _configured_model_version(config, requested),
            "synthetic_only": bool(config.runtime.quick_test),
            "samples": {},
        },
    )
    expected_version = _configured_model_version(config, requested)
    if (
        manifest.get("requested_backend") != requested
        or manifest.get("configured_model_version") != expected_version
        or manifest.get("synthetic_only") != bool(config.runtime.quick_test)
    ):
        manifest = {
            "version": 1,
            "requested_backend": requested,
            "configured_model_version": expected_version,
            "synthetic_only": bool(config.runtime.quick_test),
            "samples": {},
        }
    manifest.setdefault("samples", {})
    generated = 0
    skipped = 0
    failures = []
    for sample_id, render_entry in samples:
        destination_dir = cache_root / sample_id
        mesh_path = destination_dir / "base_mesh.obj"
        existing = manifest["samples"].get(sample_id, {})
        if existing.get("status") == "complete" and mesh_path.is_file():
            try:
                loaded = load_mesh(mesh_path)
                if len(loaded.vertices) and len(loaded.faces):
                    skipped += 1
                    continue
            except Exception:
                pass
        try:
            sample_dir = render_root / render_entry["sample_directory"]
            if config.runtime.quick_test:
                _synthetic_base_mesh(
                    Path(render_entry["ground_truth_mesh"]), mesh_path, sample_id
                )
                actual_backend = "synthetic-perturbation"
                model_version = "v1-test-only"
                inference_time = 0.0
                peak_vram = 0.0
            else:
                result = run_inference(
                    sample_dir / "rgba.png",
                    destination_dir,
                    config.inference,
                    requested_backend=requested,
                )
                generated_obj = destination_dir / result["files"]["obj"]
                if generated_obj != mesh_path:
                    os.replace(generated_obj, mesh_path)
                actual_backend = str(result["backend"])
                model_version = str(
                    result.get("backend_metadata", {}).get("model_id", "unknown")
                )
                inference_time = float(result.get("inference_time_seconds", 0.0))
                peak_vram = float(result.get("peak_vram_gb", 0.0))
            manifest["samples"][sample_id] = {
                "status": "complete",
                "mesh": str(mesh_path.relative_to(cache_root)).replace("\\", "/"),
                "backend": actual_backend,
                "model_version": model_version,
                "model_id": render_entry["model_id"],
                "view_id": render_entry["view_index"],
                "source_image": str((sample_dir / "rgba.png").resolve()),
                "inference_time_seconds": inference_time,
                "peak_vram_gb": peak_vram,
            }
            generated += 1
        except Exception as error:
            failures.append({"sample_id": sample_id, "error": type(error).__name__})
            manifest["samples"][sample_id] = {
                "status": "failed", "error": type(error).__name__
            }
        _atomic_json(manifest_path, manifest)
    return {
        "manifest": str(manifest_path),
        "generated": generated,
        "skipped": skipped,
        "failed": len(failures),
        "samples": len(samples),
    }
