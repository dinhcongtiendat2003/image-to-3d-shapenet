"""Sequential, interruption-safe rendering for prepared ShapeNet meshes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image_to_3d_shapenet.config import ProjectConfig
from image_to_3d_shapenet.dataset.meshes import load_mesh, model_id_from_path
from image_to_3d_shapenet.utils import ProjectPaths

from .camera import camera_for_mesh, orbit_angles
from .factory import create_renderer


REQUIRED_SAMPLE_FILES = (
    "rgb.png",
    "rgba.png",
    "mask.png",
    "depth.npy",
    "normal.png",
    "camera.json",
    "sample.json",
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(array).save(temporary, format="PNG")
    os.replace(temporary, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _split_lookup(split_path: Path) -> dict[str, str]:
    payload = _load_json(split_path, {})
    lookup: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for model_id in payload.get(split, []):
            lookup[str(model_id)] = split
    return lookup


def _background_for_sample(
    config: ProjectConfig,
    model_id: str,
    view_index: int,
    split: str,
) -> tuple[int, int, int]:
    if not (config.render.random_background_train and split == "train"):
        return config.render.background_rgb
    digest = hashlib.sha256(
        f"{config.runtime.seed}:{model_id}:{view_index}".encode("utf-8")
    ).digest()
    return tuple(48 + int(value) % 192 for value in digest[:3])


def _sample_is_complete(sample_dir: Path) -> bool:
    if not all((sample_dir / name).is_file() for name in REQUIRED_SAMPLE_FILES):
        return False
    metadata = _load_json(sample_dir / "sample.json", {})
    return metadata.get("status") == "complete"


def _save_result(
    sample_dir: Path,
    result: Any,
    camera_payload: dict[str, Any],
    sample_payload: dict[str, Any],
) -> None:
    rgb = np.asarray(result.rgba[..., :3], dtype=np.uint8)
    normal_png = np.zeros_like(rgb)
    foreground = result.mask > 0
    encoded = np.clip((result.normals * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    normal_png[foreground] = encoded[foreground]
    _atomic_png(sample_dir / "rgb.png", rgb)
    _atomic_png(sample_dir / "rgba.png", np.asarray(result.rgba, dtype=np.uint8))
    _atomic_png(sample_dir / "mask.png", np.asarray(result.mask, dtype=np.uint8))
    _atomic_npy(sample_dir / "depth.npy", np.asarray(result.depth, dtype=np.float32))
    _atomic_png(sample_dir / "normal.png", normal_png)
    _atomic_json(sample_dir / "camera.json", camera_payload)
    _atomic_json(sample_dir / "sample.json", sample_payload)


def render_dataset(
    config: ProjectConfig,
    project_root: str | Path,
    *,
    backend_name: str | None = None,
) -> dict[str, Any]:
    """Render prepared meshes one at a time and checkpoint every view."""
    paths = ProjectPaths.from_config(project_root, config.paths).create()
    processed_root = paths.data / "processed" / config.dataset.synset_id
    meshes = sorted(processed_root.glob("*/model_normalized.obj"))
    if not meshes and config.runtime.quick_test:
        from image_to_3d_shapenet.dataset import prepare_dataset

        prepare_dataset(config, project_root, synthetic=True)
        meshes = sorted(processed_root.glob("*/model_normalized.obj"))
    if not meshes:
        raise RuntimeError(
            "Không tìm thấy mesh đã chuẩn hóa. Chạy scripts/prepare_dataset.py trước."
        )
    if config.render.max_render_meshes is not None:
        meshes = meshes[: config.render.max_render_meshes]

    render_root = paths.outputs / config.render.output_subdir
    render_root.mkdir(parents=True, exist_ok=True)
    manifest_path = render_root / "manifest.json"
    manifest = _load_json(
        manifest_path,
        {
            "version": 1,
            "category_id": config.dataset.synset_id,
            "category_name": config.dataset.category_name,
            "image_size": config.runtime.image_size,
            "views_per_mesh": config.dataset.views_per_mesh,
            "samples": {},
        },
    )
    if not isinstance(manifest.get("samples"), dict):
        raise ValueError("render manifest samples must be a mapping")
    manifest.update(
        {
            "version": 1,
            "category_id": config.dataset.synset_id,
            "category_name": config.dataset.category_name,
            "image_size": config.runtime.image_size,
            "views_per_mesh": config.dataset.views_per_mesh,
        }
    )
    split_by_model = _split_lookup(paths.outputs / "splits.json")
    renderer = create_renderer(
        backend_name or config.render.backend,
        pyrender_platform=config.render.pyrender_platform,
    )
    rendered = 0
    skipped = 0
    failed: list[dict[str, str]] = []
    angles = orbit_angles(
        config.dataset.views_per_mesh,
        config.render.elevation_min_degrees,
        config.render.elevation_max_degrees,
    )
    try:
        for mesh_path in meshes:
            model_id = model_id_from_path(mesh_path)
            split = split_by_model.get(model_id, "unassigned")
            mesh = load_mesh(mesh_path)
            for view_index, (azimuth, elevation) in enumerate(angles):
                sample_id = f"{model_id}_view_{view_index:03d}"
                sample_dir = render_root / model_id / f"view_{view_index:03d}"
                if _sample_is_complete(sample_dir):
                    skipped += 1
                    if sample_id not in manifest["samples"]:
                        manifest["samples"][sample_id] = _load_json(
                            sample_dir / "sample.json", {}
                        )
                        _atomic_json(manifest_path, manifest)
                    continue
                try:
                    camera = camera_for_mesh(
                        mesh,
                        width=config.runtime.image_size,
                        height=config.runtime.image_size,
                        azimuth_degrees=azimuth,
                        elevation_degrees=elevation,
                        fov_degrees=config.render.fov_degrees,
                        framing_margin=config.render.framing_margin,
                    )
                    background = _background_for_sample(
                        config, model_id, view_index, split
                    )
                    result = renderer.render(
                        mesh,
                        camera,
                        background_rgb=background,
                        transparent_background=config.render.transparent_background,
                        ambient_light=config.render.ambient_light,
                        directional_light=config.render.directional_light,
                    )
                    result.validate(config.runtime.image_size, config.runtime.image_size)
                    files = {name: name for name in REQUIRED_SAMPLE_FILES}
                    sample = {
                        "status": "complete",
                        "sample_id": sample_id,
                        "model_id": model_id,
                        "category_id": config.dataset.synset_id,
                        "category_name": config.dataset.category_name,
                        "split": split,
                        "view_index": view_index,
                        "azimuth_degrees": float(azimuth),
                        "elevation_degrees": float(elevation),
                        "ground_truth_mesh": str(mesh_path.resolve()),
                        "backend": result.backend,
                        "normal_space": result.normal_space,
                        "background_rgb": list(background),
                        "warnings": list(result.warnings),
                        "files": files,
                    }
                    _save_result(sample_dir, result, camera.to_dict(), sample)
                    manifest["samples"][sample_id] = {
                        **sample,
                        "sample_directory": str(sample_dir.relative_to(render_root)),
                    }
                    _atomic_json(manifest_path, manifest)
                    rendered += 1
                except Exception as error:
                    failure = {"sample_id": sample_id, "error": type(error).__name__}
                    failed.append(failure)
                    manifest.setdefault("failures", []).append(failure)
                    _atomic_json(manifest_path, manifest)
    finally:
        renderer.close()

    return {
        "render_root": str(render_root),
        "manifest": str(manifest_path),
        "meshes": len(meshes),
        "views_per_mesh": config.dataset.views_per_mesh,
        "rendered": rendered,
        "skipped": skipped,
        "failed": len(failed),
    }

