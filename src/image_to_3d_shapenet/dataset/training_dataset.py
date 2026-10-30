"""Batch-1 loader for rendered ShapeNet samples and cached base meshes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset

from image_to_3d_shapenet.config import ProjectConfig
from image_to_3d_shapenet.dataset.meshes import load_mesh
from image_to_3d_shapenet.utils import ProjectPaths


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _image_tensor(path: Path, mode: str, resolution: int, *, nearest: bool = False) -> Tensor:
    with Image.open(path) as opened:
        image = opened.convert(mode)
        image = image.resize(
            (resolution, resolution),
            Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32).copy()
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def _mesh_tensors(path: Path) -> tuple[Tensor, Tensor]:
    mesh = load_mesh(path)
    vertices = torch.tensor(np.asarray(mesh.vertices).copy(), dtype=torch.float32)
    faces = torch.tensor(np.asarray(mesh.faces).copy(), dtype=torch.long)
    return vertices, faces


class RenderedRefinementDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        project_root: str | Path,
        config: ProjectConfig,
        *,
        split: str | None = None,
        maximum: int | None = None,
        image_resolution: int | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.config = config
        paths = ProjectPaths.from_config(self.project_root, config.paths).create()
        self.render_root = paths.outputs / config.render.output_subdir
        self.cache_root = paths.outputs / config.training.base_mesh_cache_subdir
        render_manifest = _json(self.render_root / "manifest.json")
        cache_manifest = _json(self.cache_root / "manifest.json")
        cache_samples = cache_manifest.get("samples", {})
        records = []
        for sample_id, render_entry in sorted(render_manifest.get("samples", {}).items()):
            cache_entry = cache_samples.get(sample_id)
            if render_entry.get("status") != "complete" or not cache_entry:
                continue
            if cache_entry.get("status") != "complete":
                continue
            if split is not None and render_entry.get("split") != split:
                continue
            records.append((sample_id, render_entry, cache_entry))
        if maximum is not None:
            records = records[:maximum]
        if not records:
            raise RuntimeError(f"no complete rendered/cache samples for split={split!r}")
        self.records = records
        self.image_resolution = image_resolution or config.training.image_resolution

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id, render_entry, cache_entry = self.records[index]
        sample_dir = self.render_root / render_entry["sample_directory"]
        camera = _json(sample_dir / "camera.json")
        original_width = int(camera["width"])
        original_height = int(camera["height"])
        resolution = self.image_resolution
        intrinsics = torch.tensor(camera["intrinsics"], dtype=torch.float32)
        intrinsics[0] *= resolution / original_width
        intrinsics[1] *= resolution / original_height
        extrinsics = torch.tensor(camera["extrinsics_world_to_camera"], dtype=torch.float32)

        depth = torch.from_numpy(
            np.load(sample_dir / "depth.npy", allow_pickle=False).astype(np.float32)
        )[None, None]
        depth = F.interpolate(depth, size=(resolution, resolution), mode="nearest")[0]
        base_path = self.cache_root / cache_entry["mesh"]
        ground_truth_path = Path(render_entry["ground_truth_mesh"])
        if not ground_truth_path.is_file():
            ground_truth_path = (
                self.project_root / config.paths.data_dir / "processed"
                / config.dataset.synset_id / render_entry["model_id"] / "model_normalized.obj"
            )
        base_vertices, base_faces = _mesh_tensors(base_path)
        target_vertices, target_faces = _mesh_tensors(ground_truth_path)
        return {
            "rgb": _image_tensor(sample_dir / "rgb.png", "RGB", resolution),
            "mask": _image_tensor(sample_dir / "mask.png", "L", resolution, nearest=True),
            "depth": depth,
            "normal": _image_tensor(sample_dir / "normal.png", "RGB", resolution) * 2.0 - 1.0,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "camera_metadata": camera,
            "base_vertices": base_vertices,
            "base_faces": base_faces,
            "target_vertices": target_vertices,
            "target_faces": target_faces,
            "ground_truth_mesh": str(ground_truth_path),
            "base_mesh_prediction": str(base_path),
            "model_id": str(render_entry["model_id"]),
            "view_id": int(render_entry["view_index"]),
            "sample_id": sample_id,
            "split": str(render_entry.get("split", "unassigned")),
            "base_backend": str(cache_entry["backend"]),
            "base_model_version": str(cache_entry["model_version"]),
            "base_inference_time_seconds": float(cache_entry.get("inference_time_seconds", 0.0)),
            "base_peak_vram_gb": float(cache_entry.get("peak_vram_gb", 0.0)),
        }


def single_item_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("refinement topology loader requires batch_size=1")
    return items[0]
