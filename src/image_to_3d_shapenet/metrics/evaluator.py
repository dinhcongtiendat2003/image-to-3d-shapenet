"""Sequential, resumable ShapeNet evaluation and ablation runner."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import trimesh

from image_to_3d_shapenet.config import ProjectConfig, RefinementConfig
from image_to_3d_shapenet.dataset import RenderedRefinementDataset
from image_to_3d_shapenet.mesh import compute_mesh_geometry
from image_to_3d_shapenet.models import ImageConditionedMeshRefiner
from image_to_3d_shapenet.rendering.camera import CameraSpec
from image_to_3d_shapenet.rendering.software import SoftwareRasterizerBackend
from image_to_3d_shapenet.utils import ProjectPaths, peak_vram_gb, reset_peak_vram

from .mesh_metrics import compute_mesh_metrics
from .reporting import write_incremental_reports


def _mesh(vertices: torch.Tensor, faces: torch.Tensor) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=vertices.detach().float().cpu().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    )


def _camera(metadata: dict[str, Any]) -> CameraSpec:
    return CameraSpec(
        width=int(metadata["width"]), height=int(metadata["height"]),
        azimuth_degrees=float(metadata["azimuth_degrees"]),
        elevation_degrees=float(metadata["elevation_degrees"]),
        distance=float(metadata["distance"]), near=float(metadata["near"]),
        far=float(metadata["far"]),
        intrinsics=np.asarray(metadata["intrinsics"], dtype=np.float64),
        world_to_camera=np.asarray(metadata["extrinsics_world_to_camera"], dtype=np.float64),
        camera_to_world=np.asarray(metadata["camera_to_world"], dtype=np.float64),
        projection=np.asarray(metadata["projection_matrix"], dtype=np.float64),
    )


def _render(mesh: trimesh.Trimesh, camera: CameraSpec) -> tuple[np.ndarray, np.ndarray]:
    renderer = SoftwareRasterizerBackend()
    result = renderer.render(
        mesh, camera, background_rgb=(245, 245, 245), transparent_background=False,
        ambient_light=0.35, directional_light=2.0,
    )
    return result.rgba[..., :3], result.mask


def _sharp_preview(mesh: trimesh.Trimesh, size: int) -> Image.Image:
    vertices = torch.tensor(np.asarray(mesh.vertices).copy(), dtype=torch.float32).unsqueeze(0)
    faces = torch.tensor(np.asarray(mesh.faces).copy(), dtype=torch.long)
    geometry = compute_mesh_geometry(vertices, faces)
    points = np.asarray(mesh.vertices)[:, :2]
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    points = (points - minimum) / np.maximum(maximum - minimum, 1e-8)
    points = points * (size * 0.8) + size * 0.1
    points[:, 1] = size - points[:, 1]
    canvas = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(canvas)
    edges = geometry.edges.numpy()
    sharp = geometry.sharp_edges[0].numpy()
    for edge, is_sharp in zip(edges, sharp):
        draw.line(
            (tuple(points[edge[0]]), tuple(points[edge[1]])),
            fill=(220, 35, 35) if is_sharp else (70, 100, 160),
            width=3 if is_sharp else 1,
        )
    return canvas


def _comparison(
    output: Path,
    sample: dict[str, Any],
    ground_truth: trimesh.Trimesh,
    base: trimesh.Trimesh,
    refined: trimesh.Trimesh,
    configuration: str,
) -> None:
    camera = _camera(sample["camera_metadata"])
    size = camera.width
    input_image = Image.fromarray(
        np.clip(sample["rgb"].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
    ).resize((size, size))
    panels = [input_image]
    for mesh in (ground_truth, base, refined):
        rgb, _ = _render(mesh, camera)
        panels.append(Image.fromarray(rgb))
    panels.append(_sharp_preview(refined, size))
    canvas = Image.new("RGB", (size * 5, size + 22), "white")
    labels = ("input", "ground truth", "base mesh", "refined mesh", "sharp edges")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(panel.resize((size, size)), (index * size, 22))
        draw.text((index * size + 4, 4), label, fill="black")
    destination = output / "comparisons" / f"{sample['sample_id']}__{configuration}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _checkpoint_model(path: Path, device: torch.device) -> ImageConditionedMeshRefiner:
    state = torch.load(path, map_location="cpu", weights_only=False)
    refinement = RefinementConfig(**state["config"]["refinement"])
    model = ImageConditionedMeshRefiner(refinement)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model


def _existing(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics_path = output / "metrics.json"
    failures_path = output / "failures.json"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        rows = json.loads(metrics_path.read_text(encoding="utf-8")).get("per_sample", [])
    except (OSError, json.JSONDecodeError):
        pass
    try:
        failures = json.loads(failures_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return rows, failures


def evaluate(
    config: ProjectConfig,
    project_root: str | Path,
    *,
    checkpoint: str | Path | None = None,
    base_refiner_checkpoint: str | Path | None = None,
    no_edge_checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths = ProjectPaths.from_config(root, config.paths).create()
    output = Path(output_dir).expanduser().resolve() if output_dir else (
        paths.outputs / config.evaluation.output_subdir
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        dataset = RenderedRefinementDataset(
            root, config, split="test", maximum=config.evaluation.max_test_samples,
            image_resolution=config.training.image_resolution,
        )
    except RuntimeError:
        if not config.runtime.quick_test:
            raise
        dataset = RenderedRefinementDataset(
            root, config, maximum=config.evaluation.max_test_samples,
            image_resolution=config.training.image_resolution,
        )
    rows, failures = _existing(output)
    completed = {
        (row["sample_id"], row["configuration"], row["backend"]) for row in rows
    }
    metadata = {
        "category_id": config.dataset.synset_id,
        "category_name": config.dataset.category_name,
        "normalization": "ground-truth transform applied to prediction and target",
        "surface_samples": config.evaluation.surface_samples,
        "edge_samples": config.evaluation.edge_samples,
        "sequential": True,
    }
    checkpoint_by_configuration = {
        "base": None,
        "base_refiner": Path(base_refiner_checkpoint or checkpoint) if (base_refiner_checkpoint or checkpoint) else None,
        "refiner_no_edge": Path(no_edge_checkpoint) if no_edge_checkpoint else None,
        "refiner_full": Path(checkpoint) if checkpoint else None,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparison_counts: dict[str, int] = {}
    for configuration in config.evaluation.configurations:
        checkpoint_path = checkpoint_by_configuration[configuration]
        model = None
        if configuration != "base":
            if checkpoint_path is None or not checkpoint_path.is_file():
                failures.append(
                    {
                        "sample_id": "__configuration__",
                        "configuration": configuration,
                        "error": "MissingCheckpoint",
                    }
                )
                write_incremental_reports(output, rows, failures, metadata)
                continue
            model = _checkpoint_model(checkpoint_path, device)
        try:
            for index in range(len(dataset)):
                try:
                    sample = dataset[index]
                except Exception as error:
                    failures.append(
                        {
                            "sample_id": f"index_{index}",
                            "configuration": configuration,
                            "backend": "unknown",
                            "error": type(error).__name__,
                            "message": str(error)[:300],
                        }
                    )
                    write_incremental_reports(output, rows, failures, metadata)
                    continue
                actual_backend = sample["base_backend"]
                filters = config.evaluation.backend_filters
                if "cached" not in filters and actual_backend not in filters:
                    continue
                key = (sample["sample_id"], configuration, actual_backend)
                if key in completed:
                    continue
                try:
                    base = _mesh(sample["base_vertices"], sample["base_faces"])
                    target = _mesh(sample["target_vertices"], sample["target_faces"])
                    inference_time = sample["base_inference_time_seconds"]
                    peak_vram = sample["base_peak_vram_gb"]
                    if model is None:
                        prediction = base.copy()
                    else:
                        reset_peak_vram()
                        start = time.perf_counter()
                        with torch.no_grad(), torch.autocast(
                            device_type=device.type, dtype=torch.float16,
                            enabled=config.runtime.mixed_precision and device.type == "cuda",
                        ):
                            result = model(
                                sample["rgb"].unsqueeze(0).to(device),
                                sample["base_vertices"].unsqueeze(0).to(device),
                                sample["base_faces"].to(device),
                                sample["intrinsics"].unsqueeze(0).to(device),
                                sample["extrinsics"].unsqueeze(0).to(device),
                            )
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        inference_time += time.perf_counter() - start
                        peak_vram = max(peak_vram, peak_vram_gb())
                        prediction = _mesh(result.refined_vertices[0], sample["base_faces"])
                    _, predicted_mask = _render(prediction, _camera(sample["camera_metadata"]))
                    target_mask = sample["mask"][0].numpy()
                    if predicted_mask.shape != target_mask.shape:
                        predicted_mask = np.asarray(
                            Image.fromarray(predicted_mask).resize(
                                (target_mask.shape[1], target_mask.shape[0]),
                                Image.Resampling.NEAREST,
                            )
                        )
                    metrics = compute_mesh_metrics(
                        prediction, target,
                        surface_samples=config.evaluation.surface_samples,
                        edge_samples=config.evaluation.edge_samples,
                        sharp_edge_threshold_degrees=config.evaluation.sharp_edge_threshold_degrees,
                        fscore_thresholds=config.evaluation.fscore_thresholds,
                        predicted_silhouette=predicted_mask,
                        target_silhouette=target_mask,
                        inference_time_seconds=inference_time,
                        peak_vram_gb=peak_vram,
                    )
                    row = {
                        "sample_id": sample["sample_id"],
                        "model_id": sample["model_id"],
                        "view_id": sample["view_id"],
                        "category_id": config.dataset.synset_id,
                        "category_name": config.dataset.category_name,
                        "configuration": configuration,
                        "backend": actual_backend,
                        "model_version": sample["base_model_version"],
                        **metrics.to_dict(),
                    }
                    rows.append(row)
                    completed.add(key)
                    comparison_count = comparison_counts.get(configuration, 0)
                    if comparison_count < config.evaluation.comparison_samples:
                        _comparison(output, sample, target, base, prediction, configuration)
                        comparison_counts[configuration] = comparison_count + 1
                    write_incremental_reports(output, rows, failures, metadata)
                except Exception as error:
                    failures.append(
                        {
                            "sample_id": sample.get("sample_id", f"index_{index}"),
                            "configuration": configuration,
                            "backend": sample.get("base_backend", "unknown"),
                            "error": type(error).__name__,
                            "message": str(error)[:300],
                        }
                    )
                    write_incremental_reports(output, rows, failures, metadata)
        finally:
            if model is not None:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    summary = write_incremental_reports(output, rows, failures, metadata)
    return {
        "output_dir": str(output),
        "successful_samples": len(rows),
        "failures": len(failures),
        "summary_rows": len(summary),
        "device": str(device),
    }
