"""Deterministic scientific mesh metrics with no custom CUDA extensions."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import numpy as np
import torch
import trimesh

from image_to_3d_shapenet.mesh import compute_mesh_geometry


class MetricInputError(ValueError):
    pass


@dataclass(frozen=True)
class MeshMetricResult:
    chamfer_distance: float
    fscore_1_percent: float
    fscore_2_percent: float
    normal_consistency: float
    silhouette_iou: float
    edge_chamfer_distance: float
    dihedral_angle_error_degrees: float
    vertex_count: int
    face_count: int
    inference_time_seconds: float
    peak_vram_gb: float
    watertight: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validated(mesh: trimesh.Trimesh, name: str) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh):
        raise MetricInputError(f"{name} must be a Trimesh")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise MetricInputError(f"{name} mesh is empty")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise MetricInputError(f"{name} vertices are malformed")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.isfinite(faces).all():
        raise MetricInputError(f"{name} faces are malformed")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise MetricInputError(f"{name} face index is out of range")
    triangles = vertices[faces]
    areas2 = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if not np.isfinite(areas2).all() or np.all(areas2 <= 1e-12):
        raise MetricInputError(f"{name} has no valid surface area")
    return mesh


def normalize_mesh_pair(
    prediction: trimesh.Trimesh,
    ground_truth: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Apply the ground-truth coordinate transform to both meshes."""
    prediction = _validated(prediction, "prediction").copy()
    ground_truth = _validated(ground_truth, "ground_truth").copy()
    gt_vertices = np.asarray(ground_truth.vertices, dtype=np.float64)
    bounds = np.stack((gt_vertices.min(axis=0), gt_vertices.max(axis=0)))
    center = bounds.mean(axis=0)
    scale_extent = float((bounds[1] - bounds[0]).max())
    if scale_extent <= 1e-12:
        raise MetricInputError("ground_truth has zero extent")
    transform = np.eye(4)
    transform[:3, :3] *= 1.0 / scale_extent
    transform[:3, 3] = -center / scale_extent
    prediction.apply_transform(transform)
    ground_truth.apply_transform(transform)
    return prediction, ground_truth


def _surface_samples(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    doubled_area = np.linalg.norm(cross, axis=1)
    probability = doubled_area / doubled_area.sum()
    cdf = np.cumsum(probability)
    quantiles = (np.arange(count, dtype=np.float64) + 0.5) / count
    selected = np.searchsorted(cdf, quantiles).clip(max=len(faces) - 1)
    first = np.mod(quantiles * 0.754877666, 1.0)
    second = np.mod(quantiles * 0.569840296 + 0.31, 1.0)
    root = np.sqrt(first)
    barycentric = np.stack((1.0 - root, root * (1.0 - second), root * second), axis=1)
    points = (triangles[selected] * barycentric[:, :, None]).sum(axis=1)
    normals = cross[selected] / np.maximum(doubled_area[selected, None], 1e-12)
    return points.astype(np.float32), normals.astype(np.float32)


def _nearest(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = torch.cdist(torch.from_numpy(first), torch.from_numpy(second))
    values, indices = distances.min(dim=1)
    return values.numpy(), indices.numpy()


def _fscore(forward: np.ndarray, backward: np.ndarray, threshold: float) -> float:
    precision = float((forward <= threshold).mean())
    recall = float((backward <= threshold).mean())
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _sharp_edge_samples(
    mesh: trimesh.Trimesh,
    count: int,
    threshold_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = torch.tensor(np.asarray(mesh.vertices).copy(), dtype=torch.float32).unsqueeze(0)
    faces = torch.tensor(np.asarray(mesh.faces).copy(), dtype=torch.long)
    geometry = compute_mesh_geometry(
        vertices, faces, sharp_threshold_degrees=threshold_degrees
    )
    mask = geometry.sharp_edges[0]
    edges = geometry.edges[mask].numpy()
    angles = geometry.dihedral_angles[0, mask].numpy()
    if len(edges) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    ids = np.arange(count) % len(edges)
    chosen = edges[ids]
    t = np.mod((np.arange(count) + 0.5) / count * len(edges), 1.0)[:, None]
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    points = mesh_vertices[chosen[:, 0]] * (1.0 - t) + mesh_vertices[chosen[:, 1]] * t
    return points, angles[ids]


def silhouette_iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    prediction = np.asarray(prediction) > 0
    ground_truth = np.asarray(ground_truth) > 0
    if prediction.shape != ground_truth.shape:
        raise MetricInputError("silhouette shapes do not match")
    union = np.logical_or(prediction, ground_truth).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, ground_truth).sum() / union)


def compute_mesh_metrics(
    prediction: trimesh.Trimesh,
    ground_truth: trimesh.Trimesh,
    *,
    surface_samples: int = 2048,
    edge_samples: int = 512,
    sharp_edge_threshold_degrees: float = 35.0,
    fscore_thresholds: tuple[float, float] = (0.01, 0.02),
    predicted_silhouette: np.ndarray | None = None,
    target_silhouette: np.ndarray | None = None,
    inference_time_seconds: float = 0.0,
    peak_vram_gb: float = 0.0,
) -> MeshMetricResult:
    prediction, ground_truth = normalize_mesh_pair(prediction, ground_truth)
    pred_points, pred_normals = _surface_samples(prediction, surface_samples)
    gt_points, gt_normals = _surface_samples(ground_truth, surface_samples)
    pred_to_gt, pred_indices = _nearest(pred_points, gt_points)
    gt_to_pred, gt_indices = _nearest(gt_points, pred_points)
    chamfer = float(np.mean(pred_to_gt**2) + np.mean(gt_to_pred**2))
    normal_forward = np.abs((pred_normals * gt_normals[pred_indices]).sum(axis=1))
    normal_backward = np.abs((gt_normals * pred_normals[gt_indices]).sum(axis=1))
    normal_consistency = float((normal_forward.mean() + normal_backward.mean()) * 0.5)

    pred_edges, pred_angles = _sharp_edge_samples(
        prediction, edge_samples, sharp_edge_threshold_degrees
    )
    gt_edges, gt_angles = _sharp_edge_samples(
        ground_truth, edge_samples, sharp_edge_threshold_degrees
    )
    if len(pred_edges) == 0 and len(gt_edges) == 0:
        edge_chamfer = 0.0
        angle_error = 0.0
    elif len(pred_edges) == 0 or len(gt_edges) == 0:
        edge_chamfer = 1.0
        angle_error = 180.0
    else:
        edge_forward, edge_pred_indices = _nearest(pred_edges, gt_edges)
        edge_backward, edge_gt_indices = _nearest(gt_edges, pred_edges)
        edge_chamfer = float(np.mean(edge_forward**2) + np.mean(edge_backward**2))
        forward_angle = np.abs(pred_angles - gt_angles[edge_pred_indices])
        backward_angle = np.abs(gt_angles - pred_angles[edge_gt_indices])
        angle_error = float(np.rad2deg((forward_angle.mean() + backward_angle.mean()) * 0.5))

    if predicted_silhouette is None or target_silhouette is None:
        iou = math.nan
    else:
        iou = silhouette_iou(predicted_silhouette, target_silhouette)
    return MeshMetricResult(
        chamfer_distance=chamfer,
        fscore_1_percent=_fscore(pred_to_gt, gt_to_pred, fscore_thresholds[0]),
        fscore_2_percent=_fscore(pred_to_gt, gt_to_pred, fscore_thresholds[1]),
        normal_consistency=normal_consistency,
        silhouette_iou=iou,
        edge_chamfer_distance=edge_chamfer,
        dihedral_angle_error_degrees=angle_error,
        vertex_count=int(len(prediction.vertices)),
        face_count=int(len(prediction.faces)),
        inference_time_seconds=float(inference_time_seconds),
        peak_vram_gb=float(peak_vram_gb),
        watertight=bool(prediction.is_watertight),
    )
