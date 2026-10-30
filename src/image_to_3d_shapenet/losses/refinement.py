"""Edge-aware mesh refinement losses with a stock-PyTorch fallback."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from image_to_3d_shapenet.config import RefinementConfig
from image_to_3d_shapenet.mesh.geometry import (
    compute_mesh_geometry,
    laplacian_coordinates,
)


class PointDistanceBackend:
    name = "native"

    def __call__(self, first: Tensor, second: Tensor) -> Tensor:
        distances = torch.cdist(first, second).square()
        return distances.min(dim=2).values.mean() + distances.min(dim=1).values.mean()


class PyTorch3DPointDistance(PointDistanceBackend):
    name = "pytorch3d"

    def __init__(self) -> None:
        try:
            from pytorch3d.loss import chamfer_distance
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "PyTorch3D loss requested but unavailable; use loss_backend=native or auto"
            ) from error
        self._chamfer = chamfer_distance

    def __call__(self, first: Tensor, second: Tensor) -> Tensor:
        return self._chamfer(first, second)[0]


def build_point_distance_backend(name: str) -> PointDistanceBackend:
    if name == "native":
        return PointDistanceBackend()
    if name == "pytorch3d":
        return PyTorch3DPointDistance()
    if name == "auto":
        try:
            return PyTorch3DPointDistance()
        except RuntimeError:
            return PointDistanceBackend()
    raise ValueError(f"unknown loss backend: {name}")


def sample_mesh_surface(vertices: Tensor, faces: Tensor, count: int) -> Tensor:
    """Deterministic differentiable surface samples; topology selection has no gradient."""
    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)
    if faces.ndim == 3:
        faces = faces[0]
    triangles = vertices[:, faces]
    cross = torch.cross(
        triangles[:, :, 1] - triangles[:, :, 0],
        triangles[:, :, 2] - triangles[:, :, 0],
        dim=-1,
    )
    areas = cross.norm(dim=-1).clamp_min(1e-12)
    quantiles = (torch.arange(count, device=vertices.device, dtype=vertices.dtype) + 0.5) / count
    samples = []
    for batch_index in range(vertices.shape[0]):
        cdf = torch.cumsum(areas[batch_index] / areas[batch_index].sum(), dim=0)
        selected = torch.searchsorted(cdf, quantiles).clamp_max(len(faces) - 1)
        chosen = triangles[batch_index, selected]
        first = torch.frac(quantiles * 0.754877666)
        second = torch.frac(quantiles * 0.569840296 + 0.31)
        root = torch.sqrt(first)
        barycentric = torch.stack((1.0 - root, root * (1.0 - second), root * second), dim=-1)
        samples.append((chosen * barycentric.unsqueeze(-1)).sum(dim=1))
    return torch.stack(samples)


def sample_edge_points(vertices: Tensor, edges: Tensor, mask: Tensor, count: int) -> Tensor:
    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    output = []
    for batch_index in range(vertices.shape[0]):
        selected_edges = edges[mask[batch_index]]
        if len(selected_edges) == 0:
            output.append(vertices[batch_index, :1].expand(count, -1))
            continue
        ids = torch.arange(count, device=vertices.device) % len(selected_edges)
        chosen = selected_edges[ids]
        t = ((torch.arange(count, device=vertices.device, dtype=vertices.dtype) + 0.5) / count)
        t = torch.frac(t * len(selected_edges)).unsqueeze(-1)
        points = (
            vertices[batch_index, chosen[:, 0]] * (1.0 - t)
            + vertices[batch_index, chosen[:, 1]] * t
        )
        output.append(points)
    return torch.stack(output)


@dataclass(frozen=True)
class RefinementLossOutput:
    total: Tensor
    components: dict[str, Tensor]
    backend: str


class EdgeAwareRefinementLoss:
    def __init__(self, config: RefinementConfig) -> None:
        self.config = config
        self.distance = build_point_distance_backend(config.loss_backend)

    def __call__(
        self,
        base_vertices: Tensor,
        refined_vertices: Tensor,
        faces: Tensor,
        target_vertices: Tensor,
        target_faces: Tensor,
        *,
        predicted_silhouette: Tensor | None = None,
        target_silhouette: Tensor | None = None,
    ) -> RefinementLossOutput:
        if base_vertices.ndim == 2:
            base_vertices = base_vertices.unsqueeze(0)
        if refined_vertices.ndim == 2:
            refined_vertices = refined_vertices.unsqueeze(0)
        if target_vertices.ndim == 2:
            target_vertices = target_vertices.unsqueeze(0)
        base_geometry = compute_mesh_geometry(
            base_vertices, faces,
            sharp_threshold_degrees=self.config.sharp_edge_threshold_degrees,
        )
        refined_geometry = compute_mesh_geometry(
            refined_vertices, faces,
            sharp_threshold_degrees=self.config.sharp_edge_threshold_degrees,
        )
        target_geometry = compute_mesh_geometry(
            target_vertices, target_faces,
            sharp_threshold_degrees=self.config.sharp_edge_threshold_degrees,
        )

        predicted_points = sample_mesh_surface(refined_vertices, faces, self.config.chamfer_samples)
        target_points = sample_mesh_surface(target_vertices, target_faces, self.config.chamfer_samples)
        chamfer = self.distance(predicted_points, target_points)

        vertex_distances = torch.cdist(refined_vertices, target_vertices)
        nearest = vertex_distances.argmin(dim=2)
        target_normals = torch.gather(
            target_geometry.vertex_normals,
            1,
            nearest.unsqueeze(-1).expand(-1, -1, 3),
        )
        normal = (1.0 - F.cosine_similarity(refined_geometry.vertex_normals, target_normals, dim=-1)).mean()

        if predicted_silhouette is not None and target_silhouette is not None:
            silhouette = F.binary_cross_entropy(
                predicted_silhouette.clamp(1e-6, 1.0 - 1e-6), target_silhouette
            )
        else:
            silhouette = refined_vertices.sum() * 0.0

        predicted_sharp = sample_edge_points(
            refined_vertices,
            refined_geometry.edges,
            base_geometry.sharp_edges,
            self.config.sharp_edge_samples,
        )
        target_sharp = sample_edge_points(
            target_vertices,
            target_geometry.edges,
            target_geometry.sharp_edges,
            self.config.sharp_edge_samples,
        )
        sharp_distance = self.distance(predicted_sharp, target_sharp)
        angle_preservation = (
            (refined_geometry.dihedral_angles - base_geometry.dihedral_angles).square()
            * base_geometry.sharp_edges.to(refined_vertices.dtype)
        ).sum() / base_geometry.sharp_edges.sum().clamp_min(1)
        # Dedicated samples and an explicit multiplier emphasize sharp geometry.
        edge = self.config.sharp_edge_sampling_weight * sharp_distance + angle_preservation

        base_laplacian = laplacian_coordinates(
            base_vertices, base_geometry.edges, base_geometry.sharp_edges
        )
        refined_laplacian = laplacian_coordinates(
            refined_vertices, base_geometry.edges, base_geometry.sharp_edges
        )
        laplacian = (refined_laplacian - base_laplacian).square().mean()
        displacement = (refined_vertices - base_vertices).square().mean()

        boundary_mask = base_geometry.boundary_edges
        if boundary_mask.any():
            boundary_edges = base_geometry.edges[boundary_mask]
            base_lengths = (
                base_vertices[:, boundary_edges[:, 1]] - base_vertices[:, boundary_edges[:, 0]]
            ).norm(dim=-1)
            refined_lengths = (
                refined_vertices[:, boundary_edges[:, 1]] - refined_vertices[:, boundary_edges[:, 0]]
            ).norm(dim=-1)
            boundary = (refined_lengths - base_lengths).square().mean()
        else:
            boundary = refined_vertices.sum() * 0.0

        components = {
            "chamfer": chamfer,
            "normal": normal,
            "silhouette": silhouette,
            "edge": edge,
            "laplacian": laplacian,
            "displacement": displacement,
            "boundary": boundary,
        }
        total = (
            self.config.lambda_chamfer * chamfer
            + self.config.lambda_normal * normal
            + self.config.lambda_silhouette * silhouette
            + self.config.lambda_edge * edge
            + self.config.lambda_laplacian * laplacian
            + self.config.lambda_displacement * displacement
            + self.config.lambda_boundary * boundary
        )
        return RefinementLossOutput(total=total, components=components, backend=self.distance.name)
