"""Differentiable mesh geometry utilities implemented with stock PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class EdgeTopology:
    edges: Tensor
    adjacent_faces: Tensor
    boundary: Tensor


@dataclass(frozen=True)
class MeshGeometry:
    face_normals: Tensor
    vertex_normals: Tensor
    curvature: Tensor
    edges: Tensor
    adjacent_faces: Tensor
    dihedral_angles: Tensor
    sharp_edges: Tensor
    boundary_edges: Tensor
    vertex_dihedral: Tensor
    vertex_sharpness: Tensor


def _batched_vertices(vertices: Tensor) -> Tensor:
    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [V, 3] or [B, V, 3]")
    return vertices


def _shared_faces(faces: Tensor) -> Tensor:
    if faces.ndim == 3:
        if faces.shape[0] != 1:
            if not torch.equal(faces, faces[:1].expand_as(faces)):
                raise ValueError("all batch items must share topology")
        faces = faces[0]
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape [F, 3] or shared [B, F, 3]")
    return faces.long()


def build_edge_topology(faces: Tensor) -> EdgeTopology:
    faces = _shared_faces(faces)
    face_edges = torch.cat(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=0
    )
    face_edges = face_edges.sort(dim=1).values
    face_ids = torch.arange(len(faces), device=faces.device).repeat(3)
    edges, inverse = torch.unique(face_edges, dim=0, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=len(edges))
    order = torch.argsort(inverse, stable=True)
    starts = torch.cumsum(counts, dim=0) - counts
    adjacent = torch.full((len(edges), 2), -1, dtype=torch.long, device=faces.device)
    adjacent[:, 0] = face_ids[order[starts]]
    has_second = counts > 1
    adjacent[has_second, 1] = face_ids[order[starts[has_second] + 1]]
    return EdgeTopology(edges=edges, adjacent_faces=adjacent, boundary=~has_second)


def compute_face_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    vertices = _batched_vertices(vertices)
    faces = _shared_faces(faces)
    triangles = vertices[:, faces]
    normals = torch.cross(
        triangles[:, :, 1] - triangles[:, :, 0],
        triangles[:, :, 2] - triangles[:, :, 0],
        dim=-1,
    )
    return F.normalize(normals, dim=-1, eps=1e-8)


def compute_vertex_normals(vertices: Tensor, faces: Tensor, face_normals: Tensor | None = None) -> Tensor:
    vertices = _batched_vertices(vertices)
    faces = _shared_faces(faces)
    if face_normals is None:
        face_normals = compute_face_normals(vertices, faces)
    result = torch.zeros_like(vertices)
    for corner in range(3):
        result.index_add_(1, faces[:, corner], face_normals)
    return F.normalize(result, dim=-1, eps=1e-8)


def compute_dihedral_angles(face_normals: Tensor, topology: EdgeTopology) -> Tensor:
    first = face_normals[:, topology.adjacent_faces[:, 0]]
    second_ids = topology.adjacent_faces[:, 1].clamp_min(0)
    second = face_normals[:, second_ids]
    cosine = (first * second).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angles = torch.acos(cosine)
    return torch.where(topology.boundary.unsqueeze(0), torch.zeros_like(angles), angles)


def _aggregate_edge_scalar(values: Tensor, edges: Tensor, vertex_count: int) -> Tensor:
    batch = values.shape[0]
    result = values.new_zeros((batch, vertex_count))
    counts = values.new_zeros((batch, vertex_count))
    ones = torch.ones_like(values)
    for endpoint in range(2):
        result.index_add_(1, edges[:, endpoint], values)
        counts.index_add_(1, edges[:, endpoint], ones)
    return (result / counts.clamp_min(1.0)).unsqueeze(-1)


def compute_mesh_geometry(
    vertices: Tensor,
    faces: Tensor,
    *,
    sharp_threshold_degrees: float = 35.0,
) -> MeshGeometry:
    vertices = _batched_vertices(vertices)
    faces = _shared_faces(faces)
    topology = build_edge_topology(faces)
    face_normals = compute_face_normals(vertices, faces)
    vertex_normals = compute_vertex_normals(vertices, faces, face_normals)
    dihedral = compute_dihedral_angles(face_normals, topology)
    threshold = torch.deg2rad(vertices.new_tensor(sharp_threshold_degrees))
    sharp = (dihedral > threshold) & ~topology.boundary.unsqueeze(0)

    neighbor_sum = torch.zeros_like(vertices)
    neighbor_count = vertices.new_zeros(vertices.shape[:2] + (1,))
    for endpoint, other in ((0, 1), (1, 0)):
        neighbor_sum.index_add_(1, topology.edges[:, endpoint], vertices[:, topology.edges[:, other]])
        neighbor_count.index_add_(
            1,
            topology.edges[:, endpoint],
            vertices.new_ones((vertices.shape[0], len(topology.edges), 1)),
        )
    neighbor_mean = neighbor_sum / neighbor_count.clamp_min(1.0)
    curvature = (vertices - neighbor_mean).norm(dim=-1, keepdim=True)
    vertex_dihedral = _aggregate_edge_scalar(dihedral, topology.edges, vertices.shape[1])
    vertex_sharpness = _aggregate_edge_scalar(
        sharp.to(vertices.dtype), topology.edges, vertices.shape[1]
    )
    return MeshGeometry(
        face_normals=face_normals,
        vertex_normals=vertex_normals,
        curvature=curvature,
        edges=topology.edges,
        adjacent_faces=topology.adjacent_faces,
        dihedral_angles=dihedral,
        sharp_edges=sharp,
        boundary_edges=topology.boundary,
        vertex_dihedral=vertex_dihedral,
        vertex_sharpness=vertex_sharpness,
    )


def laplacian_coordinates(vertices: Tensor, edges: Tensor, sharp_edges: Tensor) -> Tensor:
    """Uniform Laplacian excluding messages that cross sharp edges."""
    vertices = _batched_vertices(vertices)
    if sharp_edges.ndim == 1:
        sharp_edges = sharp_edges.unsqueeze(0)
    output = torch.zeros_like(vertices)
    counts = vertices.new_zeros(vertices.shape[:2] + (1,))
    for batch_index in range(vertices.shape[0]):
        keep = ~sharp_edges[batch_index]
        selected = edges[keep]
        if selected.numel() == 0:
            continue
        source = torch.cat((selected[:, 0], selected[:, 1]))
        neighbor = torch.cat((selected[:, 1], selected[:, 0]))
        output[batch_index].index_add_(0, source, vertices[batch_index, neighbor])
        counts[batch_index].index_add_(
            0, source, torch.ones((len(source), 1), device=vertices.device, dtype=vertices.dtype)
        )
    neighbor_mean = output / counts.clamp_min(1.0)
    return torch.where(counts > 0, vertices - neighbor_mean, torch.zeros_like(vertices))
