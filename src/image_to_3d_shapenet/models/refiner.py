"""Image-conditioned, sharp-edge-aware graph mesh refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from image_to_3d_shapenet.config import RefinementConfig
from image_to_3d_shapenet.mesh.geometry import MeshGeometry, compute_mesh_geometry

from .image_encoder import build_image_encoder, project_features_to_vertices


@dataclass(frozen=True)
class RefinementOutput:
    base_vertices: Tensor
    refined_vertices: Tensor
    displacement: Tensor
    faces: Tensor
    geometry: MeshGeometry
    visibility: Tensor
    projected_image_features: Tensor


class EdgeAwareGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: Tensor,
        edges: Tensor,
        dihedral: Tensor,
        sharp: Tensor,
        sharp_message_weight: float,
    ) -> Tensor:
        batch, vertex_count, hidden = features.shape
        aggregated = torch.zeros_like(features)
        counts = features.new_zeros((batch, vertex_count, 1))
        for batch_index in range(batch):
            for source_column, target_column in ((0, 1), (1, 0)):
                source = edges[:, source_column]
                target = edges[:, target_column]
                edge_info = torch.stack(
                    (dihedral[batch_index], sharp[batch_index].to(features.dtype)), dim=-1
                )
                message_input = torch.cat(
                    (features[batch_index, source], features[batch_index, target], edge_info), dim=-1
                )
                gate = torch.where(
                    sharp[batch_index].unsqueeze(-1),
                    features.new_full((len(edges), 1), sharp_message_weight),
                    features.new_ones((len(edges), 1)),
                )
                messages = self.message(message_input) * gate
                aggregated[batch_index].index_add_(0, target, messages)
                counts[batch_index].index_add_(
                    0, target, torch.ones((len(edges), 1), device=features.device, dtype=features.dtype)
                )
        aggregated = aggregated / counts.clamp_min(1.0)
        residual = self.update(torch.cat((features, aggregated), dim=-1))
        return self.norm(features + residual)


class ImageConditionedMeshRefiner(nn.Module):
    def __init__(self, config: RefinementConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = build_image_encoder(
            config.encoder,
            config.feature_dim,
            pretrained=config.pretrained_encoder,
            freeze=config.freeze_encoder,
            allow_fallback=config.allow_encoder_fallback,
        )
        geometry_feature_dim = 3 + 3 + 1 + 1 + 1 + 1
        input_dim = geometry_feature_dim + self.encoder.output_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim), nn.SiLU(), nn.LayerNorm(config.hidden_dim)
        )
        self.blocks = nn.ModuleList(
            EdgeAwareGraphBlock(config.hidden_dim) for _ in range(config.num_blocks)
        )
        self.displacement_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, 3)
        )

    def forward(
        self,
        image: Tensor,
        vertices: Tensor,
        faces: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        visibility: Tensor | None = None,
    ) -> RefinementOutput:
        if vertices.ndim == 2:
            vertices = vertices.unsqueeze(0)
        if self.config.freeze_encoder:
            with torch.no_grad():
                image_features = self.encoder(image)
        else:
            image_features = self.encoder(image)
        projected, camera_visibility = project_features_to_vertices(
            image_features,
            vertices,
            intrinsics,
            extrinsics,
            camera_convention=self.config.camera_convention,
        )
        if visibility is not None:
            supplied_visibility = visibility
            if supplied_visibility.ndim == 2:
                supplied_visibility = supplied_visibility.unsqueeze(-1)
            if supplied_visibility.shape != projected.shape[:2] + (1,):
                raise ValueError("visibility must have shape [B, V] or [B, V, 1]")
            visibility = camera_visibility * supplied_visibility.to(camera_visibility.dtype)
        else:
            visibility = camera_visibility
        projected = projected * visibility
        geometry = compute_mesh_geometry(
            vertices,
            faces,
            sharp_threshold_degrees=self.config.sharp_edge_threshold_degrees,
        )
        node_features = torch.cat(
            (
                vertices,
                geometry.vertex_normals,
                geometry.curvature,
                geometry.vertex_dihedral,
                geometry.vertex_sharpness,
                visibility,
                projected,
            ),
            dim=-1,
        )
        hidden = self.input_projection(node_features)
        for block in self.blocks:
            hidden = block(
                hidden,
                geometry.edges,
                geometry.dihedral_angles,
                geometry.sharp_edges,
                self.config.sharp_edge_message_weight,
            )
        displacement = torch.tanh(self.displacement_head(hidden)) * self.config.max_displacement
        return RefinementOutput(
            base_vertices=vertices,
            refined_vertices=vertices + displacement,
            displacement=displacement,
            faces=faces,
            geometry=geometry,
            visibility=visibility,
            projected_image_features=projected,
        )
