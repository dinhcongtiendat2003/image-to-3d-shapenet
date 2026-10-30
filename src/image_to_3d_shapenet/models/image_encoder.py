"""Small multi-scale image encoders with a dependency-safe ResNet18 option."""

from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LightweightImageEncoder(nn.Module):
    def __init__(self, feature_dim: int = 32) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, feature_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(feature_dim), nn.SiLU(),
                    nn.Conv2d(feature_dim, feature_dim, 3, padding=1), nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(feature_dim, feature_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(feature_dim), nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(feature_dim, feature_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(feature_dim), nn.SiLU(),
                ),
            ]
        )
        self.output_dim = feature_dim * 3
        self.frozen = False

    def train(self, mode: bool = True):
        super().train(False if self.frozen else mode)
        return self

    def forward(self, image: Tensor) -> list[Tensor]:
        features = []
        value = image
        for stage in self.stages:
            value = stage(value)
            features.append(value)
        return features


class ResNet18MultiScaleEncoder(nn.Module):
    def __init__(self, feature_dim: int, *, pretrained: bool, freeze: bool) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layers = nn.ModuleList((backbone.layer1, backbone.layer2, backbone.layer3))
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, feature_dim, 1) for channels in (64, 128, 256)
        )
        self.output_dim = feature_dim * 3
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False
        )
        self.frozen = freeze
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False if self.frozen else mode)
        return self

    def forward(self, image: Tensor) -> list[Tensor]:
        value = (image - self.mean) / self.std
        value = self.stem(value)
        outputs = []
        for layer, projection in zip(self.layers, self.projections):
            value = layer(value)
            outputs.append(projection(value))
        return outputs


def build_image_encoder(
    name: str,
    feature_dim: int,
    *,
    pretrained: bool,
    freeze: bool,
    allow_fallback: bool,
) -> nn.Module:
    if name == "lightweight":
        encoder = LightweightImageEncoder(feature_dim)
    elif name == "resnet18":
        try:
            encoder = ResNet18MultiScaleEncoder(
                feature_dim, pretrained=pretrained, freeze=freeze
            )
        except Exception as error:
            if not allow_fallback:
                raise RuntimeError(
                    f"ResNet18 encoder unavailable ({type(error).__name__}); install compatible torchvision"
                ) from None
            warnings.warn(
                f"ResNet18 unavailable ({type(error).__name__}); using lightweight encoder",
                RuntimeWarning,
                stacklevel=2,
            )
            encoder = LightweightImageEncoder(feature_dim)
    else:
        raise ValueError(f"unknown image encoder: {name}")
    if freeze:
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        if hasattr(encoder, "frozen"):
            encoder.frozen = True
        encoder.eval()
    return encoder


def project_features_to_vertices(
    feature_maps: list[Tensor],
    vertices: Tensor,
    intrinsics: Tensor,
    extrinsics: Tensor,
    *,
    camera_convention: str = "opengl",
) -> tuple[Tensor, Tensor]:
    """Project world vertices and bilinearly sample every feature scale."""
    if vertices.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("invalid vertex or intrinsic shape")
    if extrinsics.shape[-2:] != (4, 4):
        raise ValueError("extrinsics must be world-to-camera [B, 4, 4]")
    ones = torch.ones_like(vertices[..., :1])
    camera = torch.matmul(
        torch.cat((vertices, ones), dim=-1), extrinsics.transpose(1, 2)
    )[..., :3]
    if camera_convention == "opengl":
        depth = -camera[..., 2]
        camera_for_projection = torch.stack(
            (camera[..., 0], -camera[..., 1], depth), dim=-1
        )
    elif camera_convention == "opencv":
        depth = camera[..., 2]
        camera_for_projection = camera
    else:
        raise ValueError("camera_convention must be opengl or opencv")
    safe_depth = depth.clamp_min(1e-6)
    pixels_h = torch.matmul(camera_for_projection, intrinsics.transpose(1, 2))
    u = pixels_h[..., 0] / safe_depth
    v = pixels_h[..., 1] / safe_depth
    # Intrinsics refer to the input image; infer its size from principal point.
    input_width = (intrinsics[:, 0, 2] * 2.0 + 1.0).clamp_min(2.0)
    input_height = (intrinsics[:, 1, 2] * 2.0 + 1.0).clamp_min(2.0)
    x = 2.0 * u / (input_width[:, None] - 1.0) - 1.0
    y = 2.0 * v / (input_height[:, None] - 1.0) - 1.0
    grid = torch.stack((x, y), dim=-1)
    visibility = (
        (depth > 1e-6) & (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)
    )
    sampled = []
    sample_grid = grid.unsqueeze(2)
    for feature_map in feature_maps:
        value = F.grid_sample(
            feature_map, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        sampled.append(value.squeeze(-1).transpose(1, 2))
    projected = torch.cat(sampled, dim=-1) * visibility.unsqueeze(-1)
    return projected, visibility.unsqueeze(-1).to(projected.dtype)
