from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
import trimesh

from image_to_3d_shapenet.config import RefinementConfig
from image_to_3d_shapenet.losses import EdgeAwareRefinementLoss, build_point_distance_backend
from image_to_3d_shapenet.mesh import compute_mesh_geometry, save_refinement_debug
from image_to_3d_shapenet.models import ImageConditionedMeshRefiner


def cube_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices = torch.tensor(np.asarray(cube.vertices).copy(), dtype=torch.float32).unsqueeze(0)
    faces = torch.tensor(np.asarray(cube.faces).copy(), dtype=torch.long)
    return vertices, faces


def small_config() -> RefinementConfig:
    return RefinementConfig(
        encoder="lightweight",
        pretrained_encoder=False,
        freeze_encoder=False,
        feature_dim=8,
        hidden_dim=16,
        num_blocks=2,
        max_displacement=0.04,
        chamfer_samples=32,
        sharp_edge_samples=24,
        loss_backend="native",
    )


def camera(size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.tensor(
        [[[30.0, 0.0, (size - 1) / 2], [0.0, 30.0, (size - 1) / 2], [0.0, 0.0, 1.0]]]
    )
    extrinsics = torch.eye(4).unsqueeze(0)
    extrinsics[:, 2, 3] = -3.0
    return intrinsics, extrinsics


def test_cube_detects_twelve_ninety_degree_sharp_edges() -> None:
    vertices, faces = cube_tensors()
    geometry = compute_mesh_geometry(vertices, faces, sharp_threshold_degrees=35.0)
    sharp_angles = torch.rad2deg(geometry.dihedral_angles[geometry.sharp_edges])
    assert geometry.edges.shape[0] == 18  # 12 cube edges + 6 face diagonals
    assert int(geometry.sharp_edges.sum()) == 12
    assert torch.allclose(sharp_angles, torch.full_like(sharp_angles, 90.0), atol=1e-3)


def test_refiner_shape_displacement_loss_and_backward() -> None:
    torch.manual_seed(42)
    config = small_config()
    model = ImageConditionedMeshRefiner(config)
    vertices, faces = cube_tensors()
    image = torch.rand((1, 3, 32, 32))
    intrinsics, extrinsics = camera()
    output = model(
        image,
        vertices,
        faces,
        intrinsics,
        extrinsics,
        visibility=torch.ones((1, vertices.shape[1])),
    )
    assert output.refined_vertices.shape == vertices.shape
    assert output.displacement.shape == vertices.shape
    assert output.faces.shape == faces.shape
    assert int(output.visibility.sum()) == vertices.shape[1]
    assert output.displacement.abs().max() <= config.max_displacement + 1e-7
    assert torch.isfinite(output.refined_vertices).all()

    target = vertices * torch.tensor((1.02, 0.98, 1.01))
    losses = EdgeAwareRefinementLoss(config)(
        vertices, output.refined_vertices, faces, target, faces
    )
    assert losses.backend == "native"
    assert torch.isfinite(losses.total)
    assert all(torch.isfinite(value) for value in losses.components.values())
    assert set(losses.components) == {
        "chamfer", "normal", "silhouette", "edge", "laplacian",
        "displacement", "boundary",
    }
    losses.total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_tiny_overfit_step_updates_parameters() -> None:
    torch.manual_seed(7)
    config = replace(small_config(), chamfer_samples=16, sharp_edge_samples=12)
    model = ImageConditionedMeshRefiner(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    vertices, faces = cube_tensors()
    image = torch.ones((1, 3, 32, 32)) * 0.5
    intrinsics, extrinsics = camera()
    target = vertices * 0.98
    before = model.displacement_head[-1].weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    output = model(image, vertices, faces, intrinsics, extrinsics)
    loss = EdgeAwareRefinementLoss(config)(vertices, output.refined_vertices, faces, target, faces).total
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.displacement_head[-1].weight.detach())


def test_native_distance_backend_and_explicit_pytorch3d_message() -> None:
    assert build_point_distance_backend("native").name == "native"
    try:
        backend = build_point_distance_backend("pytorch3d")
    except RuntimeError as error:
        assert "PyTorch3D" in str(error)
    else:
        assert backend.name == "pytorch3d"


def test_visual_debug_saves_before_after_and_sharp_edges(tmp_path) -> None:
    vertices, faces = cube_tensors()
    files = save_refinement_debug(vertices, vertices * 0.98, faces, tmp_path)
    assert set(files) == {"before", "after", "sharp_edges"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in files.values())
    assert len(trimesh.load(files["after"], force="mesh").faces) == len(faces)


def test_invalid_refiner_config_rejected() -> None:
    with pytest.raises(ValueError, match="num_blocks"):
        RefinementConfig(num_blocks=1)
