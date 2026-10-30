from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from image_to_3d_shapenet import load_config
from image_to_3d_shapenet.config import TrainingConfig
from image_to_3d_shapenet.dataset import (
    RenderedRefinementDataset,
    generate_base_mesh_cache,
)
from image_to_3d_shapenet.losses import EdgeAwareRefinementLoss
from image_to_3d_shapenet.models import ImageConditionedMeshRefiner
from image_to_3d_shapenet.training import (
    OOMFallbackManager,
    ResourceState,
    load_checkpoint,
    refinement_step,
    save_checkpoint,
    validate_epoch,
)


ROOT = Path(__file__).resolve().parents[1]


def _quick_config(tmp_path: Path):
    config = load_config(ROOT / "configs" / "quick_test.yaml")
    return replace(
        config,
        paths=replace(config.paths, data_dir="data", cache_dir="cache", output_dir="outputs"),
        render=replace(config.render, backend="software"),
        training=replace(config.training, image_resolution=32, surface_samples=16),
        refinement=replace(
            config.refinement, feature_dim=8, hidden_dim=16,
            chamfer_samples=16, sharp_edge_samples=8,
        ),
    )


def _sample() -> dict:
    cube = trimesh.creation.box()
    vertices = torch.tensor(np.asarray(cube.vertices).copy(), dtype=torch.float32)
    faces = torch.tensor(np.asarray(cube.faces).copy(), dtype=torch.long)
    intrinsics = torch.tensor(
        [[30.0, 0.0, 15.5], [0.0, 30.0, 15.5], [0.0, 0.0, 1.0]]
    )
    extrinsics = torch.eye(4)
    extrinsics[2, 3] = -3.0
    return {
        "rgb": torch.rand(3, 32, 32),
        "mask": torch.ones(1, 32, 32),
        "depth": torch.ones(1, 32, 32),
        "normal": torch.zeros(3, 32, 32),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "base_vertices": vertices * 0.9,
        "base_faces": faces,
        "target_vertices": vertices,
        "target_faces": faces,
        "model_id": "cube",
        "view_id": 0,
    }


def test_cache_manifest_resume_and_dataset_reads_all_modalities(tmp_path: Path) -> None:
    config = _quick_config(tmp_path)
    first = generate_base_mesh_cache(config, tmp_path, maximum=4)
    assert first == {**first, "generated": 4, "skipped": 0, "failed": 0, "samples": 4}
    second = generate_base_mesh_cache(config, tmp_path, maximum=4)
    assert second["generated"] == 0
    assert second["skipped"] == 4
    manifest = json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))
    assert manifest["synthetic_only"] is True
    assert all(item["backend"] == "synthetic-perturbation" for item in manifest["samples"].values())

    dataset = RenderedRefinementDataset(tmp_path, config, maximum=4, image_resolution=32)
    item = dataset[0]
    required = {
        "rgb", "mask", "depth", "normal", "intrinsics", "extrinsics",
        "ground_truth_mesh", "base_mesh_prediction", "model_id", "view_id",
        "base_vertices", "base_faces", "target_vertices", "target_faces",
    }
    assert required <= set(item)
    assert item["rgb"].shape == (3, 32, 32)
    assert torch.isfinite(item["depth"]).all()


def test_checkpoint_roundtrip_restores_global_step(tmp_path: Path) -> None:
    config = _quick_config(tmp_path)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = save_checkpoint(
        tmp_path / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, epoch=3, global_step=17,
        best_metric={"chamfer": 0.2, "edge": 0.3}, config=config,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    state = load_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, restore_random=False,
    )
    assert state["epoch"] == 3
    assert state["global_step"] == 17
    assert state["best_metric"]["chamfer"] == 0.2
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())


def test_training_and_validation_step_are_finite() -> None:
    config = load_config(ROOT / "configs" / "quick_test.yaml")
    refinement = replace(
        config.refinement, feature_dim=8, hidden_dim=16,
        chamfer_samples=16, sharp_edge_samples=8,
    )
    model = ImageConditionedMeshRefiner(refinement)
    criterion = EdgeAwareRefinementLoss(refinement)
    sample = _sample()
    losses, _ = refinement_step(model, criterion, sample)
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert any(parameter.grad is not None for parameter in model.parameters())
    metrics = validate_epoch(
        model, criterion, [sample], torch.device("cpu"),
        ResourceState(surface_samples=16, max_vertices=512, image_resolution=32),
    )
    assert set(metrics) == {"loss", "chamfer", "edge"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_oom_fallback_order_and_exhaustion() -> None:
    config = TrainingConfig(
        surface_samples=64, minimum_surface_samples=16,
        max_vertices=64, minimum_vertices=16,
        image_resolution=64, minimum_image_resolution=16,
    )
    manager = OOMFallbackManager(config)
    actions = [manager.reduce() for _ in range(7)]
    assert actions[0].startswith("cleared")
    assert actions[1:3] == [
        "reduced surface samples to 32", "reduced surface samples to 16"
    ]
    assert actions[3:5] == ["reduced max vertices to 32", "reduced max vertices to 16"]
    assert actions[5:7] == [
        "reduced image resolution to 32", "reduced image resolution to 16"
    ]
    try:
        manager.reduce()
    except RuntimeError as error:
        assert "OOM persists" in str(error)
    else:
        raise AssertionError("exhausted OOM policy must fail clearly")
