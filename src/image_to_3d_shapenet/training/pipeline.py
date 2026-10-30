"""Colab-first batch-1 training and validation for the mesh refiner."""

from __future__ import annotations

import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml

from image_to_3d_shapenet.config import ProjectConfig
from image_to_3d_shapenet.dataset import (
    RenderedRefinementDataset,
    generate_base_mesh_cache,
    single_item_collate,
)
from image_to_3d_shapenet.losses import EdgeAwareRefinementLoss, RefinementLossOutput
from image_to_3d_shapenet.models import ImageConditionedMeshRefiner
from image_to_3d_shapenet.utils import ProjectPaths, set_random_seed

from .checkpoint import load_checkpoint, save_checkpoint
from .oom import OOMFallbackManager, ResourceState, is_out_of_memory


def _device_sample(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in sample.items()}


def _limit_mesh(vertices: Tensor, faces: Tensor, maximum: int) -> tuple[Tensor, Tensor]:
    if len(vertices) <= maximum:
        return vertices, faces
    keep_faces = faces[(faces < maximum).all(dim=1)]
    if len(keep_faces) == 0:
        chosen = []
        used: set[int] = set()
        for face in faces.tolist():
            candidate = used | set(face)
            if len(candidate) <= maximum:
                chosen.append(face)
                used = candidate
        if not chosen:
            raise RuntimeError(f"cannot reduce mesh to {maximum} vertices")
        keep_faces = torch.tensor(chosen, dtype=faces.dtype, device=faces.device)
        selected = torch.tensor(sorted(used), dtype=torch.long, device=faces.device)
    else:
        selected = torch.arange(maximum, device=faces.device)
    remap = torch.full((len(vertices),), -1, dtype=torch.long, device=faces.device)
    remap[selected] = torch.arange(len(selected), device=faces.device)
    return vertices[selected], remap[keep_faces]


def adapt_sample(sample: dict[str, Any], resources: ResourceState) -> dict[str, Any]:
    result = dict(sample)
    current = sample["rgb"].shape[-1]
    if current != resources.image_resolution:
        size = (resources.image_resolution, resources.image_resolution)
        for key in ("rgb", "normal"):
            result[key] = F.interpolate(sample[key][None], size=size, mode="bilinear", align_corners=False)[0]
        for key in ("mask", "depth"):
            result[key] = F.interpolate(sample[key][None], size=size, mode="nearest")[0]
        scale = resources.image_resolution / current
        result["intrinsics"] = sample["intrinsics"].clone()
        result["intrinsics"][0] *= scale
        result["intrinsics"][1] *= scale
    result["base_vertices"], result["base_faces"] = _limit_mesh(
        sample["base_vertices"], sample["base_faces"], resources.max_vertices
    )
    result["target_vertices"], result["target_faces"] = _limit_mesh(
        sample["target_vertices"], sample["target_faces"], resources.max_vertices
    )
    return result


def refinement_step(
    model: ImageConditionedMeshRefiner,
    criterion: EdgeAwareRefinementLoss,
    sample: dict[str, Any],
) -> tuple[RefinementLossOutput, Any]:
    output = model(
        sample["rgb"].unsqueeze(0),
        sample["base_vertices"].unsqueeze(0),
        sample["base_faces"],
        sample["intrinsics"].unsqueeze(0),
        sample["extrinsics"].unsqueeze(0),
    )
    losses = criterion(
        sample["base_vertices"].unsqueeze(0),
        output.refined_vertices,
        sample["base_faces"],
        sample["target_vertices"].unsqueeze(0),
        sample["target_faces"],
    )
    return losses, output


@torch.no_grad()
def validate_epoch(
    model: ImageConditionedMeshRefiner,
    criterion: EdgeAwareRefinementLoss,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    resources: ResourceState,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "chamfer": 0.0, "edge": 0.0}
    count = 0
    for raw in loader:
        sample = adapt_sample(_device_sample(raw, device), resources)
        losses, _ = refinement_step(model, criterion, sample)
        totals["loss"] += float(losses.total)
        totals["chamfer"] += float(losses.components["chamfer"])
        totals["edge"] += float(losses.components["edge"])
        count += 1
    if count == 0:
        raise RuntimeError("validation loader is empty")
    return {name: value / count for name, value in totals.items()}


def _save_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_loss_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 640, 360
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((50, 20, width - 20, height - 40), outline="black")
    values = [float(row["train_loss"]) for row in rows] + [float(row["val_loss"]) for row in rows]
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    for key, color in (("train_loss", "blue"), ("val_loss", "red")):
        points = []
        for index, row in enumerate(rows):
            x = 50 + index * (width - 70) / max(len(rows) - 1, 1)
            y = height - 40 - (float(row[key]) - low) / span * (height - 60)
            points.append((x, y))
        if len(points) == 1:
            draw.ellipse((points[0][0] - 2, points[0][1] - 2, points[0][0] + 2, points[0][1] + 2), fill=color)
        else:
            draw.line(points, fill=color, width=2)
    canvas.save(path)


def _criterion(config: ProjectConfig, resources: ResourceState) -> EdgeAwareRefinementLoss:
    return EdgeAwareRefinementLoss(
        replace(
            config.refinement,
            chamfer_samples=resources.surface_samples,
            sharp_edge_samples=max(8, resources.surface_samples // 2),
        )
    )


def train_refiner(
    config: ProjectConfig,
    project_root: str | Path,
    *,
    resume: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    quick_overfit: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    set_random_seed(config.runtime.seed, deterministic=True)
    if config.runtime.quick_test:
        generate_base_mesh_cache(config, root, maximum=config.training.quick_samples)
    paths = ProjectPaths.from_config(root, config.paths).create()
    if checkpoint_dir is not None:
        output_dir = Path(checkpoint_dir).expanduser().resolve()
    elif resume is not None:
        output_dir = Path(resume).expanduser().resolve().parent
    else:
        output_dir = paths.outputs / config.training.checkpoint_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8"
    )
    dataset = RenderedRefinementDataset(
        root, config,
        maximum=config.training.quick_samples if config.runtime.quick_test else None,
        image_resolution=config.training.image_resolution,
    )
    train_indices = [i for i, (_, entry, _) in enumerate(dataset.records) if entry.get("split") == "train"]
    validation_indices = [i for i, (_, entry, _) in enumerate(dataset.records) if entry.get("split") == "validation"]
    if quick_overfit:
        train_indices = [0]
        validation_indices = [0]
    else:
        if not train_indices:
            train_indices = list(range(max(1, len(dataset) - 1)))
        if not validation_indices:
            validation_indices = [len(dataset) - 1]
    loader_options = {
        "batch_size": 1,
        "num_workers": config.training.num_workers,
        "collate_fn": single_item_collate,
    }
    train_loader = DataLoader(Subset(dataset, train_indices), shuffle=False, **loader_options)
    validation_loader = DataLoader(Subset(dataset, validation_indices), shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config.runtime.mixed_precision and device.type == "cuda")
    model = ImageConditionedMeshRefiner(config.refinement).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate, weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    oom = OOMFallbackManager(config.training)
    criterion = _criterion(config, oom.state)
    start_epoch = 0
    global_step = 0
    best = {"chamfer": float("inf"), "edge": float("inf")}
    if resume is not None:
        state = load_checkpoint(
            resume, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, map_location=device,
        )
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best.update(state["best_metric"])

    rows: list[dict[str, Any]] = []
    history_path = output_dir / "training_history.csv"
    if resume is not None and history_path.is_file():
        with history_path.open("r", newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    stale_epochs = 0
    epochs = min(config.training.epochs, 3) if quick_overfit else config.training.epochs
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        batch_count = 0
        for batch_index, raw in enumerate(train_loader):
            while True:
                try:
                    sample = adapt_sample(_device_sample(raw, device), oom.state)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                        losses, _ = refinement_step(model, criterion, sample)
                        scaled_loss = losses.total / config.training.gradient_accumulation_steps
                    scaler.scale(scaled_loss).backward()
                    break
                except RuntimeError as error:
                    if not is_out_of_memory(error):
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    print(f"OOM fallback: {oom.reduce()}")
                    criterion = _criterion(config, oom.state)
            train_total += float(losses.total.detach())
            batch_count += 1
            should_step = (
                (batch_index + 1) % config.training.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        while True:
            try:
                validation = validate_epoch(
                    model, criterion, validation_loader, device, oom.state
                )
                break
            except RuntimeError as error:
                if not is_out_of_memory(error):
                    raise
                print(f"OOM fallback: {oom.reduce()}")
                criterion = _criterion(config, oom.state)
        scheduler.step(validation["chamfer"])
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_total / max(batch_count, 1),
            "val_loss": validation["loss"],
            "val_chamfer": validation["chamfer"],
            "val_edge": validation["edge"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "surface_samples": oom.state.surface_samples,
            "max_vertices": oom.state.max_vertices,
            "image_resolution": oom.state.image_resolution,
        }
        rows.append(row)
        improved = False
        if validation["chamfer"] < best["chamfer"]:
            best["chamfer"] = validation["chamfer"]
            save_checkpoint(
                output_dir / "best_chamfer.pt", model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, epoch=epoch, global_step=global_step,
                best_metric=best, config=config,
            )
            improved = True
        if validation["edge"] < best["edge"]:
            best["edge"] = validation["edge"]
            save_checkpoint(
                output_dir / "best_edge.pt", model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, epoch=epoch, global_step=global_step,
                best_metric=best, config=config,
            )
            improved = True
        save_checkpoint(
            output_dir / "last.pt", model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, epoch=epoch, global_step=global_step,
            best_metric=best, config=config,
        )
        _save_history(history_path, rows)
        _save_loss_plot(output_dir / "loss_plot.png", rows)
        stale_epochs = 0 if improved else stale_epochs + 1
        if stale_epochs >= config.training.early_stopping_patience:
            break
    return {
        "checkpoint_dir": str(output_dir),
        "epochs_completed": len(rows),
        "global_step": global_step,
        "best_metric": best,
        "mixed_precision": amp_enabled,
        "device": str(device),
        "oom_actions": list(oom.actions),
        "history": rows,
    }
