"""Atomic, resumable training checkpoints including reproducibility state."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from image_to_3d_shapenet.config import ProjectConfig


def _random_state() -> dict[str, Any]:
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["cuda"] = torch.cuda.get_rng_state_all()
    return value


def _restore_random_state(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and "cuda" in value:
        torch.cuda.set_rng_state_all(value["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_metric: dict[str, float],
    config: ProjectConfig,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(
        {
            "version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_metric": dict(best_metric),
            "config": asdict(config),
            "random_state": _random_state(),
        },
        temporary,
    )
    os.replace(temporary, target)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_random: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_random and "random_state" in checkpoint:
        _restore_random_state(checkpoint["random_state"])
    return checkpoint
