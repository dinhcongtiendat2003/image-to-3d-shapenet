"""Deterministic, disjoint model-level dataset splits."""

from __future__ import annotations

import json
import random
from pathlib import Path


def create_splits(
    model_ids: list[str],
    *,
    seed: int = 42,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
) -> dict[str, list[str]]:
    unique = sorted(set(model_ids))
    if len(unique) != len(model_ids):
        raise ValueError("model IDs must be unique")
    random.Random(seed).shuffle(unique)
    train_end = int(len(unique) * 0.70)
    val_end = train_end + int(len(unique) * 0.15)
    result = {
        "train": unique[:train_end],
        "validation": unique[train_end:val_end],
        "test": unique[val_end:],
    }
    limits = {"train": max_train, "validation": max_val, "test": max_test}
    return {
        name: values[: limits[name]] if limits[name] is not None else values
        for name, values in result.items()
    }


def save_splits(splits: dict[str, list[str]], path: str | Path, *, seed: int) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15}, **splits}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target

