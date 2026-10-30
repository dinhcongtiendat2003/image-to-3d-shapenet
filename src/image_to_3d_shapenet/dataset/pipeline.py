"""End-to-end ShapeNetCore category preparation without rendering or models."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from image_to_3d_shapenet.config import ProjectConfig
from image_to_3d_shapenet.utils import ProjectPaths

from .archive import safe_extract_zip
from .download import download_category
from .meshes import (
    find_normalized_meshes,
    model_id_from_path,
    save_normalized_mesh,
    validate_mesh,
)
from .splits import create_splits, save_splits
from .synthetic import create_synthetic_meshes


def _limit_sources(paths: list[Path], maximum: int | None, seed: int) -> list[Path]:
    ordered = sorted(paths)
    if maximum is None or len(ordered) <= maximum:
        return ordered
    random.Random(seed).shuffle(ordered)
    return sorted(ordered[:maximum])


def prepare_dataset(
    config: ProjectConfig,
    project_root: str | Path,
    *,
    synthetic: bool | None = None,
) -> dict[str, Any]:
    """Prepare real data or an offline synthetic quick test sequentially."""
    paths = ProjectPaths.from_config(project_root, config.paths).create()
    use_synthetic = config.runtime.quick_test if synthetic is None else synthetic
    if use_synthetic:
        source_root = paths.data / "synthetic" / config.dataset.synset_id
        create_synthetic_meshes(source_root)
        source_kind = "synthetic"
    else:
        archive = download_category(config.dataset, paths.cache / "huggingface")
        source_root = paths.data / "raw" / config.dataset.synset_id
        safe_extract_zip(
            archive,
            source_root,
            max_extracted_bytes=int(config.dataset.max_extracted_size_gb * 1024**3),
            max_members=config.dataset.max_archive_members,
            min_free_bytes=int(config.dataset.min_free_disk_gb * 1024**3),
        )
        source_kind = "shapenetcore"

    discovered = _limit_sources(
        find_normalized_meshes(source_root),
        config.dataset.max_meshes,
        config.runtime.seed,
    )
    if not discovered:
        raise RuntimeError(f"Không tìm thấy model_normalized.obj trong {source_root}")

    processed_root = paths.data / "processed" / config.dataset.synset_id
    validations: list[dict[str, Any]] = []
    valid_ids: list[str] = []
    normalized_count = 0
    for source in discovered:
        validation = validate_mesh(
            source,
            max_degenerate_ratio=config.dataset.max_degenerate_face_ratio,
            max_file_bytes=int(config.dataset.max_mesh_file_mb * 1024**2),
        )
        validations.append(validation.to_dict())
        if not validation.valid:
            continue
        model_id = model_id_from_path(source)
        if model_id in valid_ids:
            raise ValueError(f"duplicate model ID: {model_id}")
        model_root = processed_root / model_id
        output_mesh = model_root / "model_normalized.obj"
        transform = model_root / "transform.json"
        if not (output_mesh.is_file() and transform.is_file()):
            save_normalized_mesh(
                source,
                output_mesh,
                transform,
                mode=config.dataset.normalization,
            )
            normalized_count += 1
        valid_ids.append(model_id)

    invalid = [item for item in validations if not item["valid"]]
    invalid_path = paths.outputs / "invalid_meshes.json"
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.write_text(json.dumps(invalid, indent=2), encoding="utf-8")

    splits = create_splits(
        valid_ids,
        seed=config.runtime.seed,
        max_train=config.dataset.max_train_meshes,
        max_val=config.dataset.max_val_meshes,
        max_test=config.dataset.max_test_meshes,
    )
    split_path = save_splits(splits, paths.outputs / "splits.json", seed=config.runtime.seed)
    report = {
        "source": source_kind,
        "category_name": config.dataset.category_name,
        "synset_id": config.dataset.synset_id,
        "archive_filename": None if use_synthetic else config.dataset.zip_filename,
        "discovered_meshes": len(discovered),
        "valid_meshes": len(valid_ids),
        "invalid_meshes": len(invalid),
        "normalized_this_run": normalized_count,
        "split_counts": {name: len(values) for name, values in splits.items()},
        "split_file": str(split_path),
        "invalid_file": str(invalid_path),
    }
    report_path = paths.outputs / "dataset_preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

