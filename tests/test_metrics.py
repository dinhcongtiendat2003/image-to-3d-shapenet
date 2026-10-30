from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from image_to_3d_shapenet import load_config
from image_to_3d_shapenet.metrics import (
    MetricInputError,
    compute_mesh_metrics,
    evaluate,
    silhouette_iou,
    write_incremental_reports,
)


ROOT = Path(__file__).resolve().parents[1]


def cube() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


def metric(prediction: trimesh.Trimesh):
    mask = np.ones((16, 16), dtype=np.uint8)
    return compute_mesh_metrics(
        prediction,
        cube(),
        surface_samples=256,
        edge_samples=96,
        predicted_silhouette=mask,
        target_silhouette=mask,
    )


def test_identical_cube_has_near_optimal_metrics() -> None:
    result = metric(cube())
    assert result.chamfer_distance == pytest.approx(0.0, abs=1e-7)
    assert result.fscore_1_percent == pytest.approx(1.0)
    assert result.fscore_2_percent == pytest.approx(1.0)
    assert result.normal_consistency == pytest.approx(1.0, abs=1e-6)
    assert result.silhouette_iou == pytest.approx(1.0)
    assert result.edge_chamfer_distance == pytest.approx(0.0, abs=1e-7)
    assert result.dihedral_angle_error_degrees == pytest.approx(0.0, abs=1e-5)
    assert result.vertex_count == 8
    assert result.face_count == 12
    assert result.watertight is True


def test_shifted_and_scaled_cubes_are_worse() -> None:
    identical = metric(cube())
    shifted = cube()
    shifted.apply_translation((0.1, 0.0, 0.0))
    scaled = cube()
    scaled.apply_scale(1.2)
    shifted_result = metric(shifted)
    scaled_result = metric(scaled)
    assert shifted_result.chamfer_distance > identical.chamfer_distance
    assert scaled_result.chamfer_distance > identical.chamfer_distance
    assert shifted_result.fscore_1_percent < identical.fscore_1_percent
    assert scaled_result.fscore_2_percent < identical.fscore_2_percent


def test_empty_and_malformed_meshes_are_rejected() -> None:
    empty = trimesh.Trimesh(vertices=[], faces=[], process=False)
    with pytest.raises(MetricInputError, match="empty"):
        compute_mesh_metrics(empty, cube(), surface_samples=16, edge_samples=8)
    malformed = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    with pytest.raises(MetricInputError, match="malformed"):
        compute_mesh_metrics(malformed, cube(), surface_samples=16, edge_samples=8)


def test_silhouette_iou() -> None:
    mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    other = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    assert silhouette_iou(mask, mask) == 1.0
    assert silhouette_iou(mask, other) == pytest.approx(1 / 3)


def test_reports_include_markdown_latex_json_csv_and_plots(tmp_path: Path) -> None:
    result = metric(cube()).to_dict()
    row = {
        "sample_id": "cube_view_000", "model_id": "cube", "view_id": 0,
        "category_id": "03001627", "category_name": "chair",
        "configuration": "refiner_full", "backend": "triposr",
        "model_version": "test", **result,
    }
    summary = write_incremental_reports(tmp_path, [row], [], {"test": True})
    assert any(item["level"] == "model" for item in summary)
    for name in (
        "metrics_per_sample.csv", "metrics_summary.csv", "metrics.json",
        "benchmark_table.md", "benchmark_table.tex", "failures.json",
    ):
        assert (tmp_path / name).is_file()
    assert "| Config |" in (tmp_path / "benchmark_table.md").read_text(encoding="utf-8")
    assert "\\begin{tabular}" in (tmp_path / "benchmark_table.tex").read_text(encoding="utf-8")
    assert list((tmp_path / "plots").glob("*.png"))


def test_evaluator_records_bad_samples_without_aborting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenDataset:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 2

        def __getitem__(self, index):
            raise ValueError(f"broken mesh {index}")

    monkeypatch.setattr(
        "image_to_3d_shapenet.metrics.evaluator.RenderedRefinementDataset",
        BrokenDataset,
    )
    config = load_config(ROOT / "configs" / "quick_test.yaml")
    config = replace(
        config,
        paths=replace(config.paths, data_dir="data", cache_dir="cache", output_dir="outputs"),
        evaluation=replace(config.evaluation, configurations=("base",)),
    )
    output = tmp_path / "evaluation"
    report = evaluate(config, tmp_path, output_dir=output)
    assert report["successful_samples"] == 0
    assert report["failures"] == 2
    failures = json.loads((output / "failures.json").read_text(encoding="utf-8"))
    assert len(failures) == 2
    assert all(item["error"] == "ValueError" for item in failures)
