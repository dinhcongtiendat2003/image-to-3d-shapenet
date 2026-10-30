import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from image_to_3d_shapenet import load_config
from image_to_3d_shapenet.rendering import render_dataset


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path):
    config = load_config(ROOT / "configs" / "quick_test.yaml")
    return replace(
        config,
        dataset=replace(config.dataset, views_per_mesh=1),
        paths=replace(
            config.paths,
            data_dir="data",
            cache_dir="cache",
            output_dir="outputs",
        ),
        render=replace(config.render, backend="software", max_render_meshes=1),
    )


def test_render_pipeline_manifest_and_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = render_dataset(config, tmp_path)
    assert first["rendered"] == 1
    assert first["skipped"] == 0
    assert first["failed"] == 0

    manifest_path = Path(first["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["samples"]) == 1
    sample = next(iter(manifest["samples"].values()))
    assert sample["category_id"] == "03001627"
    assert sample["model_id"]
    assert sample["ground_truth_mesh"].endswith("model_normalized.obj")
    assert sample["backend"] == "software"
    sample_dir = Path(first["render_root"]) / sample["sample_directory"]
    for name in ("rgb.png", "rgba.png", "mask.png", "depth.npy", "normal.png", "camera.json", "sample.json"):
        assert (sample_dir / name).is_file()
    assert Image.open(sample_dir / "rgb.png").size == (128, 128)
    assert Image.open(sample_dir / "mask.png").getbbox() is not None
    assert np.isfinite(np.load(sample_dir / "depth.npy", allow_pickle=False)).all()

    second = render_dataset(config, tmp_path)
    assert second["rendered"] == 0
    assert second["skipped"] == 1
    manifest_again = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest_again["samples"]) == 1

    (sample_dir / "normal.png").unlink()
    repaired = render_dataset(config, tmp_path)
    assert repaired["rendered"] == 1
    assert repaired["skipped"] == 0
    assert (sample_dir / "normal.png").is_file()
    assert len(json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]) == 1

