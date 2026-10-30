from dataclasses import replace
from pathlib import Path

from image_to_3d_shapenet import load_config
from image_to_3d_shapenet.dataset import prepare_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_pipeline_runs_without_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config = load_config(ROOT / "configs" / "quick_test.yaml")
    config = replace(
        config,
        paths=replace(
            config.paths,
            data_dir="data",
            cache_dir="cache",
            output_dir="outputs",
        ),
    )
    report = prepare_dataset(config, tmp_path)
    assert report["source"] == "synthetic"
    assert report["discovered_meshes"] == 3
    assert report["valid_meshes"] == 2
    assert report["invalid_meshes"] == 1
    assert (tmp_path / "outputs" / "splits.json").is_file()
    assert (tmp_path / "outputs" / "invalid_meshes.json").is_file()
    assert len(list((tmp_path / "data" / "processed").rglob("transform.json"))) == 2

