from pathlib import Path

import pytest

from image_to_3d_shapenet import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["quick_test.yaml", "chair_colab.yaml", "inference.yaml"])
def test_all_configs_load(name: str) -> None:
    config = load_config(ROOT / "configs" / name)
    assert config.project_name == "image-to-3d-shapenet"
    assert config.runtime.batch_size == 1
    assert config.runtime.mixed_precision is True
    assert config.export.formats == ("obj", "glb", "stl")
    assert config.render.backend in {"auto", "pyrender", "software"}
    assert config.inference.backend in {
        "auto", "triposr", "stable-fast-3d", "hunyuan3d-2mini"
    }
    assert config.inference.geometry_only is True
    assert 2 <= config.refinement.num_blocks <= 4
    assert config.refinement.max_displacement > 0
    assert config.refinement.sharp_edge_threshold_degrees == 35.0
    assert config.runtime.batch_size == 1
    assert config.training.gradient_accumulation_steps > 0
    assert 2 <= config.training.quick_samples <= 5
    assert config.evaluation.fscore_thresholds == (0.01, 0.02)
    assert config.evaluation.max_test_samples > 0
    assert config.mesh_postprocess.sharp_edge_threshold_degrees == 35.0
    assert config.mesh_postprocess.smooth is False
    assert config.app.preset == "Colab Free"
    assert config.app.default_backend == "triposr"
    assert config.app.default_geometry_only is True


def test_chair_colab_required_values() -> None:
    config = load_config(ROOT / "configs" / "chair_colab.yaml")
    assert config.dataset.repo_id == "ShapeNet/ShapeNetCore"
    assert config.dataset.category_name == "chair"
    assert config.dataset.synset_id == "03001627"
    assert config.dataset.zip_filename == "03001627.zip"
    assert config.runtime.image_size == 256
    assert config.dataset.views_per_mesh == 8
    assert config.runtime.seed == 42
    assert config.dataset.max_train_meshes == 70
    assert config.dataset.max_val_meshes == 15
    assert config.dataset.max_test_meshes == 15
    assert config.runtime.image_size == 256
    assert config.render.max_render_meshes == 100


def test_invalid_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("runtime:\n  batch_size: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="batch_size"):
        load_config(path)

