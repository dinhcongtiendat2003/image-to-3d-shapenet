"""Lightweight smoke coverage for the default Colab-free configuration path."""

from pathlib import Path

from image_to_3d_shapenet import load_config
from image_to_3d_shapenet.utils import ProjectPaths, cuda_available, set_random_seed


def test_quick_config_cpu_fallback_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "quick_test.yaml")
    set_random_seed(config.runtime.seed)
    paths = ProjectPaths.from_config(tmp_path, config.paths).create()

    assert config.runtime.quick_test is True
    assert config.runtime.batch_size == 1
    assert config.runtime.mixed_precision is True
    assert config.dataset.max_meshes == 3
    assert isinstance(cuda_available(), bool)
    assert paths.outputs.is_dir()

