import logging
import random
from pathlib import Path

import pytest

from image_to_3d_shapenet.config import PathsConfig
from image_to_3d_shapenet.utils import (
    ProjectPaths,
    configure_logging,
    cuda_available,
    get_env,
    gpu_name,
    log_peak_vram,
    peak_vram_gb,
    reset_peak_vram,
    set_random_seed,
    total_vram_gb,
)


def test_seed_reproducible() -> None:
    set_random_seed(42)
    first = random.random()
    set_random_seed(42)
    assert random.random() == first


def test_environment_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROJECT_TEST_VALUE", "available")
    assert get_env("PROJECT_TEST_VALUE") == "available"
    monkeypatch.delenv("PROJECT_TEST_VALUE")
    with pytest.raises(RuntimeError, match="PROJECT_TEST_VALUE"):
        get_env("PROJECT_TEST_VALUE", required=True)


def test_project_paths_create_relative_directories(tmp_path: Path) -> None:
    paths = ProjectPaths.from_config(tmp_path, PathsConfig()).create()
    assert paths.root == tmp_path.resolve()
    assert paths.data.is_dir()
    assert paths.cache.is_dir()
    assert paths.outputs.is_dir()


def test_system_utilities_have_cpu_fallback(caplog: pytest.LogCaptureFixture) -> None:
    available = cuda_available()
    assert isinstance(available, bool)
    assert gpu_name() is None or isinstance(gpu_name(), str)
    assert total_vram_gb() >= 0.0
    reset_peak_vram()
    assert peak_vram_gb() >= 0.0
    with caplog.at_level(logging.INFO):
        assert log_peak_vram() >= 0.0


def test_logging_configuration_is_idempotent() -> None:
    first = configure_logging("DEBUG")
    count = len(first.handlers)
    second = configure_logging("INFO")
    assert first is second
    assert len(second.handlers) == count

