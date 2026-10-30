from __future__ import annotations

from dataclasses import replace

import pytest
import trimesh
from PIL import Image

from image_to_3d_shapenet.config import InferenceConfig
from image_to_3d_shapenet.inference import AutoBackend
from image_to_3d_shapenet.inference.base import (
    BackendGenerationError,
    BaseImageTo3DBackend,
    InferenceOptions,
)
from image_to_3d_shapenet.inference.registry import (
    BackendRegistry,
    BackendSpec,
    create_default_registry,
)


class _SuccessBackend(BaseImageTo3DBackend):
    name = "triposr"
    unloaded = False

    def load(self) -> None:
        self._model = object()

    def generate(self, image, options):
        started = self._start_timer()
        return self._finish_result(
            started,
            trimesh.creation.box(),
            options=options,
            texture_status="unit-test",
        )

    def unload(self) -> None:
        type(self).unloaded = True
        super().unload()


class _LoadFailureBackend(_SuccessBackend):
    name = "triposr"

    def load(self) -> None:
        raise RuntimeError("intentional test failure")


class _GenerationFailureBackend(_SuccessBackend):
    name = "triposr"

    def generate(self, image, options):
        raise BackendGenerationError("intentional test failure")


class _StableSuccessBackend(_SuccessBackend):
    name = "stable-fast-3d"


class _HunyuanSuccessBackend(_SuccessBackend):
    name = "hunyuan3d-2mini"


def _registry(first= _SuccessBackend, second=_StableSuccessBackend) -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(BackendSpec("triposr", first, "json", "test/one", 0.0))
    registry.register(BackendSpec("stable-fast-3d", second, "json", "test/two", 0.0))
    registry.register(
        BackendSpec(
            "hunyuan3d-2mini",
            _HunyuanSuccessBackend,
            "json",
            "test/three",
            0.0,
            high_quality_only=True,
        )
    )
    return registry


def _weights(*_args) -> tuple[bool, None]:
    return True, None


def test_production_registry_contains_only_real_backend_names() -> None:
    names = create_default_registry(InferenceConfig()).names()
    assert names == ("triposr", "stable-fast-3d", "hunyuan3d-2mini")
    assert not any("fake" in name for name in names)


def test_registry_rejects_duplicate_name() -> None:
    registry = _registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(BackendSpec("triposr", _SuccessBackend, "json", "x", 0.0))


def test_auto_selects_triposr_and_unloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("image_to_3d_shapenet.inference.registry.cuda_available", lambda: False)
    config = replace(InferenceConfig(), allow_cpu=True, remove_background=False)
    auto = AutoBackend(config, registry=_registry(), weight_probe=_weights)
    auto.load()
    assert auto.get_metadata()["selected"] == "triposr"
    result = auto.generate(Image.new("RGBA", (16, 16), "white"), InferenceOptions())
    assert result.backend_name == "triposr"
    assert result.vertex_count == 8
    auto.unload()
    assert auto.is_loaded is False


def test_auto_falls_back_after_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("image_to_3d_shapenet.inference.registry.cuda_available", lambda: False)
    config = replace(InferenceConfig(), allow_cpu=True)
    auto = AutoBackend(
        config,
        registry=_registry(first=_LoadFailureBackend),
        weight_probe=_weights,
    )
    auto.load()
    assert auto.get_metadata()["selected"] == "stable-fast-3d"
    assert "load failed" in auto.get_metadata()["attempts"][0]
    auto.unload()


def test_auto_falls_back_after_generation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("image_to_3d_shapenet.inference.registry.cuda_available", lambda: False)
    config = replace(InferenceConfig(), allow_cpu=True)
    auto = AutoBackend(
        config,
        registry=_registry(first=_GenerationFailureBackend),
        weight_probe=_weights,
    )
    auto.load()
    result = auto.generate(Image.new("RGBA", (16, 16)), InferenceOptions())
    assert result.backend_name == "stable-fast-3d"
    assert any("generation failed" in warning for warning in result.warnings)
    auto.unload()


def test_high_quality_prefers_hunyuan_when_environment_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("image_to_3d_shapenet.inference.registry.cuda_available", lambda: True)
    monkeypatch.setattr("image_to_3d_shapenet.inference.registry.total_vram_gb", lambda: 16.0)
    config = replace(InferenceConfig(), mode="high_quality")
    auto = AutoBackend(config, registry=_registry(), weight_probe=_weights)
    auto.load()
    assert auto.get_metadata()["selected"] == "hunyuan3d-2mini"
    auto.unload()
