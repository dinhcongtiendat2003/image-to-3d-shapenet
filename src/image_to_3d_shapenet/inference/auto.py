"""Automatic backend selection and failure-safe sequential fallback."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from image_to_3d_shapenet.config import InferenceConfig

from .base import (
    BackendUnavailableError,
    BaseImageTo3DBackend,
    GenerationResult,
    InferenceOptions,
)
from .preprocess import PreprocessedImage, preprocess_image
from .registry import (
    BackendRegistry,
    WeightProbe,
    create_default_registry,
    default_weight_probe,
)


class AutoBackend(BaseImageTo3DBackend):
    name = "auto"

    def __init__(
        self,
        config: InferenceConfig,
        *,
        requested: str | None = None,
        registry: BackendRegistry | None = None,
        weight_probe: WeightProbe = default_weight_probe,
    ) -> None:
        device = "cuda" if self._cuda_available() else "cpu"
        super().__init__(config, device)
        self.requested = requested or config.backend
        self.registry = registry or create_default_registry(config)
        self.weight_probe = weight_probe
        self._active: BaseImageTo3DBackend | None = None
        self._attempts: list[str] = []
        self._candidates = self._candidate_order()
        self._candidate_index = -1

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except (ImportError, OSError):
            return False

    def _candidate_order(self) -> list[str]:
        if self.requested != "auto":
            order = [self.requested, "triposr", "stable-fast-3d"]
        elif self.config.mode == "high_quality":
            order = ["hunyuan3d-2mini", "triposr", "stable-fast-3d"]
        else:
            order = ["triposr", "stable-fast-3d"]
        return list(dict.fromkeys(name for name in order if name in self.registry.names()))

    def _load_next(self) -> None:
        if self._active is not None:
            self._active.unload()
            self._active = None
        for index in range(self._candidate_index + 1, len(self._candidates)):
            name = self._candidates[index]
            self._candidate_index = index
            probe = self.registry.probe(
                name,
                self.config,
                weight_probe=self.weight_probe,
            )
            if not probe.available:
                self._attempts.append(f"{name}: {'; '.join(probe.reasons)}")
                continue
            spec = self.registry.get(name)
            candidate = spec.backend_class(self.config, self.device)
            try:
                candidate.load()
            except Exception as error:
                candidate.unload()
                self._attempts.append(f"{name}: load failed ({type(error).__name__})")
                continue
            self._active = candidate
            return
        detail = " | ".join(self._attempts) or "no registered candidates"
        raise BackendUnavailableError(f"No real image-to-3D backend is available. {detail}")

    def load(self) -> None:
        if self._active is None:
            self._load_next()

    @property
    def is_loaded(self) -> bool:
        return self._active is not None and self._active.is_loaded

    def preprocess(self, image: str | Path | Image.Image) -> PreprocessedImage:
        return preprocess_image(
            image,
            size=self.config.input_size,
            foreground_ratio=self.config.foreground_ratio,
            remove_background=self.config.remove_background,
        )

    def generate(
        self,
        image: PreprocessedImage | Image.Image,
        options: InferenceOptions,
    ) -> GenerationResult:
        if self._active is None:
            self.load()
        while self._active is not None:
            active_name = self._active.name
            try:
                result = self._active.generate(image, options)
                if self._attempts:
                    result = replace(
                        result,
                        warnings=(*result.warnings, *tuple(self._attempts)),
                    )
                return result
            except Exception as error:
                self._attempts.append(
                    f"{active_name}: generation failed ({type(error).__name__})"
                )
                self._load_next()
        raise BackendUnavailableError("all real backends failed")

    def unload(self) -> None:
        if self._active is not None:
            self._active.unload()
            self._active = None

    def get_memory_usage(self) -> dict[str, float]:
        return self._active.get_memory_usage() if self._active else {"peak_vram_gb": 0.0}

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requested": self.requested,
            "selected": self._active.name if self._active else None,
            "attempts": tuple(self._attempts),
            "device": self.device,
        }

