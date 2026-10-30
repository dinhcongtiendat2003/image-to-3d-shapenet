"""Adapter for the official VAST-AI-Research/TripoSR implementation."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import (
    BackendUnavailableError,
    BaseImageTo3DBackend,
    GenerationResult,
    InferenceOptions,
)
from .preprocess import PreprocessedImage


class TripoSRBackend(BaseImageTo3DBackend):
    name = "triposr"

    def load(self) -> None:
        try:
            from tsr.system import TSR

            model = TSR.from_pretrained(
                self.config.triposr_model_id,
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            model.renderer.set_chunk_size(self.config.triposr_chunk_size)
            model.to(self.device)
            model.eval()
            self._model = model
        except Exception as error:
            self.unload()
            raise BackendUnavailableError(
                f"TripoSR load failed ({type(error).__name__}); install official tsr dependencies and weights"
            ) from None

    def generate(
        self,
        image: PreprocessedImage | Image.Image,
        options: InferenceOptions,
    ) -> GenerationResult:
        if not self.is_loaded:
            raise BackendUnavailableError("TripoSR is not loaded")
        import torch

        pil = image.image if isinstance(image, PreprocessedImage) else image.convert("RGBA")
        rgba = np.asarray(pil, dtype=np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:4] + 0.5 * (1.0 - rgba[..., 3:4])
        condition = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
        started = self._start_timer()
        with torch.no_grad():
            scene_codes = self._model([condition], device=self.device)
            meshes = self._model.extract_mesh(
                scene_codes,
                True,
                resolution=self.config.triposr_mc_resolution,
            )
        return self._finish_result(
            started,
            meshes[0],
            options=options,
            texture_status="vertex_colors",
            metadata={
                "model_id": self.config.triposr_model_id,
                "mc_resolution": self.config.triposr_mc_resolution,
                "chunk_size": self.config.triposr_chunk_size,
            },
        )

