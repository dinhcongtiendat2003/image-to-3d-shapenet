"""Adapter for Stability AI's official gated Stable Fast 3D codebase."""

from __future__ import annotations

from PIL import Image

from .base import (
    BackendUnavailableError,
    BaseImageTo3DBackend,
    GenerationResult,
    InferenceOptions,
)
from .preprocess import PreprocessedImage


class StableFast3DBackend(BaseImageTo3DBackend):
    name = "stable-fast-3d"

    def load(self) -> None:
        try:
            from sf3d.system import SF3D

            model = SF3D.from_pretrained(
                self.config.stable_fast_3d_model_id,
                config_name="config.yaml",
                weight_name="model.safetensors",
            )
            model.to(self.device)
            model.eval()
            self._model = model
        except Exception as error:
            self.unload()
            raise BackendUnavailableError(
                f"Stable Fast 3D load failed ({type(error).__name__}); verify gated access and official dependencies"
            ) from None

    def generate(
        self,
        image: PreprocessedImage | Image.Image,
        options: InferenceOptions,
    ) -> GenerationResult:
        if not self.is_loaded:
            raise BackendUnavailableError("Stable Fast 3D is not loaded")
        import torch

        pil = image.image if isinstance(image, PreprocessedImage) else image.convert("RGBA")
        started = self._start_timer()
        with torch.no_grad():
            meshes, _ = self._model.run_image(
                [pil],
                bake_resolution=self.config.stable_fast_3d_texture_resolution,
                remesh="none",
                vertex_count=-1,
            )
        return self._finish_result(
            started,
            meshes,
            options=options,
            texture_status="uv_texture",
            metadata={"model_id": self.config.stable_fast_3d_model_id},
        )

