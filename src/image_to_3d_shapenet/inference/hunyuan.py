"""Shape-only adapter for Hunyuan3D-2mini-Turbo."""

from __future__ import annotations

from PIL import Image

from .base import (
    BackendUnavailableError,
    BaseImageTo3DBackend,
    GenerationResult,
    InferenceOptions,
)
from .preprocess import PreprocessedImage


class Hunyuan3D2MiniBackend(BaseImageTo3DBackend):
    name = "hunyuan3d-2mini"

    def load(self) -> None:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            self._model = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.config.hunyuan_model_id,
                subfolder=self.config.hunyuan_subfolder,
                use_safetensors=True,
                device=self.device,
            )
        except Exception as error:
            self.unload()
            raise BackendUnavailableError(
                f"Hunyuan3D load failed ({type(error).__name__}); adapter remains disabled and fallback is allowed"
            ) from None

    def generate(
        self,
        image: PreprocessedImage | Image.Image,
        options: InferenceOptions,
    ) -> GenerationResult:
        if not self.is_loaded:
            raise BackendUnavailableError("Hunyuan3D-2mini is not loaded")
        import torch

        pil = image.image if isinstance(image, PreprocessedImage) else image.convert("RGBA")
        generator = torch.Generator(device=self.device).manual_seed(options.seed)
        started = self._start_timer()
        with torch.no_grad():
            meshes = self._model(
                image=pil,
                num_inference_steps=self.config.hunyuan_steps,
                octree_resolution=self.config.hunyuan_octree_resolution,
                num_chunks=20000,
                generator=generator,
                output_type="trimesh",
            )
        return self._finish_result(
            started,
            meshes[0],
            options=options,
            texture_status="none",
            metadata={
                "model_id": self.config.hunyuan_model_id,
                "subfolder": self.config.hunyuan_subfolder,
                "steps": self.config.hunyuan_steps,
            },
        )

