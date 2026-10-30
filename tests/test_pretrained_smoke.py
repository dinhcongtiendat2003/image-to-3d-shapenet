"""Opt-in smoke test: a real pretrained model must return a non-empty mesh."""

import os

import pytest
from PIL import Image

from image_to_3d_shapenet.config import InferenceConfig
from image_to_3d_shapenet.inference import AutoBackend, InferenceOptions
from image_to_3d_shapenet.utils import cuda_available


@pytest.mark.skipif(
    os.getenv("RUN_PRETRAINED_SMOKE") != "1" or not cuda_available(),
    reason="requires RUN_PRETRAINED_SMOKE=1, CUDA, official backend code, and weights",
)
def test_real_pretrained_backend_generates_mesh() -> None:
    image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    from PIL import ImageDraw

    ImageDraw.Draw(image).rounded_rectangle((55, 45, 200, 220), radius=20, fill="gray")
    backend = AutoBackend(InferenceConfig(remove_background=False))
    try:
        backend.load()
        result = backend.generate(backend.preprocess(image), InferenceOptions())
        assert result.vertex_count > 0
        assert result.face_count > 0
        assert result.backend_name in {"triposr", "stable-fast-3d", "hunyuan3d-2mini"}
    finally:
        backend.unload()
