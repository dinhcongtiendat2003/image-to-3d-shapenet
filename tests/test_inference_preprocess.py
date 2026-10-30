from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_to_3d_shapenet.inference.preprocess import (
    ImagePreprocessingError,
    preprocess_image,
)


def test_preprocess_removes_background_centers_and_preserves_aspect() -> None:
    source = Image.new("RGB", (120, 80), "white")
    source.paste((20, 40, 200), (20, 25, 100, 55))

    def remover(image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        pixels = np.asarray(rgba).copy()
        pixels[..., 3] = np.where(np.all(pixels[..., :3] > 240, axis=-1), 0, 255)
        return Image.fromarray(pixels)

    result = preprocess_image(
        source,
        size=100,
        foreground_ratio=0.8,
        remover=remover,
    )
    alpha = np.asarray(result.image.getchannel("A"))
    ys, xs = np.nonzero(alpha > 128)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    assert result.image.mode == "RGBA"
    assert result.image.size == (100, 100)
    assert result.background_removed is True
    assert abs(float(xs.mean()) - 49.5) < 1.0
    assert abs(float(ys.mean()) - 49.5) < 1.0
    assert width / height == pytest.approx(80 / 30, rel=0.08)


def test_existing_alpha_does_not_invoke_remover(tmp_path: Path) -> None:
    image = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (10, 5, 30, 15))
    path = tmp_path / "input.png"
    image.save(path)

    def should_not_run(_: Image.Image) -> Image.Image:
        raise AssertionError("remover should not run for an alpha image")

    result = preprocess_image(path, size=64, remover=should_not_run)
    assert result.background_removed is False
    assert result.foreground_bbox == (10, 5, 30, 15)


def test_preprocess_rejects_unsupported_file(tmp_path: Path) -> None:
    path = tmp_path / "input.bmp"
    Image.new("RGB", (8, 8)).save(path)
    with pytest.raises(ImagePreprocessingError, match="PNG"):
        preprocess_image(path)
