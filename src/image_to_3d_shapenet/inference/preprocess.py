"""Aspect-preserving image preprocessing shared by real image-to-3D backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


class ImagePreprocessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreprocessedImage:
    image: Image.Image
    original_size: tuple[int, int]
    foreground_bbox: tuple[int, int, int, int]
    scale: float
    background_removed: bool


def _open_image(source: str | Path | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.copy()
    path = Path(source)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ImagePreprocessingError("input must be PNG, JPG, or JPEG")
    if not path.is_file():
        raise FileNotFoundError(f"input image not found: {path}")
    with Image.open(path) as opened:
        return opened.convert("RGBA")


def preprocess_image(
    source: str | Path | Image.Image,
    *,
    size: int = 512,
    foreground_ratio: float = 0.85,
    remove_background: bool = True,
    remover: Callable[[Image.Image], Image.Image] | None = None,
) -> PreprocessedImage:
    """Remove background, crop alpha, pad square, center, and preserve aspect ratio."""
    image = _open_image(source).convert("RGBA")
    original_size = image.size
    alpha = np.asarray(image.getchannel("A"))
    has_transparency = bool(np.any(alpha < 250))
    removed = False
    if remove_background and not has_transparency:
        if remover is None:
            try:
                from rembg import remove
            except ImportError:
                raise ImagePreprocessingError(
                    "rembg is required for opaque PNG/JPG background removal"
                ) from None
            remover = remove
        output = remover(image)
        image = output if isinstance(output, Image.Image) else Image.open(output)
        image = image.convert("RGBA")
        removed = True

    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.nonzero(alpha > 8)
    if len(xs) == 0:
        raise ImagePreprocessingError("background removal produced an empty foreground")
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    cropped = image.crop(bbox)
    target_foreground = max(1, int(round(size * foreground_ratio)))
    scale = min(target_foreground / cropped.width, target_foreground / cropped.height)
    resized_size = (
        max(1, int(round(cropped.width * scale))),
        max(1, int(round(cropped.height * scale))),
    )
    resized = cropped.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset = ((size - resized.width) // 2, (size - resized.height) // 2)
    canvas.alpha_composite(resized, dest=offset)
    return PreprocessedImage(
        image=canvas,
        original_size=original_size,
        foreground_bbox=bbox,
        scale=scale,
        background_removed=removed,
    )

