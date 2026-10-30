"""Pretrained single-image-to-3D inference backends."""

from .auto import AutoBackend
from .base import (
    BackendGenerationError,
    BackendUnavailableError,
    BaseImageTo3DBackend,
    GenerationResult,
    InferenceOptions,
)
from .hunyuan import Hunyuan3D2MiniBackend
from .pipeline import run_inference
from .preprocess import PreprocessedImage, preprocess_image
from .registry import BackendRegistry, BackendSpec, create_default_registry
from .stable_fast_3d import StableFast3DBackend
from .triposr import TripoSRBackend

__all__ = [
    "AutoBackend",
    "BackendGenerationError",
    "BackendRegistry",
    "BackendSpec",
    "BackendUnavailableError",
    "BaseImageTo3DBackend",
    "GenerationResult",
    "Hunyuan3D2MiniBackend",
    "InferenceOptions",
    "PreprocessedImage",
    "StableFast3DBackend",
    "TripoSRBackend",
    "create_default_registry",
    "preprocess_image",
    "run_inference",
]


