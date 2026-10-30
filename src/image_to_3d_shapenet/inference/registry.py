"""Production backend registry and environment/weight probes."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import HfApi, try_to_load_from_cache

from image_to_3d_shapenet.config import InferenceConfig
from image_to_3d_shapenet.utils import cuda_available, total_vram_gb

from .base import BaseImageTo3DBackend
from .hunyuan import Hunyuan3D2MiniBackend
from .stable_fast_3d import StableFast3DBackend
from .triposr import TripoSRBackend


@dataclass(frozen=True)
class BackendSpec:
    name: str
    backend_class: type[BaseImageTo3DBackend]
    module: str
    model_id: str
    minimum_vram_gb: float
    gated: bool = False
    high_quality_only: bool = False
    cache_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendProbe:
    name: str
    available: bool
    reasons: tuple[str, ...]


def _optional_hf_token() -> str | None:
    value = os.getenv("HF_TOKEN")
    if value and value.strip():
        return value.strip()
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        value = userdata.get("HF_TOKEN")
    except Exception:
        return None
    return str(value).strip() if value else None


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _activate_local_source(spec_name: str) -> None:
    """Expose an explicitly configured or project-cache upstream checkout."""
    locations = {
        "triposr": ("TRIPOSR_SOURCE_DIR", "TripoSR"),
        "stable-fast-3d": ("SF3D_SOURCE_DIR", "stable-fast-3d"),
        "hunyuan3d-2mini": ("HUNYUAN3D_SOURCE_DIR", "Hunyuan3D-2"),
    }
    if spec_name not in locations:
        return
    variable, directory = locations[spec_name]
    candidates = []
    configured = os.getenv(variable)
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path.cwd() / ".cache" / "backends" / directory)
    for candidate in candidates:
        if candidate.is_dir():
            resolved = str(candidate.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            return


def default_weight_probe(spec: BackendSpec, allow_download: bool) -> tuple[bool, str | None]:
    cached = all(
        isinstance(try_to_load_from_cache(spec.model_id, filename), str)
        for filename in spec.cache_files
    ) if spec.cache_files else False
    if cached:
        return True, None
    if not allow_download:
        return False, "weights are not cached and downloads are disabled"
    token = _optional_hf_token()
    if spec.gated and not token:
        return False, "gated weights require HF_TOKEN and accepted license"
    try:
        HfApi().model_info(spec.model_id, token=token)
    except Exception as error:
        return False, f"model access probe failed ({type(error).__name__})"
    return True, None


WeightProbe = Callable[[BackendSpec, bool], tuple[bool, str | None]]


class BackendRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, BackendSpec] = {}

    def register(self, spec: BackendSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"backend already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> BackendSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(f"unknown backend: {name}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def probe(
        self,
        name: str,
        config: InferenceConfig,
        *,
        weight_probe: WeightProbe = default_weight_probe,
    ) -> BackendProbe:
        spec = self.get(name)
        reasons: list[str] = []
        _activate_local_source(spec.name)
        has_cuda = cuda_available()
        vram = total_vram_gb()
        if spec.high_quality_only and config.mode != "high_quality":
            reasons.append("available only in high_quality mode")
        if not _module_available(spec.module):
            reasons.append(f"missing dependency module: {spec.module}")
        if not has_cuda and (not config.allow_cpu or spec.name == "hunyuan3d-2mini"):
            reasons.append("CUDA is unavailable")
        if has_cuda and vram < spec.minimum_vram_gb:
            reasons.append(
                f"requires at least {spec.minimum_vram_gb:.1f} GiB VRAM; found {vram:.1f}"
            )
        if not reasons:
            weights_ok, weight_reason = weight_probe(spec, config.allow_download)
            if not weights_ok:
                reasons.append(weight_reason or "weights unavailable")
        return BackendProbe(name=name, available=not reasons, reasons=tuple(reasons))


def create_default_registry(config: InferenceConfig) -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(
        BackendSpec(
            name="triposr",
            backend_class=TripoSRBackend,
            module="tsr.system",
            model_id=config.triposr_model_id,
            minimum_vram_gb=5.5,
            cache_files=("config.yaml", "model.ckpt"),
        )
    )
    registry.register(
        BackendSpec(
            name="stable-fast-3d",
            backend_class=StableFast3DBackend,
            module="sf3d.system",
            model_id=config.stable_fast_3d_model_id,
            minimum_vram_gb=6.0,
            gated=True,
            cache_files=("config.yaml", "model.safetensors"),
        )
    )
    registry.register(
        BackendSpec(
            name="hunyuan3d-2mini",
            backend_class=Hunyuan3D2MiniBackend,
            module="hy3dgen.shapegen",
            model_id=config.hunyuan_model_id,
            minimum_vram_gb=8.0,
            gated=True,
            high_quality_only=True,
        )
    )
    return registry

