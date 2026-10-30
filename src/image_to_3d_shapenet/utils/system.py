"""CUDA and GPU memory inspection with a safe CPU-only fallback."""

from __future__ import annotations

import logging
from typing import Any


def _torch() -> Any | None:
    try:
        import torch
    except (ImportError, OSError):
        return None
    return torch


def cuda_available() -> bool:
    torch = _torch()
    return bool(torch is not None and torch.cuda.is_available())


def gpu_name(device: int = 0) -> str | None:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(device))


def total_vram_gb(device: int = 0) -> float:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.get_device_properties(device).total_memory / 1024**3)


def reset_peak_vram(device: int = 0) -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_vram_gb(device: int = 0) -> float:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / 1024**3)


def log_peak_vram(logger: logging.Logger | None = None, device: int = 0) -> float:
    peak = peak_vram_gb(device)
    (logger or logging.getLogger(__name__)).info("Peak GPU VRAM: %.3f GiB", peak)
    return peak

