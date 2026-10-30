"""Reproducibility helpers with optional NumPy and PyTorch integration."""

from __future__ import annotations

import os
import random


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except (ImportError, OSError):
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
    except (ImportError, OSError):
        pass

