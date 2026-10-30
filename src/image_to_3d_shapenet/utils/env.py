"""Environment variable access that never logs secret values."""

from __future__ import annotations

import os


def get_env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"required environment variable is not set: {name}")
    return value

