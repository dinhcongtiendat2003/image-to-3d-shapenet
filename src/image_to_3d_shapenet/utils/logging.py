"""Consistent application logging."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int | str = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("image_to_3d_shapenet")
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger

