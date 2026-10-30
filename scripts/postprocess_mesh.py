#!/usr/bin/env python
"""Repair and export one mesh without requiring optional native extensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_to_3d_shapenet import load_config
from image_to_3d_shapenet.mesh import postprocess_mesh_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Sharp-edge-aware mesh post-processing")
    parser.add_argument("--input", required=True, help="Input mesh readable by trimesh")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="configs/inference.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    result = postprocess_mesh_file(args.input, args.output, config.mesh_postprocess)
    print(json.dumps({
        "output_dir": result["output_dir"],
        "before": result["report"]["before"],
        "after": result["report"]["after"],
        "warnings": result["report"]["warnings"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
