"""Evaluate cached base meshes and edge-aware refiner checkpoints sequentially."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_to_3d_shapenet import load_config  # noqa: E402
from image_to_3d_shapenet.metrics import evaluate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/chair_colab.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base-refiner-checkpoint", default=None)
    parser.add_argument("--no-edge-checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    report = evaluate(
        load_config(config_path), PROJECT_ROOT,
        checkpoint=args.checkpoint,
        base_refiner_checkpoint=args.base_refiner_checkpoint,
        no_edge_checkpoint=args.no_edge_checkpoint,
        output_dir=args.output,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
