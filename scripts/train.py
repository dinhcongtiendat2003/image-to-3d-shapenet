"""Train or resume the edge-aware mesh refiner."""

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
from image_to_3d_shapenet.training import train_refiner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/chair_colab.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--quick-overfit", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    report = train_refiner(
        load_config(config_path), PROJECT_ROOT, resume=args.resume,
        checkpoint_dir=args.checkpoint_dir, quick_overfit=args.quick_overfit,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
