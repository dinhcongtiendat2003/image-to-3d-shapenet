"""Prepare one ShapeNetCore category or run the offline synthetic quick test."""

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
from image_to_3d_shapenet.dataset import prepare_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true", help="Never access ShapeNetCore")
    mode.add_argument("--real", action="store_true", help="Require gated ShapeNetCore access")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = load_config(config_path)
    selected_mode = True if args.synthetic else False if args.real else None
    try:
        report = prepare_dataset(config, PROJECT_ROOT, synthetic=selected_mode)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

