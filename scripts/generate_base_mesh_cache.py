"""Generate or resume the per-render base-mesh cache."""

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
from image_to_3d_shapenet.dataset import generate_base_mesh_cache  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/chair_colab.yaml")
    parser.add_argument(
        "--backend", default=None,
        choices=("auto", "triposr", "stable-fast-3d", "hunyuan3d-2mini"),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    report = generate_base_mesh_cache(
        load_config(config_path), PROJECT_ROOT,
        backend=args.backend, maximum=args.max_samples,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
