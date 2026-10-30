"""Generate a real base mesh from one image using a pretrained backend."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_to_3d_shapenet import load_config  # noqa: E402
from image_to_3d_shapenet.inference import run_inference  # noqa: E402
from image_to_3d_shapenet.utils import cuda_available, gpu_name, total_vram_gb  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "triposr", "stable-fast-3d", "hunyuan3d-2mini"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--high-quality", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    project = load_config(config_path)
    inference = replace(
        project.inference,
        backend=args.backend,
        mode="high_quality" if args.high_quality else project.inference.mode,
        allow_cpu=args.allow_cpu or project.inference.allow_cpu,
        allow_download=not args.local_files_only,
    )
    print(
        json.dumps(
            {
                "cuda": cuda_available(),
                "gpu": gpu_name(),
                "total_vram_gb": total_vram_gb(),
                "requested_backend": args.backend,
                "mode": inference.mode,
            }
        )
    )
    try:
        result = run_inference(
            args.image,
            args.output,
            inference,
            requested_backend=args.backend,
        )
    except Exception as error:
        print(f"Inference thất bại ({type(error).__name__}): {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

