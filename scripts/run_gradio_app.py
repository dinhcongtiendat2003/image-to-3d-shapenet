#!/usr/bin/env python
"""Launch the Colab-first Gradio image-to-3D application."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_to_3d_shapenet.app.gradio_app import build_app, create_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the image-to-3D Gradio UI")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--share", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--server-name", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    service = create_service(config_path)
    demo = build_app(config_path, service=service)
    demo.launch(
        server_name=args.server_name or service.config.app.server_name,
        server_port=args.server_port or service.config.app.server_port,
        share=args.share,
        allowed_paths=[str(service.output_root)],
    )


if __name__ == "__main__":
    main()
