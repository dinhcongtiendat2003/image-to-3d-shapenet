"""Tiny batch-1 forward/backward/overfit smoke test for the mesh refiner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

from image_to_3d_shapenet import load_config  # noqa: E402
from image_to_3d_shapenet.losses import EdgeAwareRefinementLoss  # noqa: E402
from image_to_3d_shapenet.mesh import save_refinement_debug  # noqa: E402
from image_to_3d_shapenet.models import ImageConditionedMeshRefiner  # noqa: E402
from image_to_3d_shapenet.utils import set_random_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quick_test.yaml")
    parser.add_argument("--output", default="outputs/quick_test/refiner")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    set_random_seed(config.runtime.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config.runtime.mixed_precision and device.type == "cuda")

    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices = torch.tensor(np.asarray(cube.vertices).copy(), dtype=torch.float32, device=device).unsqueeze(0)
    faces = torch.tensor(np.asarray(cube.faces).copy(), dtype=torch.long, device=device)
    target = vertices * torch.tensor((1.02, 0.98, 1.01), device=device)
    size = config.runtime.image_size
    image = torch.zeros((1, 3, size, size), device=device)
    image[:, :, size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 0.7
    intrinsics = torch.tensor(
        [[[100.0, 0.0, (size - 1) / 2], [0.0, 100.0, (size - 1) / 2], [0.0, 0.0, 1.0]]],
        device=device,
    )
    extrinsics = torch.eye(4, device=device).unsqueeze(0)
    extrinsics[:, 2, 3] = -3.0
    model = ImageConditionedMeshRefiner(config.refinement).to(device)
    criterion = EdgeAwareRefinementLoss(config.refinement)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    losses = []
    output = None
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(image, vertices, faces, intrinsics, extrinsics)
            loss_output = criterion(vertices, output.refined_vertices, faces, target, faces)
        if not torch.isfinite(loss_output.total):
            raise RuntimeError("non-finite refinement loss")
        scaler.scale(loss_output.total).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss_output.total.detach().cpu()))
    assert output is not None
    files = save_refinement_debug(
        vertices,
        output.refined_vertices,
        faces,
        PROJECT_ROOT / args.output,
        sharp_threshold_degrees=config.refinement.sharp_edge_threshold_degrees,
    )
    report = {
        "device": str(device),
        "mixed_precision": amp_enabled,
        "steps": args.steps,
        "losses": losses,
        "max_abs_displacement": float(output.displacement.detach().abs().max().cpu()),
        "sharp_edges": int(output.geometry.sharp_edges.sum().item()),
        "files": {name: str(path) for name, path in files.items()},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
