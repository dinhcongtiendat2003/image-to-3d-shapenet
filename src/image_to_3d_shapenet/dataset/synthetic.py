"""Small real mesh fixtures for offline quick tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def _write(mesh: trimesh.Trimesh, root: Path, model_id: str) -> Path:
    target = root / model_id / "models" / "model_normalized.obj"
    target.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(target, file_type="obj")
    return target


def create_synthetic_meshes(root: str | Path) -> list[Path]:
    """Create a cube, rectangular box, and mesh with a degenerate component."""
    target = Path(root)
    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
    faulty_vertices = np.vstack((cube.vertices, [[3.0, 3.0, 3.0]]))
    faulty_faces = np.vstack((cube.faces, [[8, 8, 8]]))
    faulty = trimesh.Trimesh(vertices=faulty_vertices, faces=faulty_faces, process=False)
    return [
        _write(cube, target, "synthetic_cube"),
        _write(box, target, "synthetic_box"),
        _write(faulty, target, "synthetic_faulty_component"),
    ]

