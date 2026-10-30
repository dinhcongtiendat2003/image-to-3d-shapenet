from pathlib import Path

import numpy as np
import trimesh

from image_to_3d_shapenet.dataset.meshes import normalize_mesh, validate_mesh
from image_to_3d_shapenet.dataset.synthetic import create_synthetic_meshes


def test_mesh_normalization_is_uniform_and_reversible() -> None:
    mesh = trimesh.creation.box(extents=(4.0, 2.0, 1.0))
    mesh.apply_translation((10.0, -3.0, 7.0))
    normalized, metadata = normalize_mesh(mesh, mode="unit_cube")
    extents = normalized.bounds[1] - normalized.bounds[0]
    assert np.isclose(extents.max(), 1.0)
    assert np.allclose(extents / extents.max(), [1.0, 0.5, 0.25])
    assert np.allclose(normalized.bounds.mean(axis=0), 0.0)
    forward = np.asarray(metadata["original_to_normalized"])
    inverse = np.asarray(metadata["normalized_to_original"])
    assert np.allclose(forward @ inverse, np.eye(4))


def test_faulty_synthetic_component_is_detected(tmp_path: Path) -> None:
    paths = create_synthetic_meshes(tmp_path)
    valid = [validate_mesh(path, max_degenerate_ratio=0.05) for path in paths]
    assert valid[0].valid is True
    assert valid[1].valid is True
    assert valid[2].valid is False
    assert "degenerate" in (valid[2].reason or "")

