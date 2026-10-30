"""ShapeNet mesh discovery, validation and reversible normalization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class MeshValidation:
    path: str
    valid: bool
    vertices: int = 0
    faces: int = 0
    degenerate_faces: int = 0
    degenerate_ratio: float = 0.0
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_normalized_meshes(root: str | Path) -> list[Path]:
    return sorted(Path(root).rglob("model_normalized.obj"))


def model_id_from_path(path: str | Path) -> str:
    mesh_path = Path(path)
    if mesh_path.parent.name == "models":
        return mesh_path.parent.parent.name
    return mesh_path.parent.name


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(Path(path), process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("mesh scene has no geometry")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"unsupported mesh type: {type(loaded).__name__}")
    return loaded


def validate_mesh(
    path: str | Path,
    *,
    max_degenerate_ratio: float = 0.05,
    max_file_bytes: int | None = None,
) -> MeshValidation:
    source = str(Path(path))
    try:
        if max_file_bytes is not None and Path(path).stat().st_size > max_file_bytes:
            raise ValueError("mesh file exceeds configured memory safety limit")
        mesh = load_mesh(path)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] != 3:
            raise ValueError("mesh has no valid vertices")
        if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] != 3:
            raise ValueError("mesh has no triangular faces")
        if not np.isfinite(vertices).all():
            raise ValueError("mesh vertices contain NaN or Inf")
        if not np.isfinite(faces).all():
            raise ValueError("mesh faces contain NaN or Inf")
        repeated = (
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 0] == faces[:, 2])
        )
        triangles = vertices[faces]
        doubled_area = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        degenerate = repeated | (doubled_area <= 1e-12)
        degenerate_count = int(degenerate.sum())
        ratio = degenerate_count / int(faces.shape[0])
        if ratio > max_degenerate_ratio:
            raise ValueError(
                f"degenerate face ratio {ratio:.6f} exceeds {max_degenerate_ratio:.6f}"
            )
        return MeshValidation(
            path=source,
            valid=True,
            vertices=int(vertices.shape[0]),
            faces=int(faces.shape[0]),
            degenerate_faces=degenerate_count,
            degenerate_ratio=ratio,
        )
    except Exception as error:
        return MeshValidation(path=source, valid=False, reason=str(error))


def normalize_mesh(
    mesh: trimesh.Trimesh,
    *,
    mode: str = "unit_cube",
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Center and uniformly scale a mesh, returning reversible transforms."""
    if mode not in {"unit_cube", "unit_sphere"}:
        raise ValueError("mode must be unit_cube or unit_sphere")
    result = mesh.copy()
    vertices = np.asarray(result.vertices, dtype=np.float64)
    if vertices.size == 0 or not np.isfinite(vertices).all():
        raise ValueError("cannot normalize empty or non-finite vertices")
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    center = bounds.mean(axis=0)
    centered = vertices - center
    if mode == "unit_cube":
        denominator = float((bounds[1] - bounds[0]).max())
    else:
        denominator = float(np.linalg.norm(centered, axis=1).max())
    if denominator <= 1e-12:
        raise ValueError("cannot normalize a zero-size mesh")
    scale = 1.0 / denominator
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = -center * scale
    result.apply_transform(transform)
    result.remove_unreferenced_vertices()
    try:
        result.fix_normals(multibody=True)
    except (ValueError, IndexError):
        pass
    metadata = {
        "mode": mode,
        "center": center.tolist(),
        "uniform_scale": scale,
        "original_to_normalized": transform.tolist(),
        "normalized_to_original": np.linalg.inv(transform).tolist(),
        "original_bounds": bounds.tolist(),
    }
    return result, metadata


def save_normalized_mesh(
    source: str | Path,
    output_mesh: str | Path,
    transform_json: str | Path,
    *,
    mode: str,
) -> dict[str, Any]:
    mesh = load_mesh(source)
    normalized, metadata = normalize_mesh(mesh, mode=mode)
    mesh_path = Path(output_mesh)
    metadata_path = Path(transform_json)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.export(mesh_path, file_type="obj")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

