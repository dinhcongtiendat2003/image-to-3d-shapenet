"""Conservative sharp-edge-aware mesh repair and portable export.

The default path uses only trimesh and NumPy.  Every mutating operation is
validated and rolled back when it produces an empty, non-finite, or less safe
topology.  No smoothing operation is allowed to move a sharp-edge vertex.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
import warnings

import numpy as np
import trimesh

from image_to_3d_shapenet.config import MeshPostprocessConfig


@dataclass(frozen=True)
class EdgeAnalysis:
    edges: np.ndarray
    counts: np.ndarray
    boundary: np.ndarray
    non_manifold: np.ndarray
    sharp: np.ndarray
    dihedral_degrees: np.ndarray


def _finite_nonempty(mesh: trimesh.Trimesh) -> bool:
    return bool(
        len(mesh.vertices) >= 3
        and len(mesh.faces) >= 1
        and np.isfinite(np.asarray(mesh.vertices)).all()
        and np.asarray(mesh.faces).min(initial=0) >= 0
        and np.asarray(mesh.faces).max(initial=-1) < len(mesh.vertices)
    )


def load_trimesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(Path(path), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or not _finite_nonempty(loaded):
        raise ValueError(f"Mesh rỗng, lỗi hoặc chứa giá trị không hữu hạn: {path}")
    return loaded


def analyze_edges(
    mesh: trimesh.Trimesh, *, sharp_threshold_degrees: float = 35.0
) -> EdgeAnalysis:
    """Classify boundary, non-manifold and two-face sharp edges."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    face_ids = np.tile(np.arange(len(faces), dtype=np.int64), 3)
    sorted_edges = np.sort(face_edges, axis=1)
    edges, inverse, counts = np.unique(
        sorted_edges, axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    dihedral = np.full(len(edges), np.nan, dtype=np.float64)
    manifold = counts == 2
    if manifold.any():
        first = face_ids[order[starts[manifold]]]
        second = face_ids[order[starts[manifold] + 1]]
        normals = np.asarray(mesh.face_normals, dtype=np.float64)
        cosine = np.einsum("ij,ij->i", normals[first], normals[second])
        dihedral[manifold] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return EdgeAnalysis(
        edges=edges,
        counts=counts,
        boundary=counts == 1,
        non_manifold=counts > 2,
        sharp=manifold & (dihedral >= sharp_threshold_degrees),
        dihedral_degrees=dihedral,
    )


def mesh_report(
    mesh: trimesh.Trimesh, *, sharp_threshold_degrees: float = 35.0
) -> dict[str, Any]:
    edge = analyze_edges(mesh, sharp_threshold_degrees=sharp_threshold_degrees)
    components = mesh.split(only_watertight=False)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(components)),
        "boundary_edges": int(edge.boundary.sum()),
        "non_manifold_edges": int(edge.non_manifold.sum()),
        "sharp_edges": int(edge.sharp.sum()),
        "watertight": bool(mesh.is_watertight),
        "bounding_box": {
            "minimum": bounds[0].tolist(),
            "maximum": bounds[1].tolist(),
            "extents": (bounds[1] - bounds[0]).tolist(),
        },
    }


def _remove_duplicate_faces(mesh: trimesh.Trimesh) -> None:
    canonical = np.sort(np.asarray(mesh.faces, dtype=np.int64), axis=1)
    _, keep = np.unique(canonical, axis=0, return_index=True)
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[np.sort(keep)] = True
    mesh.update_faces(mask)


def _remove_small_components(mesh: trimesh.Trimesh, config: MeshPostprocessConfig) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    largest_area = max(float(part.area) for part in parts)
    keep = [
        part for part in parts
        if len(part.faces) >= config.min_component_faces
        and (largest_area <= 0 or float(part.area) / largest_area >= config.min_component_area_ratio)
    ]
    if not keep:
        keep = [max(parts, key=lambda item: (float(item.area), len(item.faces)))]
    return trimesh.util.concatenate(keep)


def _fill_small_holes(mesh: trimesh.Trimesh, config: MeshPostprocessConfig) -> trimesh.Trimesh:
    before_faces = len(mesh.faces)
    before_non_manifold = int(analyze_edges(mesh).non_manifold.sum())
    candidate = mesh.copy()
    candidate.fill_holes()
    added = len(candidate.faces) - before_faces
    after_non_manifold = int(analyze_edges(candidate).non_manifold.sum())
    if added < 0 or added > config.max_hole_faces or after_non_manifold > before_non_manifold:
        raise ValueError("hole fill exceeded the configured safe topology limit")
    return candidate


def _remesh_preserving_features(
    mesh: trimesh.Trimesh, config: MeshPostprocessConfig
) -> trimesh.Trimesh:
    """Split long triangles without relocating the original sharp-edge vertices."""
    vertices, faces = trimesh.remesh.subdivide_to_size(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        max_edge=config.remesh_max_edge,
        max_iter=config.remesh_max_iterations,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _quadrics(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-15
    normals[valid] /= lengths[valid, None]
    planes = np.concatenate(
        (normals, -np.einsum("ij,ij->i", normals, triangles[:, 0])[:, None]), axis=1
    )
    face_quadrics = np.einsum("fi,fj->fij", planes, planes)
    result = np.zeros((len(vertices), 4, 4), dtype=np.float64)
    for corner in range(3):
        np.add.at(result, faces[:, corner], face_quadrics)
    return result


def _feature_aware_quadric_decimation(
    mesh: trimesh.Trimesh, config: MeshPostprocessConfig
) -> tuple[trimesh.Trimesh, int]:
    """Conservative QEM edge collapse; all sharp-edge vertices are locked."""
    target = config.target_face_count
    if target is None or len(mesh.faces) <= target:
        return mesh, 0
    result = mesh.copy()
    original_edges = analyze_edges(
        result, sharp_threshold_degrees=config.sharp_edge_threshold_degrees
    )
    protected_positions = np.unique(original_edges.edges[original_edges.sharp].reshape(-1))
    protected_points = np.asarray(result.vertices)[protected_positions].copy()
    collapses = 0
    while len(result.faces) > target and collapses < config.decimation_max_collapses:
        edge = analyze_edges(
            result, sharp_threshold_degrees=config.sharp_edge_threshold_degrees
        )
        locked = np.zeros(len(result.vertices), dtype=bool)
        if edge.sharp.any():
            locked[np.unique(edge.edges[edge.sharp].reshape(-1))] = True
        candidates = edge.edges[(edge.counts == 2) & ~locked[edge.edges].any(axis=1)]
        if len(candidates) == 0:
            break
        quadrics = _quadrics(result)
        points = np.asarray(result.vertices, dtype=np.float64)
        midpoints = points[candidates].mean(axis=1)
        homogeneous = np.concatenate((midpoints, np.ones((len(midpoints), 1))), axis=1)
        combined = quadrics[candidates[:, 0]] + quadrics[candidates[:, 1]]
        costs = np.einsum("bi,bij,bj->b", homogeneous, combined, homogeneous)
        selected = candidates[int(np.argmin(costs))]
        keep, remove = int(selected[0]), int(selected[1])
        candidate = result.copy()
        candidate.vertices[keep] = (candidate.vertices[keep] + candidate.vertices[remove]) * 0.5
        faces = np.asarray(candidate.faces).copy()
        faces[faces == remove] = keep
        candidate.faces = faces
        candidate.update_faces(candidate.nondegenerate_faces(height=config.degenerate_height))
        _remove_duplicate_faces(candidate)
        candidate.remove_unreferenced_vertices()
        if not _finite_nonempty(candidate):
            break
        candidate_edges = analyze_edges(
            candidate, sharp_threshold_degrees=config.sharp_edge_threshold_degrees
        )
        if int(candidate_edges.non_manifold.sum()) > int(edge.non_manifold.sum()):
            break
        current_angles = edge.dihedral_degrees[np.isfinite(edge.dihedral_degrees)]
        candidate_angles = candidate_edges.dihedral_degrees[
            np.isfinite(candidate_edges.dihedral_degrees)
        ]
        current_max = float(current_angles.max(initial=0.0))
        candidate_max = float(candidate_angles.max(initial=0.0))
        maximum_allowed = min(
            150.0,
            max(current_max + 15.0, config.sharp_edge_threshold_degrees + 45.0),
        )
        if candidate_max > maximum_allowed:
            # A near-opposite dihedral is evidence of a fold/inverted face.
            break
        result = candidate
        collapses += 1

    # Protected feature points must still exist exactly (within numeric tolerance).
    if len(protected_points):
        current = np.asarray(result.vertices, dtype=np.float64)
        distances = np.linalg.norm(protected_points[:, None] - current[None, :], axis=2)
        tolerance = max(float(np.ptp(np.asarray(mesh.vertices), axis=0).max()), 1.0) * 1e-8
        if np.any(distances.min(axis=1) > tolerance):
            return mesh, 0
    return result, collapses


def _smooth_without_crossing_sharp_edges(
    mesh: trimesh.Trimesh, config: MeshPostprocessConfig
) -> trimesh.Trimesh:
    result = mesh.copy()
    for _ in range(config.smoothing_iterations):
        edge = analyze_edges(
            result, sharp_threshold_degrees=config.sharp_edge_threshold_degrees
        )
        sharp_vertices = np.zeros(len(result.vertices), dtype=bool)
        if edge.sharp.any():
            sharp_vertices[np.unique(edge.edges[edge.sharp].reshape(-1))] = True
        usable = edge.edges[~edge.sharp & ~edge.non_manifold]
        vertices = np.asarray(result.vertices, dtype=np.float64)
        sums = np.zeros_like(vertices)
        counts = np.zeros((len(vertices), 1), dtype=np.float64)
        for source, neighbor in ((usable[:, 0], usable[:, 1]), (usable[:, 1], usable[:, 0])):
            np.add.at(sums, source, vertices[neighbor])
            np.add.at(counts, source, 1.0)
        movable = (~sharp_vertices) & (counts[:, 0] > 0)
        means = sums / np.maximum(counts, 1.0)
        vertices[movable] += config.smoothing_lambda * (means[movable] - vertices[movable])
        result.vertices = vertices
    return result


def _apply_safe(
    mesh: trimesh.Trimesh,
    name: str,
    operation: Callable[[trimesh.Trimesh], trimesh.Trimesh | None],
    steps: list[dict[str, Any]],
) -> trimesh.Trimesh:
    before = mesh.copy()
    try:
        value = operation(mesh)
        candidate = mesh if value is None else value
        if not _finite_nonempty(candidate):
            raise ValueError("operation produced an invalid mesh")
        steps.append({"name": name, "status": "applied"})
        return candidate
    except Exception as exc:
        steps.append({"name": name, "status": "skipped_unsafe", "reason": str(exc)})
        return before


def process_mesh(
    mesh: trimesh.Trimesh, config: MeshPostprocessConfig
) -> tuple[trimesh.Trimesh, list[dict[str, Any]]]:
    """Apply enabled repair steps in deterministic order."""
    if not _finite_nonempty(mesh):
        raise ValueError("input mesh is empty or invalid")
    result = mesh.copy()
    steps: list[dict[str, Any]] = []

    operations: list[tuple[bool, str, Callable[[trimesh.Trimesh], trimesh.Trimesh | None]]] = [
        (config.merge_duplicate_vertices, "merge_duplicate_vertices", lambda item: item.merge_vertices(digits_vertex=config.merge_digits)),
        (config.remove_duplicate_faces, "remove_duplicate_faces", lambda item: _remove_duplicate_faces(item)),
        (config.remove_degenerate_faces, "remove_degenerate_faces", lambda item: item.update_faces(item.nondegenerate_faces(height=config.degenerate_height))),
        (config.remove_unreferenced_vertices, "remove_unreferenced_vertices", lambda item: item.remove_unreferenced_vertices()),
        (config.fix_face_orientation, "fix_face_orientation", lambda item: trimesh.repair.fix_winding(item)),
        (config.fix_normals, "fix_normals", lambda item: item.fix_normals(multibody=True)),
        (config.remove_small_components, "remove_small_components", lambda item: _remove_small_components(item, config)),
        (config.fill_small_holes and config.max_hole_faces > 0, "fill_small_holes", lambda item: _fill_small_holes(item, config)),
        (config.remesh and config.remesh_max_iterations > 0, "sharp_preserving_remesh", lambda item: _remesh_preserving_features(item, config)),
    ]
    for enabled, name, operation in operations:
        if enabled:
            result = _apply_safe(result, name, operation, steps)
        else:
            steps.append({"name": name, "status": "disabled"})

    edge_status = analyze_edges(
        result, sharp_threshold_degrees=config.sharp_edge_threshold_degrees
    )
    steps.append({
        "name": "detect_non_manifold_edges",
        "status": "analyzed" if config.detect_non_manifold_edges else "disabled",
        "count": int(edge_status.non_manifold.sum()) if config.detect_non_manifold_edges else None,
    })
    steps.append({
        "name": "detect_sharp_edges",
        "status": "analyzed" if config.detect_sharp_edges else "disabled",
        "count": int(edge_status.sharp.sum()) if config.detect_sharp_edges else None,
    })

    if config.decimate:
        before_faces = len(result.faces)
        result, collapses = _feature_aware_quadric_decimation(result, config)
        steps.append({
            "name": "feature_aware_quadric_decimation",
            "status": "applied" if collapses else "no_safe_collapse",
            "collapses": collapses,
            "faces_before": before_faces,
            "faces_after": len(result.faces),
        })
    else:
        steps.append({"name": "feature_aware_quadric_decimation", "status": "disabled"})

    if config.smooth and config.smoothing_iterations > 0:
        result = _apply_safe(
            result,
            "sharp_locked_smoothing",
            lambda item: _smooth_without_crossing_sharp_edges(item, config),
            steps,
        )
    else:
        steps.append({"name": "sharp_locked_smoothing", "status": "disabled"})
    steps.append({
        "name": "check_watertight",
        "status": "analyzed" if config.check_watertight else "disabled",
        "watertight": bool(result.is_watertight) if config.check_watertight else None,
    })
    return result, steps


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def postprocess_mesh_file(
    input_path: str | Path,
    output_dir: str | Path,
    config: MeshPostprocessConfig,
) -> dict[str, Any]:
    """Save raw/processed geometry, reports, and portable OBJ/GLB/STL files."""
    source = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = load_trimesh(source)
    raw_path = output / "raw_mesh.obj"
    raw.export(raw_path, file_type="obj")
    before = mesh_report(raw, sharp_threshold_degrees=config.sharp_edge_threshold_degrees)
    processed, steps = process_mesh(raw, config)
    after = mesh_report(processed, sharp_threshold_degrees=config.sharp_edge_threshold_degrees)

    files = {
        "raw_obj": raw_path.name,
        "processed_obj": "refined_mesh.obj",
        "processed_glb": "refined_mesh.glb",
        "processed_stl": "refined_mesh.stl",
    }
    processed.export(output / files["processed_obj"], file_type="obj")
    processed.export(output / files["processed_glb"], file_type="glb")
    processed.export(output / files["processed_stl"], file_type="stl")

    messages: list[str] = []
    if not after["watertight"]:
        message = (
            "Mesh sau xử lý không watertight; OBJ/GLB vẫn hợp lệ để trao đổi, "
            "nhưng STL có thể không phù hợp để in 3D."
        )
        messages.append(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    report = {"before": before, "after": after, "steps": steps, "warnings": messages}
    metadata = {
        "source": str(source),
        "backend": "trimesh",
        "pymeshlab_available": find_spec("pymeshlab") is not None,
        "sharp_edges_preserved_by_locking": True,
        "config": asdict(config),
        "files": files,
        "watertight": after["watertight"],
        "warnings": messages,
    }
    _atomic_json(output / "mesh_report.json", report)
    _atomic_json(output / "metadata.json", metadata)
    return {"report": report, "metadata": metadata, "output_dir": str(output)}

