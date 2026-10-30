from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from image_to_3d_shapenet.config import MeshPostprocessConfig
from image_to_3d_shapenet.mesh import (
    analyze_edges,
    mesh_report,
    postprocess_mesh_file,
    process_mesh,
)


def cube() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


def test_cube_keeps_twelve_right_angle_edges() -> None:
    source = cube()
    config = MeshPostprocessConfig(decimate=True, target_face_count=8)
    processed, _ = process_mesh(source, config)
    analysis = analyze_edges(processed, sharp_threshold_degrees=35.0)
    angles = analysis.dihedral_degrees[analysis.sharp]
    assert len(angles) == 12
    assert np.allclose(angles, 90.0, atol=1e-5)
    assert len(processed.faces) == 12  # unsafe feature collapse is intentionally refused


def test_cleanup_removes_duplicate_degenerate_and_small_component() -> None:
    source = cube()
    vertices = np.vstack((source.vertices, source.vertices[0], [[10, 10, 10]]))
    duplicate_vertex = len(source.vertices)
    faces = np.vstack((
        source.faces,
        source.faces[0],
        [duplicate_vertex, source.faces[0, 1], source.faces[0, 2]],
        [0, 0, 1],
    ))
    faulty = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    processed, steps = process_mesh(faulty, MeshPostprocessConfig())
    report = mesh_report(processed)
    assert report["vertices"] == 8
    assert report["faces"] == 12
    assert report["components"] == 1
    assert report["boundary_edges"] == 0
    assert report["non_manifold_edges"] == 0
    assert report["watertight"] is True
    assert all(step["status"] != "skipped_unsafe" for step in steps[:4])


def test_feature_aware_decimation_preserves_cube_corners() -> None:
    source = cube()
    vertices, faces = np.asarray(source.vertices), np.asarray(source.faces)
    for _ in range(2):
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    dense = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    config = MeshPostprocessConfig(
        decimate=True,
        target_face_count=96,
        decimation_max_collapses=128,
        fill_small_holes=False,
    )
    processed, _ = process_mesh(dense, config)
    assert len(processed.faces) <= len(dense.faces)
    original_corners = np.asarray(source.vertices)
    distances = np.linalg.norm(
        original_corners[:, None] - np.asarray(processed.vertices)[None, :], axis=2
    )
    assert np.all(distances.min(axis=1) < 1e-8)
    sharp = analyze_edges(processed).dihedral_degrees
    assert np.allclose(sharp[np.isfinite(sharp) & (sharp > 35.0)], 90.0, atol=1e-4)


def test_quadric_decimation_reaches_target_on_smooth_watertight_mesh() -> None:
    source = trimesh.creation.icosphere(subdivisions=2)
    config = MeshPostprocessConfig(
        fill_small_holes=False,
        decimate=True,
        target_face_count=250,
        decimation_max_collapses=100,
    )
    processed, steps = process_mesh(source, config)
    decimation = next(step for step in steps if "decimation" in step["name"])
    assert len(processed.faces) == 250
    assert processed.is_watertight
    assert decimation["status"] == "applied"
    assert decimation["collapses"] == 35


def test_exports_round_trip_and_report_is_exact(tmp_path: Path) -> None:
    source = tmp_path / "cube.obj"
    cube().export(source)
    result = postprocess_mesh_file(source, tmp_path / "result", MeshPostprocessConfig())
    output = Path(result["output_dir"])
    for filename in (
        "raw_mesh.obj", "refined_mesh.obj", "refined_mesh.glb", "refined_mesh.stl",
        "mesh_report.json", "metadata.json",
    ):
        assert (output / filename).is_file()
    for filename in ("refined_mesh.obj", "refined_mesh.glb", "refined_mesh.stl"):
        loaded = trimesh.load(output / filename, force="mesh")
        assert isinstance(loaded, trimesh.Trimesh)
        assert len(loaded.vertices) == 8
        assert len(loaded.faces) == 12
    report = json.loads((output / "mesh_report.json").read_text(encoding="utf-8"))
    assert report["before"]["watertight"] is True
    assert report["after"]["sharp_edges"] == 12
    assert report["warnings"] == []


def test_non_watertight_stl_has_explicit_warning(tmp_path: Path) -> None:
    open_cube = cube()
    open_cube.update_faces(np.arange(len(open_cube.faces)) != 0)
    source = tmp_path / "open.obj"
    open_cube.export(source)
    config = replace(MeshPostprocessConfig(), fill_small_holes=False)
    with pytest.warns(UserWarning, match="không watertight"):
        result = postprocess_mesh_file(source, tmp_path / "result", config)
    assert result["report"]["after"]["watertight"] is False
    assert result["report"]["warnings"]
    assert (tmp_path / "result" / "refined_mesh.stl").is_file()

