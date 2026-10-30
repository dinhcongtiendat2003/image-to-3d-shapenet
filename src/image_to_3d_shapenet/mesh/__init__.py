"""Differentiable mesh geometry and debug export helpers."""

from .geometry import (
    EdgeTopology,
    MeshGeometry,
    build_edge_topology,
    compute_dihedral_angles,
    compute_face_normals,
    compute_mesh_geometry,
    compute_vertex_normals,
    laplacian_coordinates,
)
from .debug import save_refinement_debug
from .postprocess import (
    EdgeAnalysis,
    analyze_edges,
    load_trimesh,
    mesh_report,
    postprocess_mesh_file,
    process_mesh,
)

__all__ = [
    "EdgeTopology",
    "MeshGeometry",
    "build_edge_topology",
    "compute_dihedral_angles",
    "compute_face_normals",
    "compute_mesh_geometry",
    "compute_vertex_normals",
    "laplacian_coordinates",
    "save_refinement_debug",
    "EdgeAnalysis",
    "analyze_edges",
    "load_trimesh",
    "mesh_report",
    "postprocess_mesh_file",
    "process_mesh",
]

