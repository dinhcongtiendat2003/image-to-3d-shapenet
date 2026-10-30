"""Training losses for edge-aware mesh refinement."""

from .refinement import (
    EdgeAwareRefinementLoss,
    PointDistanceBackend,
    PyTorch3DPointDistance,
    RefinementLossOutput,
    build_point_distance_backend,
    sample_edge_points,
    sample_mesh_surface,
)

__all__ = [
    "EdgeAwareRefinementLoss",
    "PointDistanceBackend",
    "PyTorch3DPointDistance",
    "RefinementLossOutput",
    "build_point_distance_backend",
    "sample_edge_points",
    "sample_mesh_surface",
]

