"""Scientific geometry and silhouette metrics."""

from .mesh_metrics import (
    MeshMetricResult,
    MetricInputError,
    compute_mesh_metrics,
    normalize_mesh_pair,
    silhouette_iou,
)
from .evaluator import evaluate
from .reporting import summarize, write_incremental_reports

__all__ = [
    "MeshMetricResult", "MetricInputError", "compute_mesh_metrics",
    "normalize_mesh_pair", "silhouette_iou",
    "evaluate", "summarize", "write_incremental_reports",
]

