"""Small dependency-light visual diagnostics for refinement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor
import trimesh

from .geometry import compute_mesh_geometry


def _numpy(value: Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _save_obj(vertices: Tensor, faces: Tensor, path: Path) -> None:
    mesh = trimesh.Trimesh(vertices=_numpy(vertices), faces=_numpy(faces).astype(np.int64), process=False)
    mesh.export(path, file_type="obj")


def save_refinement_debug(
    base_vertices: Tensor,
    refined_vertices: Tensor,
    faces: Tensor,
    output_dir: str | Path,
    *,
    sharp_threshold_degrees: float = 35.0,
    image_size: int = 512,
) -> dict[str, Path]:
    """Save before/after OBJ and an orthographic sharp-edge visualization."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if base_vertices.ndim == 3:
        base_vertices = base_vertices[0]
    if refined_vertices.ndim == 3:
        refined_vertices = refined_vertices[0]
    if faces.ndim == 3:
        faces = faces[0]
    before_path = output / "mesh_before.obj"
    after_path = output / "mesh_after.obj"
    preview_path = output / "sharp_edges.png"
    _save_obj(base_vertices, faces, before_path)
    _save_obj(refined_vertices, faces, after_path)

    geometry = compute_mesh_geometry(
        refined_vertices, faces, sharp_threshold_degrees=sharp_threshold_degrees
    )
    vertices = _numpy(refined_vertices)
    edges = _numpy(geometry.edges).astype(np.int64)
    sharp = _numpy(geometry.sharp_edges[0]).astype(bool)
    xy = vertices[:, :2]
    minimum, maximum = xy.min(axis=0), xy.max(axis=0)
    extent = np.maximum(maximum - minimum, 1e-8)
    points = (xy - minimum) / extent
    points = points * (image_size * 0.8) + image_size * 0.1
    points[:, 1] = image_size - points[:, 1]
    canvas = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(canvas)
    for edge, is_sharp in zip(edges, sharp):
        color = (220, 35, 35) if is_sharp else (80, 110, 160)
        width = 3 if is_sharp else 1
        draw.line((tuple(points[edge[0]]), tuple(points[edge[1]])), fill=color, width=width)
    canvas.save(preview_path)
    return {"before": before_path, "after": after_path, "sharp_edges": preview_path}
