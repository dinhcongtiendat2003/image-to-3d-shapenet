"""Deterministic orbit cameras with complete reconstruction metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class CameraSpec:
    width: int
    height: int
    azimuth_degrees: float
    elevation_degrees: float
    distance: float
    near: float
    far: float
    intrinsics: np.ndarray
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    projection: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "azimuth_degrees": self.azimuth_degrees,
            "elevation_degrees": self.elevation_degrees,
            "distance": self.distance,
            "near": self.near,
            "far": self.far,
            "intrinsics": self.intrinsics.tolist(),
            "extrinsics_world_to_camera": self.world_to_camera.tolist(),
            "camera_to_world": self.camera_to_world.tolist(),
            "projection_matrix": self.projection.tolist(),
            "camera_convention": "OpenGL: right-handed, camera looks along -Z, +Y is up",
        }


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("cannot normalize zero vector")
    return vector / length


def orbit_angles(
    views: int,
    elevation_min_degrees: float,
    elevation_max_degrees: float,
) -> list[tuple[float, float]]:
    if views <= 0:
        raise ValueError("views must be positive")
    midpoint = (elevation_min_degrees + elevation_max_degrees) * 0.5
    amplitude = (elevation_max_degrees - elevation_min_degrees) * 0.5
    return [
        (
            360.0 * index / views,
            midpoint + amplitude * np.sin(2.0 * np.pi * index / views),
        )
        for index in range(views)
    ]


def camera_for_mesh(
    mesh: trimesh.Trimesh,
    *,
    width: int,
    height: int,
    azimuth_degrees: float,
    elevation_degrees: float,
    fov_degrees: float,
    framing_margin: float,
) -> CameraSpec:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    radius = max(float(np.linalg.norm(bounds[1] - bounds[0]) * 0.5), 1e-4)
    half_fov = np.deg2rad(fov_degrees) * 0.5
    distance = radius * framing_margin / np.sin(half_fov)
    azimuth = np.deg2rad(azimuth_degrees)
    elevation = np.deg2rad(elevation_degrees)
    offset = distance * np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation),
            np.cos(elevation) * np.sin(azimuth),
        ],
        dtype=np.float64,
    )
    position = center + offset
    z_axis = _normalize(position - center)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(up, z_axis))) > 0.99:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = _normalize(np.cross(up, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 0] = x_axis
    camera_to_world[:3, 1] = y_axis
    camera_to_world[:3, 2] = z_axis
    camera_to_world[:3, 3] = position
    world_to_camera = np.linalg.inv(camera_to_world)

    focal = 0.5 * height / np.tan(half_fov)
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5
    intrinsics = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    near = max(0.01, distance - radius * 1.5)
    far = distance + radius * 2.5
    projection = np.zeros((4, 4), dtype=np.float64)
    projection[0, 0] = 2.0 * focal / width
    projection[1, 1] = 2.0 * focal / height
    projection[0, 2] = 1.0 - 2.0 * cx / width
    projection[1, 2] = 2.0 * cy / height - 1.0
    projection[2, 2] = -(far + near) / (far - near)
    projection[2, 3] = -(2.0 * far * near) / (far - near)
    projection[3, 2] = -1.0
    return CameraSpec(
        width=width,
        height=height,
        azimuth_degrees=float(azimuth_degrees),
        elevation_degrees=float(elevation_degrees),
        distance=distance,
        near=near,
        far=far,
        intrinsics=intrinsics,
        world_to_camera=world_to_camera,
        camera_to_world=camera_to_world,
        projection=projection,
    )

