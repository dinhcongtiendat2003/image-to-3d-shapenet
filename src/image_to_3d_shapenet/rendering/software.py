"""Pure NumPy z-buffer fallback requiring no OpenGL or CUDA extensions."""

from __future__ import annotations

import numpy as np
import trimesh

from .base import RendererBackend, RenderResult
from .camera import CameraSpec


class SoftwareRasterizerBackend(RendererBackend):
    name = "software"

    def render(
        self,
        mesh: trimesh.Trimesh,
        camera: CameraSpec,
        *,
        background_rgb: tuple[int, int, int],
        transparent_background: bool,
        ambient_light: float,
        directional_light: float,
    ) -> RenderResult:
        height, width = camera.height, camera.width
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = np.asarray(background_rgb, dtype=np.uint8)
        rgba[..., 3] = 0 if transparent_background else 255
        z_buffer = np.full((height, width), np.inf, dtype=np.float64)
        normals = np.zeros((height, width, 3), dtype=np.float32)
        mask = np.zeros((height, width), dtype=np.uint8)

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
        camera_vertices = (camera.world_to_camera @ homogeneous.T).T[:, :3]
        depth_values = -camera_vertices[:, 2]
        fx, fy = camera.intrinsics[0, 0], camera.intrinsics[1, 1]
        cx, cy = camera.intrinsics[0, 2], camera.intrinsics[1, 2]
        screen = np.empty((len(vertices), 2), dtype=np.float64)
        safe_depth = np.maximum(depth_values, 1e-12)
        screen[:, 0] = fx * camera_vertices[:, 0] / safe_depth + cx
        screen[:, 1] = cy - fy * camera_vertices[:, 1] / safe_depth
        light_direction = np.array([0.35, 0.8, 0.65], dtype=np.float64)
        light_direction /= np.linalg.norm(light_direction)

        for face in faces:
            triangle_depth = depth_values[face]
            if np.any(triangle_depth <= camera.near) or np.any(triangle_depth >= camera.far):
                continue
            points = screen[face]
            x0 = max(0, int(np.floor(points[:, 0].min())))
            x1 = min(width - 1, int(np.ceil(points[:, 0].max())))
            y0 = max(0, int(np.floor(points[:, 1].min())))
            y1 = min(height - 1, int(np.ceil(points[:, 1].max())))
            if x1 < x0 or y1 < y0:
                continue
            denominator = (
                (points[1, 1] - points[2, 1]) * (points[0, 0] - points[2, 0])
                + (points[2, 0] - points[1, 0]) * (points[0, 1] - points[2, 1])
            )
            if abs(float(denominator)) <= 1e-12:
                continue
            grid_x, grid_y = np.meshgrid(
                np.arange(x0, x1 + 1, dtype=np.float64) + 0.5,
                np.arange(y0, y1 + 1, dtype=np.float64) + 0.5,
            )
            weight0 = (
                (points[1, 1] - points[2, 1]) * (grid_x - points[2, 0])
                + (points[2, 0] - points[1, 0]) * (grid_y - points[2, 1])
            ) / denominator
            weight1 = (
                (points[2, 1] - points[0, 1]) * (grid_x - points[2, 0])
                + (points[0, 0] - points[2, 0]) * (grid_y - points[2, 1])
            ) / denominator
            weight2 = 1.0 - weight0 - weight1
            inside = (weight0 >= -1e-8) & (weight1 >= -1e-8) & (weight2 >= -1e-8)
            interpolated_depth = 1.0 / (
                weight0 / triangle_depth[0]
                + weight1 / triangle_depth[1]
                + weight2 / triangle_depth[2]
            )
            local_z = z_buffer[y0 : y1 + 1, x0 : x1 + 1]
            visible = inside & (interpolated_depth < local_z)
            if not np.any(visible):
                continue
            world_triangle = vertices[face]
            face_normal = np.cross(
                world_triangle[1] - world_triangle[0],
                world_triangle[2] - world_triangle[0],
            )
            normal_length = float(np.linalg.norm(face_normal))
            if normal_length <= 1e-12:
                continue
            face_normal /= normal_length
            shade = np.clip(
                ambient_light + directional_light * 0.45 * abs(float(np.dot(face_normal, light_direction))),
                0.0,
                1.0,
            )
            color = np.asarray([176, 184, 196], dtype=np.float64) * shade
            local_z[visible] = interpolated_depth[visible]
            local_rgba = rgba[y0 : y1 + 1, x0 : x1 + 1]
            local_rgba[visible, :3] = np.clip(color, 0, 255).astype(np.uint8)
            local_rgba[visible, 3] = 255
            normals[y0 : y1 + 1, x0 : x1 + 1][visible] = face_normal.astype(np.float32)
            mask[y0 : y1 + 1, x0 : x1 + 1][visible] = 255

        depth = np.where(np.isfinite(z_buffer), z_buffer, 0.0).astype(np.float32)
        return RenderResult(
            rgba=rgba,
            depth=depth,
            normals=normals,
            mask=mask,
            backend=self.name,
            normal_space="world",
        )

