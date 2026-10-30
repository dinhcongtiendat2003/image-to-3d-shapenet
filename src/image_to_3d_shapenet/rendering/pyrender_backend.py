"""Optional EGL/OSMesa offscreen backend implemented with pyrender."""

from __future__ import annotations

import os
import sys

import numpy as np
import trimesh

from .base import RendererBackend, RenderResult
from .camera import CameraSpec


class PyrenderBackend(RendererBackend):
    name = "pyrender"

    def __init__(self, platform: str = "egl") -> None:
        if platform not in {"egl", "osmesa", "pyglet"}:
            raise ValueError("unsupported pyrender platform")
        if platform != "pyglet" and sys.platform.startswith("linux"):
            os.environ.setdefault("PYOPENGL_PLATFORM", platform)
        import pyrender  # Imported after selecting the OpenGL platform.

        self._pyrender = pyrender

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
        pyrender = self._pyrender
        background = [value / 255.0 for value in background_rgb]
        background.append(0.0 if transparent_background else 1.0)
        scene = pyrender.Scene(
            bg_color=background,
            ambient_light=np.full(3, ambient_light, dtype=np.float32),
        )
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.69, 0.72, 0.77, 1.0),
            metallicFactor=0.0,
            roughnessFactor=0.85,
        )
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
        fx, fy = camera.intrinsics[0, 0], camera.intrinsics[1, 1]
        cx, cy = camera.intrinsics[0, 2], camera.intrinsics[1, 2]
        intrinsics_camera = pyrender.IntrinsicsCamera(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            znear=camera.near,
            zfar=camera.far,
        )
        scene.add(intrinsics_camera, pose=camera.camera_to_world)
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=directional_light),
            pose=camera.camera_to_world,
        )
        renderer = pyrender.OffscreenRenderer(camera.width, camera.height)
        try:
            flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES
            color, depth = renderer.render(scene, flags=flags)
            normal_color, _ = renderer.render(
                scene,
                flags=flags | pyrender.RenderFlags.FACE_NORMALS | pyrender.RenderFlags.FLAT,
            )
        finally:
            renderer.delete()
        rgba = np.array(color, dtype=np.uint8, copy=True)
        if rgba.shape[-1] == 3:
            alpha = np.full((*rgba.shape[:2], 1), 255, dtype=np.uint8)
            rgba = np.concatenate((rgba, alpha), axis=2)
        depth_array = np.asarray(depth, dtype=np.float32)
        mask = (depth_array > 0).astype(np.uint8) * 255
        rgba[..., 3] = np.where(mask > 0, 255, 0 if transparent_background else 255)
        normal_rgb = np.asarray(normal_color[..., :3], dtype=np.float32)
        normals = normal_rgb / 127.5 - 1.0
        normals[mask == 0] = 0.0
        lengths = np.linalg.norm(normals, axis=2, keepdims=True)
        foreground = (mask > 0) & (lengths[..., 0] > 1e-8)
        normals[foreground] /= lengths[foreground]
        return RenderResult(
            rgba=rgba,
            depth=depth_array,
            normals=normals.astype(np.float32),
            mask=mask,
            backend=self.name,
            normal_space="camera",
        )

