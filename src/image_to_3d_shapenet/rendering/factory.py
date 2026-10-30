"""Renderer selection with automatic fallback to the software backend."""

from __future__ import annotations

import trimesh

from .base import RendererBackend, RenderResult
from .camera import CameraSpec
from .pyrender_backend import PyrenderBackend
from .software import SoftwareRasterizerBackend


class AutoRendererBackend(RendererBackend):
    name = "auto"

    def __init__(self, pyrender_platform: str = "egl") -> None:
        self._fallback = SoftwareRasterizerBackend()
        self._active: RendererBackend
        self._initialization_warning: str | None = None
        try:
            self._active = PyrenderBackend(pyrender_platform)
        except Exception as error:
            self._active = self._fallback
            self._initialization_warning = f"pyrender unavailable ({type(error).__name__}); software fallback"

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
        kwargs = {
            "background_rgb": background_rgb,
            "transparent_background": transparent_background,
            "ambient_light": ambient_light,
            "directional_light": directional_light,
        }
        warning = self._initialization_warning
        try:
            result = self._active.render(mesh, camera, **kwargs)
        except Exception as error:
            if self._active is self._fallback:
                raise
            warning = f"pyrender failed ({type(error).__name__}); software fallback"
            self._active.close()
            self._active = self._fallback
            result = self._active.render(mesh, camera, **kwargs)
        if warning:
            return RenderResult(
                rgba=result.rgba,
                depth=result.depth,
                normals=result.normals,
                mask=result.mask,
                backend=result.backend,
                normal_space=result.normal_space,
                warnings=(*result.warnings, warning),
            )
        return result

    def close(self) -> None:
        self._active.close()


def create_renderer(name: str, *, pyrender_platform: str = "egl") -> RendererBackend:
    if name == "auto":
        return AutoRendererBackend(pyrender_platform)
    if name == "pyrender":
        return PyrenderBackend(pyrender_platform)
    if name == "software":
        return SoftwareRasterizerBackend()
    raise ValueError(f"unknown renderer backend: {name}")

