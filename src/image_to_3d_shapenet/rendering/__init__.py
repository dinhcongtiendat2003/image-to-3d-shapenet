"""Headless multi-backend mesh rendering."""

from .base import RendererBackend, RenderResult
from .camera import CameraSpec, camera_for_mesh, orbit_angles
from .factory import AutoRendererBackend, create_renderer
from .pyrender_backend import PyrenderBackend
from .pipeline import render_dataset
from .software import SoftwareRasterizerBackend

__all__ = [
    "AutoRendererBackend",
    "CameraSpec",
    "PyrenderBackend",
    "RenderResult",
    "RendererBackend",
    "SoftwareRasterizerBackend",
    "camera_for_mesh",
    "create_renderer",
    "orbit_angles",
    "render_dataset",
]


