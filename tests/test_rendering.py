import numpy as np
import pytest
import trimesh

from image_to_3d_shapenet.rendering import (
    SoftwareRasterizerBackend,
    camera_for_mesh,
    orbit_angles,
)
from image_to_3d_shapenet.rendering import factory


def _cube_camera(size: int = 128):
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    camera = camera_for_mesh(
        mesh,
        width=size,
        height=size,
        azimuth_degrees=30.0,
        elevation_degrees=20.0,
        fov_degrees=50.0,
        framing_margin=1.15,
    )
    return mesh, camera


def test_camera_matrices_are_valid_and_reproducible() -> None:
    _, camera = _cube_camera()
    rotation = camera.camera_to_world[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert np.allclose(camera.world_to_camera @ camera.camera_to_world, np.eye(4))
    assert camera.intrinsics.shape == (3, 3)
    assert camera.projection.shape == (4, 4)
    assert np.isfinite(camera.projection).all()
    assert 0.0 < camera.near < camera.far


def test_orbit_views_cover_full_circle() -> None:
    angles = orbit_angles(8, 15.0, 30.0)
    assert [value[0] for value in angles] == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    assert all(15.0 <= value[1] <= 30.0 for value in angles)


def test_software_renderer_outputs_finite_nonempty_cube() -> None:
    mesh, camera = _cube_camera()
    result = SoftwareRasterizerBackend().render(
        mesh,
        camera,
        background_rgb=(245, 245, 245),
        transparent_background=True,
        ambient_light=0.35,
        directional_light=2.0,
    )
    result.validate(128, 128)
    assert result.rgba.shape == (128, 128, 4)
    assert result.depth.dtype == np.float32
    assert np.isfinite(result.depth).all()
    assert np.any(result.depth > 0)
    assert np.any(result.mask > 0)
    assert np.all(result.rgba[result.mask == 0, 3] == 0)
    foreground_normals = result.normals[result.mask > 0]
    assert np.allclose(np.linalg.norm(foreground_normals, axis=1), 1.0, atol=1e-5)


def test_software_renderer_supports_512_resolution() -> None:
    mesh, camera = _cube_camera(512)
    result = SoftwareRasterizerBackend().render(
        mesh,
        camera,
        background_rgb=(245, 245, 245),
        transparent_background=False,
        ambient_light=0.35,
        directional_light=2.0,
    )
    result.validate(512, 512)
    assert result.rgba.shape == (512, 512, 4)


def test_auto_renderer_falls_back_when_pyrender_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenPyrender:
        def __init__(self, platform: str) -> None:
            raise RuntimeError("simulated unavailable context")

    monkeypatch.setattr(factory, "PyrenderBackend", BrokenPyrender)
    mesh, camera = _cube_camera()
    result = factory.AutoRendererBackend().render(
        mesh,
        camera,
        background_rgb=(245, 245, 245),
        transparent_background=True,
        ambient_light=0.35,
        directional_light=2.0,
    )
    result.validate(128, 128)
    assert result.backend == "software"
    assert "software fallback" in result.warnings[0]

