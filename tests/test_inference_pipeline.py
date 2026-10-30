from __future__ import annotations

import json

import trimesh
from PIL import Image

from image_to_3d_shapenet.config import InferenceConfig
from image_to_3d_shapenet.inference.base import GenerationResult
from image_to_3d_shapenet.inference.pipeline import run_inference
from image_to_3d_shapenet.inference.preprocess import preprocess_image


class _TestOnlyAutoBackend:
    def __init__(self) -> None:
        self.unloaded = False

    def preprocess(self, image):
        return preprocess_image(image, size=64, remove_background=False)

    def load(self) -> None:
        pass

    def generate(self, image, options) -> GenerationResult:
        mesh = trimesh.creation.icosphere(subdivisions=1)
        return GenerationResult(
            mesh=mesh,
            backend_name="test-only",
            inference_time_seconds=0.01,
            peak_vram_gb=0.0,
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            texture_status="none",
        )

    def unload(self) -> None:
        self.unloaded = True


def test_pipeline_exports_loadable_obj_and_glb(tmp_path) -> None:
    source = tmp_path / "source.png"
    image = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    image.paste((20, 100, 200, 255), (5, 3, 35, 17))
    image.save(source)
    backend = _TestOnlyAutoBackend()

    payload = run_inference(
        source,
        tmp_path / "result",
        InferenceConfig(remove_background=False),
        backend=backend,
    )

    output = tmp_path / "result"
    assert payload["backend"] == "test-only"
    assert backend.unloaded is True
    for name in ("base_mesh.obj", "base_mesh.glb", "metadata.json", "input_preprocessed.png"):
        assert (output / name).is_file()
    assert len(trimesh.load(output / "base_mesh.obj", force="mesh").faces) > 0
    assert len(trimesh.load(output / "base_mesh.glb", force="mesh").faces) > 0
    assert json.loads((output / "metadata.json").read_text())["geometry_only"] is True
