from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import trimesh
from PIL import Image

from image_to_3d_shapenet import load_config
from image_to_3d_shapenet.app.pipeline import (
    AppInferenceService,
    AppOptions,
    AppPipelineError,
)
from image_to_3d_shapenet.inference.base import GenerationResult
from image_to_3d_shapenet.inference.preprocess import preprocess_image


ROOT = Path(__file__).resolve().parents[1]


class _TestOnlyBackend:
    def __init__(self, config, requested: str) -> None:
        self.config = config
        self.requested = requested
        self.loaded = False
        self.unloaded = False
        self.geometry_only: bool | None = None

    def preprocess(self, image):
        return preprocess_image(image, size=64, remove_background=False)

    def load(self) -> None:
        self.loaded = True

    def generate(self, image, options) -> GenerationResult:
        self.geometry_only = options.geometry_only
        mesh = trimesh.creation.box()
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


class _Factory:
    def __init__(self, backend_class=_TestOnlyBackend) -> None:
        self.backend_class = backend_class
        self.instances: list[_TestOnlyBackend] = []

    def __call__(self, config, requested: str):
        value = self.backend_class(config, requested)
        self.instances.append(value)
        return value


class _FailureBackend(_TestOnlyBackend):
    def load(self) -> None:
        raise RuntimeError("test-only failure")


def rgba_image() -> Image.Image:
    image = Image.new("RGBA", (40, 30), (0, 0, 0, 0))
    image.paste((30, 120, 220, 255), (5, 4, 35, 27))
    return image


def service(tmp_path: Path, factory: _Factory | None = None) -> AppInferenceService:
    return AppInferenceService(
        load_config(ROOT / "configs" / "quick_test.yaml"),
        tmp_path / "sessions",
        backend_factory=factory or _Factory(),
        cuda_provider=lambda: False,
        vram_provider=lambda: 0.0,
    )


def test_test_backend_pipeline_exports_viewer_and_download_files(tmp_path: Path) -> None:
    factory = _Factory()
    app = service(tmp_path, factory)
    result = app.generate(rgba_image(), AppOptions(repair_mesh=True, target_face_count=12))
    assert factory.instances[0].loaded and factory.instances[0].unloaded
    assert factory.instances[0].geometry_only is True
    assert result.session_dir.parent == app.output_root
    assert result.preprocessed_image.is_file()
    for path in (result.obj_path, result.glb_path, result.stl_path, result.metadata_path):
        assert path.is_file()
    for path in (result.obj_path, result.glb_path, result.stl_path):
        mesh = trimesh.load(path, force="mesh")
        assert len(mesh.vertices) == 8
        assert len(mesh.faces) == 12
    assert result.table[0][0] == "test-only"
    assert result.table[0][5] is True


def test_colab_free_policy_disables_heavy_options_without_vram(tmp_path: Path) -> None:
    factory = _Factory()
    result = service(tmp_path, factory).generate(
        rgba_image(),
        AppOptions(generate_texture=True, edge_refinement=True, repair_mesh=False),
    )
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["effective_options"]["generate_texture"] is False
    assert metadata["effective_options"]["edge_refinement"] is False
    assert any("Texture đã tắt" in item for item in metadata["warnings"])
    assert any("Refinement đã tắt" in item for item in metadata["warnings"])
    assert factory.instances[0].geometry_only is True


def test_sessions_are_unique_and_previous_outputs_are_not_deleted(tmp_path: Path) -> None:
    app = service(tmp_path)
    first = app.generate(rgba_image(), AppOptions(repair_mesh=False))
    second = app.generate(rgba_image(), AppOptions(repair_mesh=False))
    assert first.session_id != second.session_id
    assert first.session_dir != second.session_dir
    assert first.glb_path.is_file() and second.glb_path.is_file()
    assert first.session_dir.parent == second.session_dir.parent == app.output_root


def test_upload_filename_cannot_control_output_path(tmp_path: Path) -> None:
    source = tmp_path / "..evil.name.png"
    rgba_image().save(source)
    app = service(tmp_path)
    result = app.generate(source, AppOptions(repair_mesh=False))
    assert result.session_dir.parent == app.output_root
    assert not any("evil" in path.name for path in result.session_dir.iterdir())
    invalid = tmp_path / "mesh.obj"
    invalid.write_text("not an image", encoding="utf-8")
    with pytest.raises(AppPipelineError, match="PNG"):
        app.generate(invalid, AppOptions())


def test_backend_exception_is_friendly_and_model_is_unloaded(tmp_path: Path) -> None:
    factory = _Factory(_FailureBackend)
    app = service(tmp_path, factory)
    with pytest.raises(AppPipelineError, match="backend thật") as error:
        app.generate(rgba_image(), AppOptions())
    assert "test-only failure" not in str(error.value)
    assert factory.instances[0].unloaded is True


def test_gradio_app_builds_with_required_components(tmp_path: Path) -> None:
    code = (
        "import json,tempfile; from pathlib import Path; "
        "from image_to_3d_shapenet import load_config; "
        "from image_to_3d_shapenet.app.pipeline import AppInferenceService; "
        "from image_to_3d_shapenet.app.gradio_app import build_app; "
        "c=load_config('configs/quick_test.yaml'); "
        "s=AppInferenceService(c,Path(tempfile.mkdtemp())/'sessions'); "
        "d=build_app('configs/quick_test.yaml',service=s); x=d.get_config_file(); "
        "print(json.dumps({'labels':[i.get('props',{}).get('label') for i in x['components']],"
        "'types':[i['type'] for i in x['components']]}))"
    )
    environment = os.environ.copy()
    environment["PYTHONFAULTHANDLER"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment,
        text=True, capture_output=True, check=True,
    )
    config = json.loads(completed.stdout.strip().splitlines()[-1])
    labels = set(config["labels"])
    types = set(config["types"])
    assert "Ảnh gốc — kéo thả hoặc upload" in labels
    assert "Ảnh sau tách nền/crop" in labels
    assert "GLB Viewer" in labels
    assert "model3d" in types
    assert "downloadbutton" in types


def test_notebook_has_one_runnable_code_cell() -> None:
    notebook = json.loads(
        (ROOT / "notebooks" / "05_inference_gradio.ipynb").read_text(encoding="utf-8")
    )
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 1
    compile("".join(code[0]["source"]), "05_inference_gradio.ipynb", "exec")

