"""Session-isolated application inference pipeline used by the Gradio UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import gc
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable
from uuid import uuid4

import numpy as np
from PIL import Image
import trimesh

from image_to_3d_shapenet.config import ProjectConfig, RefinementConfig
from image_to_3d_shapenet.inference import AutoBackend
from image_to_3d_shapenet.inference.base import BaseImageTo3DBackend, GenerationResult, InferenceOptions
from image_to_3d_shapenet.mesh import mesh_report, process_mesh
from image_to_3d_shapenet.utils import cuda_available, total_vram_gb


BackendFactory = Callable[[Any, str], BaseImageTo3DBackend]


@dataclass(frozen=True)
class AppOptions:
    backend: str = "triposr"
    edge_refinement: bool = False
    generate_texture: bool = False
    repair_mesh: bool = True
    target_face_count: int = 50_000
    sharp_edge_angle: float = 35.0
    refinement_strength: float = 1.0


@dataclass(frozen=True)
class AppResult:
    session_id: str
    session_dir: Path
    preprocessed_image: Path
    obj_path: Path
    glb_path: Path
    stl_path: Path
    metadata_path: Path
    table: list[list[Any]]
    status: str
    metadata: dict[str, Any]


class AppPipelineError(RuntimeError):
    """Error safe to translate to a short user-facing message."""


def _default_backend_factory(config: Any, requested: str) -> BaseImageTo3DBackend:
    return AutoBackend(config, requested=requested)


def _as_mesh(value: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Trimesh):
        return value.copy()
    geometries = tuple(value.geometry.values())
    if not geometries:
        raise AppPipelineError("Backend không trả về geometry hợp lệ.")
    return trimesh.util.concatenate(geometries)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _export(value: trimesh.Trimesh | trimesh.Scene, path: Path, file_type: str) -> None:
    payload = value.export(file_type=file_type)
    temporary = path.with_name(path.name + ".tmp")
    if isinstance(payload, str):
        temporary.write_text(payload, encoding="utf-8")
    else:
        temporary.write_bytes(payload)
    os.replace(temporary, path)


def _cleanup_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, OSError):
        pass


class AppInferenceService:
    """Run one real backend at a time and retain every session independently."""

    _model_lock = threading.Lock()

    def __init__(
        self,
        config: ProjectConfig,
        output_root: str | Path,
        *,
        backend_factory: BackendFactory | None = None,
        vram_provider: Callable[[], float] = total_vram_gb,
        cuda_provider: Callable[[], bool] = cuda_available,
    ) -> None:
        self.config = config
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.backend_factory = backend_factory or _default_backend_factory
        self.vram_provider = vram_provider
        self.cuda_provider = cuda_provider

    def _session_dir(self) -> tuple[str, Path]:
        for _ in range(10):
            session_id = uuid4().hex
            path = (self.output_root / session_id).resolve()
            if path.parent != self.output_root:
                raise AppPipelineError("Không thể tạo thư mục phiên an toàn.")
            try:
                path.mkdir(parents=False, exist_ok=False)
                return session_id, path
            except FileExistsError:
                continue
        raise AppPipelineError("Không thể cấp mã phiên duy nhất.")

    def _read_image(self, value: str | Path | Image.Image | np.ndarray) -> Image.Image:
        maximum = int(self.config.app.max_upload_mb * 1024**2)
        if isinstance(value, Image.Image):
            image = value.copy()
        elif isinstance(value, np.ndarray):
            if value.nbytes > maximum:
                raise AppPipelineError("Ảnh tải lên vượt giới hạn dung lượng.")
            image = Image.fromarray(value)
        else:
            path = Path(value).expanduser().resolve()
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                raise AppPipelineError("Chỉ chấp nhận ảnh PNG, JPG hoặc JPEG.")
            if not path.is_file() or path.stat().st_size > maximum:
                raise AppPipelineError("Ảnh không tồn tại hoặc vượt giới hạn dung lượng.")
            with Image.open(path) as opened:
                image = opened.convert("RGBA")
        if image.width * image.height > 50_000_000:
            raise AppPipelineError("Ảnh có độ phân giải quá lớn.")
        return image.convert("RGBA")

    def _policy(self, options: AppOptions) -> tuple[AppOptions, list[str], float]:
        warnings_list: list[str] = []
        vram = float(self.vram_provider()) if self.cuda_provider() else 0.0
        texture = options.generate_texture
        refinement = options.edge_refinement
        if texture and vram < self.config.app.texture_min_vram_gb:
            texture = False
            warnings_list.append(
                f"Texture đã tắt: cần {self.config.app.texture_min_vram_gb:.1f} GiB VRAM, hiện có {vram:.1f} GiB."
            )
        if refinement and vram < self.config.app.refinement_min_vram_gb:
            refinement = False
            warnings_list.append(
                f"Refinement đã tắt: cần {self.config.app.refinement_min_vram_gb:.1f} GiB VRAM, hiện có {vram:.1f} GiB."
            )
        return replace(options, generate_texture=texture, edge_refinement=refinement), warnings_list, vram

    def _refine(self, mesh: trimesh.Trimesh, image: Image.Image, strength: float) -> trimesh.Trimesh:
        checkpoint_value = self.config.app.refinement_checkpoint or os.getenv("REFINER_CHECKPOINT")
        if not checkpoint_value:
            raise AppPipelineError("Refinement được yêu cầu nhưng chưa cấu hình REFINER_CHECKPOINT.")
        checkpoint_path = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise AppPipelineError("Không tìm thấy checkpoint refinement.")
        import torch

        from image_to_3d_shapenet.models import ImageConditionedMeshRefiner
        from image_to_3d_shapenet.rendering import camera_for_mesh

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_config = checkpoint.get("config", {}).get("refinement", {})
        refinement_config = RefinementConfig(**raw_config) if raw_config else self.config.refinement
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ImageConditionedMeshRefiner(refinement_config).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        center = np.asarray(mesh.bounds, dtype=np.float32).mean(axis=0)
        scale = max(float(np.ptp(vertices, axis=0).max()), 1e-6)
        normalized = (vertices - center) / scale
        normalized_mesh = trimesh.Trimesh(vertices=normalized, faces=np.asarray(mesh.faces), process=False)
        size = min(int(self.config.inference.input_size), 512)
        camera = camera_for_mesh(
            normalized_mesh, width=size, height=size, azimuth_degrees=90.0,
            elevation_degrees=0.0, fov_degrees=50.0, framing_margin=1.15,
        )
        rgba = image.resize((size, size), Image.Resampling.LANCZOS)
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        array = np.asarray(background.convert("RGB"), dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
        vertex_tensor = torch.from_numpy(normalized).unsqueeze(0).to(device)
        face_tensor = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64)).to(device)
        intrinsics = torch.from_numpy(camera.intrinsics.astype(np.float32)).unsqueeze(0).to(device)
        extrinsics = torch.from_numpy(camera.world_to_camera.astype(np.float32)).unsqueeze(0).to(device)
        with torch.inference_mode():
            result = model(image_tensor, vertex_tensor, face_tensor, intrinsics, extrinsics)
        refined = (normalized + result.displacement[0].detach().cpu().numpy() * float(strength)) * scale + center
        output = trimesh.Trimesh(vertices=refined, faces=np.asarray(mesh.faces), process=False)
        del model, checkpoint, result
        _cleanup_gpu()
        return output

    def generate(self, image_value: str | Path | Image.Image | np.ndarray, options: AppOptions) -> AppResult:
        if options.backend not in {"auto", "triposr", "stable-fast-3d", "hunyuan3d-2mini"}:
            raise AppPipelineError("Backend không hợp lệ.")
        if not 4 <= int(options.target_face_count) <= 1_000_000:
            raise AppPipelineError("Target face count phải nằm trong [4, 1.000.000].")
        if not 1.0 <= float(options.sharp_edge_angle) < 180.0:
            raise AppPipelineError("Sharp-edge angle phải nằm trong [1, 180).")
        if not 0.0 <= float(options.refinement_strength) <= 1.0:
            raise AppPipelineError("Refinement strength phải nằm trong [0, 1].")
        image = self._read_image(image_value)
        session_id, session_dir = self._session_dir()
        image.save(session_dir / "input_original.png", format="PNG")
        effective, policy_warnings, available_vram = self._policy(options)
        inference_config = replace(
            self.config.inference, backend=effective.backend,
            mode="high_quality" if effective.backend == "hunyuan3d-2mini" else "standard",
            geometry_only=not effective.generate_texture,
        )
        backend = self.backend_factory(inference_config, effective.backend)
        result: GenerationResult | None = None
        with self._model_lock:
            try:
                prepared = backend.preprocess(image)
                preprocessed_path = session_dir / "input_preprocessed.png"
                prepared.image.save(preprocessed_path, format="PNG")
                backend.load()
                result = backend.generate(
                    prepared, InferenceOptions(geometry_only=inference_config.geometry_only, seed=inference_config.seed)
                )
                result.validate()
            except AppPipelineError:
                raise
            except Exception as exc:
                raise AppPipelineError(
                    "Không thể tạo mesh bằng backend thật. Kiểm tra CUDA, dependency, weights/quyền model. "
                    f"Chi tiết: {type(exc).__name__}."
                ) from exc
            finally:
                backend.unload()
                _cleanup_gpu()
        assert result is not None
        warnings_list = [*policy_warnings, *result.warnings]
        final_value: trimesh.Trimesh | trimesh.Scene = result.mesh
        final_mesh = _as_mesh(result.mesh)
        refinement_applied = False
        if effective.edge_refinement:
            try:
                final_mesh = self._refine(final_mesh, prepared.image, effective.refinement_strength)
                final_value = final_mesh
                refinement_applied = True
                if effective.generate_texture:
                    warnings_list.append("Texture bị loại sau refinement vì vertex đã thay đổi.")
            except AppPipelineError as exc:
                warnings_list.append(str(exc))
        repair_steps: list[dict[str, Any]] = []
        if effective.repair_mesh:
            repair_config = replace(
                self.config.mesh_postprocess,
                sharp_edge_threshold_degrees=float(effective.sharp_edge_angle),
                decimate=len(final_mesh.faces) > int(effective.target_face_count),
                target_face_count=int(effective.target_face_count),
            )
            final_mesh, repair_steps = process_mesh(final_mesh, repair_config)
            final_value = final_mesh
            if effective.generate_texture:
                warnings_list.append("Texture bị loại khi repair mesh được áp dụng.")
        report = mesh_report(final_mesh, sharp_threshold_degrees=float(effective.sharp_edge_angle))
        if not report["watertight"]:
            warnings_list.append("Mesh không watertight; STL đã xuất nhưng có thể không phù hợp để in 3D.")
        obj_path = session_dir / "final_mesh.obj"
        glb_path = session_dir / "final_mesh.glb"
        stl_path = session_dir / "final_mesh.stl"
        metadata_path = session_dir / "metadata.json"
        _export(final_mesh, obj_path, "obj")
        _export(final_value, glb_path, "glb")
        _export(final_mesh, stl_path, "stl")
        warning_text = " | ".join(dict.fromkeys(warnings_list))
        metadata = {
            "session_id": session_id, "preset": self.config.app.preset,
            "requested_options": asdict(options), "effective_options": asdict(effective),
            "available_vram_gb": available_vram, "backend": result.backend_name,
            "inference_time_seconds": result.inference_time_seconds,
            "peak_vram_gb": result.peak_vram_gb, "texture_status": result.texture_status,
            "refinement_applied": refinement_applied, "repair_steps": repair_steps,
            "mesh_report": report, "warnings": list(dict.fromkeys(warnings_list)),
            "files": {"obj": obj_path.name, "glb": glb_path.name, "stl": stl_path.name, "preprocessed": preprocessed_path.name},
        }
        _atomic_json(metadata_path, metadata)
        table = [[
            result.backend_name, round(float(result.inference_time_seconds), 3),
            round(float(result.peak_vram_gb), 3), int(report["vertices"]),
            int(report["faces"]), bool(report["watertight"]), warning_text,
        ]]
        status = f"Hoàn tất phiên `{session_id}`. Mesh đã sẵn sàng để xem hoặc tải xuống."
        return AppResult(
            session_id, session_dir, preprocessed_path, obj_path, glb_path, stl_path,
            metadata_path, table, status, metadata,
        )
