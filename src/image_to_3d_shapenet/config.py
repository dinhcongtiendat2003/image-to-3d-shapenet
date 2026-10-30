"""Typed YAML configuration with lightweight dataclass validation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    image_size: int = 256
    batch_size: int = 1
    mixed_precision: bool = True
    quick_test: bool = False
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("runtime.seed must be non-negative")
        if self.image_size <= 0:
            raise ValueError("runtime.image_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("runtime.batch_size must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("runtime.device must be auto, cpu, or cuda")


@dataclass(frozen=True)
class DatasetConfig:
    repo_id: str = "ShapeNet/ShapeNetCore"
    category_name: str = "chair"
    synset_id: str = "03001627"
    zip_filename: str = "03001627.zip"
    views_per_mesh: int = 8
    max_meshes: int | None = None
    max_train_meshes: int | None = None
    max_val_meshes: int | None = None
    max_test_meshes: int | None = None
    normalization: str = "unit_cube"
    max_degenerate_face_ratio: float = 0.05
    max_mesh_file_mb: float = 512.0
    max_archive_size_gb: float = 12.0
    max_extracted_size_gb: float = 25.0
    min_free_disk_gb: float = 2.0
    max_archive_members: int = 250_000

    def __post_init__(self) -> None:
        if not self.repo_id or not self.category_name:
            raise ValueError("dataset repo_id and category_name are required")
        if not (self.synset_id.isdigit() and len(self.synset_id) == 8):
            raise ValueError("dataset.synset_id must be an 8-digit string")
        if not self.zip_filename.endswith(".zip"):
            raise ValueError("dataset.zip_filename must end in .zip")
        if self.views_per_mesh <= 0:
            raise ValueError("dataset.views_per_mesh must be positive")
        if self.max_meshes is not None and self.max_meshes <= 0:
            raise ValueError("dataset.max_meshes must be positive or null")
        for name in ("max_train_meshes", "max_val_meshes", "max_test_meshes"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"dataset.{name} must be positive or null")
        if self.normalization not in {"unit_cube", "unit_sphere"}:
            raise ValueError("dataset.normalization must be unit_cube or unit_sphere")
        if not 0.0 <= self.max_degenerate_face_ratio <= 1.0:
            raise ValueError("dataset.max_degenerate_face_ratio must be in [0, 1]")
        if self.max_mesh_file_mb <= 0:
            raise ValueError("dataset.max_mesh_file_mb must be positive")
        if self.max_archive_size_gb <= 0 or self.max_extracted_size_gb <= 0:
            raise ValueError("dataset archive size limits must be positive")
        if self.min_free_disk_gb < 0:
            raise ValueError("dataset.min_free_disk_gb must be non-negative")
        if self.max_archive_members <= 0:
            raise ValueError("dataset.max_archive_members must be positive")


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str = "data"
    cache_dir: str = ".cache"
    output_dir: str = "outputs"


@dataclass(frozen=True)
class ExportConfig:
    formats: tuple[str, ...] = ("obj", "glb", "stl")

    def __post_init__(self) -> None:
        normalized = tuple(value.lower() for value in self.formats)
        invalid = set(normalized) - {"obj", "glb", "stl"}
        if invalid:
            raise ValueError(f"unsupported export formats: {sorted(invalid)}")
        object.__setattr__(self, "formats", normalized)


@dataclass(frozen=True)
class RenderConfig:
    backend: str = "auto"
    output_subdir: str = "renders"
    max_render_meshes: int | None = None
    fov_degrees: float = 50.0
    elevation_min_degrees: float = 15.0
    elevation_max_degrees: float = 30.0
    framing_margin: float = 1.15
    transparent_background: bool = True
    random_background_train: bool = False
    background_rgb: tuple[int, int, int] = (245, 245, 245)
    ambient_light: float = 0.35
    directional_light: float = 2.0
    pyrender_platform: str = "egl"

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "pyrender", "software"}:
            raise ValueError("render.backend must be auto, pyrender, or software")
        if self.max_render_meshes is not None and self.max_render_meshes <= 0:
            raise ValueError("render.max_render_meshes must be positive or null")
        if not 1.0 < self.fov_degrees < 179.0:
            raise ValueError("render.fov_degrees must be in (1, 179)")
        if self.elevation_min_degrees > self.elevation_max_degrees:
            raise ValueError("render elevation range is invalid")
        if self.framing_margin <= 1.0:
            raise ValueError("render.framing_margin must be greater than 1")
        if len(self.background_rgb) != 3 or any(not 0 <= value <= 255 for value in self.background_rgb):
            raise ValueError("render.background_rgb must contain three values in [0, 255]")
        if self.ambient_light < 0 or self.directional_light < 0:
            raise ValueError("render light intensities must be non-negative")
        if self.pyrender_platform not in {"egl", "osmesa", "pyglet"}:
            raise ValueError("render.pyrender_platform must be egl, osmesa, or pyglet")
        object.__setattr__(self, "background_rgb", tuple(self.background_rgb))


@dataclass(frozen=True)
class InferenceConfig:
    backend: str = "auto"
    mode: str = "standard"
    geometry_only: bool = True
    input_size: int = 512
    foreground_ratio: float = 0.85
    remove_background: bool = True
    allow_cpu: bool = False
    allow_download: bool = True
    seed: int = 42
    triposr_model_id: str = "stabilityai/TripoSR"
    triposr_chunk_size: int = 8192
    triposr_mc_resolution: int = 256
    stable_fast_3d_model_id: str = "stabilityai/stable-fast-3d"
    stable_fast_3d_texture_resolution: int = 1024
    hunyuan_model_id: str = "tencent/Hunyuan3D-2mini"
    hunyuan_subfolder: str = "hunyuan3d-dit-v2-mini-turbo"
    hunyuan_steps: int = 5
    hunyuan_octree_resolution: int = 256

    def __post_init__(self) -> None:
        choices = {"auto", "triposr", "stable-fast-3d", "hunyuan3d-2mini"}
        if self.backend not in choices:
            raise ValueError(f"inference.backend must be one of {sorted(choices)}")
        if self.mode not in {"standard", "high_quality"}:
            raise ValueError("inference.mode must be standard or high_quality")
        if self.input_size <= 0 or self.input_size > 2048:
            raise ValueError("inference.input_size must be in [1, 2048]")
        if not 0.1 <= self.foreground_ratio <= 1.0:
            raise ValueError("inference.foreground_ratio must be in [0.1, 1.0]")
        for name in (
            "triposr_chunk_size",
            "triposr_mc_resolution",
            "stable_fast_3d_texture_resolution",
            "hunyuan_steps",
            "hunyuan_octree_resolution",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"inference.{name} must be positive")


@dataclass(frozen=True)
class RefinementConfig:
    encoder: str = "resnet18"
    pretrained_encoder: bool = True
    freeze_encoder: bool = True
    allow_encoder_fallback: bool = True
    camera_convention: str = "opengl"
    feature_dim: int = 32
    hidden_dim: int = 64
    num_blocks: int = 3
    max_displacement: float = 0.05
    sharp_edge_threshold_degrees: float = 35.0
    sharp_edge_message_weight: float = 0.1
    chamfer_samples: int = 512
    sharp_edge_samples: int = 256
    sharp_edge_sampling_weight: float = 2.0
    loss_backend: str = "auto"
    lambda_chamfer: float = 1.0
    lambda_normal: float = 0.1
    lambda_silhouette: float = 0.1
    lambda_edge: float = 0.5
    lambda_laplacian: float = 0.02
    lambda_displacement: float = 0.01
    lambda_boundary: float = 0.05

    def __post_init__(self) -> None:
        if self.encoder not in {"resnet18", "lightweight"}:
            raise ValueError("refinement.encoder must be resnet18 or lightweight")
        if self.camera_convention not in {"opengl", "opencv"}:
            raise ValueError("refinement.camera_convention must be opengl or opencv")
        if self.feature_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("refinement feature dimensions must be positive")
        if not 2 <= self.num_blocks <= 4:
            raise ValueError("refinement.num_blocks must be in [2, 4]")
        if self.max_displacement <= 0:
            raise ValueError("refinement.max_displacement must be positive")
        if not 0 < self.sharp_edge_threshold_degrees < 180:
            raise ValueError("refinement sharp edge threshold must be in (0, 180)")
        if not 0 <= self.sharp_edge_message_weight <= 1:
            raise ValueError("refinement.sharp_edge_message_weight must be in [0, 1]")
        if self.chamfer_samples <= 0 or self.sharp_edge_samples <= 0:
            raise ValueError("refinement sample counts must be positive")
        if self.sharp_edge_sampling_weight < 1.0:
            raise ValueError("refinement.sharp_edge_sampling_weight must be at least 1")
        if self.loss_backend not in {"auto", "native", "pytorch3d"}:
            raise ValueError("refinement.loss_backend must be auto, native, or pytorch3d")
        for name in (
            "lambda_chamfer", "lambda_normal", "lambda_silhouette", "lambda_edge",
            "lambda_laplacian", "lambda_displacement", "lambda_boundary",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"refinement.{name} must be non-negative")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-5
    surface_samples: int = 512
    minimum_surface_samples: int = 64
    max_vertices: int = 20000
    minimum_vertices: int = 1000
    image_resolution: int = 256
    minimum_image_resolution: int = 64
    quick_samples: int = 4
    num_workers: int = 0
    checkpoint_subdir: str = "checkpoints"
    base_mesh_cache_subdir: str = "base_mesh_cache"

    def __post_init__(self) -> None:
        positive = (
            "epochs", "gradient_accumulation_steps", "scheduler_patience",
            "early_stopping_patience", "surface_samples", "minimum_surface_samples",
            "max_vertices", "minimum_vertices", "image_resolution",
            "minimum_image_resolution", "quick_samples",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("training learning_rate/weight_decay are invalid")
        if self.gradient_clip_norm <= 0:
            raise ValueError("training.gradient_clip_norm must be positive")
        if not 0 < self.scheduler_factor < 1:
            raise ValueError("training.scheduler_factor must be in (0, 1)")
        if self.early_stopping_min_delta < 0:
            raise ValueError("training.early_stopping_min_delta must be non-negative")
        if self.minimum_surface_samples > self.surface_samples:
            raise ValueError("training minimum surface samples exceed initial samples")
        if self.minimum_vertices > self.max_vertices:
            raise ValueError("training minimum vertices exceed max vertices")
        if self.minimum_image_resolution > self.image_resolution:
            raise ValueError("training minimum image resolution exceeds initial resolution")
        if not 2 <= self.quick_samples <= 5:
            raise ValueError("training.quick_samples must be in [2, 5]")
        if self.num_workers < 0:
            raise ValueError("training.num_workers must be non-negative")


@dataclass(frozen=True)
class EvaluationConfig:
    max_test_samples: int = 10
    surface_samples: int = 2048
    edge_samples: int = 512
    sharp_edge_threshold_degrees: float = 35.0
    fscore_thresholds: tuple[float, float] = (0.01, 0.02)
    output_subdir: str = "evaluation"
    configurations: tuple[str, ...] = (
        "base", "base_refiner", "refiner_no_edge", "refiner_full"
    )
    backend_filters: tuple[str, ...] = (
        "cached", "triposr", "stable-fast-3d", "hunyuan3d-2mini"
    )
    comparison_samples: int = 4

    def __post_init__(self) -> None:
        if self.max_test_samples <= 0 or self.surface_samples <= 0 or self.edge_samples <= 0:
            raise ValueError("evaluation sample limits must be positive")
        if not 0 < self.sharp_edge_threshold_degrees < 180:
            raise ValueError("evaluation sharp-edge threshold must be in (0, 180)")
        if len(self.fscore_thresholds) != 2 or any(value <= 0 for value in self.fscore_thresholds):
            raise ValueError("evaluation requires two positive F-score thresholds")
        allowed = {"base", "base_refiner", "refiner_no_edge", "refiner_full"}
        if not self.configurations or set(self.configurations) - allowed:
            raise ValueError("evaluation contains an unknown configuration")
        backends = {"cached", "triposr", "stable-fast-3d", "hunyuan3d-2mini"}
        if not self.backend_filters or set(self.backend_filters) - backends:
            raise ValueError("evaluation contains an unknown backend filter")
        if self.comparison_samples <= 0:
            raise ValueError("evaluation.comparison_samples must be positive")


@dataclass(frozen=True)
class MeshPostprocessConfig:
    """Conservative, dependency-light mesh cleanup and export settings."""

    merge_duplicate_vertices: bool = True
    remove_duplicate_faces: bool = True
    remove_degenerate_faces: bool = True
    remove_unreferenced_vertices: bool = True
    fix_face_orientation: bool = True
    fix_normals: bool = True
    remove_small_components: bool = True
    fill_small_holes: bool = True
    detect_non_manifold_edges: bool = True
    detect_sharp_edges: bool = True
    check_watertight: bool = True
    remesh: bool = False
    decimate: bool = False
    smooth: bool = False
    sharp_edge_threshold_degrees: float = 35.0
    merge_digits: int = 8
    degenerate_height: float = 1e-10
    min_component_faces: int = 4
    min_component_area_ratio: float = 0.01
    max_hole_faces: int = 4
    remesh_max_edge: float = 0.05
    remesh_max_iterations: int = 1
    target_face_count: int | None = None
    decimation_max_collapses: int = 5000
    smoothing_iterations: int = 0
    smoothing_lambda: float = 0.25

    def __post_init__(self) -> None:
        if not 0 < self.sharp_edge_threshold_degrees < 180:
            raise ValueError("mesh_postprocess sharp-edge threshold must be in (0, 180)")
        if self.merge_digits < 0 or self.merge_digits > 15:
            raise ValueError("mesh_postprocess.merge_digits must be in [0, 15]")
        if self.degenerate_height <= 0:
            raise ValueError("mesh_postprocess.degenerate_height must be positive")
        if self.min_component_faces <= 0:
            raise ValueError("mesh_postprocess.min_component_faces must be positive")
        if not 0 <= self.min_component_area_ratio <= 1:
            raise ValueError("mesh_postprocess.min_component_area_ratio must be in [0, 1]")
        if self.max_hole_faces < 0 or self.remesh_max_iterations < 0:
            raise ValueError("mesh_postprocess hole/remesh limits must be non-negative")
        if self.remesh_max_edge <= 0:
            raise ValueError("mesh_postprocess.remesh_max_edge must be positive")
        if self.target_face_count is not None and self.target_face_count < 4:
            raise ValueError("mesh_postprocess.target_face_count must be at least 4 or null")
        if self.decimation_max_collapses <= 0:
            raise ValueError("mesh_postprocess.decimation_max_collapses must be positive")
        if self.smoothing_iterations < 0 or not 0 < self.smoothing_lambda <= 1:
            raise ValueError("mesh_postprocess smoothing settings are invalid")


@dataclass(frozen=True)
class AppConfig:
    preset: str = "Colab Free"
    output_subdir: str = "gradio_sessions"
    default_backend: str = "triposr"
    default_geometry_only: bool = True
    texture_min_vram_gb: float = 8.0
    refinement_min_vram_gb: float = 8.0
    max_upload_mb: float = 25.0
    refinement_checkpoint: str | None = None
    server_name: str = "0.0.0.0"
    server_port: int = 7860

    def __post_init__(self) -> None:
        if self.preset != "Colab Free":
            raise ValueError("app.preset must be 'Colab Free'")
        if self.default_backend not in {
            "auto", "triposr", "stable-fast-3d", "hunyuan3d-2mini"
        }:
            raise ValueError("app.default_backend is invalid")
        if self.texture_min_vram_gb < 0 or self.refinement_min_vram_gb < 0:
            raise ValueError("app VRAM thresholds must be non-negative")
        if self.max_upload_mb <= 0:
            raise ValueError("app.max_upload_mb must be positive")
        if not 1 <= self.server_port <= 65535:
            raise ValueError("app.server_port must be in [1, 65535]")


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str = "image-to-3d-shapenet"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    mesh_postprocess: MeshPostprocessConfig = field(default_factory=MeshPostprocessConfig)
    app: AppConfig = field(default_factory=AppConfig)


def _build(cls: type[Any], raw: Mapping[str, Any] | None, section: str) -> Any:
    values = dict(raw or {})
    known = {item.name for item in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys in {section}: {sorted(unknown)}")
    if cls is ExportConfig and "formats" in values:
        values["formats"] = tuple(values["formats"])
    if cls is RenderConfig and "background_rgb" in values:
        values["background_rgb"] = tuple(values["background_rgb"])
    if cls is EvaluationConfig:
        for name in ("fscore_thresholds", "configurations", "backend_filters"):
            if name in values:
                values[name] = tuple(values[name])
    return cls(**values)


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a project YAML config without resolving machine paths."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError("config root must be a YAML mapping")
    allowed = {
        "project_name", "runtime", "dataset", "paths", "export", "render", "inference",
        "refinement",
        "training",
        "evaluation",
        "mesh_postprocess",
        "app",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    return ProjectConfig(
        project_name=str(raw.get("project_name", "image-to-3d-shapenet")),
        runtime=_build(RuntimeConfig, raw.get("runtime"), "runtime"),
        dataset=_build(DatasetConfig, raw.get("dataset"), "dataset"),
        paths=_build(PathsConfig, raw.get("paths"), "paths"),
        export=_build(ExportConfig, raw.get("export"), "export"),
        render=_build(RenderConfig, raw.get("render"), "render"),
        inference=_build(InferenceConfig, raw.get("inference"), "inference"),
        refinement=_build(RefinementConfig, raw.get("refinement"), "refinement"),
        training=_build(TrainingConfig, raw.get("training"), "training"),
        evaluation=_build(EvaluationConfig, raw.get("evaluation"), "evaluation"),
        mesh_postprocess=_build(
            MeshPostprocessConfig, raw.get("mesh_postprocess"), "mesh_postprocess"
        ),
        app=_build(AppConfig, raw.get("app"), "app"),
    )

