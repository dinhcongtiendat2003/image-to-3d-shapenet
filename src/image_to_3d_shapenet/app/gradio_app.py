"""Gradio interface for real image-to-3D inference with safe downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from image_to_3d_shapenet import load_config

from .pipeline import AppInferenceService, AppOptions, AppPipelineError


BACKEND_LABELS = {
    "Auto": "auto",
    "Hunyuan3D-2mini": "hunyuan3d-2mini",
    "Stable Fast 3D": "stable-fast-3d",
    "TripoSR": "triposr",
}


def create_service(config_path: str | Path) -> AppInferenceService:
    path = Path(config_path).expanduser().resolve()
    config = load_config(path)
    project_root = path.parent.parent
    output_root = project_root / config.paths.output_dir / config.app.output_subdir
    return AppInferenceService(config, output_root)


def _callback(service: AppInferenceService):
    def generate(
        image: Any,
        backend_label: str,
        edge_refinement: bool,
        generate_texture: bool,
        repair_mesh: bool,
        target_face_count: int,
        sharp_edge_angle: float,
        refinement_strength: float,
    ) -> tuple[Any, ...]:
        empty = (None, None, None, None, None, None, [])
        if image is None:
            return (*empty, "❌ Hãy tải lên một ảnh PNG hoặc JPG trước.")
        try:
            result = service.generate(
                image,
                AppOptions(
                    backend=BACKEND_LABELS.get(backend_label, "auto"),
                    edge_refinement=bool(edge_refinement),
                    generate_texture=bool(generate_texture),
                    repair_mesh=bool(repair_mesh),
                    target_face_count=int(target_face_count),
                    sharp_edge_angle=float(sharp_edge_angle),
                    refinement_strength=float(refinement_strength),
                ),
            )
        except AppPipelineError as exc:
            return (*empty, f"❌ {exc}")
        except Exception as exc:
            return (*empty, f"❌ Lỗi không mong đợi ({type(exc).__name__}). Xem log runtime để chẩn đoán.")
        return (
            str(result.preprocessed_image), str(result.glb_path), str(result.obj_path),
            str(result.glb_path), str(result.stl_path), str(result.metadata_path),
            result.table, f"✅ {result.status}",
        )

    return generate


def build_app(
    config_path: str | Path = "configs/inference.yaml",
    *,
    service: AppInferenceService | None = None,
):
    import gradio as gr

    active_service = service or create_service(config_path)
    default_label = next(
        label for label, value in BACKEND_LABELS.items()
        if value == active_service.config.app.default_backend
    )
    with gr.Blocks(title="Single Image to 3D — Colab Free") as demo:
        gr.Markdown(
            "# Single Image → 3D\n"
            "Preset **Colab Free**: TripoSR, geometry-only, batch 1. "
            "Texture/refinement tự tắt khi không đủ VRAM."
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Ảnh gốc — kéo thả hoặc upload", type="pil",
                    sources=["upload", "clipboard"],
                )
                processed_image = gr.Image(label="Ảnh sau tách nền/crop", interactive=False)
            with gr.Column():
                backend = gr.Dropdown(
                    choices=list(BACKEND_LABELS), value=default_label, label="Backend"
                )
                edge_refinement = gr.Checkbox(value=False, label="Edge-aware refinement")
                texture = gr.Checkbox(
                    value=not active_service.config.app.default_geometry_only,
                    label="Generate texture",
                )
                repair = gr.Checkbox(value=True, label="Repair mesh")
                target_faces = gr.Slider(
                    1_000, 100_000, value=50_000, step=1_000, label="Target face count"
                )
                sharp_angle = gr.Slider(
                    10, 90, value=35, step=1, label="Sharp-edge angle (degrees)"
                )
                strength = gr.Slider(
                    0, 1, value=1, step=0.05, label="Refinement strength"
                )
                generate_button = gr.Button("Generate 3D", variant="primary")
                status = gr.Markdown("Sẵn sàng.")
        viewer = gr.Model3D(label="GLB Viewer", clear_color=[0.95, 0.95, 0.95, 1.0])
        with gr.Row():
            download_obj = gr.DownloadButton("Tải OBJ")
            download_glb = gr.DownloadButton("Tải GLB")
            download_stl = gr.DownloadButton("Tải STL")
            download_metadata = gr.DownloadButton("Tải metadata JSON")
        result_table = gr.Dataframe(
            headers=[
                "backend", "inference time (s)", "peak VRAM (GiB)",
                "vertices", "faces", "watertight", "warning",
            ],
            datatype=["str", "number", "number", "number", "number", "bool", "str"],
            row_count=1, column_count=7, interactive=False,
            label="Kết quả",
        )
        generate_button.click(
            fn=_callback(active_service),
            inputs=[
                input_image, backend, edge_refinement, texture, repair,
                target_faces, sharp_angle, strength,
            ],
            outputs=[
                processed_image, viewer, download_obj, download_glb, download_stl,
                download_metadata, result_table, status,
            ],
        )
    return demo.queue(default_concurrency_limit=1)

