# Final Audit

Ngày kiểm định: **2026-10-30 (Asia/Bangkok)**

Kết luận tổng thể: **PARTIAL — chưa đủ bằng chứng để tuyên bố hoàn thành trên
Google Colab Free**. Quick pipeline đã chạy end-to-end trên CPU local từ một bản
sao sạch. Colab GPU, ShapeNetCore gated thật và pretrained inference thật chưa
thể kiểm chứng trong môi trường audit này, nên được ghi `NOT VERIFIED` thay vì
giả định thành công.

## Ma trận yêu cầu

| Yêu cầu | Trạng thái | Bằng chứng | File liên quan | Hạn chế |
|---|---|---|---|---|
| Kiểm tra toàn repository | PASS | Full tests, compile, secret/size/path/weight scan | `tests/`, `pyproject.toml`, `.gitignore` | Workspace không có Git metadata nên không audit commit status |
| 5 notebook 01–05 tồn tại | PASS | `tests/test_notebooks.py`: 6 test pass | `notebooks/01_...` đến `05_...` | Static validation/compile; chưa Run all trên Colab thật |
| Quick test, path portable, dependency pin | PASS | Notebook test kiểm tra marker và compile từng code cell | `requirements-colab.txt`, 5 notebook | `pip install` phụ thuộc availability của PyPI/GitHub |
| GPU/HF_TOKEN/Drive/error/resume trong notebook | PASS (code audit) | Tất cả notebook có check/mount/error/resume path | 5 notebook | Drive mount/share URL chưa kiểm chứng trên Colab |
| ShapeNetCore một category, gated access | IMPLEMENTED / NOT VERIFIED | Downloader dùng đúng `03001627.zip`, unit tests pass | `dataset/download.py`, `chair_colab.yaml` | Không có quyền/token trong audit; chưa tải Chair thật |
| Safe extraction, validate, normalize, split | PASS | Unit tests và fresh synthetic prepare: 3 discovered, 2 valid, 1 rejected | `dataset/`, `tests/test_dataset_*` | Synthetic không đại diện mọi lỗi ShapeNetCore |
| Headless render và resume | PASS (quick local) | Fresh run: 4/4 sample, 0 failure; renderer tests pass | `rendering/`, `02_render_shapenet.ipynb` | Colab EGL/GPU chưa đo; software fallback có thể chậm |
| Pretrained backend thật và fallback | IMPLEMENTED / NOT VERIFIED | Registry/selection/failure tests; production registry không có fake | `inference/`, `tests/test_inference_backends.py` | Chưa chạy TripoSR/SF3D/Hunyuan weights thật |
| Edge-aware refiner forward/backward | PASS | Refiner unit tests; quick train loss giảm | `models/`, `losses/`, `tests/test_refinement.py` | Checkpoint quick là synthetic, không chứng minh quality Chair |
| Cache base mesh | PASS (quick) | Fresh cache 4/4; manifest/resume tests | `dataset/base_mesh_cache.py` | Quick cache gắn nhãn test-only, không phải pretrained result |
| Train + checkpoint | PASS (quick local) | 3 epoch, global step 3, best Chamfer 0.000880 | `training/`, `03_train_refiner.ipynb` | CPU, mixed precision false |
| Checkpoint resume | PASS | Fresh resume load `last.pt`, giữ global step 3; unit test kiểm tra restore | `training/checkpoint.py`, `tests/test_training_pipeline.py` | Run đã hoàn tất epoch nên không tăng thêm step |
| Scientific evaluation + resume | PASS (quick local) | 4 success, 0 failure, 6 summary rows; rerun vẫn 4 CSV rows | `metrics/`, `04_evaluate.ipynb` | Chưa benchmark Chair thật/nhiều backend |
| OBJ/GLB/STL | PASS | Fresh export và round-trip: 8 vertices, 12 faces, watertight ở cả 3 format | `mesh/postprocess.py`, `tests/test_mesh_postprocess.py` | Texture preservation sau repair/refinement bị giới hạn |
| Gradio startup/download pipeline | PASS (local/test backend) | Server mở `127.0.0.1:7863` rồi đóng; app tests pass | `app/`, `05_inference_gradio.ipynb` | Share URL và generation model thật chưa xác minh |
| README tiếng Việt + Mermaid | PASS | Manual audit các mục bắt buộc | `README.md` | Số đo là local CPU, không đại diện Colab GPU |
| Report outline tiếng Việt | PASS | Đủ 13 mục yêu cầu | `docs/report_outline_vi.md` | Chưa phải báo cáo thực nghiệm hoàn chỉnh |
| License/attribution | PASS (link audit) | Link official và tóm tắt constraint | `README.md`, `NOTICE.md` | Người dùng phải đọc bản license hiện hành |
| Không secret/token hardcode | PASS | Regex scan không thấy token-like value | toàn repository | `HF_TOKEN` chỉ xuất hiện như tên biến/secret |
| Không dataset/model weights/output lớn | PASS sau cleanup | Không file weight; file lớn nhất dưới 1 MB; outputs chỉ `.gitkeep` | `.gitignore`, `outputs/.gitkeep` | Synthetic data nhỏ còn dưới `data/` và bị gitignore |
| Không personal path | PASS | Scan không thấy Windows/Unix home prefix hoặc đường workspace | source/config/notebook/docs | Runtime report không được nhúng absolute temp path |
| Không mock production result | PASS | Fake/test backend chỉ ở tests; synthetic cache gắn `test-only` và chỉ quick mode | `tests/`, `dataset/base_mesh_cache.py` | Quick result tuyệt đối không dùng làm production claim |

## Fresh quick pipeline đã chạy

Môi trường: Windows 11, Python 3.12.10, RAM tổng 7.21 GB, CPU-only, CUDA false,
VRAM 0 GB, disk trống 217.55 GB. Bản sao audit ban đầu không có `data/`,
`outputs/` hoặc `.cache/`.

| Stage | Kết quả thực tế | Thời gian | Peak RAM | Disk workspace |
|---|---|---:|---:|---:|
| `prepare_dataset.py` | 3 discovered, 2 valid, 1 invalid | 5.25 s | 221.8 MB | 0.51 MB |
| `render_dataset.py` | 4 rendered, 0 failed | 9.77 s | 405.3 MB | 0.82 MB |
| `generate_base_mesh_cache.py` | 4 generated, 0 failed | 5.38 s | 243.1 MB | 0.82 MB |
| `train.py --quick-overfit` | 3 epoch, global step 3 | 13.35 s | 315.9 MB | 1.94 MB |
| `train.py --resume` | checkpoint restore pass | 11.11 s | 301.4 MB | 1.94 MB |
| `evaluate.py` | 4 success, 0 failure | 4.76 s | 236.5 MB | 2.03 MB |
| `postprocess_mesh.py` | OBJ/GLB/STL pass | 3.20 s | 208.1 MB | 2.03 MB |

Tổng: **52.82 giây**. Evaluation resume riêng: **5.72 giây**, CSV vẫn 4 hàng.
App startup/close: **14.15 giây**. Không có số đo VRAM vì host không có CUDA.

Train loss giảm `0.003344 → 0.002189 → 0.001801`. Validation Chamfer giảm
`0.001120 → 0.000978 → 0.000880`.

## License đã kiểm tra

- ShapeNetCore official gated terms:
  https://huggingface.co/datasets/ShapeNet/ShapeNetCore/blob/main/README.md
  — non-commercial research/education, hạn chế redistribution.
- TripoSR official license:
  https://github.com/VAST-AI-Research/TripoSR/blob/main/LICENSE — MIT.
- Stable Fast 3D official license:
  https://github.com/Stability-AI/stable-fast-3d/blob/main/LICENSE.md
  — Stability AI Community License.
- Hunyuan3D-2 official license:
  https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE
  — Tencent Hunyuan 3D 2.0 Community License.

## Dependency/chế độ chưa xác minh

- Colab GPU runtime, peak VRAM và thời gian thực tế.
- TripoSR `torchmcubes` build và weights thật.
- Stable Fast 3D gated weights/source.
- Hunyuan3D-2mini gated weights, territory/license eligibility và VRAM.
- `pymeshlab` optional.
- ShapeNetCore Chair gated ZIP thật.
- Gradio public share link.

Stable Fast 3D/texture được đánh dấu **Optional**. Hunyuan3D-2mini và cấu hình
resolution/sample lớn được đánh dấu **High Quality**, không phải luồng mặc định
Colab Free.

## Lệnh local đã kiểm chứng

```bash
python scripts/prepare_dataset.py --config configs/quick_test.yaml
python scripts/render_dataset.py --config configs/quick_test.yaml
python scripts/generate_base_mesh_cache.py --config configs/quick_test.yaml --max-samples 4
python scripts/train.py --config configs/quick_test.yaml --quick-overfit
python scripts/train.py --config configs/quick_test.yaml --quick-overfit --resume <last.pt>
python scripts/evaluate.py --config configs/quick_test.yaml --checkpoint <best_edge.pt>
python scripts/postprocess_mesh.py --input <base_mesh.obj> --output <dir> --config configs/quick_test.yaml
python scripts/run_gradio_app.py --config configs/inference.yaml
python -m pytest -q
```

## Lệnh Colab

Sau khi clone repository, chọn GPU và thêm secret `HF_TOKEN`, mở lần lượt các
notebook 01–05 rồi **Run all**. Đây là luồng dự kiến; do audit host không phải
Colab, trạng thái cuối vẫn là `NOT VERIFIED ON COLAB`.
