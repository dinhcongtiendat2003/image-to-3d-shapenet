# Project Status

Ngày cập nhật: 2026-10-30 (Asia/Bangkok)

## Mục tiêu

Xây dựng hệ thống single-image to 3D dùng ShapeNetCore làm ground truth chính,
ưu tiên hình học/cạnh sắc và chạy được trên Google Colab miễn phí. Ảnh đơn chỉ
cho phép ước lượng phần bị che khuất, không thể khôi phục tuyệt đối.

## Milestone hiện tại

Kiểm định cuối về khả năng tái tạo, notebook Run all, quick pipeline, tài nguyên,
license và vệ sinh repository. Trạng thái tổng thể: **PARTIAL — NOT VERIFIED ON
COLAB GPU**, không tuyên bố hoàn thành production.

## Đã hoàn thành

- Gradio UI có upload/drag-drop, ảnh gốc/đã tách nền, bốn lựa chọn backend,
  refinement/texture/repair, ba slider, status, GLB viewer, downloads và bảng kết quả.
- Preset Colab Free mặc định TripoSR và geometry-only; texture/refinement tự tắt
  có cảnh báo khi thiếu VRAM.
- AutoBackend giữ fallback tuần tự giữa backend thật; không có production fake mesh.
- Global model lock và Gradio queue concurrency 1 ngăn load backend đồng thời;
  unload, garbage collection và CUDA cache cleanup chạy cả khi exception.
- Mỗi Generate tạo output UUID riêng; upload chỉ được đọc như image và không
  quyết định filename/path output; không xóa session khác.
- Repair/refinement là tùy chọn; kết quả luôn xuất `final_mesh.obj/.glb/.stl`
  cùng metadata, kể cả khi browser viewer không hoạt động.
- Notebook `05_inference_gradio.ipynb` có một cell setup/launch và bật share link.
- Smoke pipeline dùng backend test inject từ pytest; production factory vẫn là
  `AutoBackend` với adapter pretrained thật.

- Hậu xử lý cấu hình được theo từng bước: duplicate vertex/face, degenerate face,
  unreferenced vertex, winding, normal, component nhỏ và lỗ nhỏ an toàn.
- Mỗi mutation được kiểm tra; mesh lỗi/rỗng/non-finite được rollback thay vì phá
  topology âm thầm.
- Báo cáo trước/sau gồm vertex, face, component, boundary/non-manifold/sharp edge,
  watertight và bounding box.
- Remesh bằng subdivision không di chuyển cạnh gốc; quadric decimation khóa vertex
  cạnh sắc, từ chối fold/crease mới và non-manifold mới.
- Smoothing mặc định tắt; khi bật không di chuyển hoặc truyền qua cạnh sắc.
- Xuất raw OBJ và processed OBJ/GLB/STL có thể load lại, kèm JSON report/metadata.
- Mesh không watertight vẫn xuất được nhưng STL có warning rõ ràng.
- `pymeshlab` hoàn toàn tùy chọn; fallback `trimesh` luôn sẵn sàng.

- Metric thuần PyTorch/NumPy: Chamfer, F-score 1%/2%, normal consistency,
  silhouette IoU, sharp-edge Chamfer và dihedral-angle error.
- Ghi vertex/face count, inference time, peak VRAM và watertight status.
- Chuẩn hóa pair đúng hệ tọa độ: transform của ground truth được áp dụng đồng
  thời lên prediction/target; lỗi dịch/scale không bị normalize mất.
- Surface và sharp-edge sampling deterministic, xử lý từng mesh tuần tự.
- Aggregation theo từng sample, model, category và toàn test set.
- Ablation qua config: `base`, `base_refiner`, `refiner_no_edge`, `refiner_full`;
  checkpoint riêng cho no-edge/full.
- Kết quả nhóm theo backend thực tế trong cache; TripoSR, Stable Fast 3D và
  Hunyuan3D-2mini được đánh giá tuần tự, không load đồng thời.
- Incremental resume bằng khóa sample/config/backend; CSV/JSON/failures được ghi
  atomically sau mỗi sample.
- Sample/mesh lỗi được ghi `failures.json` và benchmark tiếp tục.
- Xuất per-sample/summary CSV, JSON, Markdown, LaTeX, 5 plot PNG và comparison
  panels gồm input/GT/base/refined/sharp-edge visualization.
- Notebook `04_evaluate.ipynb` chạy từ trên xuống, tự chuẩn bị quick cache/
  checkpoint nếu thiếu, evaluate hai lần để kiểm tra resume và hiển thị artifacts.

## Test đã chạy

Môi trường local: Python 3.12.10, PyTorch 2.2.2 CPU, không có CUDA.

```text
Baseline trước milestone: python -m pytest -q
PASS: 49 passed, 1 skipped

Focused metrics/config tests
PASS: 11 passed in 3.22s

Focused mesh postprocess/config tests
PASS: 11 passed

Focused Gradio/app/config tests
PASS: 12 passed in 19.82s

Focused notebook audit
PASS: 6 passed

python -m pytest -q
PASS: 74 passed, 1 skipped in 25.40s
SKIP duy nhất: pretrained image-to-3D smoke cần CUDA/source/weights

python -m compileall -q src scripts
PASS

Notebook 04 nbformat validation + compile code cells
PASS

Postprocess CLI trên cube synthetic
PASS: 8 vertices, 12 faces, 12 cạnh sắc 90°, watertight trước/sau
PASS: raw OBJ và refined OBJ/GLB/STL/JSON được xuất, không warning

Gradio local startup smoke
PASS: mở http://127.0.0.1:7862/, sau đó đóng server sạch

Notebook 05 nbformat validation + single-cell compile
PASS

Secret scan
PASS

Final repository scan
PASS: 0 token-like secret, 0 personal path, 0 model weight, 0 file >10 MB
outputs chỉ còn .gitkeep
```

## Fresh end-to-end audit

Chạy từ bản sao sạch không có `data/`, `outputs/`, `.cache/` trên Windows 11,
Python 3.12.10, CPU-only, RAM tổng 7.21 GB, CUDA/VRAM bằng 0:

```text
prepare dataset: 5.25 s, peak RAM 221.8 MB
render 4 samples: 9.77 s, peak RAM 405.3 MB
cache 4 base meshes: 5.38 s, peak RAM 243.1 MB
quick train 3 epochs: 13.35 s, peak RAM 315.9 MB
checkpoint resume: 11.11 s, peak RAM 301.4 MB
evaluate 4 predictions: 4.76 s, peak RAM 236.5 MB
OBJ/GLB/STL export: 3.20 s, peak RAM 208.1 MB
total: 52.82 s; workspace cuối: 2.03 MB
evaluation resume: 5.72 s, vẫn 4 rows, không duplicate
Gradio startup/close: 14.15 s
```

Train loss giảm `0.003344 → 0.001801`; validation Chamfer giảm
`0.001120 → 0.000880`. Export round-trip cả ba format: 8 vertices, 12 faces,
watertight. Generated audit output đã được dọn; chạy lại lệnh quick để tái tạo.

Metric sanity:

```text
identical cube: Chamfer 9.55e-09, F@1%=1.0
translated 0.1: Chamfer 0.01679, F@1%=0.0
scaled 1.2: Chamfer 0.02635, F@2%=0.0
empty/malformed meshes: lỗi có kiểm soát
```

Quick evaluator thực tế:

```text
2 test views × (base + refiner_full) = 4 per-sample rows
failures=0, summary rows=6
chạy lần hai: vẫn 4 rows, không duplicate
Markdown/LaTeX/JSON/CSV/5 plots/4 comparison panels: PASS
```

Notebook `04_evaluate.ipynb` Run all local CPU: PASS từ prepare/render/cache,
train checkpoint nếu thiếu, evaluate/resume đến hiển thị bảng, plot, comparison.

## Lỗi và giới hạn còn lại

- Chưa chạy evaluator trên ShapeNetCore Chair thật vì không có dataset gated,
  pretrained backend weights hoặc GPU trong môi trường local.
- Chưa xác minh peak VRAM/timing trên Colab GPU; CPU smoke ghi VRAM bằng 0.
- Quick cache/checkpoint là synthetic test-only; số liệu quick không đại diện chất
  lượng pretrained backend hoặc khả năng tổng quát.
- Hai ablation checkpoint `base_refiner` và `refiner_no_edge` chưa được huấn luyện
  trong quick smoke; interface ghi `MissingCheckpoint` nếu người dùng yêu cầu mà
  chưa cung cấp.
- Local environment phát warning Transformers bên ngoài repository do yêu cầu
  PyTorch mới hơn 2.2.2; evaluation vẫn hoàn tất.
- Hậu xử lý mới đã smoke trên CPU với cube/mesh lỗi nhân tạo; chưa kiểm chứng trên
  tập Chair production có topology phức tạp hoặc trực tiếp trong Colab runtime.
- `pymeshlab` không được cài trong môi trường local và chỉ là tùy chọn; đường mặc
  định cùng toàn bộ export đã được kiểm chứng bằng `trimesh 5.2.0`.
- Feature-aware QEM ưu tiên an toàn: khi mọi candidate đụng cạnh sắc hoặc tạo
  crease/non-manifold mới, nó giữ mesh nguyên trạng thay vì ép đạt face target.
- App pipeline đã smoke bằng backend test đúng yêu cầu; chưa chạy generation qua
  TripoSR/Stable Fast 3D/Hunyuan thật do local không có CUDA/source/weights.
- Chưa xác minh Gradio share URL thực tế trên Colab; notebook một-cell đã được
  validate/compile và local server đã startup thành công.
- Edge-aware refinement trong app cần `app.refinement_checkpoint` hoặc biến môi
  trường `REFINER_CHECKPOINT`; nếu thiếu, app giữ base mesh và cảnh báo rõ.
- Năm notebook mục tiêu đã được validate/compile và code-audit nhưng chưa Run all
  trên Google Colab thật; không có số đo Colab RAM/VRAM/thời gian/share URL.
- ShapeNetCore Chair thật, TripoSR, Stable Fast 3D và Hunyuan3D-2mini chưa chạy
  trong final audit do thiếu quyền gated, CUDA và weights.
- Không cấu hình linter trong `pyproject.toml`; vì vậy lint là N/A, không báo PASS.
- Workspace cha không phải Git repository nên không có trạng thái commit.

## Lệnh tiếp tục

```bash
python scripts/prepare_dataset.py --config configs/quick_test.yaml
python scripts/render_dataset.py --config configs/quick_test.yaml
python scripts/generate_base_mesh_cache.py --config configs/quick_test.yaml --max-samples 4
python scripts/train.py --config configs/quick_test.yaml --quick-overfit

python scripts/run_gradio_app.py --config configs/inference.yaml

python scripts/run_gradio_app.py --config configs/inference.yaml --share

python scripts/postprocess_mesh.py \
  --input outputs/example/base_mesh.obj \
  --output outputs/example/final \
  --config configs/inference.yaml

python scripts/evaluate.py \
  --config configs/quick_test.yaml \
  --checkpoint outputs/quick_test/training_smoke/best_edge.pt

python scripts/evaluate.py \
  --config configs/chair_colab.yaml \
  --checkpoint checkpoints/best_edge.pt \
  --base-refiner-checkpoint checkpoints/base_refiner.pt \
  --no-edge-checkpoint checkpoints/no_edge.pt

python -m pytest -q
```

Trên Colab: chọn GPU, thêm secret `HF_TOKEN`, rồi Run all lần lượt notebook
01→05. Mặc định quick synthetic; chỉ chuyển sang `chair_colab.yaml` sau khi quyền
ShapeNetCore đã được duyệt. Xem ma trận bằng chứng trong `FINAL_AUDIT.md`.
