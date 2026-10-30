# Đề cương báo cáo: Single-Image to 3D với Edge-Aware Refinement

## 1. Đặt vấn đề

- Bài toán tái tạo mesh 3D từ một ảnh RGB.
- Tính bất định do che khuất, thiếu scale tuyệt đối và nhập nhằng hình học.
- Mục tiêu: base mesh từ pretrained prior, sau đó tinh chỉnh hình học có bảo toàn
  cạnh sắc bằng ShapeNetCore; không tuyên bố phục hồi tuyệt đối phần không quan sát.
- Phạm vi Colab Free: một category Chair, batch 1, mixed precision và resume.

## 2. Nghiên cứu liên quan

- Single-view reconstruction cổ điển: voxel, point cloud, mesh deformation.
- Implicit surface, radiance field và feed-forward image-to-3D.
- TripoSR, Stable Fast 3D, Hunyuan3D-2: mục tiêu, trade-off tài nguyên/license.
- Graph neural network trên mesh, image-feature projection và differentiable loss.
- Bảo toàn feature edge trong smoothing, remeshing và decimation.

## 3. ShapeNetCore

- ShapeNetCore v2, taxonomy WordNet và category Chair `03001627`.
- Điều khoản gated, phi thương mại, quy trình xin quyền và không redistribute.
- Tải đúng một category ZIP bằng `hf_hub_download`, safe extraction và resume.
- Kiểm tra mesh, chuẩn hóa unit cube/sphere và lưu transform nghịch đảo.
- Split deterministic 70/15/15, seed 42, không trùng model ID.
- Synthetic cube/box chỉ để smoke test, không đại diện dữ liệu production.

## 4. Phương pháp tổng thể

- Sơ đồ: ảnh → preprocess → pretrained base mesh → edge-aware refiner → repair
  → OBJ/GLB/STL.
- Render ground truth: RGB, mask, depth, normal và camera metadata.
- Cache base mesh để foundation backend không chạy lại mỗi epoch.
- AutoBackend và chiến lược fallback tuần tự, không load nhiều model đồng thời.
- Chuẩn hóa hệ tọa độ giữa prediction và ground truth.

## 5. Edge-aware refinement

- ResNet18/lightweight multi-scale image encoder.
- Chiếu feature ảnh lên vertex bằng intrinsics/extrinsics.
- Node feature: position, normal, curvature, visibility, dihedral và sharpness.
- Graph blocks, residual update và `tanh * max_displacement`.
- Phát hiện sharp edge theo góc dihedral mặc định 35°.
- Giảm message qua cạnh sắc, giữ topology và khóa cạnh trong hậu xử lý.

## 6. Hàm loss

Trình bày:

\[
L = \lambda_c L_{chamfer} + \lambda_n L_{normal}
  + \lambda_s L_{silhouette} + \lambda_e L_{edge}
  + \lambda_l L_{laplacian} + \lambda_d L_{displacement}
  + \lambda_b L_{boundary}.
\]

- Chamfer trên surface samples.
- Normal consistency và silhouette IoU surrogate.
- Edge Chamfer với oversampling quanh sharp edge.
- Bảo toàn dihedral angle.
- Laplacian trọng số thấp, không truyền qua cạnh sắc.
- Regularization displacement và boundary.

## 7. Thiết kế thực nghiệm

- Môi trường: phiên bản Python/PyTorch/CUDA, GPU, RAM, VRAM và disk.
- Quick synthetic audit và production Chair protocol phải tách riêng.
- Batch 1, gradient accumulation, AMP, clipping, scheduler, early stopping.
- Checkpoint/resume và OOM fallback.
- Backend/configuration: base, base+refiner, no-edge-loss, full refiner.
- Seed, surface samples, resolution và giới hạn test set.

## 8. Metric

- Chamfer Distance.
- F-score tại 1% và 2% unit scale.
- Normal Consistency, Silhouette IoU.
- Edge Chamfer Distance và Dihedral Angle Error.
- Vertex/face count, inference time, peak VRAM, watertight status.
- Báo cáo per sample, model, category và toàn test set; failures tách riêng.

## 9. Ablation

- Base pretrained backend.
- Base + refiner không edge loss.
- Base + full edge-aware refiner.
- Ảnh hưởng sharp threshold, max displacement và Laplacian weight.
- So sánh backend tuần tự: TripoSR, Stable Fast 3D, Hunyuan3D-2mini.
- Chi phí RAM/VRAM/thời gian và chất lượng hình học.

## 10. Kết quả

- Bảng Markdown/LaTeX từ evaluator.
- Đường loss train/validation và overfit sanity.
- So sánh input/GT/base/refined/sharp-edge visualization.
- Báo cáo kết quả lỗi, sample bị loại và confidence interval nếu đủ mẫu.
- Tách rõ kết quả quick synthetic với kết quả ShapeNetCore thật.

## 11. Hạn chế

- Ambiguity và hallucination ở mặt khuất trong single-view.
- Domain gap ảnh render–ảnh thật.
- Topology cố định của refiner phiên bản đầu.
- Phụ thuộc license/quyền weights và Colab GPU biến động.
- Software renderer chậm; metric sampling là xấp xỉ.
- Không suy rộng kết quả Chair sang toàn bộ ShapeNetCore.

## 12. Hướng phát triển

- Multi-view/video conditioning và uncertainty estimation.
- Topology adaptation, differentiable remeshing và material refinement.
- Domain adaptation ảnh thật, segmentation robust hơn.
- Edge-aware loss có semantic part boundary.
- Benchmark nhiều category/backend và đo trên Colab GPU thực tế.

## 13. Tài liệu tham khảo

- Chang et al., *ShapeNet: An Information-Rich 3D Model Repository*, 2015.
- Tochilkin et al., *TripoSR: Fast 3D Object Reconstruction from a Single Image*, 2024.
- Stable Fast 3D repository/paper và Stability AI Community License.
- Hunyuan3D 2.0 paper/repository và Tencent Community License.
- Các công trình mesh deformation, graph convolution, Chamfer/F-score và
  feature-preserving mesh processing được dùng trong phần phương pháp.

Khi hoàn thiện bibliography, dùng link/paper chính thức và ghi ngày truy cập cho
dataset/model license vì điều khoản upstream có thể thay đổi.
