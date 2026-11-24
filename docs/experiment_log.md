# Sổ thí nghiệm

Mỗi run một dòng. Chỉ số trên **Validation** (không dùng Test). Macro F1 tính trên 22 lớp; nhóm nhựa = 6 lớp liên quan nhựa (ADR-001). Số liệu lấy từ `results/metrics/<EXP_ID>_val.json`.

| EXP_ID | Ngày | Người chạy | Môi trường | Run trên Drive | Dữ liệu | Seed | Thay đổi so với mốc | Macro F1 val | Nhóm nhựa | Ghi chú |
|---|---|---|---|---|---|---:|---|---:|---:|---|
| E01 | 05/11/2026 | Yến | Colab T4, TF 2.21 | `runs/E01_20261105-1513` | v1_crop128 | 42 | — (mốc Sprint 2) | 0,0757 | 0,0000 | best epoch 7 |
| E02 | 05/11/2026 | Yến | Colab T4, TF 2.21 | `runs/E02_20261105-1515` | v1_crop128 | 42 | + augmentation | 0,0648 | 0,0165 | H2 không xác nhận |
| E03 | 08/11/2026 | Bách | CPU local, PyTorch | `runs/E03_20261108-1459` | v1_crop128 | 42 | mô hình: MobileNetV2 ImageNet (+ framework, chuẩn hoá ImageNet, đầu phân loại) | **0,2034** | 0,1228 | chọn `best_phase1.pt` (epoch 2, val_loss 2,205); xem mục dưới |
| E03a–E03f | — | Bách | — | — | v1_crop128 | 42 (E03f: 43) | mỗi run đổi một siêu tham số so với E03 | xem `docs/experiment_plan.md` | | các run và trạng thái được liệt kê trong kế hoạch thực nghiệm |
| E03G | Chưa chạy | Bách | Chưa xác nhận | Chưa có | v1_crop128 | 42 | tăng epoch tối đa 15/20 → 30/40; patience 6 → 12 ở cả hai pha | Chưa có | Chưa có | đã có cấu hình; chờ chạy thực nghiệm và metrics Validation |
| E04 | — | — | — | — | DATA_VERSION mới (#16, 09/11) | 42 | dữ liệu | chưa chạy | | tuỳ chọn |

## Ghi chú E03

- Pha 1 (backbone đóng băng) dừng ở epoch 8, val_loss tốt nhất epoch 2 (2,205). Pha 2 (mở từ block 14, lr 1e-5/1e-4) dừng ở epoch 7, val_loss tốt nhất epoch 1 (2,275). Theo quy tắc "checkpoint theo val_loss", `best.pt` = `best_phase1.pt`.
- Macro F1 val của pha 2 cao hơn (0,219 ở epoch 1; tối đa 0,233 ở epoch 7) nhưng val_loss tệ hơn và epoch 7 không được lưu checkpoint. Chênh lệch này cần kiểm chứng bằng tinh chỉnh (không kết luận từ một seed).
- Khác biệt ngoài "mô hình" so với E01: framework Keras → PyTorch; chuẩn hoá ImageNet mean/std; đầu phân loại `Dropout → Linear(1280, 22)`. Cùng split, seed 42, class weights `balanced`, không augmentation, batch 64, 128×128.
- `config.json` ghi kế hoạch đo thời gian suy luận trên CUDA nhưng run thực tế là CPU: 32,4 ms/ảnh (batch 1, 4 luồng). Phải đo E01 trong cùng điều kiện trước khi so sánh tốc độ.
- File checkpoint (`*.pt`, ~9 MB) không commit; nằm trên Drive.

## Ghi chú E03G

- E03G dùng MobileNetV2 pretrained ImageNet và giữ các thiết lập cốt lõi của E03 (dữ liệu `v1_crop128`, input 128×128×3, 22 lớp, batch size 64, seed 42).
- Phase 1: tối đa 30 epoch, Early Stopping patience 12. Phase 2: tối đa 40 epoch, Early Stopping patience 12.
- Cả hai pha theo dõi `val_loss` theo chiều `min` và khôi phục trọng số tốt nhất (`restore_best_weights: true`). Tổng số epoch tối đa theo cấu hình là 70; số epoch thực tế có thể ít hơn do Early Stopping.
- Trạng thái: mới ghi nhận cấu hình; chưa có run, checkpoint hay chỉ số Validation. Không suy diễn kết quả E03G từ kết quả E03 hoặc từ kiểm thử tự động `pytest`.
