# Kế hoạch thực nghiệm

**Issue:** #4, #10 · **Phạm vi:** [ADR-001](decisions/ADR-001-class-scope.md) · **Dữ liệu:** `DATA_VERSION = v1_crop128`

## 1. Câu hỏi nghiên cứu

1. CNN nhỏ tự xây dựng đạt Macro F1 bao nhiêu trên 22 lớp TrashCan-Instance khi chia dữ liệu theo video?
2. Data Augmentation có cải thiện Macro F1 (tổng thể và nhóm 6 lớp liên quan nhựa) so với baseline không?
3. Transfer learning (Sprint 3) cải thiện bao nhiêu so với CNN tự xây dựng?

Giả thuyết H2: E02 có Macro F1 Validation cao hơn E01, rõ nhất ở các lớp ít mẫu.

## 2. Danh sách thí nghiệm

| EXP_ID | Mô tả | Thay đổi so với mốc | Cố định | Sprint |
|---|---|---|---|---|
| E01 | CNN baseline + class weights | — | dữ liệu v1, seed 42, 128×128, Adam 1e-3, batch 64, ≤ 40 epoch, EarlyStopping(patience 6) | 2 |
| E02 | E01 + augmentation | Lật ngang, xoay ±10%, zoom ±10%, độ tương phản ±20% (chỉ Train) | như E01 | 2 |
| E03 | Transfer learning (MobileNetV2, ImageNet), 2 pha | Mô hình (kèm: framework PyTorch, chuẩn hoá ImageNet, đầu phân loại — xem `E03_transfer_mobilenetv2.yaml`) | dữ liệu v1, split, seed 42, 128×128, class weights, không augmentation | 3 |
| E03a–f | Tinh chỉnh E03 (≤ 6 run, mục 6) | một siêu tham số mỗi run (dropout hoặc learning rate) | như E03 | 3 |
| E04 | Mô hình tốt nhất + dữ liệu cải tiến | `DATA_VERSION` | mô hình, seed | 3 |

Cấu hình: `configs/experiments/<EXP_ID>_*.yaml`.

## 3. Kiến trúc E01

```
Input 128×128×3 (pixel 0..255)
→ Rescaling(1/255)
→ [Conv2D(32,3) → BatchNorm → ReLU → MaxPool(2)]
→ [Conv2D(64,3) → BatchNorm → ReLU → MaxPool(2)]
→ [Conv2D(128,3) → BatchNorm → ReLU → MaxPool(2)]
→ [Conv2D(256,3) → BatchNorm → ReLU] → GlobalAveragePooling
→ Dropout(0.4) → Dense(128, ReLU) → Dropout(0.4) → Dense(22, softmax)
Loss: sparse_categorical_crossentropy · Optimizer: Adam
```

## 4. Quy tắc

- Chọn checkpoint tốt nhất theo `val_loss`; so sánh thí nghiệm bằng **Macro F1 Validation**.
- Mỗi run lưu `config.json`, `history.csv`, `best.keras`, `class_mapping.json` trong `runs/<EXP_ID>_<YYYYMMDD-HHMM>/` trên Drive.
- Kết quả chỉ ghi vào README/báo cáo sau khi `results/metrics/<EXP_ID>_val.json` được merge.
- Test chạy một lần cho mô hình cuối (Sprint 3).

## 5. Sổ thí nghiệm

| EXP_ID | Ngày | Người chạy | Run trên Drive | Macro F1 val | Macro F1 nhóm nhựa (val) |
|---|---|---|---|---:|---:|
| E01 | 05/11/2026 | Yến (Colab T4) | `runs/E01_20261105-1513` | 0,0757 | 0,0000 |
| E02 | 05/11/2026 | Yến (Colab T4) | `runs/E02_20261105-1515` | 0,0648 | 0,0165 |
| E03 | 08/11/2026 | Bách (CPU, PyTorch) | `runs/E03_20261108-1459` | 0,2034 | 0,1228 |

Kết luận Sprint 2: H2 không được xác nhận (E02 < E01). Hai mô hình chỉ cao hơn mức ngẫu nhiên (≈ 0,045) một chút; nguyên nhân và đề xuất: `docs/evaluation_report.md` mục 3–4. Mốc so sánh cho Sprint 3 là E01.

Sổ đầy đủ (môi trường, ghi chú từng run): `docs/experiment_log.md`.

## 6. Tinh chỉnh có kiểm soát (S3-CNN, mục B)

Mô hình tốt nhất hiện tại: **E03** (Macro F1 val 0,2034 so với E01 0,0757). Tìm kiếm một-yếu-tố-mỗi-lần quanh E03, tối đa 6 run, cùng split/seed/dữ liệu:

| EXP_ID | Thay đổi so với E03 | Giá trị |
|---|---|---|
| E03a | dropout đầu phân loại | 0,4 → 0,2 |
| E03b | dropout đầu phân loại | 0,4 → 0,6 |
| E03c | learning rate pha 1 | 1e-3 → 3e-4 |
| E03d | learning rate pha 1 | 1e-3 → 3e-3 |
| E03e | learning rate pha 2 (features / classifier) | 1e-5 / 1e-4 → 3e-5 / 3e-4 |
| E03f | learning rate pha 2 (features / classifier) | 1e-5 / 1e-4 → 3e-6 / 3e-5 |

Quy tắc: chọn theo Macro F1 Validation; chỉ coi là cải thiện khi hơn E03 rõ rệt (gợi ý ≥ 0,02, vì một seed và lớp support nhỏ gây nhiễu); checkpoint mỗi run vẫn chọn theo `val_loss`; không chạm tập Test. Run pha 2 nạp `best_phase1.pt` của chính biến thể đó. Nếu chọn được cấu hình tốt hơn, chạy E04 (cấu hình đó + `DATA_VERSION` mới từ #16) thay đổi đúng yếu tố dữ liệu.

## 7. Ứng viên mô hình cuối (hạn 12/11)

Tạo bằng `python scripts/summarize_runs.py <thư mục các run>`; cột Macro F1 từ `results/metrics/<EXP_ID>_val.json`. Ô "—" là chưa đo.

| EXP_ID | Mô hình | Tham số | Kích thước file (MB) | Suy luận (ms/ảnh, CPU, batch 1) | Macro F1 val | Nhóm nhựa val |
|---|---|---:|---:|---:|---:|---:|
| E01 | CNN tự xây dựng | ≈ 0,425 M (tính từ kiến trúc) | — | — | 0,0757 | 0,0000 |
| E03 | MobileNetV2 (chọn `best_phase1.pt`) | 2,252 M | 9,23 | 32,4 | 0,2034 | 0,1228 |
| E03a–f, E04 | chờ chạy | | | | | |

Điều kiện đo: E03 đo trên CPU 4 luồng. Cần đo E01 (và các ứng viên sau) với cùng điều kiện để so sánh tốc độ/kích thước công bằng.
