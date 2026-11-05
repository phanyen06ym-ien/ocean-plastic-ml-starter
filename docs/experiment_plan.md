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
| E03 | Transfer learning (MobileNetV2, ImageNet) | Backbone, resize lên kích thước backbone | dữ liệu, split, seed | 3 |
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

Kết luận Sprint 2: H2 không được xác nhận (E02 < E01). Hai mô hình chỉ cao hơn mức ngẫu nhiên (≈ 0,045) một chút; nguyên nhân và đề xuất: `docs/evaluation_report.md` mục 3–4. Mốc so sánh cho Sprint 3 là E01.
