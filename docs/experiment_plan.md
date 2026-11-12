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
| E03 | Transfer learning (MobileNetV2, ImageNet) | Backbone (chuẩn hoá mean/std ImageNet, giữ 128×128), framework PyTorch | dữ liệu, split, seed 42 | 3 |
| E04 | Mô hình tốt nhất + dữ liệu cải tiến | `DATA_VERSION` | mô hình, seed | 3 |

Cấu hình: `configs/experiments/<EXP_ID>_*.yaml` (E03: `configs/experiments/E03.yaml`).

### 2.1. Tinh chỉnh E03 (S3-CNN)

Mỗi run chỉ đổi một yếu tố so với E03, bằng tuỳ chọn `--set` của `src/train.py` (không tạo file YAML riêng):

```
python -m src.train --config configs/experiments/E03.yaml --processed-dir <DATA> --runs-root runs --exp-id <ID> --set <KEY=VALUE>
```

| EXP_ID | Thay đổi so với E03 | `--set` | Trạng thái |
|---|---|---|---|
| E03a | Dropout 0,6 (E03: 0,4) | `model.head.dropout=0.6` | đã chạy |
| E03b | Dropout 0,2 | `model.head.dropout=0.2` | không chạy |
| E03c | Learning rate pha 1 = 3e-4 (E03: 1e-3) | `training.phase1.learning_rate=3e-4` | đã chạy |
| E03d | Learning rate backbone pha 2 = 3e-5 (E03: 1e-5) | `training.phase2.learning_rate.features=3e-5` | đã chạy |
| E03e | Tổ hợp cấu hình tốt nhất | — | không chạy (chưa cấu hình nào tốt hơn rõ rệt) |
| E03f | Seed 43 (E03: 42), cùng cấu hình; ước lượng nhiễu giữa các seed | `seed=43` | đã chạy |

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

### Kiến trúc E03 (Sprint 3)

```
Input 128×128×3 → chuẩn hoá mean/std ImageNet
→ MobileNetV2 features (torchvision, ImageNet) → GlobalAveragePooling
→ Dropout(0.4) → Linear(1280, 22)
Loss: cross-entropy có class weights "balanced" (như E01) · Optimizer: Adam (eps 1e-7)
Pha 1: đóng băng backbone, lr 1e-3 · Pha 2: mở features[14:], lr 1e-5 (backbone) và 1e-4 (đầu), BatchNorm giữ eval
```

Khác E01 ngoài yếu tố mô hình: framework TensorFlow/Keras → PyTorch; chuẩn hoá `Rescaling(1/255)` → mean/std ImageNet; đầu phân loại Dropout → Linear thay cho Dropout → Dense(128) → Dropout → softmax; giữ 128×128 (không phóng lên cỡ gốc của backbone) để chỉ đổi một yếu tố đầu vào.

## 4. Quy tắc

- Chọn checkpoint tốt nhất theo `val_loss`; so sánh thí nghiệm bằng **Macro F1 Validation**.
- Mỗi run lưu `config.json`, `history.csv`, `best.keras` (E01, E02) hoặc `best.pt` (E03 trở đi), `class_mapping.json` trong `runs/<EXP_ID>_<YYYYMMDD-HHMM>/` trên Drive.
- Từ E03, mỗi run còn lưu `bestf1_phase1.pt` và `bestf1_phase2.pt` (epoch có Macro F1 Validation cao nhất của từng pha). Các checkpoint này chỉ để báo cáo bổ sung, không đổi quy tắc chọn ở trên.
- Kết quả chỉ ghi vào README/báo cáo sau khi `results/metrics/<EXP_ID>_val.json` được merge.
- Test chạy một lần cho mô hình cuối (Sprint 3).

## 5. Sổ thí nghiệm

| EXP_ID | Ngày | Người chạy | Run trên Drive | Macro F1 val | Macro F1 nhóm nhựa (val) |
|---|---|---|---|---:|---:|
| E01 | 05/11/2026 | Yến (Colab T4) | `runs/E01_20261105-1513` | 0,0757 | 0,0000 |
| E02 | 05/11/2026 | Yến (Colab T4) | `runs/E02_20261105-1515` | 0,0648 | 0,0165 |
| E03 | 08/11/2026 | Bách (CPU, PyTorch) | `runs/E03_20261108-1459` | 0,2034 | 0,1228 |
| E03a | 10/11/2026 | Bách (CPU, PyTorch) | `runs/E03a_<điền thời gian>` | 0,2099 | 0,1377 |
| E03c | 10/11/2026 | Bách (CPU, PyTorch) | `runs/E03c_<điền thời gian>` | 0,1955 | 0,1083 |
| E03d | 10/11/2026 | Bách (CPU, PyTorch) | `runs/E03d_20261110-1445` | 0,2034 | 0,1228 |
| E03f | 10/11/2026 | Bách (CPU, PyTorch) | `runs/E03f_20261110-1744` | 0,2122 | 0,1168 |

Kết luận Sprint 2: H2 không được xác nhận (E02 < E01). Hai mô hình chỉ cao hơn mức ngẫu nhiên (≈ 0,045) một chút; nguyên nhân và đề xuất: `docs/evaluation_report.md` mục 3–4. Mốc so sánh cho Sprint 3 là E01.

Kết luận Sprint 3 (Validation, một seed mỗi cấu hình): transfer learning cải thiện rõ so với E01 (Macro F1 0,2034 so với 0,0757; nhóm nhựa 0,1228 so với 0,0000). Tinh chỉnh dropout và learning rate không cho khác biệt rõ rệt: hai seed của cùng cấu hình (E03 và E03f) đã chênh nhau 0,009. E03d trùng kết quả E03 vì checkpoint được chọn theo `val_loss` luôn rơi vào pha 1 nên learning rate pha 2 không có tác dụng.

## 6. Ứng viên mô hình cuối (S3-CNN)

Tất cả số liệu trên tập Validation (v1_crop128, 1.782 ảnh, 22 lớp). Tập Test chưa dùng.
Thời gian suy luận: CPU 4 luồng, batch 1, ảnh 128×128, đo trên máy của Bách (dao động khoảng ±10% giữa các lần đo).

| EXP_ID | Mô tả | Checkpoint (quy tắc chọn) | Số tham số | Kích thước file | Suy luận CPU (ms/ảnh) | Macro F1 val | Macro F1 nhóm nhựa |
|---|---|---|---:|---:|---:|---:|---:|
| E01 | CNN tự xây + class weights (Sprint 2) | `best.keras` (val_loss) | 425.590 (*) | chưa đo | chưa đo | 0,0757 | 0,0000 |
| E03 | MobileNetV2, 2 pha | `best.pt` (val_loss; pha 1, epoch 2) | 2.252.054 | 9,23 MB | 32,4 | 0,2034 | 0,1228 |
| E03a | E03 + dropout 0,6 | `best.pt` (val_loss; pha 1, epoch 2) | 2.252.054 | 9,23 MB | 29,3 | 0,2099 | 0,1377 |
| E03c | E03 + lr pha 1 = 3e-4 | `best.pt` (val_loss; pha 1, epoch 6) | 2.252.054 | 9,23 MB | 30,2 | 0,1955 | 0,1083 |
| E03d | E03 + lr backbone pha 2 = 3e-5 | `best.pt` (val_loss; pha 1, epoch 2, trùng E03) | 2.252.054 | 9,23 MB | 29,9 | 0,2034 | 0,1228 |
| E03f | E03 với seed 43 | `best.pt` (val_loss; pha 1, epoch 1) | 2.252.054 | 9,23 MB | 29,5 | 0,2122 | 0,1168 |
| E03d-F1 | E03d, chọn theo Macro F1 | `bestf1_phase2.pt` (Macro F1; pha 2, epoch 6) | 2.252.054 | 9,23 MB | 29,9 | 0,2398 | 0,1076 |
| E03f-F1 | E03f, chọn theo Macro F1 | `bestf1_phase2.pt` (Macro F1; pha 2, epoch 9) | 2.252.054 | 9,23 MB | 29,5 | 0,2372 | 0,1357 |

(*) Tính từ kiến trúc trong code (4 khối Conv-BN, GAP, Dense 128, softmax 22), gồm cả trọng số không huấn luyện của BatchNorm (424.630 trọng số huấn luyện được). Kích thước file và thời gian suy luận của E01 cần đo cùng điều kiện (CPU, batch 1) hoặc lấy từ file của Ân.

Số liệu của E03d-F1 và E03f-F1 lấy từ `bestf1_phase2_val.json` (chạy `src/eval_torch.py` trên tập Validation), khớp với log huấn luyện: Accuracy 0,4994 và 0,4826; val_loss 2,583 và 2,384.

**Ghi chú**
- Hai lần chạy cùng cấu hình khác seed (E03 và E03f) chênh nhau 0,009 Macro F1, nên chênh lệch nhỏ hơn mức này giữa các cấu hình không đủ để kết luận.
- Chọn checkpoint theo `val_loss` luôn rơi vào pha 1; chọn theo Macro F1 val cho kết quả cao hơn khoảng 0,02 đến 0,04 ở cả 5 lần chạy (0,232 đến 0,240).
- Macro F1 nhóm nhựa của hai ứng viên F1 không cùng chiều: E03d-F1 thấp hơn E03d (0,1076 so với 0,1228), E03f-F1 cao hơn E03f (0,1357 so với 0,1168). Nhóm nhựa chỉ có 192 mẫu nên dao động mạnh.
- Mô hình E03 có nhiều tham số hơn E01 khoảng 5,3 lần; cần cân nhắc với định hướng chạy trên robot.
- Quy tắc chọn checkpoint cho ứng viên cuối (mục 4: `val_loss`; `docs/evaluation_plan.md` mục 3.2 và 4.3: Macro F1) cần Yến xác nhận trong buổi họp chốt mô hình.
- E04 (dữ liệu cải tiến) chưa chạy, chờ `DATA_VERSION` mới từ #16. Việc chọn siêu tham số dùng tập Validation, nên tập Test chỉ chạy một lần cho mô hình cuối.
