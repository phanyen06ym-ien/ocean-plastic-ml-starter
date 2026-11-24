# Kế hoạch thực nghiệm

**Issue:** #4, #10 · **Phạm vi:** [ADR-001](decisions/ADR-001-class-scope.md) · **Dữ liệu:** `DATA_VERSION = v1_crop128`

## 1. Câu hỏi nghiên cứu

1. CNN nhỏ tự xây dựng đạt Macro F1 bao nhiêu trên 22 lớp TrashCan-Instance khi chia dữ liệu theo video?
2. Data Augmentation có cải thiện Macro F1 (tổng thể và nhóm 6 lớp liên quan nhựa) so với baseline không?
3. Transfer learning (Sprint 3) cải thiện bao nhiêu so với CNN tự xây dựng?
4. Việc kéo dài thời gian huấn luyện và tăng Early Stopping patience (E03G) có cải thiện khả năng tổng quát hóa so với E03 không?

Giả thuyết H2: E02 có Macro F1 Validation cao hơn E01, rõ nhất ở các lớp ít mẫu. E03G kiểm tra giả thuyết phụ rằng E03 có thể dừng quá sớm; đây là câu hỏi thực nghiệm, không giả định trước rằng lịch dài hơn sẽ tốt hơn.

## 2. Danh sách thí nghiệm

| EXP_ID | Mô tả | Thay đổi so với mốc | Cố định | Sprint |
|---|---|---|---|---|
| E01 | CNN baseline + class weights | — | dữ liệu v1, seed 42, 128×128, Adam 1e-3, batch 64, ≤ 40 epoch, EarlyStopping(patience 6) | 2 |
| E02 | E01 + augmentation | Lật ngang, xoay ±10%, zoom ±10%, độ tương phản ±20% (chỉ Train) | như E01 | 2 |
| E03 | Transfer learning (MobileNetV2, ImageNet) | Backbone (chuẩn hoá mean/std ImageNet, giữ 128×128), framework PyTorch | dữ liệu, split, seed 42 | 3 |
| E03G | E03 + extended training | Pha 1 tối đa 30 epoch, pha 2 tối đa 40 epoch; patience 12 mỗi pha | cấu hình lõi E03, dữ liệu, split, seed 42; monitor `val_loss` | 3 |
| E04 | Mô hình tốt nhất + dữ liệu cải tiến | `DATA_VERSION` | mô hình, seed | 3 |

Cấu hình: `configs/experiments/<EXP_ID>.yaml` (E03: `configs/experiments/E03.yaml`; E03G: `configs/experiments/E03G.yaml`).

### 2.1. Tinh chỉnh E03 (S3-CNN)

Mỗi run chỉ đổi một yếu tố so với E03, bằng tuỳ chọn `--set` của `src/train.py` (không tạo file YAML riêng):

```bash
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
| E03G | Kéo dài lịch huấn luyện, tăng patience | YAML riêng `configs/experiments/E03G.yaml` | đã chạy |

**Cấu hình E03G**
- Pha 1: tối đa 30 epoch; Early Stopping patience 12.
- Pha 2: tối đa 40 epoch; Early Stopping patience 12.
- Monitor `val_loss`, mode `min`, `restore_best_weights: true`.
- Giữ cấu hình lõi E03: MobileNetV2 pretrained ImageNet, `v1_crop128`, 128×128×3, 22 lớp, batch 64, seed 42.
- `num_workers: 0` để tương thích môi trường Windows và tránh lỗi pickle DataLoader.

## 3. Kiến trúc E01

```text
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

```text
Input 128×128×3 → chuẩn hoá mean/std ImageNet
→ MobileNetV2 features (torchvision, ImageNet) → GlobalAveragePooling
→ Dropout(0.4) → Linear(1280, 22)
Loss: cross-entropy có class weights "balanced" (như E01) · Optimizer: Adam (eps 1e-7)
Pha 1: đóng băng backbone, lr 1e-3 · Pha 2: mở features[14:], lr 1e-5 (backbone) và 1e-4 (đầu), BatchNorm giữ eval
```

Khác E01 ngoài yếu tố mô hình: framework TensorFlow/Keras → PyTorch; chuẩn hoá `Rescaling(1/255)` → mean/std ImageNet; đầu phân loại Dropout → Linear thay cho Dropout → Dense(128) → Dropout → softmax; giữ 128×128 (không phóng lên cỡ gốc của backbone) để chỉ đổi một yếu tố đầu vào.

## 4. Quy tắc đánh giá và lưu checkpoint

- Chỉ dùng tập Validation để so sánh thí nghiệm và chọn mô hình; không dùng tập Test để chọn epoch, siêu tham số hoặc checkpoint.
- Quy tắc trong cấu hình huấn luyện hiện tại: chọn checkpoint theo `val_loss` (min). Macro F1 Validation là chỉ số chính để so sánh chất lượng các thí nghiệm.
- Từ E03, mỗi run có thể lưu thêm `bestf1_phase1.pt` và `bestf1_phase2.pt` (epoch có Macro F1 Validation cao nhất của từng pha). Đây là checkpoint phân tích bổ sung, không tự động thay thế checkpoint theo `val_loss`.
- Mỗi run lưu `config.json`, `history.csv`, checkpoint (`best.keras` cho E01/E02 hoặc `best.pt` cho E03 trở đi), và `class_mapping.json` trong `runs/<EXP_ID>_<YYYYMMDD-HHMM>/` trên Drive.
- Kết quả chính thức cần đối chiếu với file metric Validation tương ứng trong `results/metrics/`.
- Test chạy một lần cho mô hình cuối sau khi reviewer/nhóm chốt tiêu chí và checkpoint.

**Điểm cần reviewer xác nhận:** tài liệu đánh giá của dự án có chỗ xếp hạng ứng viên theo Macro F1 Validation, trong khi quy tắc lưu checkpoint trong kế hoạch này là `val_loss`. E03G cho thấy sự khác biệt có ý nghĩa: epoch có Macro F1 cao nhất trong pha 2 không phải epoch có `val_loss` thấp nhất. Không tự ý thay đổi tiêu chí chọn checkpoint trước khi được xác nhận.

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
| E03d-F1 | 10/11/2026 | Bách (CPU, PyTorch) | theo run E03d | 0,2398 | 0,1076 |
| E03f-F1 | 10/11/2026 | Bách (CPU, PyTorch) | theo run E03f | 0,2372 | 0,1357 |
| E03G | 24/11/2026 | Bách (CPU local, PyTorch) | đường dẫn run cần bổ sung | 0,2034 | 0,1228 |

## 6. Phân tích kết quả và ứng viên

Tất cả số liệu dưới đây là Validation trên `v1_crop128` (1.782 ảnh, 22 lớp). Tập Test chưa dùng.

| EXP_ID | Mô tả | Checkpoint / tiêu chí | Số tham số | Kích thước file | CPU inference (ms/ảnh) | Macro F1 val | Macro F1 nhóm nhựa |
|---|---|---|---:|---:|---:|---:|---:|
| E01 | CNN tự xây + class weights | `best.keras` (`val_loss`) | 425.590 (*) | chưa đo | chưa đo | 0,0757 | 0,0000 |
| E03 | MobileNetV2, 2 pha | `best.pt` (`val_loss`, pha 1 epoch 2) | 2.252.054 | 9,23 MB | 32,4 | 0,2034 | 0,1228 |
| E03a | E03 + dropout 0,6 | `best.pt` (`val_loss`, pha 1 epoch 2) | 2.252.054 | 9,23 MB | 29,3 | 0,2099 | 0,1377 |
| E03c | E03 + lr pha 1 = 3e-4 | `best.pt` (`val_loss`, pha 1 epoch 6) | 2.252.054 | 9,23 MB | 30,2 | 0,1955 | 0,1083 |
| E03d | E03 + lr backbone pha 2 = 3e-5 | `best.pt` (`val_loss`, pha 1 epoch 2) | 2.252.054 | 9,23 MB | 29,9 | 0,2034 | 0,1228 |
| E03f | E03 với seed 43 | `best.pt` (`val_loss`, pha 1 epoch 1) | 2.252.054 | 9,23 MB | 29,5 | 0,2122 | 0,1168 |
| E03d-F1 | E03d, chọn theo Macro F1 | `bestf1_phase2.pt`, pha 2 epoch 6 | 2.252.054 | 9,23 MB | 29,9 | 0,2398 | 0,1076 |
| E03f-F1 | E03f, chọn theo Macro F1 | `bestf1_phase2.pt`, pha 2 epoch 9 | 2.252.054 | 9,23 MB | 29,5 | 0,2372 | 0,1357 |
| E03G | Extended training | checkpoint cuối theo `val_loss` | 2.252.054 | 9,23 MB | 31,396 | 0,2034 | 0,1228 |

(*) Số tham số E01 được tính từ kiến trúc trong code (4 khối Conv-BN, GAP, Dense 128, softmax 22), gồm cả trọng số BatchNorm; cần giữ chú thích khi so sánh với số tham số trainable.

Số liệu E03d-F1 và E03f-F1 lấy từ `bestf1_phase2_val.json` (chạy `src/eval_torch.py` trên Validation), khớp log huấn luyện: Accuracy lần lượt 0,4994 và 0,4826; `val_loss` lần lượt 2,583 và 2,384.

**Nhận xét**
- Sprint 2: H2 không được xác nhận (E02 < E01). Mốc so sánh cho Sprint 3 là E01.
- E03 cải thiện rõ so với E01: Macro F1 Validation 0,2034 so với 0,0757; Macro F1 nhóm nhựa 0,1228 so với 0,0000.
- E03a/E03c/E03d không cho thấy một thay đổi siêu tham số đơn lẻ cải thiện rõ rệt. E03 và E03f cùng cấu hình nhưng khác seed chênh 0,009 Macro F1; vì vậy chênh lệch nhỏ hơn mức này cần được diễn giải thận trọng.
- Checkpoint chọn theo Macro F1 Validation của E03d-F1/E03f-F1 đạt Macro F1 0,2398/0,2372, cao hơn checkpoint chọn theo `val_loss`. Đây là hai tiêu chí khác nhau và phải được ghi rõ khi so sánh.
- E03G kiểm tra lịch huấn luyện dài hơn. Checkpoint cuối được chọn theo `val_loss` đạt Macro F1 0,2034, Accuracy 0,4052 và Plastic6 Macro F1 0,1228, tương đương E03. Trong pha 2, training loss giảm trong khi validation loss nhìn chung tăng; Macro F1 log đạt đỉnh 0,236 ở epoch 9 nhưng không trùng epoch có `val_loss` tốt nhất. Kết quả không cho thấy lợi ích của việc kéo dài training.
- E03 có khoảng 5,3 lần số tham số E01; cần cân nhắc kích thước và tốc độ nếu triển khai trên robot. E03G có cùng số tham số với E03.
- Thời gian inference đo trên CPU 4 luồng, batch 1, ảnh 128×128; dao động khoảng ±10% giữa các lần đo. Không so sánh tốc độ E01 khi chưa có phép đo cùng điều kiện.

## 7. Kết luận và công việc còn lại

1. Giữ E03 làm mốc transfer-learning; E03G không chứng minh được cải thiện khi kéo dài training.
2. Không tuyên bố E03d-F1/E03f-F1 là checkpoint cuối đã được nhóm phê duyệt nếu reviewer chưa xác nhận tiêu chí Macro F1 so với `val_loss`.
3. Reviewer xác nhận quy tắc chọn checkpoint và mô hình bàn giao.
4. Bổ sung đường dẫn run thực tế và kiểm tra file metric chuẩn cho E03G nếu pipeline đã xuất.
5. E04 (dữ liệu cải tiến) chưa chạy, chờ `DATA_VERSION` mới từ S3-DATA (#16).
6. Chỉ đánh giá Test một lần sau khi chốt mô hình cuối.
