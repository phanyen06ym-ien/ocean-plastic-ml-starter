---
key: S1-CNN
number: 6
title: [S1-CNN] CNN Architecture, Implementation & Training Pipeline
assignees: phannguyenhoangbach2, phanyen06ym-ien
labels: area:cnn, type:feat, P1-high
milestone: Sprint 1 — Foundation
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 1 — Foundation |
| Mã nhiệm vụ | `S1-CNN` |
| Người phụ trách chính | Hoàng Bách (@phannguyenhoangbach2) |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) |
| Vai trò | CNN Developer · Model Architecture · Training Pipeline |
| Ưu tiên | P1 — High |
| Nhánh | `feat/s1-cnn-baseline` |
| Thời gian | 23/10 → 08/11/2026 |
| Hạn mở PR | 05/11 |

## 2. Mục tiêu

Xây dựng CNN baseline có thể cấu hình và training pipeline chạy trên Colab, sẵn sàng nhận dữ liệu đã xử lý từ S2-DATA.

**Kết quả mong đợi:** `src/train.py` có hàm dựng/compile mô hình theo cấu hình, smoke test forward pass với nhiều số lớp, notebook 03 có pipeline nạp dữ liệu, checkpoint, early stopping, lưu history và cấu hình run lên Drive.

## 3. Phạm vi công việc

### A. Môi trường — hạn 30/10
- [ ] Kiểm tra Python, TensorFlow/Keras, GPU trên Colab; ghi phiên bản vào PR.
- [ ] Đặt seed cố định (Python, NumPy, TensorFlow).
- [ ] Đề xuất dependency cho `requirements.txt` (gửi qua PR, không ghi đè dòng của người khác).

### B. Kiến trúc — hạn 03/11
- [ ] Các khối Conv2D + ReLU + MaxPooling (+ BatchNorm nếu dùng), GlobalAveragePooling hoặc Flatten, Dense, Dropout, Softmax.
- [ ] `input_shape`, `num_classes` và hyperparameter đọc từ cấu hình; `input_size` theo `configs/data_contract.yaml` (đang lệch: notebook dùng 128, problem statement dùng 224).
- [ ] `model.summary()` và giải thích số tham số.

### C. Lập trình và kiểm tra — hạn 05/11
- [ ] `build_cnn()`, `compile_model()`, `DEFAULT_CONFIG` trong `src/train.py`.
- [ ] Loss phù hợp nhãn int (`sparse_categorical_crossentropy`); optimizer Adam; giải thích lựa chọn.
- [ ] `tests/test_train.py`: forward pass với `num_classes` ∈ {2, 4, 6}, kiểm tra shape và tổng softmax = 1.

### D. Training pipeline — hạn 06/11
- [ ] Hàm nạp Train/Val từ thư mục processed trên Drive (đường dẫn qua cấu hình, không hard-code).
- [ ] `ModelCheckpoint` (best theo `val_loss`), `EarlyStopping`, `CSVLogger`.
- [ ] Lưu `config.json`, `history.csv`, `best.keras` vào `runs/<EXP_ID>_<YYYYMMDD-HHMM>/` trên Drive.
- [ ] Hướng dẫn tiếp tục huấn luyện khi Colab ngắt kết nối.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| CNN + cấu hình | `src/train.py` | 05/11 |
| Test | `tests/test_train.py` | 05/11 |
| Notebook huấn luyện | `notebooks/03_cnn_training.ipynb` | 06/11 |
| Mô tả kiến trúc + `model.summary()` | Mục trong PR + `docs/experiment_plan.md` (phối hợp Yến) | 06/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] `pytest tests/test_train.py` chạy qua trên Colab và máy cá nhân.
- [ ] Số lớp đầu ra cấu hình được; không hard-code.
- [ ] Notebook 03 chỉ gọi hàm trong `src/train.py`, không chép lại code.
- [ ] Không commit checkpoint, không in token.
- [ ] Không báo cáo Accuracy/F1 khi chưa huấn luyện trên dữ liệu thật.
- [ ] PR đã được Yến approve và merge vào `main`.

## 6. Ngoài phạm vi

- Huấn luyện chính thức trên TrashCan — S2-CNN (chờ dữ liệu từ S2-DATA).
- Tiền xử lý dữ liệu — Hùng. Đánh giá — Ân. Demo — Ngọc Nguyên.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | #4 | Data contract (input size, normalization), kế hoạch E01/E02 |
| Nhận từ | #5 | Class mapping dự kiến |
| Bàn giao cho | #7 | Định dạng history, checkpoint |
| Bàn giao cho | #8 | Định dạng checkpoint, input/output |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 03/11 | Kiến trúc + `model.summary()` | Yến |
| 05/11 | Mở PR `feat(s1-cnn): CNN baseline and training pipeline` | Yến |
| 07/11 | PR được merge | Yến |

## 9. Quy định review

- Tiêu đề `feat(s1-cnn): ...`; nội dung theo PR template; `Closes #6`.
- Reviewer bắt buộc **@phanyen06ym-ien**; Ân review phần định dạng history/checkpoint (không bắt buộc approve).
- Báo tiến độ thứ Hai và thứ Năm.

## 10. Ghi chú

- **Ghi chú review (02/11 — Yến):** `notebooks/03_cnn_training.ipynb` đang được commit **trực tiếp lên `main`** qua Colab ("Created using Colab" — 28/10, 30/10). Từ nay chỉ lưu notebook vào nhánh `feat/s1-cnn-baseline` (Colab: *File → Save a copy in GitHub* → chọn nhánh) rồi mở PR. Nhánh `feature/s1-cnn-baseline` trong notebook chưa có trên GitHub và `src/train.py` trên `main` vẫn trống.
- Forward pass với dữ liệu ngẫu nhiên chỉ chứng minh mô hình đúng cấu trúc, không chứng minh độ chính xác.
