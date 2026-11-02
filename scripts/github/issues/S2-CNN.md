---
key: S2-CNN
title: [S2-CNN] Train CNN Baseline (E01) and Augmentation Experiment (E02)
assignees: phannguyenhoangbach2, phanyen06ym-ien
labels: area:cnn, type:exp, P0-critical
milestone: Sprint 2 — Data & Baseline
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 2 — Data & Baseline |
| Mã nhiệm vụ | `S2-CNN` |
| Người phụ trách chính | Hoàng Bách (@phannguyenhoangbach2) |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) |
| Vai trò | CNN Developer · Training |
| Ưu tiên | P0 — Critical |
| Nhánh | `exp/s2-cnn-e01-e02` |
| Thời gian | 09/11 → 22/11/2026 |
| Hạn mở PR | 19/11 |

## 2. Mục tiêu

Huấn luyện CNN baseline (E01) trên dữ liệu processed và thí nghiệm đối chứng E02 (thêm Data Augmentation, các yếu tố khác giữ nguyên).

**Kết quả mong đợi:** hai run có checkpoint, history, cấu hình trên Drive; bàn giao checkpoint tốt nhất cho EVAL và DEMO.

## 3. Phạm vi công việc

### A. Chuẩn bị — hạn 12/11
- [ ] Cập nhật `src/train.py` theo data contract v1 (input size, normalization, `num_classes` từ class mapping).
- [ ] `configs/experiments/E01_baseline.yaml`, `E02_augment.yaml` (chỉ khác phần augmentation).
- [ ] Chạy thử pipeline trên một phần nhỏ dữ liệu (sanity check: overfit được 1 batch).

### B. E01 — hạn 14/11
- [ ] Huấn luyện E01 với seed 42; EarlyStopping; lưu run `runs/E01_<YYYYMMDD-HHMM>/`.
- [ ] Bàn giao `best.keras` + `config.json` + `history.csv` cho Ân và Nguyên (đường dẫn Drive trong issue).

### C. E02 — hạn 17/11
- [ ] Augmentation chỉ trên Train (lật, xoay nhỏ, thay đổi độ sáng/màu phù hợp ảnh dưới nước).
- [ ] Huấn luyện E02 cùng seed, cùng split, cùng số epoch tối đa.

### D. Mất cân bằng lớp — hạn 19/11 (nếu cần)
- [ ] Nếu Macro F1 val của E01 thấp do lớp nhỏ → thử class weights (E01b), ghi vào sổ thí nghiệm.

### E. Tài liệu — hạn 19/11
- [ ] Mô tả kiến trúc, tham số, thời gian huấn luyện, GPU trong `docs/experiment_plan.md` (mục kết quả huấn luyện).
- [ ] Ghi mỗi run vào `docs/experiment_log.md`.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Cấu hình thí nghiệm | `configs/experiments/E01_baseline.yaml`, `E02_augment.yaml` | 12/11 |
| Run E01 | Drive `runs/E01_*/` | 14/11 |
| Run E02 | Drive `runs/E02_*/` | 17/11 |
| Code + notebook | `src/train.py`, `notebooks/03_cnn_training.ipynb` | 19/11 |
| Learning curves | `results/figures/E01_learning_curves.png`, `E02_...` | 19/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] E01 và E02 chỉ khác augmentation (Yến đối chiếu 2 file config).
- [ ] Chạy lại notebook từ đầu với cùng config cho kết quả cùng mức (chênh lệch nhỏ do GPU được chấp nhận, ghi rõ).
- [ ] Không chọn mô hình hay hyperparameter bằng tập Test.
- [ ] Không commit checkpoint.
- [ ] PR đã được Yến approve và merge vào `main`.

## 6. Ngoài phạm vi

- Tính chỉ số đánh giá chính thức — Ân. Transfer learning — S3-CNN.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | {{S2-DATA}} | Dữ liệu processed (12/11), class mapping |
| Nhận từ | {{S2-ARCH}} | Data contract v1, kế hoạch E01/E02 |
| Bàn giao cho | {{S2-EVAL}} | Checkpoint, history E01 (14/11), E02 (17/11) |
| Bàn giao cho | {{S2-DEMO}} | Checkpoint E01 (14/11) |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 12/11 | Config E01/E02 + sanity check | Yến |
| 14/11 | **Bàn giao checkpoint E01** | Ân, Nguyên |
| 17/11 | **Bàn giao checkpoint E02** | Ân |
| 19/11 | Mở PR `exp(s2-cnn): train E01 baseline and E02 augmentation` | Yến |
| 21/11 | PR được merge | Yến |

## 9. Quy định review

- Tiêu đề `exp(s2-cnn): ...`; nội dung theo PR template; `Closes` issue này.
- Reviewer bắt buộc **@phanyen06ym-ien**; chỉ Yến merge.
- Báo tiến độ thứ Hai và thứ Năm. Nếu S2-DATA trễ, dùng thời gian chờ cho mục A và báo Yến.

## 10. Ghi chú

- Không ghi số liệu hiệu năng trong PR khi chưa có file đánh giá của Ân; có thể ghi `val_loss`/`val_accuracy` từ history kèm nhãn "theo Keras history".
