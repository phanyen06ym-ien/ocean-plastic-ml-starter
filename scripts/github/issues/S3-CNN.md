---
key: S3-CNN
title: [S3-CNN] Transfer Learning (E03), Tuning & Final Model Candidates
assignees: phannguyenhoangbach2, phanyen06ym-ien
labels: area:cnn, type:exp, P0-critical
milestone: Sprint 3 — Improvement & Midterm
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 3 — Improvement & Midterm |
| Mã nhiệm vụ | `S3-CNN` |
| Người phụ trách chính | Hoàng Bách (@phannguyenhoangbach2) |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) |
| Vai trò | CNN Developer · Training · Tuning |
| Ưu tiên | P0 — Critical |
| Nhánh | `exp/s3-cnn-e03-transfer` |
| Thời gian | 05/11 → 18/11/2026 |
| Hạn mở PR | 12/11 |

## 2. Mục tiêu

So sánh CNN tự xây dựng với một mô hình transfer learning nhẹ (E03), thử tinh chỉnh có kiểm soát, và đề xuất mô hình cuối dựa trên Validation.

**Kết quả mong đợi:** run E03 (và E04 nếu dùng dữ liệu cải tiến của S3-DATA) có đầy đủ cấu hình; danh sách ứng viên mô hình cuối kèm Macro F1 val.

## 3. Phạm vi công việc

### A. E03 Transfer learning — hạn 08/11
- [ ] Backbone nhẹ phù hợp hướng robot (ví dụ MobileNetV2 hoặc EfficientNetB0, pretrained ImageNet); chuẩn hoá đầu vào đúng yêu cầu backbone, ghi rõ trong config.
- [ ] Hai pha: đóng băng backbone → mở một số tầng cuối với learning rate nhỏ.
- [ ] Cùng split, cùng seed, cùng augmentation với mô hình tốt nhất Sprint 2.

### B. Tinh chỉnh có kiểm soát — hạn 10/11
- [ ] Tìm kiếm nhỏ (≤ 6 run) trên learning rate và dropout cho mô hình tốt nhất; ghi đủ vào sổ thí nghiệm.
- [ ] E04 (tuỳ chọn): mô hình tốt nhất + `DATA_VERSION` mới của S3-DATA.

### C. Ứng viên mô hình cuối — hạn 12/11
- [ ] Bảng ứng viên: EXP_ID, số tham số, kích thước file, thời gian suy luận, Macro F1 val (từ file của Ân).
- [ ] Bàn giao checkpoint ứng viên cho Ân và Nguyên.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Config E03/E04 | `configs/experiments/E03_*.yaml`, `E04_*.yaml` | 08/11 |
| Run E03/E04 | Drive `runs/E03_*`, `runs/E04_*` | 10/11 |
| Bảng ứng viên | `docs/experiment_plan.md` (mục ứng viên) | 12/11 |
| Code | `src/train.py`, notebook 03 | 12/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] So sánh công bằng: chỉ thay đổi đúng một yếu tố giữa hai thí nghiệm được so sánh, hoặc ghi rõ các yếu tố thay đổi.
- [ ] Không sử dụng tập Test.
- [ ] Mọi run có trong `docs/experiment_log.md`.
- [ ] PR đã được Yến approve và merge vào `main`.

## 6. Ngoài phạm vi

- Lượng tử hoá/triển khai trên robot; Object Detection.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | {{S2-EVAL}} | Mô hình tốt nhất Sprint 2, khuyến nghị cải tiến |
| Nhận từ | {{S3-DATA}} | `DATA_VERSION` mới (09/11) |
| Bàn giao cho | {{S3-EVAL}} | Checkpoint ứng viên (12/11) |
| Bàn giao cho | {{S3-DEMO}} | Checkpoint cuối (13/11) |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 08/11 | E03 xong | Yến |
| 10/11 | Tinh chỉnh + E04 xong | Yến |
| 12/11 | Mở PR `exp(s3-cnn): transfer learning and final candidates`; họp chốt mô hình | Cả nhóm |
| 14/11 | PR được merge | Yến |

## 9. Quy định review

- Tiêu đề `exp(s3-cnn): ...`; nội dung theo PR template; `Closes` issue này.
- Reviewer bắt buộc **@phanyen06ym-ien**; chỉ Yến merge.
- Báo tiến độ thứ Hai và thứ Năm.

## 10. Ghi chú

- Ưu tiên mô hình nhẹ vì định hướng sau giữa kỳ là chạy trên robot thu gom.
