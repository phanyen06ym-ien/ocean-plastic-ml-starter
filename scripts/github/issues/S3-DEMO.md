---
key: S3-DEMO
title: [S3-DEMO] Final Demo App, End-to-End Test & Presentation Material
assignees: hiang0910, phanyen06ym-ien
labels: area:demo, type:feat, P1-high
milestone: Sprint 3 — Improvement & Midterm
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 3 — Improvement & Midterm |
| Mã nhiệm vụ | `S3-DEMO` |
| Người phụ trách chính | Ngọc Nguyên (@hiang0910) |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) |
| Vai trò | Inference Developer · Demo · Testing |
| Ưu tiên | P1 — High |
| Nhánh | `feat/s3-demo-final-app` |
| Thời gian | 05/11 → 18/11/2026 |
| Hạn mở PR | 15/11 |

## 2. Mục tiêu

Hoàn thiện demo dùng mô hình cuối, kiểm thử end-to-end và chuẩn bị tư liệu demo cho buổi trình bày giữa kỳ.

**Kết quả mong đợi:** demo chạy ổn định trên Colab (giao diện đơn giản, ví dụ Gradio) với checkpoint cuối; bộ ảnh demo gồm cả dự đoán đúng và sai; video/ảnh chụp dự phòng.

## 3. Phạm vi công việc

### A. Giao diện demo — hạn 10/11
- [ ] Ứng dụng demo trên Colab: tải ảnh → hiển thị top-3 lớp + confidence; nút chọn ảnh mẫu.
- [ ] Logic dự đoán gọi `src/predict.py`, không viết lại.

### B. Tích hợp mô hình cuối — hạn 14/11
- [ ] Nạp checkpoint cuối (sau ADR 12/11); cập nhật đường dẫn trong cấu hình.
- [ ] Bộ 10–15 ảnh demo từ tập Test (Yến duyệt), có cả trường hợp sai để thảo luận.

### C. Kiểm thử end-to-end — hạn 15/11
- [ ] Chạy demo trên Colab mới, CPU và GPU; đo thời gian suy luận.
- [ ] `tests/test_predict.py` chạy qua với checkpoint cuối (test đánh dấu `slow`, chạy thủ công).

### D. Tư liệu trình bày — hạn 16/11
- [ ] Video quay màn hình demo 1–2 phút (dự phòng khi mạng lỗi) — lưu Drive.
- [ ] Ảnh chụp cho slide; cập nhật `docs/demo_spec.md` và README mục Demo.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Demo app | `notebooks/05_inference_demo.ipynb` (hoặc `app/`), `src/predict.py` | 14/11 |
| Bộ ảnh demo | Drive `demo/images/` | 14/11 |
| Video demo | Drive `demo/` | 16/11 |
| Tài liệu | `docs/demo_spec.md`, README | 16/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] Demo chạy từ đầu trên Colab mới theo README, không lỗi.
- [ ] Kết quả trên ảnh demo khớp với file dự đoán của S3-EVAL.
- [ ] Có phương án dự phòng (video).
- [ ] PR đã được Yến approve và merge vào `main`.

## 6. Ngoài phạm vi

- Video thời gian thực, Object Detection, robot.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | {{S3-CNN}} | Checkpoint cuối (13/11) |
| Nhận từ | {{S3-EVAL}} | Dự đoán trên Test để so khớp |
| Bàn giao cho | {{S3-ARCH}} | Ảnh/video demo cho slide (16/11) |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 10/11 | Giao diện demo với checkpoint Sprint 2 | Yến |
| 14/11 | Tích hợp mô hình cuối | Yến |
| 15/11 | Mở PR `feat(s3-demo): final demo app` | Yến |
| 16/11 | Video demo; PR được merge | Yến |

## 9. Quy định review

- Tiêu đề `feat(s3-demo): ...`; nội dung theo PR template; `Closes` issue này.
- Reviewer bắt buộc **@phanyen06ym-ien**; chỉ Yến merge.
- Báo tiến độ thứ Hai và thứ Năm.

## 10. Ghi chú

- Không trình bày demo như hệ thống giám sát thực tế; đây là thành phần phân loại ảnh đã crop.
