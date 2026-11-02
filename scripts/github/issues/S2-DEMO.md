---
key: S2-DEMO
title: [S2-DEMO] Inference with Trained Checkpoint & Colab Demo Notebook
assignees: hiang0910, phanyen06ym-ien
labels: area:demo, type:feat, P1-high
milestone: Sprint 2 — Data & Baseline
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 2 — Data & Baseline |
| Mã nhiệm vụ | `S2-DEMO` |
| Người phụ trách chính | Ngọc Nguyên (@hiang0910) |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) |
| Vai trò | Inference Developer · Testing · Demo |
| Ưu tiên | P1 — High |
| Nhánh | `feat/s2-demo-trained-model` |
| Thời gian | 04/11/2026 |
| Hạn mở PR | 04/11 |

## 2. Mục tiêu

Tích hợp checkpoint E01 thật vào chương trình dự đoán, xây dựng notebook demo trên Colab và kiểm thử end-to-end trên ảnh TrashCan đã crop.

**Kết quả mong đợi:** `python main.py --image ...` và notebook 05 dự đoán đúng định dạng trên ảnh thật với checkpoint E01; kết quả khớp với pipeline đánh giá.

## 3. Phạm vi công việc

### A. Tích hợp — hạn 04/11
- [ ] Đọc `configs/class_mapping.json` + data contract v1; nạp `best.keras` E01 từ Drive.
- [ ] So khớp: dự đoán của `predict.py` trên 20 ảnh val **trùng** với `y_pred` của pipeline đánh giá.

### B. Demo notebook — hạn 04/11
- [ ] `notebooks/05_inference_demo.ipynb`: chọn ảnh (upload hoặc đường dẫn Drive), hiển thị ảnh, top-3 lớp + confidence.
- [ ] Chế độ thư mục: dự đoán hàng loạt, xuất CSV.

### C. Kiểm thử — hạn 04/11
- [ ] Bổ sung `tests/test_predict.py`: kiểm tra số lớp khớp mapping, ảnh xám/RGBA được chuyển RGB.
- [ ] Đo thời gian suy luận trung bình một ảnh trên Colab CPU và GPU.

### D. Tài liệu — hạn 04/11
- [ ] Cập nhật `docs/demo_spec.md` và mục chạy demo trong README.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Inference tích hợp checkpoint thật | `src/predict.py`, `main.py` | 04/11 |
| Demo notebook | `notebooks/05_inference_demo.ipynb` | 04/11 |
| Test | `tests/test_predict.py` | 04/11 |
| Đặc tả + hướng dẫn | `docs/demo_spec.md` | 04/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] Kết quả khớp 20/20 ảnh với pipeline đánh giá.
- [ ] Notebook demo chạy từ đầu trên Colab mới (không phụ thuộc biến cũ).
- [ ] Không commit checkpoint; đường dẫn Drive cấu hình được.
- [ ] PR đã được Yến approve và merge vào `main`.

## 6. Ngoài phạm vi

- Giao diện web hoàn chỉnh — S3-DEMO. Object Detection, video.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | {{S2-CNN}} | Checkpoint E01 (04/11) |
| Nhận từ | {{S2-DATA}} | Class mapping, tham số tiền xử lý |
| Nhận từ | {{S2-EVAL}} | `y_pred` trên val để so khớp |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 04/11 | Dự đoán bằng checkpoint E01 | Yến |
| 04/11 | Notebook demo | Yến |
| 04/11 | Mở PR `feat(s2-demo): inference with trained checkpoint and demo notebook` | Yến |
| 04/11 | PR được merge | Yến |

## 9. Quy định review

- Tiêu đề `feat(s2-demo): ...`; nội dung theo PR template; `Closes` issue này.
- Reviewer bắt buộc **@phanyen06ym-ien**; chỉ Yến merge.
- Báo tiến độ thứ Hai và thứ Năm.

## 10. Ghi chú

- **Lịch rút gọn (03/11/2026):** Sprint 1 kết thúc 03/11, Sprint 2 hoàn thành trong ngày 04/11, Sprint 3 chạy 05/11 → 18/11. Việc không hoàn thành đúng hạn → Yến (dự phòng) làm thay và ghi chú tại đây. Phạm vi: [ADR-001](../blob/main/docs/decisions/ADR-001-class-scope.md) — 22 lớp, 128×128, chia theo video 70/15/15, class weights.
- Trong thời gian chờ checkpoint (trước 04/11), tiếp tục với mock model và phần giao diện notebook.
