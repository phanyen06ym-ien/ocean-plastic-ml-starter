---
key: S1-DEMO
number: 8
title: [S1-DEMO] Inference Module, Testing & Prediction Demo
assignees: hiang0910, phanyen06ym-ien
labels: area:demo, type:feat, P1-high
milestone: Sprint 1 — Foundation
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 1 — Foundation |
| Mã nhiệm vụ | `S1-DEMO` |
| Người phụ trách chính | Ngọc Nguyên (@hiang0910) |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) |
| Vai trò | Inference Developer · Testing · Demo |
| Ưu tiên | P1 — High |
| Nhánh | `feat/s1-demo-inference` |
| Thời gian | 23/10 → 03/11/2026 |
| Hạn mở PR | 03/11 |

## 2. Mục tiêu

Xây dựng chương trình dự đoán: nhận ảnh một đối tượng → tiền xử lý giống lúc huấn luyện → CNN → trả về tên lớp và confidence. Sprint 1 kiểm thử bằng mock model.

**Kết quả mong đợi:** `src/predict.py` tách biệt khỏi giao diện, `main.py` chạy được từ dòng lệnh, test tự động với mock model.

## 3. Phạm vi công việc

### A. Thiết kế — hạn 03/11
- [ ] Luồng `Ảnh → Preprocessing → CNN → Softmax → (class_name, class_index, confidence, top-k)`.
- [ ] Đặc tả đầu vào/đầu ra và thông báo lỗi trong `docs/demo_spec.md`.
- [ ] Kích thước ảnh, chuẩn hoá, class mapping đọc từ `configs/` — không hard-code.

### B. Cài đặt — hạn 03/11
- [ ] `load_image()`, `preprocess()`, `load_model()`, `predict()` trong `src/predict.py`.
- [ ] Xử lý: file không tồn tại, file không phải ảnh, thiếu checkpoint, class mapping không khớp số lớp.
- [ ] Thay nội dung mẫu PyCharm trong `main.py` bằng CLI: `python main.py --image <path> --model <path> [--top-k 3]`.

### C. Kiểm thử — hạn 03/11
- [ ] `tests/test_predict.py` với mock model (Keras nhỏ tạo trong test): ảnh JPG/PNG hợp lệ, đường dẫn sai, file không phải ảnh, shape sau preprocess, tổng xác suất = 1, thứ tự nhãn đúng mapping.
- [ ] Ghi kết quả smoke test trong PR.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Đặc tả demo | `docs/demo_spec.md` | 03/11 |
| Inference module | `src/predict.py` | 03/11 |
| CLI | `main.py` | 03/11 |
| Test | `tests/test_predict.py` | 03/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] `pytest tests/test_predict.py` chạy qua.
- [ ] Preprocessing dùng chung tham số với `configs/data_contract.yaml`.
- [ ] Thông báo lỗi dễ hiểu cho các trường hợp sai đầu vào.
- [ ] PR đã được Yến approve và merge vào `main`.

## 6. Ngoài phạm vi

- Dự đoán bằng mô hình đã huấn luyện thật — S2-DEMO.
- Object Detection, video thời gian thực, robot.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | #5 | Cấu hình tiền xử lý, class mapping |
| Nhận từ | #6 | Định dạng checkpoint, input shape |
| Bàn giao cho | #7 | Định dạng kết quả dự đoán |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 03/11 | `docs/demo_spec.md` | Yến |
| 03/11 | Mở PR `feat(s1-demo): inference module, CLI and tests` | Yến |
| 03/11 | PR được merge | Yến |

## 9. Quy định review

- Tiêu đề `feat(s1-demo): ...`; nội dung theo PR template; `Closes #8`.
- Reviewer bắt buộc **@phanyen06ym-ien**.
- Báo tiến độ thứ Hai và thứ Năm.

## 10. Ghi chú

- **Lịch rút gọn (03/11/2026):** Sprint 1 kết thúc 03/11, Sprint 2 hoàn thành trong ngày 04/11, Sprint 3 chạy 05/11 → 18/11. Việc không hoàn thành đúng hạn → Yến (dự phòng) làm thay và ghi chú tại đây. Phạm vi: [ADR-001](../blob/main/docs/decisions/ADR-001-class-scope.md) — 22 lớp, 128×128, chia theo video 70/15/15, class weights.
- Đầu vào giữa kỳ là ảnh **một đối tượng đã crop**, không phải ảnh nhiều vật thể.
- Confidence là điểm softmax, không phải xác suất đã hiệu chuẩn.
