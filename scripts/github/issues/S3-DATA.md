---
key: S3-DATA
title: [S3-DATA] Imbalance Handling, Crop Strategy Study & Dataset Card
assignees: tranngochung126hqv-stack, phanyen06ym-ien
labels: area:data, type:exp, P1-high
milestone: Sprint 3 — Improvement & Midterm
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 3 — Improvement & Midterm |
| Mã nhiệm vụ | `S3-DATA` |
| Người phụ trách chính | Hùng (@tranngochung126hqv-stack) |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) |
| Vai trò | Data Engineer · Data-centric Improvement |
| Ưu tiên | P1 — High |
| Nhánh | `exp/s3-data-imbalance-crop` |
| Thời gian | 05/11 → 18/11/2026 |
| Hạn mở PR | 13/11 |

## 2. Mục tiêu

Cải thiện dữ liệu theo kết quả phân tích lỗi Sprint 2 (mất cân bằng, crop quá sát/quá nhỏ) và hoàn thiện tài liệu mô tả bộ dữ liệu cho báo cáo.

**Kết quả mong đợi:** phương án dữ liệu cải tiến (nếu có) với `DATA_VERSION` mới, có số liệu so sánh trên Validation; dataset card đầy đủ.

## 3. Phạm vi công việc

### A. Phân tích đầu vào — hạn 06/11
- [ ] Đọc `docs/evaluation_report.md`; liệt kê lỗi liên quan dữ liệu (lớp nhỏ, box nhỏ, nhãn nghi sai).

### B. Phương án cải tiến — hạn 09/11
- [ ] Oversampling lớp nhỏ trên Train hoặc class weights (phối hợp Bách — chỉ chọn một để so sánh công bằng).
- [ ] Thử `padding_ratio` khác (ví dụ thêm ngữ cảnh quanh box) → `DATA_VERSION` mới, **giữ nguyên split theo video**.
- [ ] Ghi danh sách mẫu nghi gán nhãn sai (không tự sửa nhãn gốc).

### C. Dataset card — hạn 13/11
- [ ] `docs/dataset_card.md`: nguồn, giấy phép, trích dẫn, lớp đã chọn và lý do, số mẫu từng tập, quy trình crop/split, hạn chế, rủi ro thiên lệch.
- [ ] Hình: phân bố lớp sau xử lý, ảnh mẫu mỗi lớp.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Dữ liệu cải tiến (nếu dùng) | Drive `data/processed/<DATA_VERSION_v2>/` | 09/11 |
| Code | `src/preprocessing.py`, notebook 02 | 13/11 |
| Dataset card | `docs/dataset_card.md` | 13/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] Mọi `DATA_VERSION` mới tạo lại được từ code trên `main`.
- [ ] Tập Val/Test giữ nguyên giữa các phiên bản dữ liệu (chỉ Train thay đổi) để so sánh công bằng.
- [ ] Dataset card được Yến duyệt cho báo cáo.
- [ ] PR đã được Yến approve và merge vào `main`.

## 6. Ngoài phạm vi

- Sửa nhãn gốc TrashCan; tăng cường ảnh bằng GAN.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | {{S2-EVAL}} | Phân tích lỗi Sprint 2 |
| Bàn giao cho | {{S3-CNN}} | `DATA_VERSION` mới (09/11) |
| Bàn giao cho | {{S3-ARCH}} | Dataset card |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 06/11 | Danh sách lỗi liên quan dữ liệu | Yến |
| 09/11 | Bàn giao `DATA_VERSION` mới cho Bách | Bách |
| 13/11 | Mở PR `exp(s3-data): ...` + dataset card | Yến |
| 15/11 | PR được merge | Yến |

## 9. Quy định review

- Tiêu đề `exp(s3-data): ...`; nội dung theo PR template; `Closes` issue này.
- Reviewer bắt buộc **@phanyen06ym-ien**; chỉ Yến merge.
- Báo tiến độ thứ Hai và thứ Năm.

## 10. Ghi chú

- Nếu phân tích lỗi không cho thấy vấn đề dữ liệu đáng kể, ghi rõ và tập trung vào dataset card.
