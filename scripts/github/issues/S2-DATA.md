---
key: S2-DATA
title: [S2-DATA] Processed Dataset, Class Mapping & Grouped Train/Val/Test Split
assignees: tranngochung126hqv-stack, phanyen06ym-ien
labels: area:data, type:feat, P0-critical
milestone: Sprint 2 — Data & Baseline
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 2 — Data & Baseline |
| Mã nhiệm vụ | `S2-DATA` |
| Người phụ trách chính | Hùng (@tranngochung126hqv-stack) |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) |
| Vai trò | Data Engineer · Preprocessing |
| Ưu tiên | P0 — Critical (đường găng của Sprint 2) |
| Nhánh | `feat/s2-data-processed-dataset` |
| Thời gian | 09/11 → 22/11/2026 |
| Hạn bàn giao dữ liệu | **12/11** · Hạn mở PR: 13/11 |

## 2. Mục tiêu

Tạo bộ dữ liệu ảnh đối tượng đã crop cho **các lớp rác liên quan nhựa đã chốt trong ADR-001**, chia Train/Val/Test theo video, và bàn giao cho CNN, EVAL, DEMO.

**Kết quả mong đợi:** thư mục `data/processed/<DATA_VERSION>/{train,val,test}/<class_name>/` trên Drive, `configs/class_mapping.json`, `split_manifest.csv`, báo cáo số mẫu thực tế từng lớp ở từng tập.

## 3. Phạm vi công việc

### A. Bàn giao phần còn thiếu của Sprint 1 — hạn 10/11
- [ ] PR đưa `src/eda.py`, notebook 01, `docs/dataset_report.md` (số liệu thật) lên `main` (không kèm dữ liệu).

### B. Chọn lớp và crop — hạn 11/11
- [ ] Lọc annotation thuộc các lớp trong ADR-001; lọc box nhỏ hơn `min_box_side_px`, box lỗi.
- [ ] Crop theo bbox + `padding_ratio`; resize theo `data_contract.yaml`.
- [ ] Nếu ADR-001 có lớp `other`: lấy mẫu từ các lớp ngoài phạm vi theo quy tắc đã chốt.
- [ ] Bảng số crop hợp lệ theo lớp; lớp dưới ngưỡng → báo Yến ngay.

### C. Chia dữ liệu — hạn 12/11
- [ ] Gộp train/val gốc, chia **theo video** với tỉ lệ trong data contract, seed 42; chia trước khi crop.
- [ ] Kiểm tra: không video nào nằm ở hai tập; mỗi lớp có mẫu ở cả ba tập.
- [ ] `split_manifest.csv` (image_id, video_id, ann_id, class_name, split, crop_path).
- [ ] `configs/class_mapping.json`: `category_id → class_name → class_index`.

### D. Tái lập và báo cáo — hạn 13/11
- [ ] Hàm trong `src/preprocessing.py` + notebook 02 tạo lại toàn bộ dữ liệu từ dataset gốc bằng một lần chạy.
- [ ] `tests/test_preprocessing.py`: crop hợp lệ, không rò rỉ video giữa các tập, mapping đủ lớp.
- [ ] Cập nhật `docs/dataset_report.md`: phân bố từng tập, hình ảnh mẫu mỗi lớp, hạn chế.

### E. Hỗ trợ — hạn 20/11
- [ ] Phối hợp Bách thêm class weights hoặc oversampling nếu E01 cho thấy mất cân bằng nặng.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Artefact Sprint 1 còn thiếu | PR riêng `feat(s2-data): deliver S1 audit and EDA` | 10/11 |
| Dữ liệu processed | Drive: `data/processed/<DATA_VERSION>/` | 12/11 |
| Class mapping | `configs/class_mapping.json` | 12/11 |
| Split manifest | Drive + thống kê trong `docs/dataset_report.md` | 12/11 |
| Code + test | `src/preprocessing.py`, `tests/test_preprocessing.py`, notebook 02 | 13/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] Bách, Ân, Nguyên xác nhận đọc được dữ liệu và mapping (bình luận trong issue).
- [ ] Không có video giao nhau giữa các tập (có kiểm tra tự động).
- [ ] Toàn bộ dữ liệu tạo lại được từ code trên `main`.
- [ ] Không commit ảnh hoặc annotation.
- [ ] PR đã được Yến approve và merge vào `main`.

## 6. Ngoài phạm vi

- Huấn luyện, đánh giá, demo.
- Phục hồi/tăng cường chất lượng ảnh bằng GAN.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | {{S2-ARCH}} | ADR-001, data contract v1 |
| Bàn giao cho | {{S2-CNN}} | Dữ liệu train/val, mapping |
| Bàn giao cho | {{S2-EVAL}} | Mapping, tập val/test, manifest |
| Bàn giao cho | {{S2-DEMO}} | Mapping, tham số tiền xử lý, ảnh mẫu |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 10/11 | PR artefact Sprint 1 | Yến |
| 11/11 | Bảng số crop hợp lệ theo lớp | Yến |
| 12/11 | **Bàn giao dữ liệu processed + mapping** | Bách, Ân, Nguyên |
| 13/11 | Mở PR `feat(s2-data): ...` | Yến |
| 15/11 | PR được merge | Yến |

## 9. Quy định review

- Tiêu đề `feat(s2-data): ...`; nội dung theo PR template; `Closes` issue này.
- Reviewer bắt buộc **@phanyen06ym-ien**; chỉ Yến merge.
- Báo tiến độ thứ Hai và thứ Năm. Trễ mốc 12/11 phải báo Yến **trước** ngày đó.

## 10. Ghi chú

- Số annotation ≠ số ảnh nguồn: một ảnh có thể chứa nhiều đối tượng.
- `trash_rope` không có mẫu ở val gốc → bắt buộc chia lại theo video.
