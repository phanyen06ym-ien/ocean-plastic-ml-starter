---
key: S1-DATA
number: 5
title: [S1-DATA] TrashCan Dataset Audit, EDA & Preprocessing
assignees: tranngochung126hqv-stack, phanyen06ym-ien
labels: area:data, type:feat, P0-critical
milestone: Sprint 1 — Foundation
state: open
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 1 — Foundation |
| Mã nhiệm vụ | `S1-DATA` |
| Người phụ trách chính | Hùng (@tranngochung126hqv-stack) |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) |
| Vai trò | Data Engineer · Dataset Audit · EDA · Image Preprocessing |
| Ưu tiên | P0 — Critical (đường găng: CNN, EVAL, DEMO đều chờ dữ liệu) |
| Nhánh | `feat/s1-data-audit-eda` |
| Thời gian | 23/10 → 08/11/2026 |
| Hạn mở PR | 05/11 |

## 2. Mục tiêu

Kiểm kê và phân tích **TrashCan 1.0 — Instance**, cung cấp số liệu thực để chốt các lớp rác liên quan nhựa có đủ mẫu, và xây dựng khung tiền xử lý có thể tái lập.

**Kết quả mong đợi:** báo cáo Dataset Audit + EDA với số liệu thật, bảng thống kê theo lớp, đề xuất danh sách lớp có căn cứ, quy tắc chia dữ liệu theo video, khung `src/eda.py` và `src/preprocessing.py` chạy được trên Colab.

## 3. Phạm vi công việc

### A. Thu thập và kiểm kê — hạn 30/10
- [ ] Tải TrashCan 1.0 từ nguồn chính thức (UMN Data Repository), ghi nguồn, phiên bản, giấy phép, trích dẫn.
- [ ] Kiểm tra cấu trúc thư mục, định dạng annotation COCO, danh sách category của bản Instance.
- [ ] Thống kê số ảnh, số annotation, số đối tượng theo lớp, theo train/val gốc.
- [ ] Kiểm tra ảnh hỏng, annotation lỗi, bbox không hợp lệ, khung hình trùng lặp.

### B. EDA — hạn 03/11
- [ ] Biểu đồ phân bố lớp; mức mất cân bằng.
- [ ] Phân bố kích thước ảnh và kích thước bbox; tỉ lệ box quá nhỏ theo lớp.
- [ ] Hiển thị ảnh mẫu + annotation cho từng lớp ứng viên.
- [ ] Nhận xét hạn chế của ảnh dưới nước (đục, thiếu sáng, che khuất).

### C. Đề xuất lớp — hạn 04/11
- [ ] Bảng 22 lớp: số annotation, số crop hợp lệ sau lọc, số video chứa lớp.
- [ ] Đề xuất lớp liên quan nhựa đủ mẫu theo ngưỡng đề xuất; không mặc định nhãn là chất liệu nhựa.
- [ ] Thực hiện Visual Inspection theo `docs/visual_inspection_protocol.md` cho các lớp ứng viên.
- [ ] Bàn giao số liệu cho Yến để chốt ADR-001.

### D. Khung tiền xử lý — hạn 06/11
- [ ] Quy tắc chia Train/Val/Test **theo video** (tránh data leakage), chia **trước** khi crop.
- [ ] Hàm crop theo bbox, lọc box lỗi, resize, chuẩn hoá theo `configs/data_contract.yaml`.
- [ ] Augmentation chỉ áp dụng cho Train.
- [ ] Notebook `01_trashcan_eda.ipynb`, `02_preprocessing.ipynb`; module `src/eda.py`, `src/preprocessing.py`.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Dataset Audit Report | `docs/dataset_report.md` | 04/11 |
| Thống kê lớp | `results/metrics/dataset_class_distribution.csv` | 04/11 |
| EDA notebook + module | `notebooks/01_trashcan_eda.ipynb`, `src/eda.py` | 05/11 |
| Preprocessing notebook + module | `notebooks/02_preprocessing.ipynb`, `src/preprocessing.py` | 06/11 |
| Hình EDA | `results/figures/eda_*.png` | 05/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] Mọi số liệu lấy từ dữ liệu thật, có code tái tạo.
- [ ] Đề xuất lớp có bảng số liệu + kết quả Visual Inspection.
- [ ] Quy tắc split theo video được mô tả và kiểm tra không giao nhau.
- [ ] Notebook chạy lại được trên Colab theo `docs/colab_setup.md`.
- [ ] Không commit dữ liệu TrashCan hoặc ảnh crop.
- [ ] PR đã được Yến approve và merge vào `main`.

## 6. Ngoài phạm vi

- Huấn luyện CNN — Hoàng Bách (#6). Đánh giá mô hình — Ân (#7). Demo — Ngọc Nguyên (#8).
- Tạo bộ dữ liệu đã xử lý hoàn chỉnh cho huấn luyện — chuyển sang S2-DATA.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | #4 | Định hướng phạm vi lớp, data contract |
| Bàn giao cho | #4 | Số liệu để chốt ADR-001 |
| Bàn giao cho | #6, #7, #8 | Class mapping dự kiến, kích thước ảnh, chuẩn hoá |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 30/10 | Kiểm kê xong | Yến |
| 04/11 | Bàn giao bảng thống kê + đề xuất lớp | Yến |
| 05/11 | Mở PR `feat(s1-data): dataset audit and EDA` | Yến |
| 07/11 | PR được merge | Yến |

## 9. Quy định review

- Mở PR Draft ngay khi có commit đầu tiên; tiêu đề `feat(s1-data): ...`; nội dung theo PR template; `Closes #5`.
- Reviewer bắt buộc **@phanyen06ym-ien**; chỉ Yến merge.
- Báo tiến độ thứ Hai và thứ Năm.

## 10. Ghi chú

- **Ghi chú review (02/11 — Yến):** issue được đóng ngày 25/10 nhưng `docs/dataset_report.md`, `src/eda.py`, `src/preprocessing.py`, notebook 01–02 trên `main` vẫn trống và chưa có PR từ Hùng → **chưa đạt DoD**. Các file Hùng đã chuẩn bị (`instances_*_grouped_clean.json`, `grouped_class_distribution.csv`) cần được bàn giao qua PR (code + báo cáo, không kèm dữ liệu lớn). Phần còn lại tiếp tục tại S2-DATA.
- Lớp ứng viên liên quan nhựa: `trash_bag`, `trash_bottle`, `trash_snack_wrapper`, `trash_net`, `trash_rope`, `trash_tarp` — chưa chốt.
