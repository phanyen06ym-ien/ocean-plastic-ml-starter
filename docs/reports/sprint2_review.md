# Biên bản Sprint 2 Review

| Mục | Nội dung |
|---|---|
| Sprint | Sprint 2 — Data & Baseline (kế hoạch 04/11/2026; hoàn thành 05/11/2026) |
| Người lập | Phan Hải Yến |
| Ngày | 05/11/2026 |
| Nguồn số liệu | GitHub issue/PR, `results/metrics/`, Google Sheet "Ocean Plastic ML: Kế hoạch và checklist Sprint" |

## 1. Kết quả theo issue

| Issue | Nội dung | Trạng thái | Bàn giao |
|---|---|---|---|
| #10 S2-ARCH | Tích hợp pipeline, sổ thí nghiệm, PR kết quả | Đóng | PR #29, #30, #31, #32 |
| #11 S2-DATA | Dữ liệu `v1_crop128`, class mapping, chia theo video, EDA | Đóng | PR #28, #31; Drive `data/processed/v1_crop128.zip` |
| #12 S2-CNN | Huấn luyện E01, E02 | Đóng | Drive `runs/E01_20261105-1513`, `runs/E02_20261105-1515`; PR #31 |
| #13 S2-EVAL | Đánh giá Validation, phân tích lỗi nhóm nhựa | Đóng | `docs/evaluation_report.md` (PR #31) |
| #14 S2-DEMO | Demo với checkpoint thật (notebook bước 9, CLI) | Đóng | `docs/reports/sprint2_results.md`; hình demo trên Drive `exports/` |

Kiểm thử: `pytest -q` trên Colab T4 (TensorFlow 2.21) 40/40 đạt sau khi sửa import trong `tests/test_preprocessing.py` (PR #30).

## 2. Kết quả chính (Validation)

| | Accuracy | Macro F1 (22 lớp) | Macro F1 nhóm nhựa |
|---|---:|---:|---:|
| E01 CNN + class weights | 0,219 | **0,076** | 0,000 |
| E02 E01 + augmentation | 0,161 | 0,065 | 0,017 |

- Mốc so sánh cho Sprint 3: **E01**.
- H2 (augmentation cải thiện Macro F1) không được xác nhận.
- Mô hình chỉ cao hơn mức ngẫu nhiên (≈ 0,045) một chút; nguyên nhân: học bối cảnh video, phân bố lớp Train/Val lệch do chia theo video, trọng số lớp quá lớn với lớp rất ít mẫu, EarlyStopping dừng sớm.

## 3. Vấn đề gặp phải

| Vấn đề | Ảnh hưởng | Xử lý |
|---|---|---|
| Thư mục TrashCan của Hùng trên Drive thiếu annotation gốc | Không chạy được tiền xử lý | Notebook tự tải `dataset.zip` từ UMN (PR #29) |
| Chưa có `GH_TOKEN` cho Colab | Không đẩy kết quả từ Colab | Notebook lưu kết quả lên Drive `exports/`, Yến đưa vào repo qua PR #31 |
| `pytest` lỗi thu thập trên Colab (`tests.conftest`) | Dừng ở bước 5 | PR #30 |
| Thành viên không phản hồi review, không nhận việc | Chậm 1 ngày so với kế hoạch | Yến làm thay theo CONTRIBUTING 6.1, ghi chú trên Sheet |
| Kết quả mô hình thấp | Chưa có mô hình dùng được | Phân tích lỗi, đề xuất cho Sprint 3 |

## 4. Việc chuyển sang Sprint 3

- [ ] Bật bảo vệ nhánh `main` (S1-06, cần quyền admin trên giao diện GitHub).
- [ ] Thêm tài khoản GitHub của Ân vào repo (S1-07, cần username).
- [ ] E03: học chuyển giao MobileNetV2 (#17).
- [ ] Phiên bản dữ liệu mới: chia theo video có phân tầng để Train/Val cân đối hơn (#16).
- [ ] Trọng số lớp căn bậc hai hoặc chặn 10; EarlyStopping theo Macro F1, patience dài hơn (#17).
- [ ] Báo cáo thêm kết quả gộp nhóm lớp (#18).
