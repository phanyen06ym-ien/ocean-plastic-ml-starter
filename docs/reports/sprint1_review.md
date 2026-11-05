# Biên bản Sprint 1 Review

| Mục | Nội dung |
|---|---|
| Sprint | Sprint 1 — Foundation (23/10 → 03/11/2026; merge bàn giao sáng 04/11) |
| Người lập | Phan Hải Yến |
| Ngày | 04/11/2026 |
| Nguồn số liệu | GitHub issue/PR, Google Sheet "Ocean Plastic ML: Kế hoạch và checklist Sprint" |

## 1. Kết quả

| Issue | Nội dung | Trạng thái | Bàn giao |
|---|---|---|---|
| #4 S1-ARCH | Bài toán, ADR-001, data contract v1, quy ước repo | Đóng | PR #20, #9, #22 |
| #5 S1-DATA | Dataset Audit, EDA, tiền xử lý | **Mở** | Code: PR #21; khung báo cáo: PR #25; còn thiếu số liệu thật |
| #6 S1-CNN | CNN baseline, training pipeline | Đóng | PR #21, #26 |
| #7 S1-EVAL | Evaluation pipeline | Đóng | PR #21 (code của Ân), #26 |
| #8 S1-DEMO | Inference, CLI, test | Đóng | PR #21, #24, #26 |

Kiểm thử: `pytest -q` 37/37 đạt, gồm 5 test TensorFlow (Windows, Python 3.11, TF 2.21).

## 2. Quyết định đã chốt

- ADR-001: 22 lớp TrashCan-Instance, ảnh 128×128, chia theo video 70/15/15 (seed 42), class weights; báo cáo Macro F1 tổng thể và nhóm 6 lớp liên quan nhựa.
- Lịch rút gọn: Sprint 2 trong ngày 04/11, Sprint 3 từ 05/11 đến 18/11.
- Vòng lặp xử lý việc và chuyển trách nhiệm review cho Yến khi không ai phản hồi (CONTRIBUTING mục 6.1).

## 3. Vấn đề gặp phải

| Vấn đề | Ảnh hưởng | Xử lý |
|---|---|---|
| Thành viên không review PR #9, #21, #22 | Chậm merge | Chuyển trách nhiệm cho Yến lúc 01:55, merge lúc 07:35 |
| Notebook 03 được commit thẳng lên `main` và có cell `%%writefile` ghi đè `src/train.py` | Có thể mất code mới | Refactor notebook (PR #23); bật bảo vệ `main` |
| PR xếp chồng #23, #24 merge vào nhánh của #21 thay vì `main` | Nội dung chưa lên `main` | PR #26 đưa nội dung giống hệt lên `main` |
| Chưa có tài khoản GitHub của Ân trong repo | Không gán được #7, #13, #18 | Yến tạm phụ trách |

## 4. Việc chuyển sang Sprint 2

- [ ] Bật bảo vệ nhánh `main` (Settings → Branches; admin được vượt qua).
- [ ] Thêm Ân vào repo.
- [ ] Chạy `notebooks/00_pipeline_colab.ipynb` trên Colab T4 GPU: dữ liệu thật → E01, E02 → đánh giá Validation → demo.
- [ ] Điền số liệu thật vào `docs/dataset_report.md`, đóng #5.
