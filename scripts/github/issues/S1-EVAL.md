---
key: S1-EVAL
number: 7
title: [S1-EVAL] Evaluation Pipeline, Experiment Comparison & Error Analysis Design
assignees: {{AN}}, phanyen06ym-ien
labels: area:eval, type:feat, P1-high
milestone: Sprint 1 — Foundation
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 1 — Foundation |
| Mã nhiệm vụ | `S1-EVAL` |
| Người phụ trách chính | Nguyễn Hoàng Ân {{AN_MENTION}} |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) |
| Vai trò | Model Evaluation · Experiment Comparison · Error Analysis |
| Ưu tiên | P1 — High |
| Nhánh | `feat/sprint1-evaluation-pipeline` (đã có; nhánh sau theo quy ước `feat/s<N>-eval-...`) |
| Thời gian | 23/10 → 08/11/2026 |
| Hạn mở PR | **04/11** (code đã có trên nhánh, chưa mở PR) |

## 2. Mục tiêu

Xây dựng Evaluation Pipeline có thể tái lập cho CNN phân loại nhiều lớp, gồm chỉ số, báo cáo theo lớp, confusion matrix, learning curves, bảng so sánh thí nghiệm và quy trình phân tích lỗi.

**Kết quả mong đợi:** `src/evaluate.py` + `tests/test_evaluate.py` + notebook 04 chạy được với dữ liệu mẫu đã biết kết quả; sẵn sàng đánh giá E01 ở Sprint 2.

## 3. Phạm vi công việc

### A. Kế hoạch đánh giá — hạn 03/11
- [ ] `docs/evaluation_plan.md`: vai trò Train/Val/Test; chọn mô hình bằng Validation; Test chỉ chạy một lần.
- [ ] Chỉ số: Accuracy, Precision, Recall, F1 (per-class, macro, weighted); **Macro F1 là chỉ số chính**.
- [ ] Đánh giá riêng nhóm lớp liên quan nhựa (theo ADR-001).

### B. Cài đặt — hạn 04/11
- [ ] Hàm tính chỉ số, classification report, confusion matrix (raw + normalized).
- [ ] Learning curves từ `history.csv` của S1-CNN.
- [ ] Thứ tự nhãn lấy từ `configs/class_mapping.json`; xử lý lớp không xuất hiện trong tập đánh giá.
- [ ] Xuất kết quả ra `results/metrics/<EXP_ID>_<split>.json` và hình `results/figures/<EXP_ID>_*.png`.

### C. So sánh và phân tích lỗi — hạn 05/11
- [ ] Mẫu bảng so sánh E01 vs E02 (giữ cố định dữ liệu, seed, split).
- [ ] Top cặp lớp nhầm lẫn; lưới ảnh dự đoán sai (ảnh, nhãn thật, nhãn dự đoán, confidence).
- [ ] Mẫu bảng tổng hợp nguyên nhân lỗi (giả thuyết vs đã kiểm chứng).

### D. Kiểm tra — hạn 05/11
- [ ] `tests/test_evaluate.py` với dữ liệu có kết quả kỳ vọng.
- [ ] Notebook `04_evaluation.ipynb` chạy được trên Colab.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Evaluation Plan | `docs/evaluation_plan.md` | 03/11 |
| Module + test | `src/evaluate.py`, `tests/test_evaluate.py` | 04/11 |
| Notebook | `notebooks/04_evaluation.ipynb` | 05/11 |
| Bằng chứng smoke test | `results/metrics/smoke_test_*.json` (ghi rõ là dữ liệu giả) | 05/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] `pytest tests/test_evaluate.py` chạy qua.
- [ ] Mọi file kết quả từ dữ liệu giả được đặt tên `smoke_test_*` và ghi chú rõ, không trình bày như kết quả TrashCan.
- [ ] PR không sửa `README.md`, `.gitignore`, `docs/colab_setup.md` ngoài phạm vi (tách PR riêng nếu cần).
- [ ] PR đã được Yến approve và merge vào `main`.

## 6. Ngoài phạm vi

- Đánh giá chính thức trên mô hình đã huấn luyện — S2-EVAL.
- Dữ liệu — Hùng. CNN — Hoàng Bách. Demo — Ngọc Nguyên.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | #5 | Class mapping, quy tắc split |
| Nhận từ | #6 | Định dạng history, checkpoint |
| Nhận từ | #4 | Kế hoạch thực nghiệm |
| Bàn giao cho | #8 | Định dạng thông tin dự đoán dùng cho phân tích lỗi |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 04/11 | Mở PR `feat(s1-eval): evaluation module, tests and notebook` từ nhánh hiện có | Yến |
| 06/11 | Xử lý xong nhận xét review | Yến |
| 07/11 | PR được merge | Yến |

## 9. Quy định review

- Tiêu đề `feat(s1-eval): ...`; nội dung theo PR template; `Closes #7`.
- Reviewer bắt buộc **@phanyen06ym-ien**.
- Báo tiến độ thứ Hai và thứ Năm.

## 10. Ghi chú

- **Ghi chú review (02/11 — Yến):** nhánh `feat/sprint1-evaluation-pipeline` có 3 commit (24/10) gồm `src/evaluate.py`, `tests/test_evaluate.py`, notebook 04, hình smoke test, nhưng **chưa mở PR**. Nhánh này cũng sửa `README.md`, `.gitignore`, `requirements.txt`, `docs/colab_setup.md` → cần cập nhật từ `main` và loại phần sửa ngoài phạm vi trước khi review.
- GitHub của Ân chưa là collaborator nên chưa gán được issue; Ân gửi username cho Yến.
