# Kế hoạch Sprint và phân công

**Quản lý:** Phan Hải Yến — người phụ trách phụ của mọi issue, reviewer và người merge duy nhất. **Yến là người dự phòng:** việc được giao mà không hoàn thành đúng hạn, Yến làm thay và ghi chú trong issue.

> **Lịch rút gọn (quyết định 03/11/2026):** Sprint 1 kết thúc 03/11, Sprint 2 hoàn thành trong ngày 04/11, Sprint 3 bắt đầu 05/11. Môn học không có hạn nộp cố định.

## 1. Vai trò

| Khu vực | Người phụ trách chính | Người phụ trách phụ | Phạm vi |
|---|---|---|---|
| ARCH | Phan Hải Yến (@phanyen06ym-ien) | — | Bài toán, kiến trúc, data contract, thí nghiệm, review, báo cáo |
| DATA | Hùng (@tranngochung126hqv-stack) | Yến | Audit, EDA, crop, chia theo video, dataset card |
| CNN | Hoàng Bách (@phannguyenhoangbach2) | Yến | Kiến trúc, huấn luyện E01–E04 |
| EVAL | Nguyễn Hoàng Ân | Yến | Chỉ số, so sánh, phân tích lỗi, đánh giá Test |
| DEMO | Ngọc Nguyên (@hiang0910) | Yến | Inference, CLI, notebook demo, kiểm thử |

## 2. Danh sách issue

| Sprint | ARCH | DATA | CNN | EVAL | DEMO |
|---|---|---|---|---|---|
| S1 — Foundation (23/10–03/11) | #4 | #5 | #6 | #7 | #8 |
| S2 — Data & Baseline (04/11) | #10 | #11 | #12 | #13 | #14 |
| S3 — Improvement & Midterm (05/11–18/11) | #15 | #16 | #17 | #18 | #19 |

Quyết định phạm vi: [ADR-001](decisions/ADR-001-class-scope.md) — 22 lớp, ảnh 128×128, chia theo video 70/15/15, class weights.

## 3. Timeline

### Sprint 1 — Foundation · kết thúc 03/11/2026

| Việc | Trạng thái |
|---|---|
| Quy ước repo (PR #20) | Đã merge |
| ADR-001, problem statement, class scope (PR #9) | Review và merge 03/11 |
| Code DATA / CNN / EVAL / DEMO + test | PR mở 03/11 |

### Sprint 2 — Data & Baseline · 04/11/2026 (chạy trên Colab)

Chạy `notebooks/00_pipeline_colab.ipynb` từ trên xuống:

| Giờ (dự kiến) | Bước | Kết quả | Phụ trách (dự phòng: Yến) |
|---|---|---|---|
| 08:00 | Tải TrashCan lên Drive, chạy tiền xử lý | `data/processed/v1_crop128/`, `configs/class_mapping.json`, `split_manifest.csv` | Hùng |
| 09:00 | Huấn luyện E01 (baseline + class weights) | `runs/E01_*/best.keras`, `history.csv` | Bách |
| 10:30 | Huấn luyện E02 (E01 + augmentation) | `runs/E02_*/` | Bách |
| 12:00 | Đánh giá E01, E02 trên Validation | `results/metrics/E0x_val.json`, hình | Ân |
| 13:30 | Demo với checkpoint tốt nhất | Dự đoán ảnh mẫu, CLI chạy được | Nguyên |
| 15:00 | Mở PR kết quả, Yến review, merge | Issue #10–#14 đóng | Yến |
| 17:00 | Sprint 2 Review | `docs/reports/sprint2_review.md` | Cả nhóm |

### Sprint 3 — Improvement & Midterm · 05/11 → 18/11/2026

| Ngày | Mốc | Phụ trách |
|---|---|---|
| 05/11 | Sprint 3 Planning | Cả nhóm |
| 08/11 | E03 transfer learning | Bách |
| 09/11 | `DATA_VERSION` cải tiến (nếu cần) | Hùng |
| 10/11 | Tinh chỉnh xong; giao diện demo | Bách, Nguyên |
| 11/11 | So sánh Validation E01–E04 | Ân |
| **12/11** | **Chốt mô hình cuối (ADR)** | Cả nhóm |
| 13/11 | **Đánh giá Test một lần** | Ân |
| 15/11 | Kiểm tra tái lập; demo hoàn chỉnh | Yến, Nguyên |
| 16/11 | Báo cáo + slide | Cả nhóm |
| 18/11 | Release `v0.1-midterm`; Sprint 3 Review | Yến |

## 4. Đường găng

```
Tiền xử lý (04/11 sáng) → E01 → E02 → Đánh giá Val → Demo → Merge (04/11 chiều)
E03/E04 (10/11) → ADR mô hình cuối (12/11) → Test một lần (13/11) → Báo cáo (16/11)
```

## 5. Nhịp làm việc

| Hoạt động | Thời điểm | Hình thức |
|---|---|---|
| Sprint Planning | Đầu sprint | Họp 30 phút, Yến chốt issue |
| Báo tiến độ | Thứ Hai, thứ Năm (Sprint 2: cuối ngày 04/11) | Bình luận trong issue |
| Review PR | ≤ 48 giờ (Sprint 2: trong ngày) | Yến |
| Sprint Review | Cuối sprint | Biên bản trong `docs/reports/` |
