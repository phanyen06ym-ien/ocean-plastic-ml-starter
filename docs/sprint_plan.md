# Kế hoạch Sprint và phân công

**Quản lý:** Phan Hải Yến — người phụ trách phụ của mọi issue, reviewer và người merge duy nhất.
**Nhịp sprint:** 2 tuần · Planning thứ Hai đầu sprint · Báo tiến độ trong issue thứ Hai & thứ Năm · Review cuối sprint Chủ nhật.

> Mốc thời gian là đề xuất ban đầu. Nếu hạn nộp giữa kỳ của môn khác 06/12/2026, Yến dời milestone trên GitHub và cập nhật bảng dưới đây.

## 1. Vai trò

| Khu vực | Người phụ trách chính | Người phụ trách phụ | Phạm vi |
|---|---|---|---|
| ARCH | Phan Hải Yến (@phanyen06ym-ien) | — | Bài toán, kiến trúc, data contract, thí nghiệm, review, báo cáo |
| DATA | Hùng (@tranngochung126hqv-stack) | Yến | Audit, EDA, chọn lớp, crop, split, dataset card |
| CNN | Hoàng Bách (@phannguyenhoangbach2) | Yến | Kiến trúc, huấn luyện E01–E04 |
| EVAL | Nguyễn Hoàng Ân | Yến | Chỉ số, so sánh, phân tích lỗi, đánh giá Test |
| DEMO | Ngọc Nguyên (@hiang0910) | Yến | Inference, CLI, notebook demo, kiểm thử |

## 2. Danh sách issue

| Sprint | ARCH | DATA | CNN | EVAL | DEMO |
|---|---|---|---|---|---|
| S1 — Foundation (23/10–08/11) | #4 | #5 | #6 | #7 | #8 |
| S2 — Data & Baseline (09/11–22/11) | S2-ARCH | S2-DATA | S2-CNN | S2-EVAL | S2-DEMO |
| S3 — Improvement & Midterm (23/11–06/12) | S3-ARCH | S3-DATA | S3-CNN | S3-EVAL | S3-DEMO |

Số issue Sprint 2–3 được gán khi chạy `scripts/github/setup_sprints.py`.

## 3. Timeline

### Sprint 1 — Foundation · 23/10 → 08/11/2026

| Ngày | Mốc | Phụ trách |
|---|---|---|
| 03/11 | PR quy ước repo; bật branch protection; mời Ân | Yến |
| 04/11 | Ân mở PR evaluation; Hùng gửi bảng thống kê + đề xuất lớp | Ân, Hùng |
| 05/11 | **Chốt ADR-001 (phạm vi lớp) + data contract**; Bách, Nguyên, Hùng mở PR S1 | Cả nhóm |
| 07/11 | Yến review xong mọi PR S1 | Yến |
| 08/11 | Sprint 1 Review | Cả nhóm |

### Sprint 2 — Data & Baseline · 09/11 → 22/11/2026

| Ngày | Mốc | Phụ trách |
|---|---|---|
| 09/11 | Sprint 2 Planning | Cả nhóm |
| 10/11 | Khoá class mapping + data contract v1; Hùng bàn giao artefact S1 | Yến, Hùng |
| **12/11** | **Dữ liệu processed trên Drive** (đường găng) | Hùng |
| 14/11 | Checkpoint E01 | Bách |
| 15/11 | Kết quả val E01; demo với checkpoint thật | Ân, Nguyên |
| 17/11 | Checkpoint E02 | Bách |
| 18/11 | So sánh E01 vs E02 | Ân |
| 19/11 | **Hạn mở PR toàn nhóm** | Cả nhóm |
| 21/11 | Merge xong | Yến |
| 22/11 | Sprint 2 Review | Cả nhóm |

### Sprint 3 — Improvement & Midterm · 23/11 → 06/12/2026

| Ngày | Mốc | Phụ trách |
|---|---|---|
| 23/11 | Sprint 3 Planning | Cả nhóm |
| 26/11 | E03 transfer learning | Bách |
| 27/11 | `DATA_VERSION` cải tiến | Hùng |
| 28/11 | Tinh chỉnh xong; giao diện demo | Bách, Nguyên |
| 29/11 | So sánh Validation E01–E04 | Ân |
| **30/11** | **Chốt mô hình cuối (ADR)** | Cả nhóm |
| 01/12 | **Đánh giá Test một lần** | Ân |
| 03/12 | Kiểm tra tái lập; demo hoàn chỉnh | Yến, Nguyên |
| 04/12 | Báo cáo + slide; tập dượt | Cả nhóm |
| 06/12 | Release `v0.1-midterm`; Sprint 3 Review | Yến |

## 4. Đường găng

```
S2-DATA (12/11) → S2-CNN E01 (14/11) → S2-EVAL (15/11) / S2-DEMO (15/11)
S3-CNN ứng viên (30/11) → ADR mô hình cuối (30/11) → S3-EVAL Test (01/12) → Báo cáo (04/12)
```

Trễ ở bất kỳ mốc in đậm nào phải báo Yến **trước** ngày đến hạn.

## 5. Nhịp làm việc

| Hoạt động | Thời điểm | Hình thức |
|---|---|---|
| Sprint Planning | Thứ Hai đầu sprint | Họp 30 phút, Yến chốt issue |
| Báo tiến độ | Thứ Hai, thứ Năm | Bình luận trong issue: đã làm / sẽ làm / vướng mắc |
| Checkpoint giữa sprint | Thứ Hai tuần 2 | Họp 15 phút |
| Review PR | ≤ 48 giờ | Yến |
| Sprint Review | Chủ nhật cuối sprint | Biên bản trong `docs/reports/` |
