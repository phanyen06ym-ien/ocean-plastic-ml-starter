---
key: S3-EVAL
title: [S3-EVAL] Final Test Evaluation, Model Comparison & Error Analysis Report
assignees: {{AN}}, phanyen06ym-ien
labels: area:eval, type:exp, P0-critical
milestone: Sprint 3 — Improvement & Midterm
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 3 — Improvement & Midterm |
| Mã nhiệm vụ | `S3-EVAL` |
| Người phụ trách chính | Nguyễn Hoàng Ân {{AN_MENTION}} |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) |
| Vai trò | Model Evaluation · Comparison · Error Analysis |
| Ưu tiên | P0 — Critical |
| Nhánh | `exp/s3-eval-final-test` |
| Thời gian | 23/11 → 06/12/2026 |
| Hạn mở PR | 02/12 |

## 2. Mục tiêu

Đánh giá các ứng viên trên Validation, đánh giá **mô hình cuối trên tập Test đúng một lần**, tổng hợp so sánh toàn bộ thí nghiệm và hoàn thiện phân tích lỗi cho báo cáo giữa kỳ.

**Kết quả mong đợi:** bảng so sánh E01–E04 (val), kết quả Test của mô hình cuối, phân tích lỗi theo từng lớp rác liên quan nhựa.

## 3. Phạm vi công việc

### A. Đánh giá ứng viên trên Validation — hạn 29/11
- [ ] `results/metrics/E03_val.json`, `E04_val.json` + hình.
- [ ] Bảng so sánh E01–E04: Accuracy, Macro P/R/F1, F1 từng lớp (kèm support).

### B. Đánh giá Test — hạn 01/12 (chỉ sau khi ADR mô hình cuối được merge)
- [ ] Chạy Test **một lần** cho mô hình cuối; `results/metrics/<EXP_ID>_test.json`, confusion matrix.
- [ ] Ghi thời điểm chạy, commit hash, checkpoint vào báo cáo.

### C. Phân tích lỗi cuối — hạn 02/12
- [ ] Cặp lớp nhầm nhiều nhất trên Test; lưới ảnh dự đoán sai.
- [ ] So sánh Val vs Test (chênh lệch, lý giải có kiểm soát).
- [ ] Hạn chế: số mẫu nhỏ, nhãn đối tượng ≠ chất liệu, ảnh crop từ cùng nguồn video.

### D. Báo cáo — hạn 02/12
- [ ] Cập nhật `docs/evaluation_report.md` (bản cuối) và cung cấp bảng/hình cho báo cáo giữa kỳ.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Kết quả val E03/E04 | `results/metrics/E03_val.json`, `E04_val.json` | 29/11 |
| Kết quả Test | `results/metrics/<EXP_ID>_test.json`, `results/figures/<EXP_ID>_test_*.png` | 01/12 |
| Báo cáo đánh giá cuối | `docs/evaluation_report.md` | 02/12 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] Test chỉ chạy một lần, sau ADR; Yến xác nhận trong issue.
- [ ] Mọi bảng trong báo cáo khớp file JSON.
- [ ] PR đã được Yến approve và merge vào `main`.

## 6. Ngoài phạm vi

- Huấn luyện thêm sau khi đã xem kết quả Test (không được phép).

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | {{S3-CNN}} | Checkpoint ứng viên (30/11) |
| Nhận từ | {{S3-ARCH}} | ADR mô hình cuối (30/11) |
| Bàn giao cho | {{S3-ARCH}} | Bảng, hình cho báo cáo (02/12) |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 29/11 | So sánh Validation E01–E04 | Yến |
| 01/12 | **Chạy Test một lần** | Yến |
| 02/12 | Mở PR `exp(s3-eval): final test evaluation and error analysis` | Yến |
| 04/12 | PR được merge | Yến |

## 9. Quy định review

- Tiêu đề `exp(s3-eval): ...`; nội dung theo PR template; `Closes` issue này.
- Reviewer bắt buộc **@phanyen06ym-ien**; chỉ Yến merge.
- Báo tiến độ thứ Hai và thứ Năm.

## 10. Ghi chú

- Nếu kết quả Test thấp hơn kỳ vọng, báo cáo trung thực; không quay lại tinh chỉnh dựa trên Test.
