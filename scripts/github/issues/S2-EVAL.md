---
key: S2-EVAL
title: [S2-EVAL] Validation Evaluation of E01/E02 & Plastic-Class Error Analysis
assignees: {{AN}}, phanyen06ym-ien
labels: area:eval, type:exp, P1-high
milestone: Sprint 2 — Data & Baseline
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 2 — Data & Baseline |
| Mã nhiệm vụ | `S2-EVAL` |
| Người phụ trách chính | Nguyễn Hoàng Ân {{AN_MENTION}} |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) |
| Vai trò | Model Evaluation · Error Analysis |
| Ưu tiên | P1 — High |
| Nhánh | `exp/s2-eval-validation-report` |
| Thời gian | 09/11 → 22/11/2026 |
| Hạn mở PR | 19/11 |

## 2. Mục tiêu

Đánh giá E01 và E02 trên **tập Validation** bằng Evaluation Pipeline, so sánh hai thí nghiệm và phân tích lỗi theo từng lớp rác liên quan nhựa.

**Kết quả mong đợi:** file chỉ số và hình cho E01, E02; bảng so sánh; báo cáo lỗi có ví dụ ảnh; khuyến nghị cho Sprint 3.

## 3. Phạm vi công việc

### A. Tích hợp — hạn 12/11
- [ ] Hoàn tất merge PR S1-EVAL; cập nhật `src/evaluate.py` đọc `configs/class_mapping.json`.
- [ ] Hàm nhận checkpoint + tập val → `y_true`, `y_pred`, `y_prob`.

### B. Đánh giá E01 — hạn 15/11
- [ ] `results/metrics/E01_val.json`: accuracy, per-class P/R/F1, macro & weighted.
- [ ] `results/figures/E01_confusion_matrix.png` (raw + normalized), learning curves.

### C. Đánh giá E02 và so sánh — hạn 18/11
- [ ] `results/metrics/E02_val.json` + hình.
- [ ] Bảng so sánh E01 vs E02 trong `docs/evaluation_report.md` (Macro F1 là chỉ số chính).

### D. Phân tích lỗi — hạn 19/11
- [ ] Top cặp lớp nhầm; lưới 16 ảnh dự đoán sai cho mỗi mô hình.
- [ ] Phân loại nguyên nhân (mờ, thiếu sáng, che khuất, nhỏ, giống hình dạng) — ghi rõ giả thuyết hay đã kiểm chứng.
- [ ] Khuyến nghị cải tiến cho Sprint 3.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Chỉ số E01 | `results/metrics/E01_val.json`, `results/figures/E01_*.png` | 15/11 |
| Chỉ số E02 | `results/metrics/E02_val.json`, `results/figures/E02_*.png` | 18/11 |
| Báo cáo đánh giá + phân tích lỗi | `docs/evaluation_report.md` | 19/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] **Không chạy trên tập Test** trong Sprint 2.
- [ ] Số liệu khớp khi Yến chạy lại notebook 04 với cùng checkpoint.
- [ ] Thứ tự nhãn trong mọi bảng/hình khớp class mapping.
- [ ] PR đã được Yến approve và merge vào `main`.

## 6. Ngoài phạm vi

- Đánh giá Test cuối cùng — S3-EVAL. Huấn luyện — Bách.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | {{S2-CNN}} | Checkpoint E01 (14/11), E02 (17/11) |
| Nhận từ | {{S2-DATA}} | Tập val, class mapping |
| Bàn giao cho | {{S2-ARCH}} | Kết quả cho sổ thí nghiệm, README |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 12/11 | Pipeline đọc checkpoint thật | Yến |
| 15/11 | Kết quả E01 | Yến |
| 18/11 | Kết quả E02 + so sánh | Yến |
| 19/11 | Mở PR `exp(s2-eval): validation report for E01 and E02` | Yến |
| 21/11 | PR được merge | Yến |

## 9. Quy định review

- Tiêu đề `exp(s2-eval): ...`; nội dung theo PR template; `Closes` issue này.
- Reviewer bắt buộc **@phanyen06ym-ien**; chỉ Yến merge.
- Báo tiến độ thứ Hai và thứ Năm.

## 10. Ghi chú

- Một số lớp có ít mẫu val → ghi số mẫu (support) cạnh mỗi chỉ số theo lớp; không kết luận mạnh từ lớp có support nhỏ.
