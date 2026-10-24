# Kết Quả Đánh Giá Mô Hình CNN (Evaluation Summary)

## 1. Các Chỉ Số Tổng Hợp
- **Accuracy:** 0.8500
- **Macro F1-score (Chỉ số chính):** 0.8461
- **Macro Precision:** 0.8573
- **Macro Recall:** 0.8556
- **Weighted F1-score:** 0.8494
- **Tổng số mẫu kiểm thử:** 120

## 2. Chi Tiết Từng Lớp (Per-Class Performance)
| Lớp Rác Thải | Precision | Recall | F1-Score | Số Mẫu (Support) |
|---|---|---|---|---|
| `trash_bag` | 0.9032 | 0.8000 | 0.8485 | 35 |
| `trash_bottle` | 0.8929 | 0.8333 | 0.8621 | 30 |
| `trash_can` | 0.8000 | 0.8000 | 0.8000 | 15 |
| `trash_net` | 0.7143 | 1.0000 | 0.8333 | 15 |
| `trash_wrapper` | 0.8333 | 1.0000 | 0.9091 | 15 |
| `biological_or_rov` | 1.0000 | 0.7000 | 0.8235 | 10 |
