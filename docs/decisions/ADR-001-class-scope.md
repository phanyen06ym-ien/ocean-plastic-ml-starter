# ADR-001 — Phạm vi lớp, kích thước ảnh và cách chia dữ liệu

| Mục | Nội dung |
|---|---|
| Trạng thái | **Approved** |
| Ngày | 03/11/2026 |
| Người quyết định | Phan Hải Yến (trưởng nhóm) |
| Issue | #4 (S1-ARCH), #10 (S2-ARCH) |
| Tài liệu liên quan | `docs/class_scope_analysis.md`, `docs/problem_statement.md`, `configs/data_contract.yaml` |

## 1. Bối cảnh

TrashCan 1.0 — Instance có 22 lớp đối tượng, phân bố rất lệch (`rov` và `trash_unknown_instance` chiếm khoảng một nửa số annotation). Tập val gốc thiếu mẫu ở một số lớp (`animal_shells` 1, `trash_cup` 0, `trash_rope` 0). Notebook CNN dùng ảnh 128×128 trong khi problem statement ghi 224×224. Nhóm cần một quyết định duy nhất để DATA, CNN, EVAL, DEMO dùng chung.

## 2. Các phương án đã xem xét

| Phương án | Mô tả | Lý do không chọn / chọn |
|---|---|---|
| A — 5 lớp nhựa + `other` | bag, net, bottle, tarp, rope + lớp còn lại | Ít dữ liệu (~1.400 crop), cần quy tắc lấy mẫu `other` |
| B — 5 lớp nhựa | Không có `other` | Ảnh ngoài phạm vi bị ép vào lớp nhựa |
| C — 6 lớp nhựa | Thêm snack_wrapper (84 mẫu) | Lớp quá ít mẫu, chỉ số không ổn định |
| **D — 22 lớp** | CNN đủ 22 lớp, phân tích chuyên sâu 6 lớp liên quan nhựa | **Chọn** |

## 3. Quyết định

| # | Quyết định |
|---|---|
| D1 | CNN phân loại **22 lớp** TrashCan-Instance, `class_index` 0–21 theo thứ tự `category_id` tăng dần (`configs/class_mapping.json`). |
| D2 | Báo cáo đánh giá có **hai cấp**: tổng thể 22 lớp (Macro F1 là chỉ số chính) và **nhóm 6 lớp liên quan nhựa**: `trash_bag`, `trash_bottle`, `trash_snack_wrapper`, `trash_net`, `trash_rope`, `trash_tarp`. Nhóm này chỉ là *plastic-related candidates*, không phải nhãn chất liệu. |
| D3 | Ảnh đầu vào **128 × 128 × 3**, RGB, resize thẳng (không giữ tỉ lệ), chuẩn hoá `pixel / 255.0` trong mô hình. |
| D4 | Gộp train + val gốc, **chia lại theo video** với tỉ lệ ảnh nguồn 70 / 15 / 15, seed 42. Mọi annotation của cùng một video nằm cùng một tập. |
| D5 | Crop theo bbox COCO, mở rộng 10% mỗi phía (`padding_ratio = 0.1`), bỏ box có cạnh ngắn hơn 8 px. |
| D6 | Giữ nguyên các lớp lớn (`rov`, `trash_unknown_instance`), xử lý mất cân bằng bằng **class weights** (`balanced`) khi huấn luyện. |
| D7 | Chọn mô hình chỉ bằng Validation; Test chạy một lần cho mô hình cuối. |

## 4. Hệ quả

- README đổi định hướng: phân loại đối tượng dưới nước 22 lớp, trọng tâm báo cáo là nhóm rác liên quan nhựa.
- `src/preprocessing.py` phải có hàm chia theo video và kiểm tra không rò rỉ.
- Transfer learning (Sprint 3) cần resize lên kích thước backbone yêu cầu; ghi trong config thí nghiệm.
- Kết quả theo lớp phải kèm số mẫu (support); lớp có support nhỏ không được kết luận mạnh.
- Thay đổi bất kỳ mục nào ở trên phải có ADR mới.
