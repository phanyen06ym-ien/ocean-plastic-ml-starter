# Báo cáo đánh giá Sprint 2: E01, E02 trên tập Validation (S2-EVAL)

| Mục | Nội dung |
|---|---|
| Issue | #13 (S2-EVAL) · phụ trách Ân, Yến làm thay (dự phòng) |
| Dữ liệu | `v1_crop128`, Validation 1.782 crop, 22 lớp (đủ lớp) |
| Mô hình | checkpoint `best.keras` (val_loss nhỏ nhất) của `runs/E01_20261105-1513`, `runs/E02_20261105-1515` trên Drive |
| Môi trường | Colab GPU T4, TensorFlow 2.21, commit `7520c2b`, 05/11/2026 |
| Tập Test | **Chưa dùng** (ADR-001 D7) |

## 1. Kết quả tổng hợp

| Chỉ số (Validation) | E01: CNN + class weights | E02: E01 + augmentation |
|---|---:|---:|
| Accuracy | 0,219 | 0,160 |
| **Macro F1 (22 lớp)** | **0,076** | 0,065 |
| Macro Precision / Recall | 0,100 / 0,119 | 0,086 / 0,083 |
| Weighted F1 | 0,264 | 0,167 |
| Macro F1 nhóm 6 lớp liên quan nhựa | 0,000 | 0,017 |
| Epoch dừng / epoch tốt nhất | 13 / 7 | 12 / 6 |
| val_loss tốt nhất | 2,977 | 2,971 |

Nguồn: `results/metrics/E0x_val.json`, `comparison_val.csv`, `E0x_history.csv`. Mức ngẫu nhiên theo phân bố lớp của Validation cho Macro F1 khoảng 0,045.

**Kết luận:** cả hai mô hình chỉ cao hơn mức ngẫu nhiên một chút. Giả thuyết H2 (E02 tốt hơn E01) **không được xác nhận**: E02 thấp hơn E01 ở cả Macro F1 và Accuracy, chỉ nhích ở nhóm nhựa (0 → 0,017) nhờ 2 mẫu `trash_net` và 1 mẫu `trash_bag` đúng. Checkpoint dùng cho demo Sprint 2 là E01 (Macro F1 Validation cao hơn).

## 2. Kết quả theo lớp

`*` = lớp liên quan nhựa. Support nhỏ (< 10) không đủ để kết luận.

| Lớp | Train | Val (support) | E01 Precision | E01 Recall | E01 F1 | E02 F1 |
|---|---:|---:|---:|---:|---:|---:|
| rov | 2130 | 460 | 0,771 | 0,548 | 0,640 | 0,473 |
| plant | 228 | 73 | 0,000 | 0,000 | 0,000 | 0,000 |
| animal_fish | 604 | 81 | 0,618 | 0,420 | 0,500 | 0,443 |
| animal_starfish | 360 | 33 | 0,000 | 0,000 | 0,000 | 0,000 |
| animal_shells | 233 | 1 | 0,003 | 1,000 | 0,007 | 0,000 |
| animal_crab | 185 | 3 | 0,000 | 0,000 | 0,000 | 0,000 |
| animal_eel | 194 | 103 | 0,116 | 0,078 | 0,093 | 0,282 |
| animal_etc | 162 | 24 | 0,000 | 0,000 | 0,000 | 0,000 |
| trash_clothing | 59 | 21 | 0,007 | 0,048 | 0,013 | 0,000 |
| trash_pipe | 79 | 21 | 0,000 | 0,000 | 0,000 | 0,000 |
| `*` trash_bottle | 95 | 3 | 0,000 | 0,000 | 0,000 | 0,000 |
| `*` trash_bag | 605 | 106 | 0,000 | 0,000 | 0,000 | 0,012 |
| `*` trash_snack_wrapper | 41 | 11 | 0,000 | 0,000 | 0,000 | 0,000 |
| trash_can | 385 | 59 | 0,073 | 0,271 | 0,116 | 0,093 |
| trash_cup | 14 | 36 | 0,000 | 0,000 | 0,000 | 0,000 |
| trash_container | 299 | 152 | 0,023 | 0,007 | 0,010 | 0,036 |
| trash_unknown_instance | 1970 | 479 | 0,561 | 0,154 | 0,242 | 0,000 |
| trash_branch | 249 | 13 | 0,000 | 0,000 | 0,000 | 0,000 |
| trash_wreckage | 122 | 31 | 0,029 | 0,097 | 0,044 | 0,000 |
| `*` trash_tarp | 103 | 13 | 0,000 | 0,000 | 0,000 | 0,000 |
| `*` trash_rope | 87 | 23 | 0,000 | 0,000 | 0,000 | 0,000 |
| `*` trash_net | 79 | 36 | 0,000 | 0,000 | 0,000 | 0,087 |

Chỉ 3 lớp có F1 trên 0,2 ở E01: `rov` (0,64), `animal_fish` (0,50), `trash_unknown_instance` (0,24); đây đều là lớp nhiều mẫu và xuất hiện trong nhiều video. 14/22 lớp có F1 = 0.

## 3. Phân tích lỗi

### 3.1. Mô hình dồn dự đoán vào vài lớp

| Mô hình | Lớp được dự đoán nhiều nhất (số lần / support thật) |
|---|---|
| E01 | `rov` 327 / 460 · **`animal_shells` 303 / 1** · `trash_can` 218 / 59 · `trash_clothing` 137 / 21 |
| E02 | **`animal_eel` 465 / 103** · `rov` 183 / 460 · `animal_crab` 166 / 3 · `trash_clothing` 161 / 21 |

`animal_shells` có 233 mẫu Train nhưng chỉ đến từ 7 video, còn Validation có đúng 1 mẫu. E01 gán nhãn này cho 303 ảnh Validation (đa số là `trash_unknown_instance`, `animal_eel`, `trash_container`, `trash_cup`). Điều này cho thấy mô hình học **bối cảnh của vài video** (màu nước, nền đáy biển) thay vì hình dạng đối tượng, nên khi gặp video mới thì nhầm sang lớp có bối cảnh gần giống.

### 3.2. Lệch phân bố lớp giữa Train và Validation do chia theo video

| Lớp | Train | Val | Hệ quả |
|---|---:|---:|---|
| `trash_cup` | 14 | 36 | Trọng số lớp lớn (≈ 27) nhưng chỉ 14 mẫu; Recall Val = 0 |
| `animal_shells` | 233 | 1 | Mô hình "thích" dự đoán lớp này (mục 3.1) |
| `animal_crab` | 185 | 3 | E02 dự đoán 166 lần, đúng 0 |
| `trash_bottle` | 95 | 3 | Không đủ mẫu để đánh giá |

Chia theo video là bắt buộc để tránh rò rỉ (ADR-001 D4), nhưng với lớp chỉ có 4–8 video, mỗi tập chỉ nhận 1–2 video nên phân bố rất lệch.

### 3.3. Nhóm 6 lớp liên quan nhựa (support 192)

| Lớp thật | E01 nhầm nhiều nhất sang | E02 nhầm nhiều nhất sang |
|---|---|---|
| `trash_bag` (106) | `trash_clothing` 34, `rov` 22, `trash_snack_wrapper` 12 | `trash_clothing` 34, `trash_can` 20, `animal_eel` 15 |
| `trash_net` (36) | `trash_snack_wrapper` 14, `plant` 8 | `animal_crab` 21 |
| `trash_rope` (23) | `animal_crab` 8, `animal_starfish` 6, `plant` 4 | `animal_eel` 10 |
| `trash_tarp` (13) | **`trash_bag` 13** | **`trash_bag` 11** |
| `trash_snack_wrapper` (11) | `animal_shells` 8 | `trash_bag` 6 |
| `trash_bottle` (3) | `animal_shells` 3 | `animal_eel` 3 |

- Túi nilon bị nhầm với quần áo: cả hai là vật mềm, nhăn, màu nhạt; ở 128 × 128 sau khi phóng to bbox nhỏ thì khó phân biệt.
- Bạt (`trash_tarp`) luôn bị nhầm sang `trash_bag`: nhầm lẫn *trong cùng nhóm nhựa*, nếu gộp nhóm thì không phải lỗi nghiêm trọng.
- Dây thừng, lưới bị nhầm với sinh vật (cua, sao biển, lươn) và thực vật: hình dạng mảnh, phủ rong rêu, cùng nền đáy biển.

### 3.4. Quá trình huấn luyện

- Train accuracy tăng đều (E01: 0,16 → 0,43) nhưng val_loss dao động mạnh 2,98–3,68 và val_accuracy 0,02–0,22 giữa các epoch: dấu hiệu khoảng cách miền (domain gap) giữa video Train và Validation, không phải quá khớp đơn thuần.
- EarlyStopping (patience 6) dừng ở epoch 13 và 12 khi mô hình còn chưa khớp tốt tập Train; checkpoint tốt nhất ở epoch 7 và 6.
- Trọng số lớp tới 27 (`trash_cup`) khiến loss bị chi phối bởi vài chục ảnh hiếm.
- Augmentation (E02) làm bài toán khó hơn với mô hình nhỏ nên Train accuracy thấp hơn E01 ở cùng epoch, trong khi không giải quyết được khác biệt bối cảnh giữa các video.

## 4. Đề xuất cho Sprint 3

| # | Đề xuất | Lý do (mục) | Issue |
|---|---|---|---|
| 1 | Học chuyển giao MobileNetV2 ImageNet (E03): đặc trưng tổng quát ít phụ thuộc bối cảnh | 3.1, 3.4 | #17 |
| 2 | Chia lại dữ liệu theo video **có phân tầng** (StratifiedGroupKFold) hoặc chọn split để phân bố lớp Train/Val gần nhau; tạo `DATA_VERSION` mới | 3.2 | #16 |
| 3 | Trọng số lớp dạng căn bậc hai hoặc chặn ở 10; thử focal loss | 3.4 | #17 |
| 4 | EarlyStopping theo Macro F1 Validation, patience 10–12; giảm learning rate cho E02 | 3.4 | #17 |
| 5 | Tăng cường màu (hue, độ sáng, độ đục) mô phỏng nước khác nhau giữa video | 3.1 | #16, #17 |
| 6 | Báo cáo thêm kết quả gộp nhóm (nhựa / rác khác / sinh vật / ROV) | 3.3 | #18 |

## 5. Tệp liên quan

- Ma trận nhầm lẫn: `results/figures/E0x_val_confusion_matrix.png`, số đếm `results/metrics/E0x_val_confusion_counts.csv`.
- Đường cong học: `results/figures/E0x_learning_curves.png`.
- Dự đoán từng ảnh, ảnh bị nhầm, demo: Drive `MyDrive/ocean-plastic-ml/exports/s2_results_1105-1518/` (`E0x_val_predictions.csv`, `E0x_val_misclassified.png`, `demo_val_predictions.png`).
