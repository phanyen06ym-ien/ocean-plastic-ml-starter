# Ocean Plastic Waste Classification and Monitoring

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phanyen06ym-ien/ocean-plastic-ml-starter/blob/main/notebooks/00_pipeline_colab.ipynb)

**Phân loại và giám sát rác thải nhựa đại dương** — đồ án môn Học máy và Khai phá dữ liệu.

Dự án xây dựng mô hình mạng nơ-ron tích chập (CNN) phân loại **ảnh thực tế dưới nước** theo **loại đối tượng** của bộ dữ liệu **TrashCan 1.0 — Instance** (22 lớp; mô hình cuối E04 gộp còn 18 lớp). Trọng tâm phân tích là **nhóm 6 lớp rác liên quan đến nhựa** (túi, chai, vỏ bao bì, lưới, dây, bạt). Nhóm so sánh CNN tự thiết kế (E01–E02), học chuyển giao MobileNetV2 (E03) và mô hình cải tiến EfficientNet-B0 trên dữ liệu làm lại (E04); mọi cách chia dữ liệu đều theo video để tránh rò rỉ.

## Mục lục

1. [Tiến độ](#1-tiến-độ)
2. [Bài toán](#2-bài-toán)
3. [Dữ liệu](#3-dữ-liệu)
4. [Phương pháp](#4-phương-pháp)
5. [Thực nghiệm](#5-thực-nghiệm)
6. [Kết quả và phân tích](#6-kết-quả-và-phân-tích)
7. [Tái lập kết quả](#7-tái-lập-kết-quả)
8. [Cấu trúc repo](#8-cấu-trúc-repo)
9. [Nhóm thực hiện](#9-nhóm-thực-hiện)
10. [Lộ trình và hướng phát triển](#10-lộ-trình-và-hướng-phát-triển)
11. [Tài liệu](#11-tài-liệu)
12. [Trích dẫn](#12-trích-dẫn)

---

## 1. Tiến độ

Cập nhật 21/11/2026 · Sprint 3 — hoàn thiện bản nộp giữa kỳ.

| Giai đoạn                              | Nội dung                                                                               | Trạng thái   |
| -------------------------------------- | -------------------------------------------------------------------------------------- | ------------ |
| Xác định bài toán                      | Problem statement, phạm vi 22 lớp ([phân tích](docs/class_scope_analysis.md))   | ✅ Xong      |
| Dữ liệu `v1_crop128`                   | Audit, EDA, crop 128×128, chia theo video 70/15/15                                     | ✅ Xong      |
| CNN tự thiết kế                        | E01 (baseline + class weights), E02 (+ augmentation)                                   | ✅ Xong      |
| MobileNetV2                            | E03 (2 pha) + tinh chỉnh E03a–E03f, E03G (huấn luyện dài hơn, 24/11)                   | ✅ Xong      |
| Đánh giá Validation                    | Macro F1, nhóm nhựa, confusion matrix, phân tích lỗi E01–E03                           | ✅ Xong      |
| Dữ liệu `v2_crop224` + E04             | Gộp 22 → 18 lớp, crop vuông 224×224, chia lại theo video giữ 70/15/15 cho từng lớp; EfficientNet-B0, bù lớp hiếm, chọn checkpoint theo Macro F1 | ✅ Xong      |
| Chốt mô hình cuối                      | E04 (EfficientNet-B0, 18 lớp) — Macro F1 Validation cao nhất; quyết định ADR-002 (22/11/2026) | ✅ Xong      |
| Đánh giá Test                          | E04 bản đầu (21/11): Macro F1 0,327 · E04 hiện tại (22/11, lần Test thứ hai): Macro F1 0,329, nhóm nhựa 0,132 | ✅ Xong      |
| Phân tích lỗi nhóm nhựa                | Kích thước bbox theo lớp, kết quả theo video; đổi lấy mẫu sang 1/số mẫu ([`docs/dataset_report.md`](docs/dataset_report.md) mục 9) | ✅ Xong      |
| Demo                                   | CLI `main.py` chạy với checkpoint Keras (E01–E02) và PyTorch (E03–E04)                  | ✅ Xong      |
| Notebook kèm output                    | `01_trashcan_eda`, `04_evaluation` lưu kết quả lần chạy Colab 22/11/2026                 | ✅ Xong      |
| Release `v1.0-final`                   | Tag trên GitHub tại commit `ede61ab` (mô hình cuối và kết quả), 22/11/2026               | ✅ Xong      |
| Báo cáo, slide                         |                                                                                        | ⏳           |

## 2. Bài toán

| Mục                   | Nội dung                                                                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Loại bài toán         | Phân loại ảnh có giám sát, đa lớp (supervised multi-class image classification)                                                              |
| Đầu vào               | Ảnh RGB **một đối tượng**, cắt (crop) từ khung hình video dưới nước theo bounding box của TrashCan-Instance                                  |
| Đầu ra                | Một trong **22 lớp** (E01–E03) hoặc **18 lớp** sau khi gộp lớp mơ hồ / quá hiếm (E04): tên lớp, `class_index` và vector xác suất softmax |
| Chỉ số chính          | **Macro F1** trên toàn bộ lớp và **Macro F1 nhóm 6 lớp liên quan nhựa** (6 lớp này giữ nguyên ở mọi thí nghiệm)                              |
| Chỉ số phụ            | Accuracy, Precision, Recall, Weighted F1, F1 từng lớp, Confusion Matrix; số tham số, kích thước file và thời gian suy luận (hướng tới robot) |
| Ngoài phạm vi giữa kỳ | Object Detection, phục hồi ảnh dưới nước, video thời gian thực, điều khiển robot                                                             |

Macro F1 được chọn làm chỉ số chính vì dữ liệu mất cân bằng mạnh: Accuracy cao có thể chỉ do đoán đúng hai lớp lớn. Chi tiết: [`docs/problem_statement.md`](docs/problem_statement.md).

## 3. Dữ liệu

### 3.1. Nguồn

- **TrashCan 1.0** — 7.212 ảnh là khung hình từ video quay dưới nước, annotation dạng COCO (bounding box và segmentation). Nhóm dùng bản nhãn **Instance** (theo loại đối tượng), không dùng bản Material.
- Nguồn: [UMN Data Repository](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7) · Bài báo: Hong, Fulton & Sattar, [arXiv:2007.08097](https://arxiv.org/abs/2007.08097).
- Dữ liệu **không commit lên GitHub**; lưu trên Google Drive của nhóm. Tải tự động bằng `download_trashcan` trong `src/preprocessing.py`.

### 3.2. Thống kê (phiên bản `v1_crop128`)

| Chỉ số                            |                                       Giá trị |
| --------------------------------- | --------------------------------------------: |
| Annotation (train + val gốc)      |                                        12.128 |
| Ảnh nguồn có annotation           |                                         7.083 |
| Số video                          |                                           310 |
| Crop hợp lệ / loại vì bbox < 8 px |                                   12.074 / 54 |
| Crop Train / Validation / Test    |                         8.283 / 1.782 / 2.009 |
| Crop Train / Validation / Test (`v2_crop224`, 18 lớp) | 7.815 / 1.704 / 1.751 |
| Kiểm tra rò rỉ dữ liệu            | Đạt: không video, không ảnh nào nằm ở hai tập |

Dữ liệu **mất cân bằng mạnh**: `rov` (27%) và `trash_unknown_instance` (23%) chiếm khoảng một nửa số annotation, trong khi `trash_cup` chỉ có 59 mẫu.

| Nhóm 6 lớp liên quan nhựa | Train | Val | Test | Tổng |
| ------------------------- | ----: | --: | ---: | ---: |
| `trash_bag`               |   605 | 106 |  196 |  907 |
| `trash_net`               |    79 |  36 |   12 |  127 |
| `trash_bottle`            |    95 |   3 |   28 |  126 |
| `trash_tarp`              |   103 |  13 |    5 |  121 |
| `trash_rope`              |    87 |  23 |    7 |  117 |
| `trash_snack_wrapper`     |    41 |  11 |   32 |   84 |

Tên nhãn là loại đối tượng, _không_ phải bằng chứng chất liệu (ví dụ chai có thể là nhựa hoặc thủy tinh). Bảng đủ 22 lớp: [`docs/dataset_report.md`](docs/dataset_report.md) · Lý do chọn phạm vi: [`docs/class_scope_analysis.md`](docs/class_scope_analysis.md).

### 3.3. Tiền xử lý (`src/preprocessing.py`)

| Bước | `v1_crop128` (E01–E03) | `v2_crop224` (E04) |
|---|---|---|
| Lớp | 22 lớp TrashCan-Instance | **18 lớp**: `trash_unknown_instance`, `trash_cup`, `trash_clothing`, `trash_wreckage` → `trash_other`; `animal_shells`, `animal_etc` → `animal_other`; giữ nguyên 6 lớp nhựa |
| Crop | Theo bbox COCO, lề 10% mỗi phía, resize thẳng (đối tượng có thể méo) | **Cửa sổ vuông** quanh tâm bbox, lề 15%, giữ đúng tỉ lệ đối tượng |
| Bbox tối thiểu | 8 px | 16 px |
| Kích thước | 128 × 128 RGB | **224 × 224** RGB (kích thước ảnh tiền huấn luyện ImageNet) |
| Chia tập | Gộp train + val gốc, **chia lại theo video** 70 / 15 / 15, seed 42; thử ngẫu nhiên nhiều hoán vị | Vẫn theo video, seed 42, nhưng **tìm phân công video giữ tỉ lệ 70 / 15 / 15 cho từng lớp** (lớp hiếm cũng có mẫu ở cả 3 tập) |
| Augmentation (chỉ Train) | Không (E02 có, trên CNN) | Cắt ngẫu nhiên 80–100%, lật ngang, xoay ±10°, đổi độ sáng / tương phản / màu |
| Mất cân bằng | Class weights `balanced` | Lấy mẫu theo 1/số mẫu (bản đầu: √(1/số mẫu)) + focal loss (γ = 2) |
| Chuẩn hóa | CNN tự thiết kế: pixel / 255 · MobileNetV2: mean/std ImageNet | mean/std ImageNet |

Mọi annotation của một video luôn nằm trong cùng một tập để tránh rò rỉ dữ liệu.

Thông số dùng chung được khóa trong [`configs/data_contract.yaml`](configs/data_contract.yaml) (v1) và [`configs/data_contract_v2.yaml`](configs/data_contract_v2.yaml) (v2). Lý do chọn thông số v2: [`docs/dataset_report.md`](docs/dataset_report.md).

### 3.4. Hạn chế đã biết

- `v1_crop128`: chia theo video làm lệch số mẫu giữa các tập, vài lớp chỉ có 1–4 mẫu ở Validation hoặc Test (`animal_shells` 1, `trash_bottle` 3 ở Val; `animal_starfish` 4 ở Test). `v2_crop224` khắc phục: mọi lớp có ít nhất 10 mẫu ở Val và Test, tỉ lệ ảnh 70,0 / 15,1 / 15,0%.
- `v2_crop224` bỏ 858 annotation (7,1%) có bbox cạnh < 16 px; các vật thể rất nhỏ này không được mô hình học.
- Nhiều bbox nhỏ (trung vị khoảng 90 × 66 px ở Train); một số bbox có toạ độ vượt ngoài ảnh (được giới hạn lại khi crop). Các lớp nhựa không nhỏ hơn các lớp khác (trung vị cạnh ngắn 45–127 px).
- Mẫu nhựa tập trung trong rất ít video: một video (`vid_000255`) chiếm 41% mẫu nhựa của Test, nên số đo nhóm nhựa không ổn định.
- `trash_unknown_instance` là lớp mơ hồ, dễ nhầm với các lớp rác khác.

## 4. Phương pháp

### 4.1. Quy trình

```
TrashCan-Instance (COCO JSON + ảnh)
  → Audit & EDA                     notebooks/01 · src/eda.py
  → Crop đối tượng + chia theo video notebooks/02 · src/preprocessing.py
  → Huấn luyện                      notebooks/03 · src/train.py (Keras cho E01–E02, PyTorch cho E03–E04)
  → Đánh giá Validation → chọn mô hình → đánh giá Test 1 lần
                                    notebooks/04 · src/evaluate.py
  → Dự đoán ảnh mới (demo)          src/predict.py · main.py
```

### 4.2. Mô hình

**CNN tự thiết kế (E01–E02, TensorFlow/Keras)**

```
Input 128×128×3 → Rescaling(1/255)
→ [Conv3×3 → BatchNorm → ReLU → MaxPool] × 3   (32, 64, 128 filters)
→ Conv3×3 (256) → BatchNorm → ReLU
→ GlobalAveragePooling → Dropout(0.4) → Dense(128, ReLU) → Dropout(0.4) → Dense(22, softmax)
```

Adam (lr 1e-3), batch 64, tối đa 40 epoch, EarlyStopping (patience 6), ReduceLROnPlateau (patience 3).

**Học chuyển giao MobileNetV2 (E03, PyTorch)**

- Backbone `torchvision` MobileNetV2 tiền huấn luyện ImageNet; thay bộ phân loại bằng `Dropout → Linear(1280, 22)`. Khoảng 2,25 triệu tham số, file 9,2 MB, suy luận khoảng 30 ms/ảnh trên CPU.
- **Pha 1:** đóng băng backbone, chỉ huấn luyện bộ phân loại (lr 1e-3, tối đa 15 epoch).
- **Pha 2:** mở khoá các block cuối (`features[14:]`), học với lr nhỏ (1e-5 cho backbone, 1e-4 cho bộ phân loại), giữ BatchNorm ở chế độ eval.
- Chọn checkpoint có `val_loss` thấp nhất giữa hai pha.

**Mô hình cải tiến E04 — EfficientNet-B0 (PyTorch)**

- Backbone `torchvision` EfficientNet-B0 tiền huấn luyện ImageNet (khoảng 4 triệu tham số), `Dropout(0.3) → Linear(1280, 18)`; vẫn đủ nhẹ cho robot.
- Cùng quy trình 2 pha như E03; pha 2 mở khoá `features[4:]` (5/9 khối) với lr 5e-5 (backbone) / 5e-4 (đầu phân loại).
- Bù lớp hiếm: `WeightedRandomSampler` lấy mẫu theo 1/số mẫu của lớp (mọi lớp xuất hiện đều nhau; bản đầu 21/11 dùng √(1/số mẫu)) + focal loss (γ = 2), không dùng class weights.
- Dừng sớm và chọn checkpoint theo **Macro F1 Validation** (chỉ số chính) thay vì `val_loss`.

### 4.3. Nguyên tắc thực nghiệm

- Mỗi thí nghiệm chỉ đổi **một yếu tố** so với thí nghiệm gốc; cấu hình lưu trong `configs/experiments/`.
- Cùng seed 42, cùng cách chia dữ liệu khi so sánh.
- Chọn mô hình **chỉ dựa trên Validation**; tập Test chỉ chạy một lần cho mô hình cuối.

## 5. Thực nghiệm

| EXP_ID | Mô hình         | Yếu tố thay đổi                                    | Dữ liệu       | Cấu hình                        |
| ------ | --------------- | -------------------------------------------------- | ------------- | ------------------------------- |
| E01    | CNN tự thiết kế | Baseline + class weights                           | `v1_crop128`  | `E01_baseline.yaml`             |
| E02    | CNN tự thiết kế | + Augmentation (lật ngang, xoay, zoom, tương phản) | `v1_crop128`  | `E02_augment.yaml`              |
| E03    | MobileNetV2     | Mô hình: học chuyển giao                           | `v1_crop128`  | `E03_transfer_mobilenetv2.yaml` |
| E03a–f, E03G | MobileNetV2 | Mỗi lần đổi một siêu tham số (dropout, lr, seed…); E03G: tối đa 30 + 40 epoch, patience 12 | `v1_crop128`  | E03 + `--set` (xem `results/metrics/e03_candidates.csv`) |
| E04    | EfficientNet-B0 | Dữ liệu v2 (18 lớp) + backbone + bù lớp hiếm + chọn checkpoint theo Macro F1 + augmentation | `v2_crop224` | `E04_efficientnet_b0_data_v2.yaml` |

Cấu hình từng thí nghiệm: `configs/experiments/` · Số liệu từng lần chạy: `results/metrics/<EXP_ID>_val.json`, `<EXP_ID>_history.csv`.

## 6. Kết quả và phân tích

Tập Validation: 1.782 crop, 22 lớp cho E01–E03 (`v1_crop128`); 1.704 crop, 18 lớp cho E04 (`v2_crop224`). Nguồn: `results/metrics/comparison_val.csv`, `results/metrics/e03_candidates.csv`, `results/metrics/E04_val.json`.

| EXP_ID                 |         Macro F1 |  Accuracy | Weighted F1 | Macro F1 nhóm nhựa | Macro F1 (Test) |
| ---------------------- | ---------------: | --------: | ----------: | -----------------: | --------------: |
| E01                    |            0,076 |     0,219 |       0,264 |              0,000 |               — |
| E02                    |            0,065 |     0,161 |       0,167 |              0,017 |               — |
| E03                    |            0,203 |     0,405 |       0,418 |              0,123 |               — |
| E03f (tốt nhất E03a–f) |        **0,212** | **0,412** |   **0,420** |              0,117 |               — |
| E04 bản đầu ¹ (lấy mẫu √(1/n)) | 0,339 | **0,579** | 0,546 | 0,153 | 0,327 |
| **E04** ¹ (EfficientNet-B0, 18 lớp, lấy mẫu 1/n) | **0,348** | 0,569 | **0,554** | **0,164** | **0,329** |

Mức ngẫu nhiên của Macro F1: ≈ 0,045 với 22 lớp, ≈ 0,056 với 18 lớp.

**Kết quả trên tập Test (1.751 crop từ các video chưa dùng khi huấn luyện và chọn mô hình):**

| Chỉ số | E04 bản đầu: Val | E04 bản đầu: **Test** | E04: Val | E04: **Test** |
|---|---:|---:|---:|---:|
| Macro F1 (18 lớp) | 0,339 | 0,327 | 0,348 | **0,329** |
| Accuracy | 0,579 | 0,535 | 0,569 | **0,507** |
| Weighted F1 | 0,546 | 0,497 | 0,554 | **0,486** |
| Macro F1 nhóm nhựa | 0,153 | 0,119 | 0,164 | **0,132** |
| Recall nhóm nhựa | 0,150 | 0,099 | 0,237 | **0,151** |

> **Lưu ý về tập Test.** Test đã được chạy **hai lần**: lần 1 (21/11) cho E04 bản đầu; lần 2 (22/11) cho E04 hiện tại. E04 hiện tại được tạo ra sau khi phân tích lỗi của lần 1 trên Test, nên kết quả Test lần 2 không còn hoàn toàn độc lập. Mô hình vẫn được chọn bằng Validation; cả hai kết quả được giữ (`results/metrics/E04v1_test.json`, `E04_test.json`).

Trên Test, đổi cách lấy mẫu gần như không đổi Macro F1 (0,327 → 0,329) nhưng đổi Accuracy lấy recall nhóm nhựa: Accuracy giảm 2,8 điểm, recall nhựa tăng từ 0,099 lên 0,151, số mẫu nhựa bị đoán thành `trash_other` giảm từ 129 xuống 71 (trên 217). Tăng: `trash_pipe` (F1 0,24 → 0,51), `trash_bag` (0,23 → 0,34), `trash_rope` (0,09 → 0,20). Giảm: `trash_net` (0,32 → 0,17), `animal_other` (0,29 → 0,06), `trash_other` (0,41 → 0,34). `trash_bottle` và `trash_snack_wrapper` vẫn F1 = 0; cả 10 mẫu `trash_snack_wrapper` bị đoán thành `trash_bag`. Nguồn: `results/metrics/E04_test.json`, `E04v1_test.json`, hình `results/figures/E04_test_confusion_matrix.png`.

¹ E04 khác E01–E03 ở số lớp (18 so với 22) và ở tập Validation/Test (cách chia mới), nên so sánh chỉ mang tính tham khảo. Checkpoint E04 chọn theo Macro F1 Validation (pha 2, epoch 4; bản đầu: pha 2, epoch 11); vì Validation vừa dùng để chọn checkpoint, con số này hơi lạc quan — xem kết quả Test ở trên.

**Nhận xét**

1. **Học chuyển giao hiệu quả hơn rõ rệt.** MobileNetV2 tăng Macro F1 khoảng 2,7 lần so với CNN tự thiết kế (0,203 so với 0,076) trên cùng dữ liệu.
2. **Augmentation không giúp CNN tự thiết kế** (E02 thấp hơn E01): mô hình nhỏ huấn luyện từ đầu chưa học được đặc trưng ổn định.
3. **Kết quả theo lớp rất chênh lệch** (E03f): tốt ở `rov` (F1 0,76), `animal_eel` (0,67), `animal_starfish` (0,52); bằng 0 ở nhiều lớp ít mẫu như `trash_bottle`, `trash_tarp`, `trash_rope`, `trash_snack_wrapper`, `trash_cup`.
4. **Mô hình học thuộc nhanh:** E03 đạt `val_loss` tốt nhất ngay ở epoch 2 trong khi loss trên Train vẫn giảm mạnh — mô hình học bối cảnh của từng video hơn là hình dạng vật thể. E03G (Bách, 24/11) xác nhận điều này: kéo dài huấn luyện (patience 12) vẫn chọn đúng checkpoint pha 1 epoch 2, Macro F1 0,2034 như E03; loss Validation pha 2 tăng từ 2,30 lên 2,53. Macro F1 theo epoch đạt tối đa 0,236 (pha 2, epoch 9) nhưng không được chọn vì quy tắc theo `val_loss` — lý do E04 chọn checkpoint theo Macro F1.
5. **E04 cải thiện rõ nhất.** Bản đầu của E04 kết hợp dữ liệu 224×224 chia lại theo từng lớp, gộp lớp mơ hồ / quá hiếm, EfficientNet-B0, bù lớp hiếm và chọn checkpoint theo Macro F1: Macro F1 0,339, Accuracy 57,9% (E03: 0,203 và 40,5%). Một lần thử trước đó chỉ làm lại dữ liệu và giữ MobileNetV2 (22 lớp) chỉ đạt Macro F1 0,219, Accuracy 42,7%: phần lớn cải thiện đến từ backbone mạnh hơn và việc gộp lớp. Tốt nhất ở `rov` (F1 0,91), `animal_crab` (0,69), `animal_eel` (0,59), `animal_fish` (0,58).
6. **Phân tích lỗi nhóm nhựa** ([`docs/dataset_report.md`](docs/dataset_report.md) mục 9). Vật thể nhỏ **không** phải nguyên nhân chính: các lớp nhựa có bbox trung vị 45–127 px, trong khi `animal_starfish` nhỏ nhất (72% box < 32 px) lại đạt F1 cao nhất. Nguyên nhân rõ nhất là ít mẫu (43–88 crop Train cho 5/6 lớp nhựa, `trash_bottle` và `trash_snack_wrapper` chỉ có trong 7 video) và lớp gộp `trash_other` hút nhầm (bản đầu đoán 59% mẫu nhựa Test thành `trash_other`). Mẫu nhựa còn tập trung trong vài video (một video = 41% mẫu nhựa Test), nên số đo nhóm nhựa dao động mạnh.
7. **Lấy mẫu theo 1/số mẫu** (E04 hiện tại) giảm số mẫu nhựa bị hút vào `trash_other` (Test: 129 → 71/217) và tăng recall nhựa, nhưng precision và Accuracy giảm; Macro F1 gần như không đổi. `trash_bottle`, `trash_snack_wrapper` vẫn F1 = 0. Loss Train về gần 0 trong khi loss Validation đứng quanh 1,55–1,65: mô hình vẫn học thuộc bối cảnh video. Hướng tiếp theo: bỏ `trash_unknown_instance` khỏi huấn luyện hoặc phân loại hai tầng (nhựa / không nhựa → loại nhựa), thêm dữ liệu nhựa từ video khác.

**Nguyên nhân chính nằm ở dữ liệu** và là cơ sở cho E04: ảnh 128×128 nhỏ hơn kích thước MobileNetV2 được tiền huấn luyện (224×224); phân bố lớp giữa các tập lệch mạnh do chia theo video; nhiều bbox rất nhỏ; hai lớp lớn chiếm một nửa dữ liệu.

Kết quả theo lớp với support nhỏ (dưới khoảng 10 mẫu) không đủ để kết luận. Số liệu theo lớp: `results/metrics/<EXP_ID>_val.json` · Hình: `results/figures/` · Cách đánh giá: [`docs/evaluation_plan.md`](docs/evaluation_plan.md).

## 7. Tái lập kết quả

**Google Colab (khuyến nghị, cần GPU):** mở `notebooks/00_pipeline_colab.ipynb` bằng nút _Open in Colab_ ở đầu trang và chạy từ trên xuống. Notebook mount Google Drive, tải dữ liệu, tiền xử lý, huấn luyện và đánh giá. Hướng dẫn chi tiết: [`docs/colab_setup.md`](docs/colab_setup.md).

**Máy cá nhân:**

```bash
git clone https://github.com/phanyen06ym-ien/ocean-plastic-ml-starter.git
cd ocean-plastic-ml-starter
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                            # kiểm thử

# Tiền xử lý
python -m src.preprocessing --raw-root <thư mục TrashCan> --out-dir data/processed/v1_crop128

# Huấn luyện MobileNetV2 (E03)
python -m src.train --config configs/experiments/E03_transfer_mobilenetv2.yaml --processed-dir data/processed/v1_crop128

# Dữ liệu v2 + huấn luyện E04
python -m src.preprocessing --raw-root <thư mục TrashCan> --out-dir data/processed/v2_crop224 --contract configs/data_contract_v2.yaml --mapping-out configs/class_mapping_v2.json
python -m src.train --config configs/experiments/E04_efficientnet_b0_data_v2.yaml --processed-dir data/processed/v2_crop224

# Dự đoán (checkpoint Keras)
python main.py --image path/to/crop.jpg --model runs/E01_<thời gian>/best.keras
```

Thư mục Drive của nhóm:

```
MyDrive/ocean-plastic-ml/
├── README.md                        # Mục lục thư mục Drive
├── data/
│   ├── raw/dataset.zip              # TrashCan 1.0 gốc (553 MB)
│   └── processed/                   # v1_crop128.zip (E01–E03), v2_crop224.zip (E04)
├── runs/                            # Mỗi run: best.pt / best.keras, config.json, class_mapping.json, history.csv, predictions
│   ├── E01_20261105-1513/ · E02_20261105-1515/
│   ├── E04_20261121-1102/           # E04 bản đầu (√(1/n))
│   ├── E04_20261122-0853/           # E04 — mô hình cuối (tag v1.0-final)
│   └── data_audit_v1_crop128/       # Kết quả audit của PR #35
├── exports/                         # Nơi notebook 00 ghi kết quả tạm; chép vào repo rồi xoá
├── audits/ · reports/               # Audit dữ liệu v1 và gói báo cáo giữa kỳ (13/11)
```
Checkpoint E03 (`runs/E03_20261108-1459/`) chỉ có trên máy của Bách (xem [`models/README.md`](models/README.md)).

## 8. Cấu trúc repo

```
ocean-plastic-ml-starter/
├── .github/                         # Mẫu issue, mẫu PR, CODEOWNERS
├── configs/
│   ├── data_contract.yaml           # Thông số dữ liệu v1 (E01–E03)
│   ├── data_contract_v2.yaml        # Thông số dữ liệu v2 (E04, 18 lớp)
│   ├── class_mapping_v2.json        # 18 lớp sau khi gộp (E04)
│   ├── class_mapping.json           # category_id → class_name → class_index
│   └── experiments/                 # E01, E02, E03, E04 (.yaml)
├── data/                            # Chỉ có README.md; dữ liệu lưu trên Drive (xem data/README.md)
├── docs/
│   ├── problem_statement.md         # Bài toán, phạm vi, chỉ số
│   ├── dataset_report.md            # Thống kê dữ liệu, cách chia tập
│   ├── evaluation_plan.md           # Cách đánh giá
│   ├── colab_setup.md               # Hướng dẫn chạy trên Colab
│   ├── class_scope_analysis.md      # Lý do chọn phạm vi lớp
│   └── visual_inspection_protocol.md # Kiểm tra ảnh crop và nhãn
├── notebooks/
│   ├── 00_pipeline_colab.ipynb      # Chạy toàn bộ quy trình
│   ├── 01_trashcan_eda.ipynb · 02_preprocessing.ipynb
│   └── 03_cnn_training.ipynb · 04_evaluation.ipynb
├── src/                             # Code dùng chung, notebook chỉ gọi lại
│   ├── eda.py                       # Thống kê và vẽ biểu đồ dữ liệu (notebook 01)
│   ├── preprocessing.py             # Đọc data contract, tải dữ liệu, crop, chia tập (notebook 02)
│   ├── train.py                     # Huấn luyện E01–E02 (Keras), E03 MobileNetV2 và E04 EfficientNet-B0 (PyTorch) (notebook 03)
│   ├── evaluate.py                  # Chỉ số, confusion matrix, đánh giá checkpoint (notebook 04)
│   └── predict.py                   # Dự đoán ảnh mới (main.py)
├── tests/                           # pytest cho từng module
├── models/                          # Chỉ README: danh sách checkpoint E01–E04 trên Drive
├── results/
│   ├── figures/                     # Hình cho báo cáo
│   ├── metrics/                     # <EXP_ID>_<split>.json/.csv; metrics_summary.csv so sánh E01–E04
│   └── predictions/                 # Dự đoán từng ảnh (path, y_true, y_pred): E01–E02 Val, E04/E04v1 Val và Test
├── scripts/                         # collect_runs.py, summarize_runs.py, github/
├── main.py                          # CLI demo
├── requirements.txt
├── CONTRIBUTING.md                  # Quy ước nhánh, commit, PR
└── README.md
```

## 9. Nhóm thực hiện

| Thành viên      | GitHub                                                                   | Vai trò                                            |
| --------------- | ------------------------------------------------------------------------ | -------------------------------------------------- |
| Phan Hải Yến    | [@phanyen06ym-ien](https://github.com/phanyen06ym-ien)                   | Trưởng nhóm · Kiến trúc · Quản lý dự án · Reviewer |
| Hùng            | [@tranngochung126hqv-stack](https://github.com/tranngochung126hqv-stack) | Dữ liệu: Audit, EDA, tiền xử lý                    |
| Hoàng Bách      | [@phannguyenhoangbach2](https://github.com/phannguyenhoangbach2)         | Mô hình CNN, huấn luyện                            |
| Nguyễn Hoàng Ân | _(cập nhật)_                                                             | Đánh giá, so sánh thực nghiệm, phân tích lỗi       |
| Ngọc Nguyên     | [@hiang0910](https://github.com/hiang0910)                               | Inference, kiểm thử, demo                          |

Quy ước làm việc: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 10. Lộ trình và hướng phát triển

| Sprint                           | Thời gian          | Mục tiêu                                                  |
| -------------------------------- | ------------------ | --------------------------------------------------------- |
| Sprint 1 — Foundation            | 23/10 – 03/11/2026 | Bài toán, kiến trúc, phạm vi lớp, khung code                  |
| Sprint 2 — Data & Baseline       | 04 – 05/11/2026    | Dữ liệu `v1_crop128`, E01, E02, đánh giá Validation       |
| Sprint 3 — Improvement & Midterm | 05/11/2026 – nay   | E03, dữ liệu mới + E04, chốt mô hình, Test, demo, báo cáo |

**Sau giữa kỳ:** phát hiện và định vị rác (Object Detection), cải thiện ảnh dưới nước, mô hình nhẹ cho robot thu gom rác.

## 11. Tài liệu

| Tài liệu                                                                                        | Nội dung                       |
| ----------------------------------------------------------------------------------------------- | ------------------------------ |
| [`problem_statement.md`](docs/problem_statement.md)                                             | Bài toán, phạm vi, chỉ số      |
| [`dataset_report.md`](docs/dataset_report.md)                                                   | Thống kê dữ liệu đầy đủ        |
| [`evaluation_plan.md`](docs/evaluation_plan.md)                                                 | Cách đánh giá, chỉ số          |
| [`class_scope_analysis.md`](docs/class_scope_analysis.md)                                       | Lý do chọn phạm vi 22 lớp      |
| [`visual_inspection_protocol.md`](docs/visual_inspection_protocol.md)                           | Kiểm tra ảnh crop và nhãn      |
| [`colab_setup.md`](docs/colab_setup.md)                                                         | Hướng dẫn chạy trên Colab      |

## 12. Trích dẫn

Bộ dữ liệu thuộc quyền của nhóm tác giả TrashCan; khi sử dụng vui lòng trích dẫn:

```bibtex
@article{hong2020trashcan,
  title   = {TrashCan: A Semantically-Segmented Dataset towards Visual Detection of Marine Debris},
  author  = {Hong, Jungseok and Fulton, Michael and Sattar, Junaed},
  journal = {arXiv preprint arXiv:2007.08097},
  year    = {2020}
}
```
