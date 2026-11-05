# Kết quả Sprint 2 (từ notebooks/00_pipeline_colab.ipynb)

- Ngày chạy: 05/11/2026 15:18 (giờ Colab, UTC) · Colab GPU T4 · TensorFlow 2.21.0 · commit code `7520c2b`
- `DATA_VERSION`: `v1_crop128` · Drive: `MyDrive/ocean-plastic-ml/data/processed/v1_crop128.zip`
- Run E01: `MyDrive/ocean-plastic-ml/runs/E01_20261105-1513` · Run E02: `MyDrive/ocean-plastic-ml/runs/E02_20261105-1515`
- Bản xuất đầy đủ (dự đoán từng ảnh, hình ảnh mẫu, demo): `MyDrive/ocean-plastic-ml/exports/s2_results_1105-1518/`
- `pytest -q` trên Colab: 40/40 đạt (gồm test TensorFlow)

## Dữ liệu

- Annotation: 12.128 · ảnh nguồn: 7.083 · video: 310
- Crop hợp lệ 12.074, loại 54 (bbox < 8 px)
- Crop theo tập: Train 8.283 · Val 1.782 · Test 2.009 · tỉ lệ ảnh nguồn 69,9 / 14,9 / 15,2 %
- Không rò rỉ video/ảnh; đủ 22 lớp ở Val và Test

| Lớp | Train | Val | Test | Tổng |
|---|---:|---:|---:|---:|
| rov | 2130 | 460 | 707 | 3297 |
| plant | 228 | 73 | 206 | 507 |
| animal_fish | 604 | 81 | 76 | 761 |
| animal_starfish | 360 | 33 | 4 | 397 |
| animal_shells | 233 | 1 | 14 | 248 |
| animal_crab | 185 | 3 | 121 | 309 |
| animal_eel | 194 | 103 | 37 | 334 |
| animal_etc | 162 | 24 | 49 | 235 |
| trash_clothing | 59 | 21 | 2 | 82 |
| trash_pipe | 79 | 21 | 56 | 156 |
| trash_bottle | 95 | 3 | 28 | 126 |
| trash_bag | 605 | 106 | 196 | 907 |
| trash_snack_wrapper | 41 | 11 | 32 | 84 |
| trash_can | 385 | 59 | 15 | 459 |
| trash_cup | 14 | 36 | 9 | 59 |
| trash_container | 299 | 152 | 57 | 508 |
| trash_unknown_instance | 1970 | 479 | 290 | 2739 |
| trash_branch | 249 | 13 | 74 | 336 |
| trash_wreckage | 122 | 31 | 12 | 165 |
| trash_tarp | 103 | 13 | 5 | 121 |
| trash_rope | 87 | 23 | 7 | 117 |
| trash_net | 79 | 36 | 12 | 127 |

## So sánh E01 và E02 (Validation)

| exp_id | n | Accuracy | Macro F1 | Weighted F1 | Macro F1 nhóm nhựa |
|---|---:|---:|---:|---:|---:|
| E01 | 1782 | 0,2189 | **0,0757** | 0,2641 | 0,0000 |
| E02 | 1782 | 0,1605 | 0,0648 | 0,1671 | 0,0165 |

Chỉ số chính: Macro F1 trên 22 lớp; nhóm nhựa = Macro F1 trên 6 lớp liên quan nhựa (ADR-001). Tập Test chưa được sử dụng. Phân tích: `docs/evaluation_report.md`.

## Demo (S2-DEMO)

Checkpoint E01 (Macro F1 Validation cao nhất). CLI `python main.py --image <crop> --model runs/E01_20261105-1513/best.keras` chạy được và trả top-3; ví dụ một ảnh `rov` của Validation: `animal_shells` 69,6% · `trash_can` 14,6% · `rov` 12,7% (dự đoán sai, đúng với phân tích mục 3.1 của báo cáo đánh giá). Hình 8 ảnh Validation (1/8 đúng): `results/figures/demo_val_predictions.png`; ảnh mẫu mỗi lớp: `results/figures/eda_sample_crops.png` (bản rút gọn 16 lớp; bản đủ 22 lớp trên Drive `exports/`).
