# Báo cáo dữ liệu — TrashCan 1.0 Instance (S2-DATA)

> **Trạng thái: KHUNG.** Mọi ô `⏳` là **điền sau khi chạy notebook 00** (`notebooks/00_pipeline_colab.ipynb`, mục 3–4) hoặc notebook 01–02 trên dữ liệu thật. Không điền số từ dữ liệu giả lập hay ước lượng. Nguồn của từng số được ghi ở cột hoặc dòng "Nguồn".

| Mục | Giá trị |
|---|---|
| Phụ trách | Hùng (dự phòng: Yến) |
| `DATA_VERSION` | `v1_crop128` |
| Ngày chạy | ⏳ |
| Người chạy | ⏳ |
| Commit code | ⏳ (`git rev-parse --short HEAD`) |
| Bản xử lý trên Drive | ⏳ (`MyDrive/ocean-plastic-ml/data/processed/v1_crop128.zip`) |
| Căn cứ | `docs/decisions/ADR-001-class-scope.md`, `configs/data_contract.yaml` v1 |

## 1. Nguồn dữ liệu

- TrashCan 1.0, bản nhãn **Instance** (theo loại đối tượng). Annotation COCO: `instances_train_trashcan.json`, `instances_val_trashcan.json`.
- Nguồn tải: [UMN Data Repository](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7). Bài báo: Hong, Fulton & Sattar, arXiv:2007.08097.
- File gốc đã tải: ⏳ (tên file `.zip`, dung lượng)

## 2. Quy trình xử lý (theo ADR-001)

| Bước | Tham số | Hàm |
|---|---|---|
| Class mapping | 22 lớp, `class_index` 0–21 theo `category_id` tăng dần | `build_class_mapping` |
| Gộp nguồn | train + val gốc | `build_object_table` |
| Chia tập | theo video (`vid_(\d+)`), 70/15/15 theo số ảnh nguồn, seed 42 | `grouped_split`, `check_no_leakage` |
| Crop | bbox COCO, lề 10% mỗi phía, bỏ box có cạnh < 8 px | `crop_box`, `export_crops` |
| Resize | 128×128 RGB, resize thẳng, lưu `.jpg`; chuẩn hoá /255 nằm trong mô hình | `export_crops` |

Lệnh tái lập:
```bash
python -m src.preprocessing --raw-root <thư-mục-trashcan> --out-dir data/processed/v1_crop128 --mapping-out configs/class_mapping.json
```

## 3. Tổng quan

Nguồn: `preprocessing_summary.json`.

| Chỉ số | Giá trị | Ghi chú |
|---|---:|---|
| Số annotation (train + val gốc) | ⏳ | |
| Số ảnh nguồn | ⏳ | |
| Số video | ⏳ | Dataset Audit ghi nhận 312; chênh lệch nhiều → kiểm tra `video_id_regex` |
| Annotation không tìm thấy file ảnh | ⏳ | |
| Crop hợp lệ (`ok`) | ⏳ | |
| Crop bị loại (`too_small`) | ⏳ | cạnh bbox < 8 px |
| Crop bị loại (lý do khác) | ⏳ | ghi rõ trạng thái |
| Tỉ lệ mất cân bằng (lớn nhất / nhỏ nhất) | ⏳ | `eda.imbalance_ratio` |

## 4. Phân bố annotation gốc theo lớp

Nguồn: `results/metrics/dataset_class_distribution.csv` (`eda.class_distribution`, theo `source_split`). Hình: `results/figures/eda_class_distribution.png`.

`class_index` và `category_id` lấy từ `configs/class_mapping.json`. Lớp **in đậm** thuộc nhóm 6 lớp liên quan nhựa (ADR-001 D2).

| class_index | category_id | Lớp | Train gốc | Val gốc | Tổng | Số ảnh | Số video | % |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ⏳ | ⏳ | rov | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | plant | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | animal_fish | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | animal_starfish | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | animal_shells | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | animal_crab | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | animal_eel | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | animal_etc | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | trash_clothing | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | trash_pipe | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | **trash_bottle** | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | **trash_bag** | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | **trash_snack_wrapper** | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | trash_can | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | trash_cup | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | trash_container | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | trash_unknown_instance | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | trash_branch | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | trash_wreckage | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | **trash_tarp** | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | **trash_rope** | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ⏳ | ⏳ | **trash_net** | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| | | **Tổng** | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | 100 |

Khi điền, sắp lại các dòng theo `class_index`.

## 5. Kích thước bbox

Nguồn: `eda.bbox_stats` (cạnh ngắn của bbox, pixel, trên ảnh gốc). Chỉ cần ghi các lớp đáng chú ý; bảng đầy đủ xem trong notebook 01.

| Lớp | Số box | Trung vị cạnh ngắn (px) | % box < 32 px | Nhận xét |
|---|---:|---:|---:|---|
| ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

## 6. Chia Train / Validation / Test theo video

Nguồn: `preprocessing_summary.json` → `split`, `crops_per_split`.

| Tập | Số video | Số ảnh nguồn | Tỉ lệ ảnh nguồn | Số crop hợp lệ |
|---|---:|---:|---:|---:|
| Train | ⏳ | ⏳ | ⏳ (mục tiêu 0.70) | ⏳ |
| Validation | ⏳ | ⏳ | ⏳ (mục tiêu 0.15) | ⏳ |
| Test | ⏳ | ⏳ | ⏳ (mục tiêu 0.15) | ⏳ |

- Kiểm tra rò rỉ (`check_no_leakage`: không video/ảnh nào nằm ở hai tập): ⏳ (Đạt / Không đạt)

## 7. Số crop theo lớp và tập

Nguồn: `class_split_counts.csv` → `results/metrics/dataset_split_counts.csv`. Hình: `results/figures/eda_split_counts.png`.

| Lớp | Train | Val | Test | Tổng |
|---|---:|---:|---:|---:|
| ⏳ (22 dòng, theo `class_index`) | ⏳ | ⏳ | ⏳ | ⏳ |

**Lớp không có mẫu** (nguồn: `classes_missing_in`):

| Tập | Lớp thiếu |
|---|---|
| Train | ⏳ |
| Validation | ⏳ |
| Test | ⏳ |

Cách xử lý (ADR-001 Hệ quả): lớp thiếu ở train có class weight bằng 0; lớp thiếu ở val/test bị loại khỏi Macro F1 nhóm nhựa và được liệt kê trong `classes_without_samples`. Lớp có support nhỏ không được kết luận mạnh.

## 8. Mất cân bằng và class weights

Nguồn: `runs/<EXP_ID>_*/config.json` → `train_counts`, `class_weights` (chế độ `balanced`, chặn trên 50).

| Lớp | Số crop train | Class weight |
|---|---:|---:|
| ⏳ | ⏳ | ⏳ |

## 9. Ảnh mẫu

Hình `results/figures/eda_sample_crops.png` (4 crop mỗi lớp, `eda.plot_sample_crops`, seed 42). Nhận xét định tính (độ đục, ánh sáng, đối tượng bị che, crop lẫn nền): ⏳

## 10. Hạn chế

- Tên lớp là **loại đối tượng**, không phải chất liệu; nhóm "liên quan nhựa" chỉ gồm các ứng viên (ví dụ `trash_bottle` có thể là thuỷ tinh).
- Ảnh là các khung hình liên tiếp của video: chia theo video giúp tránh rò rỉ, nhưng số video ít thì tỉ lệ thực tế có thể lệch khỏi 70/15/15.
- Crop chỉ dựa vào bbox, nên đối tượng mảnh hoặc bị che có thể lẫn nhiều nền.
- ⏳ (bổ sung sau khi xem số liệu thật: lớp quá ít mẫu, lớp thiếu ở val/test, nhãn đáng ngờ)

## 11. Tái lập và bàn giao

- [ ] `configs/class_mapping.json` được commit qua PR S2-DATA.
- [ ] `results/metrics/dataset_class_distribution.csv`, `dataset_split_counts.csv` và hình `results/figures/eda_*.png` được commit (mỗi file < 1 MB).
- [ ] Đường dẫn Drive của `v1_crop128.zip` ghi ở bảng đầu trang.
- [ ] Không commit ảnh crop, `split_manifest.csv` hay dữ liệu gốc.
