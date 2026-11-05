# Báo cáo dữ liệu: TrashCan 1.0 Instance (S2-DATA)

| Mục | Giá trị |
|---|---|
| Phụ trách | Hùng (dự phòng: Yến) |
| `DATA_VERSION` | `v1_crop128` |
| Nguồn số liệu | Báo cáo của Hùng trên Drive nhóm (`TrashCan/dataset_report.md`, cập nhật 03/11/2026), chạy `src/preprocessing.py` trên dữ liệu thật |
| Căn cứ | `docs/decisions/ADR-001-class-scope.md`, `configs/data_contract.yaml` v1 |
| Bản chính thức | Colab 05/11/2026, commit `7520c2b`, Drive `MyDrive/ocean-plastic-ml/data/processed/v1_crop128.zip` |

## 1. Nguồn dữ liệu

- TrashCan 1.0, bản nhãn **Instance** (theo loại đối tượng), annotation định dạng COCO. Ảnh đặt tên theo khung hình video, ví dụ `vid_000020_frame0000002.jpg`.
- Nguồn: [UMN Data Repository](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7). Bài báo: Hong, Fulton & Sattar, arXiv:2007.08097.
- Bản dữ liệu nhóm dùng: thư mục Drive `TrashCan/` do Hùng chia sẻ (`instance_version/`, `material_version/`, `original_data/`).

## 2. Kiểm tra dữ liệu gốc

| Chỉ tiêu | Train gốc | Val gốc |
|---|---:|---:|
| Số ảnh | 6.065 | 1.147 |
| Số annotation | 9.540 | 2.588 |
| Số lớp | 22 | 22 |
| Ảnh thiếu / không đọc được | 0 / 0 | 0 / 0 |
| Bbox vượt ngoài ảnh | 2.606 | 680 |

Bbox vượt ngoài ảnh là lỗi toạ độ annotation, không phải ảnh hỏng; `crop_box` giới hạn toạ độ theo kích thước ảnh.

### 2.1. Kích thước bbox (pixel)

| Chỉ tiêu | Train rộng | Train cao | Val rộng | Val cao |
|---|---:|---:|---:|---:|
| Trung bình | 141,75 | 95,54 | 121,95 | 83,76 |
| Trung vị | 89,98 | 66,00 | 73,00 | 52,00 |
| Nhỏ nhất | 5,00 | 5,98 | 3,00 | 3,00 |
| Lớn nhất | 481,00 | 361,00 | 481,00 | 360,00 |

## 3. Chia Train / Validation / Test theo video (bản chính thức `v1_crop128`)

Chạy `notebooks/00_pipeline_colab.ipynb` (bước 3) trên Colab ngày 05/11/2026, commit `7520c2b`. Gộp train + val gốc, chia theo video với tỉ lệ ảnh nguồn mục tiêu 70/15/15, seed 42; hàm chia thử lại (lần thử thứ 160) cho đến khi đủ 22 lớp có mặt ở cả Validation và Test. Nguồn: `results/metrics/preprocessing_summary.json`.

| Chỉ số | Giá trị |
|---|---:|
| Annotation (train + val gốc) | 12.128 |
| Ảnh nguồn có annotation | 7.083 |
| Video | 310 |
| Crop hợp lệ / bị loại vì bbox < 8 px | 12.074 / 54 |
| Tỉ lệ ảnh nguồn Train / Val / Test | 69,9% / 14,9% / 15,2% |
| Crop Train / Val / Test | 8.283 / 1.782 / 2.009 |
| Kiểm tra rò rỉ (`check_no_leakage`) | Đạt: không video, không ảnh nào nằm ở hai tập |

Dataset Audit ghi nhận 312 video; pipeline chỉ dùng ảnh có annotation (7.083 / 7.212 ảnh) và đếm được 310. Chênh lệch 2 video sẽ được đối chiếu khi viết dataset card (S3-09).

> Bảng chia ở phiên bản trước của báo cáo này (217/47/46 video) là lần chạy thử của Hùng với seed khác, **không** phải bản dùng để huấn luyện. Từ Sprint 2 chỉ dùng bản dưới đây.

## 4. Số crop theo lớp và tập

Nguồn: `results/metrics/dataset_split_counts.csv`; hình `results/figures/eda_split_counts.png`, `eda_class_distribution.png`. `*` = lớp liên quan nhựa (ADR-001). In đậm: dưới 20 mẫu Train hoặc dưới 10 mẫu Val/Test.

| Lớp | Train | Validation | Test | Tổng |
|---|---:|---:|---:|---:|
| rov | 2.130 | 460 | 707 | 3.297 |
| trash_unknown_instance | 1.970 | 479 | 290 | 2.739 |
| `*` trash_bag | 605 | 106 | 196 | 907 |
| animal_fish | 604 | 81 | 76 | 761 |
| trash_container | 299 | 152 | 57 | 508 |
| plant | 228 | 73 | 206 | 507 |
| trash_can | 385 | 59 | 15 | 459 |
| animal_starfish | 360 | 33 | **4** | 397 |
| trash_branch | 249 | 13 | 74 | 336 |
| animal_eel | 194 | 103 | 37 | 334 |
| animal_crab | 185 | **3** | 121 | 309 |
| animal_shells | 233 | **1** | 14 | 248 |
| animal_etc | 162 | 24 | 49 | 235 |
| trash_wreckage | 122 | 31 | 12 | 165 |
| trash_pipe | 79 | 21 | 56 | 156 |
| `*` trash_net | 79 | 36 | 12 | 127 |
| `*` trash_bottle | 95 | **3** | 28 | 126 |
| `*` trash_tarp | 103 | 13 | **5** | 121 |
| `*` trash_rope | 87 | 23 | **7** | 117 |
| `*` trash_snack_wrapper | 41 | 11 | 32 | 84 |
| trash_clothing | 59 | 21 | **2** | 82 |
| trash_cup | **14** | 36 | **9** | 59 |
| **Tổng** | **8.283** | **1.782** | **2.009** | **12.074** |

Tỉ lệ mất cân bằng (lớn nhất / nhỏ nhất theo tổng crop): 3.297 / 59 ≈ 56.

## 5. Kiểm tra đã thực hiện

- Ảnh crop 128×128, chế độ RGB.
- Không video nào nằm ở nhiều tập.
- Mapping đủ 22 lớp, `class_index` 0–21.
- `pytest tests/test_preprocessing.py` đạt (theo báo cáo của Hùng).

## 6. Hạn chế và lưu ý cho đánh giá

- Mất cân bằng lớn: `rov` và `trash_unknown_instance` chiếm 50% số crop; `trash_cup` chỉ có 59. Huấn luyện dùng class weights (ADR-001 D6).
- **Chia theo video làm lệch phân bố lớp giữa các tập** vì nhiều lớp chỉ xuất hiện trong ít video: `trash_cup` (4 video) có 14 Train / 36 Val; `animal_shells` (7 video) có 233 Train / 1 Val; `animal_crab` 185 / 3 / 121; `animal_starfish` 360 / 33 / 4. Đây là nguyên nhân chính khiến kết quả Validation của E01/E02 thấp (xem `docs/evaluation_report.md`).
- Đủ 22 lớp có mặt ở cả ba tập, nhưng nhiều lớp có dưới 10 mẫu ở Validation hoặc Test (in đậm ở bảng mục 4). F1 của các lớp này dao động mạnh; báo cáo phải kèm support.
- Trong nhóm lớp liên quan nhựa, `trash_bottle` chỉ có 3 mẫu Validation; `trash_tarp` 5 và `trash_rope` 7 mẫu Test.
- 54 annotation bị loại vì bbox quá nhỏ.

## 7. Tái lập và bàn giao

```bash
python -m src.preprocessing --raw-root <thư-mục-TrashCan> --out-dir data/processed/v1_crop128 --mapping-out configs/class_mapping.json
```

- [x] Số liệu từ dữ liệu thật.
- [x] Ghi commit code đã chạy: `7520c2b` (05/11/2026, Colab T4).
- [x] `v1_crop128.zip` (gồm `split_manifest.csv`) trên Drive nhóm: `MyDrive/ocean-plastic-ml/data/processed/v1_crop128.zip`.
- [x] `configs/class_mapping.json` có trong repo.
