# Báo cáo dữ liệu: TrashCan 1.0 Instance (S2-DATA)

| Mục | Giá trị |
|---|---|
| Phụ trách | Hùng (dự phòng: Yến) |
| `DATA_VERSION` | `v1_crop128` |
| Nguồn số liệu | Báo cáo của Hùng trên Drive nhóm (`TrashCan/dataset_report.md`, cập nhật 03/11/2026), chạy `src/preprocessing.py` trên dữ liệu thật |
| Căn cứ | `docs/decisions/ADR-001-class-scope.md`, `configs/data_contract.yaml` v1 |
| Còn thiếu | Commit code đã dùng, vị trí bản crop đã xử lý trên Drive (chưa được chia sẻ) |

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

## 3. Chia Train / Validation / Test theo video

Gộp train + val gốc, chia theo video, mục tiêu 70/15/15, seed 42. Không video nào xuất hiện ở hai tập.

| Tập | Annotation | Ảnh có annotation | Video | Tỉ lệ ảnh |
|---|---:|---:|---:|---:|
| Train | 8.002 | 4.645 | 217 | 65,6% |
| Validation | 2.244 | 1.244 | 47 | 17,6% |
| Test | 1.882 | 1.194 | 46 | 16,9% |
| **Tổng** | **12.128** | **7.083** | **310** | |

Tỉ lệ thực tế lệch khỏi 70/15/15 vì cả video phải nằm trong một tập. 129 trong 7.212 ảnh gốc không có annotation nên không xuất hiện ở bảng trên.

## 4. Kết quả crop

Crop theo bbox, lề 10% mỗi phía, bỏ box có cạnh ngắn hơn 8 px, resize thẳng về 128×128 RGB; chuẩn hoá /255 nằm trong mô hình.

| Tập | Crop thành công | Bbox nhỏ bị loại | Tổng annotation |
|---|---:|---:|---:|
| Train | 7.957 | 45 | 8.002 |
| Validation | 2.237 | 7 | 2.244 |
| Test | 1.880 | 2 | 1.882 |
| **Tổng** | **12.074** | **54** | **12.128** |

Tỉ lệ crop thành công 99,55%. Không có annotation thiếu ảnh, sai lớp hay lỗi xử lý.

### 4.1. Số crop theo lớp và tập

`*` = lớp liên quan nhựa (ADR-001). Lớp có dưới 10 mẫu ở Validation hoặc Test được in đậm.

| Lớp | Train | Validation | Test | Tổng |
|---|---:|---:|---:|---:|
| rov | 2.181 | 649 | 467 | 3.297 |
| trash_unknown_instance | 1.798 | 594 | 347 | 2.739 |
| `*` trash_bag | 697 | 153 | 57 | 907 |
| animal_fish | 513 | 81 | 167 | 761 |
| trash_container | 374 | 120 | 14 | 508 |
| plant | 180 | 79 | 248 | 507 |
| trash_can | 343 | **5** | 111 | 459 |
| animal_starfish | 262 | 127 | **8** | 397 |
| trash_branch | 216 | 37 | 83 | 336 |
| animal_eel | 160 | 119 | 55 | 334 |
| animal_crab | 217 | 29 | 63 | 309 |
| animal_shells | 217 | **6** | 25 | 248 |
| animal_etc | 159 | 57 | 19 | 235 |
| trash_wreckage | 73 | 90 | **2** | 165 |
| trash_pipe | 99 | **1** | 56 | 156 |
| `*` trash_net | 82 | 10 | 35 | 127 |
| `*` trash_bottle | 76 | **7** | 43 | 126 |
| `*` trash_tarp | 81 | 21 | 19 | 121 |
| `*` trash_rope | 90 | 19 | **8** | 117 |
| `*` trash_snack_wrapper | 51 | 13 | 20 | 84 |
| trash_clothing | 60 | 11 | 11 | 82 |
| trash_cup | 28 | **9** | 22 | 59 |
| **Tổng** | **7.957** | **2.237** | **1.880** | **12.074** |

## 5. Kiểm tra đã thực hiện

- Ảnh crop 128×128, chế độ RGB.
- Không video nào nằm ở nhiều tập.
- Mapping đủ 22 lớp, `class_index` 0–21.
- `pytest tests/test_preprocessing.py` đạt (theo báo cáo của Hùng).

## 6. Hạn chế và lưu ý cho đánh giá

- Mất cân bằng lớn: `rov` và `trash_unknown_instance` chiếm 50% số crop; `trash_cup` chỉ có 59. Huấn luyện dùng class weights (ADR-001 D6).
- Đủ 22 lớp có mặt ở cả ba tập, nhưng 7 lớp có dưới 10 mẫu ở Validation hoặc Test (in đậm ở bảng 4.1). F1 của các lớp này dao động mạnh; báo cáo phải kèm support và không kết luận từ riêng chúng.
- Trong nhóm lớp liên quan nhựa, `trash_bottle` chỉ có 7 mẫu Validation và `trash_rope` 8 mẫu Test.
- Phân bố lớp giữa Validation và Test khác nhau (ví dụ `trash_wreckage` 90 / 2, `plant` 79 / 248), nên chênh lệch Val/Test ở Sprint 3 có thể đến từ dữ liệu.
- 54 annotation bị loại vì bbox quá nhỏ.

## 7. Tái lập và bàn giao

```bash
python -m src.preprocessing --raw-root <thư-mục-TrashCan> --out-dir data/processed/v1_crop128 --mapping-out configs/class_mapping.json
```

- [x] Số liệu từ dữ liệu thật.
- [ ] Ghi commit code đã chạy.
- [ ] Chia sẻ `v1_crop128.zip` và `split_manifest.csv` trên Drive nhóm (`MyDrive/ocean-plastic-ml/data/processed/`).
- [ ] Đưa `configs/class_mapping.json` lên repo.
