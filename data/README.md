# data/

Thư mục chứa dữ liệu ảnh TrashCan 1.0 dùng cho huấn luyện và đánh giá.

> **Không commit dữ liệu lên GitHub.** Trên GitHub thư mục này chỉ có file README. Các thư mục con bên dưới được tạo khi chuẩn bị dữ liệu và bị `.gitignore` chặn toàn bộ. Bản dữ liệu chính thức của nhóm nằm trên Google Drive: `MyDrive/ocean-plastic-ml/data/`.

## Cấu trúc (sau khi chuẩn bị dữ liệu)

```
data/
├── README.md
├── raw/                         # Dữ liệu gốc TrashCan 1.0, không sửa
│   └── (giải nén dataset.zip)   # ảnh .jpg + instances_train_trashcan.json, instances_val_trashcan.json
└── processed/
    └── <DATA_VERSION>/          # ví dụ v1_crop128 — đầu ra của src/preprocessing.py
        ├── train/<class_name>/*.jpg
        ├── val/<class_name>/*.jpg
        ├── test/<class_name>/*.jpg
        ├── class_mapping.json
        ├── split_manifest.csv
        ├── class_split_counts.csv
        └── preprocessing_summary.json
```

| Thư mục / file                                         | Nội dung                                                                                                                                                                                            |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `raw/`                                                 | Ảnh khung hình video dưới nước và 2 file annotation COCO của bản **Instance** (22 lớp). `src/preprocessing.py` tự tìm file JSON và ảnh ở mọi thư mục con, nên giữ nguyên cấu trúc sau khi giải nén. |
| `processed/<DATA_VERSION>/`                            | Ảnh đã cắt (crop) theo bounding box, resize và chia sẵn Train / Validation / Test theo video; mỗi lớp một thư mục con. Đây là đầu vào trực tiếp của `src/train_keras.py` và `src/train.py`.         |
| `class_mapping.json`                                   | `category_id` → `class_name` → `class_index` (0–21). Bản giống hệt được commit ở `configs/class_mapping.json`.                                                                                      |
| `split_manifest.csv`                                   | Mỗi dòng một crop: annotation, ảnh nguồn, video, lớp, tập (train/val/test), trạng thái crop. Dùng để tái lập và kiểm tra rò rỉ dữ liệu.                                                             |
| `class_split_counts.csv`, `preprocessing_summary.json` | Thống kê số crop theo lớp và tập. Bản tóm tắt được commit ở `results/metrics/`.                                                                                                                     |

## Phiên bản dữ liệu

| `DATA_VERSION`    | Dùng cho              | Thông số                                                           | Crop Train / Val / Test |
| ----------------- | --------------------- | ------------------------------------------------------------------ | ----------------------- |
| `v1_crop128`      | E01, E02, E03, E03a–f | 128×128, lề 10%, bỏ bbox < 8 px, chia theo video 70/15/15, seed 42 | 8.283 / 1.782 / 2.009   |
| _(phiên bản mới)_ | E04                   | Chốt tại ADR-002                                                   | —                       |

Thông số dùng chung khai báo trong [`configs/data_contract.yaml`](../configs/data_contract.yaml). Thống kê chi tiết: [`docs/dataset_report.md`](../docs/dataset_report.md).

## Cách chuẩn bị dữ liệu

**Trên Google Colab (khuyến nghị):** chạy `notebooks/00_pipeline_colab.ipynb`. Notebook mount Drive, tải `dataset.zip` nếu chưa có (`src/download.py`), giải nén và tiền xử lý.

**Trên máy cá nhân:**

1. Tải `dataset.zip` của TrashCan 1.0 từ [UMN Data Repository](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7) và giải nén vào `data/raw/`.
2. Tiền xử lý (thư mục đầu ra được tạo tự động):

   ```bash
   python -m src.preprocessing --raw-root data/raw --out-dir data/processed/v1_crop128
   ```

3. Kiểm tra `data/processed/v1_crop128/preprocessing_summary.json`: số crop từng tập phải khớp bảng ở trên.

## Lưu ý

- Không sửa ảnh hoặc annotation trong `raw/`. Mọi thay đổi cách xử lý phải tạo `DATA_VERSION` mới và ghi ADR.
- Không dùng tập `test/` khi huấn luyện hay chọn mô hình; chỉ đánh giá một lần cho mô hình cuối (ADR-001 D7).
- Bộ dữ liệu thuộc quyền nhóm tác giả TrashCan (Hong, Fulton & Sattar, 2020); trích dẫn khi sử dụng.
