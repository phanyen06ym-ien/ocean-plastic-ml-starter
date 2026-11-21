# Ocean Plastic Waste Classification and Monitoring

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phanyen06ym-ien/ocean-plastic-ml-starter/blob/main/notebooks/00_pipeline_colab.ipynb)

**Phân loại và giám sát rác thải nhựa đại dương** — đồ án môn Học máy và Khai phá dữ liệu.

Dự án xây dựng mô hình mạng nơ-ron tích chập (CNN) phân loại **ảnh thực tế dưới nước** theo **loại đối tượng** của bộ dữ liệu **TrashCan 1.0 — Instance** (22 lớp). Trọng tâm phân tích là **nhóm 6 lớp rác liên quan đến nhựa** (túi, chai, vỏ bao bì, lưới, dây, bạt). Nhóm so sánh CNN tự thiết kế (E01–E02) với học chuyển giao MobileNetV2 (E03–E04) trên cùng một cách chia dữ liệu theo video.

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
| MobileNetV2                            | E03 (2 pha) + tinh chỉnh E03a–E03f                                                     | ✅ Xong      |
| Đánh giá Validation                    | Macro F1, nhóm nhựa, confusion matrix, phân tích lỗi E01–E03                           | ✅ Xong      |
| Dữ liệu mới + E04                      | Làm lại bộ dữ liệu để cải thiện MobileNetV2; thông số ghi trong `dataset_report.md`                   | 🔄 Đang làm  |
| Chốt mô hình cuối                      | So sánh trên Validation                                                       | ⏳ Chờ E04   |
| Đánh giá Test                          | Chạy **một lần** cho mô hình cuối                                         | ⏳ Chưa chạy |
| Demo                                   | CLI `main.py` chạy với checkpoint Keras (E01–E02); hỗ trợ checkpoint PyTorch (E03–E04) | 🔄 Đang làm  |
| Báo cáo, slide, release `v0.1-midterm` |                                                                                        | ⏳           |

## 2. Bài toán

| Mục                   | Nội dung                                                                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Loại bài toán         | Phân loại ảnh có giám sát, đa lớp (supervised multi-class image classification)                                                              |
| Đầu vào               | Ảnh RGB **một đối tượng**, cắt (crop) từ khung hình video dưới nước theo bounding box của TrashCan-Instance                                  |
| Đầu ra                | Một trong **22 lớp**: tên lớp, `class_index` và vector xác suất softmax                                                                      |
| Chỉ số chính          | **Macro F1** trên 22 lớp và **Macro F1 nhóm 6 lớp liên quan nhựa**                                                                           |
| Chỉ số phụ            | Accuracy, Precision, Recall, Weighted F1, F1 từng lớp, Confusion Matrix; số tham số, kích thước file và thời gian suy luận (hướng tới robot) |
| Ngoài phạm vi giữa kỳ | Object Detection, phục hồi ảnh dưới nước, video thời gian thực, điều khiển robot                                                             |

Macro F1 được chọn làm chỉ số chính vì dữ liệu mất cân bằng mạnh: Accuracy cao có thể chỉ do đoán đúng hai lớp lớn. Chi tiết: [`docs/problem_statement.md`](docs/problem_statement.md).

## 3. Dữ liệu

### 3.1. Nguồn

- **TrashCan 1.0** — 7.212 ảnh là khung hình từ video quay dưới nước, annotation dạng COCO (bounding box và segmentation). Nhóm dùng bản nhãn **Instance** (theo loại đối tượng), không dùng bản Material.
- Nguồn: [UMN Data Repository](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7) · Bài báo: Hong, Fulton & Sattar, [arXiv:2007.08097](https://arxiv.org/abs/2007.08097).
- Dữ liệu **không commit lên GitHub**; lưu trên Google Drive của nhóm. Tải tự động bằng `src/download.py`.

### 3.2. Thống kê (phiên bản `v1_crop128`)

| Chỉ số                            |                                       Giá trị |
| --------------------------------- | --------------------------------------------: |
| Annotation (train + val gốc)      |                                        12.128 |
| Ảnh nguồn có annotation           |                                         7.083 |
| Số video                          |                                           310 |
| Crop hợp lệ / loại vì bbox < 8 px |                                   12.074 / 54 |
| Crop Train / Validation / Test    |                         8.283 / 1.782 / 2.009 |
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

| Bước         | `v1_crop128` (E01–E03)                                                                                                                          |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Crop         | Theo bbox COCO, mở rộng 10% mỗi phía, bỏ box có cạnh < 8 px                                                                                     |
| Kích thước   | Resize thẳng về 128 × 128 RGB                                                                                                                   |
| Chia tập     | Gộp train + val gốc, **chia lại theo video** 70 / 15 / 15, seed 42 — mọi khung hình của một video nằm trong cùng một tập để tránh rò rỉ dữ liệu |
| Mất cân bằng | Giữ nguyên các lớp lớn, dùng class weights `balanced` khi huấn luyện                                                                            |
| Chuẩn hóa    | CNN tự thiết kế: pixel / 255 · MobileNetV2: mean/std ImageNet                                                                                   |

Mọi thông số dùng chung giữa các module được khóa trong [`configs/data_contract.yaml`](configs/data_contract.yaml). Phiên bản dữ liệu cho E04 sẽ được ghi trong [`docs/dataset_report.md`](docs/dataset_report.md).

### 3.4. Hạn chế đã biết

- Chia theo video làm lệch số mẫu giữa các tập: vài lớp chỉ có 1–4 mẫu ở Validation hoặc Test (`animal_shells` 1, `trash_bottle` 3 ở Val; `animal_starfish` 4 ở Test).
- Nhiều bbox nhỏ (trung vị khoảng 90 × 66 px ở Train); một số bbox có toạ độ vượt ngoài ảnh (được giới hạn lại khi crop).
- `trash_unknown_instance` là lớp mơ hồ, dễ nhầm với các lớp rác khác.

## 4. Phương pháp

### 4.1. Quy trình

```
TrashCan-Instance (COCO JSON + ảnh)
  → Audit & EDA                     notebooks/01 · src/eda.py
  → Crop đối tượng + chia theo video notebooks/02 · src/preprocessing.py
  → Huấn luyện                      notebooks/03 · src/train_keras.py (E01–E02) · src/train.py (E03–E04)
  → Đánh giá Validation → chọn mô hình → đánh giá Test 1 lần
                                    notebooks/04 · src/evaluate.py · src/evaluation_runner.py
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

**Học chuyển giao MobileNetV2 (E03–E04, PyTorch)**

- Backbone `torchvision` MobileNetV2 tiền huấn luyện ImageNet; thay bộ phân loại bằng `Dropout → Linear(1280, 22)`. Khoảng 2,25 triệu tham số, file 9,2 MB, suy luận khoảng 30 ms/ảnh trên CPU.
- **Pha 1:** đóng băng backbone, chỉ huấn luyện bộ phân loại (lr 1e-3, tối đa 15 epoch).
- **Pha 2:** mở khoá các block cuối (`features[14:]`), học với lr nhỏ (1e-5 cho backbone, 1e-4 cho bộ phân loại), giữ BatchNorm ở chế độ eval.
- Chọn checkpoint có `val_loss` thấp nhất giữa hai pha.

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
| E03a–f | MobileNetV2     | Mỗi lần đổi một siêu tham số (dropout, lr, seed…)  | `v1_crop128`  | `E03?_tune.yaml`                |
| E04    | MobileNetV2     | Dữ liệu: phiên bản mới                   | đang chuẩn bị | —                               |

Cấu hình từng thí nghiệm: `configs/experiments/` · Số liệu từng lần chạy: `results/metrics/<EXP_ID>_val.json`, `<EXP_ID>_history.csv`.

## 6. Kết quả và phân tích

Tập Validation, 1.782 crop. Nguồn: `results/metrics/comparison_val.csv`, `results/metrics/e03_candidates.csv`.

| EXP_ID                 |         Macro F1 |  Accuracy | Weighted F1 | Macro F1 nhóm nhựa | Macro F1 (Test) |
| ---------------------- | ---------------: | --------: | ----------: | -----------------: | --------------: |
| E01                    |            0,076 |     0,219 |       0,264 |              0,000 |               — |
| E02                    |            0,065 |     0,161 |       0,167 |              0,017 |               — |
| E03                    |            0,203 |     0,405 |       0,418 |              0,123 |               — |
| E03f (tốt nhất E03a–f) |        **0,212** | **0,412** |   **0,420** |              0,117 |               — |
| E04                    | _đang thực hiện_ |           |             |                    |               — |

Mức ngẫu nhiên của Macro F1 với 22 lớp ≈ 0,045.

**Nhận xét**

1. **Học chuyển giao hiệu quả hơn rõ rệt.** MobileNetV2 tăng Macro F1 khoảng 2,7 lần so với CNN tự thiết kế (0,203 so với 0,076) trên cùng dữ liệu.
2. **Augmentation không giúp CNN tự thiết kế** (E02 thấp hơn E01): mô hình nhỏ huấn luyện từ đầu chưa học được đặc trưng ổn định.
3. **Kết quả theo lớp rất chênh lệch** (E03f): tốt ở `rov` (F1 0,76), `animal_eel` (0,67), `animal_starfish` (0,52); bằng 0 ở nhiều lớp ít mẫu như `trash_bottle`, `trash_tarp`, `trash_rope`, `trash_snack_wrapper`, `trash_cup`.
4. **Mô hình học thuộc nhanh:** E03 đạt `val_loss` tốt nhất ngay ở epoch 2 trong khi loss trên Train vẫn giảm mạnh — mô hình học bối cảnh của từng video hơn là hình dạng vật thể.

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

# Dự đoán (checkpoint Keras)
python main.py --image path/to/crop.jpg --model runs/E01_<thời gian>/best.keras
```

Thư mục Drive của nhóm:

```
MyDrive/ocean-plastic-ml/
├── data/raw/trashcan/               # dataset gốc
├── data/processed/<DATA_VERSION>/   # ví dụ v1_crop128
└── runs/<EXP_ID>_<YYYYMMDD-HHMM>/   # config, history.csv, checkpoint, metrics
```

## 8. Cấu trúc repo

```
ocean-plastic-ml-starter/
├── .github/                         # Mẫu issue, mẫu PR, CODEOWNERS
├── configs/
│   ├── data_contract.yaml           # Thông số dữ liệu dùng chung
│   ├── class_mapping.json           # category_id → class_name → class_index
│   └── experiments/                 # E01, E02, E03, E03a–f (.yaml)
├── data/                            # Chỉ có README.md; dữ liệu lưu trên Drive (xem data/README.md)
├── docs/
│   ├── problem_statement.md         # Bài toán, phạm vi, chỉ số
│   ├── dataset_report.md            # Thống kê dữ liệu, cách chia tập
│   ├── evaluation_plan.md           # Cách đánh giá
│   ├── colab_setup.md               # Hướng dẫn chạy trên Colab
│   ├── class_scope_analysis.md      # Lý do chọn 22 lớp
│   └── visual_inspection_protocol.md # Kiểm tra ảnh crop và nhãn
├── notebooks/
│   ├── 00_pipeline_colab.ipynb      # Chạy toàn bộ quy trình
│   ├── 01_trashcan_eda.ipynb · 02_preprocessing.ipynb
│   └── 03_cnn_training.ipynb · 04_evaluation.ipynb
├── src/                             # Code dùng chung, notebook chỉ gọi lại
│   ├── common.py · download.py · eda.py · preprocessing.py
│   ├── train_keras.py (E01–E02) · train.py (E03–E04)
│   ├── evaluate.py · evaluation_runner.py · eval_torch.py
│   └── predict.py
├── tests/                           # pytest cho từng module
├── models/                          # KHÔNG commit trọng số (lưu Drive)
├── results/
│   ├── figures/                     # Hình cho báo cáo
│   └── metrics/                     # <EXP_ID>_<split>.json/.csv, bảng so sánh
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
