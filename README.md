# Ocean Plastic Waste Classification and Monitoring

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phanyen06ym-ien/ocean-plastic-ml-starter/blob/main/notebooks/00_pipeline_colab.ipynb)

**Phân loại và giám sát rác thải nhựa đại dương** — đồ án môn Học máy và Khai phá dữ liệu.

Dự án xây dựng mô hình CNN phân loại **ảnh thực tế dưới nước** theo **loại đối tượng** của bộ dữ liệu **TrashCan 1.0 — phiên bản Instance** (22 lớp), trong đó trọng tâm phân tích là **nhóm 6 lớp rác liên quan đến nhựa** (túi, chai, vỏ bao bì, lưới, dây, bạt). Nhóm không phân loại theo chất liệu nhựa.

> **Trạng thái:** Đã có kết quả Sprint 2 và Sprint 3 cho CNN baseline và MobileNetV2 transfer learning. E03G (extended training) đã chạy; kết quả không cho thấy cải thiện so với E03. Tiêu chí chọn checkpoint cuối (`val_loss` hay Macro F1 Validation) vẫn cần reviewer xác nhận. Tập Test chưa được dùng để chọn mô hình.

---

## 1. Bài toán

| Mục | Nội dung |
|---|---|
| Loại bài toán | Supervised multi-class image classification |
| Đầu vào | Ảnh RGB **một đối tượng** 128×128 đã được crop từ ảnh dưới nước theo annotation TrashCan-Instance |
| Đầu ra | Một trong **22 lớp** TrashCan-Instance: tên lớp, `class_index` và vector điểm softmax |
| Mô hình | CNN baseline tự xây dựng (TensorFlow/Keras), sau đó transfer learning MobileNetV2 (PyTorch) |
| Chỉ số chính | Macro F1-score trên 22 lớp và Macro F1 nhóm 6 lớp liên quan nhựa; kèm Accuracy, Precision, Recall, Confusion Matrix |
| Ngoài phạm vi | Object Detection, GAN phục hồi ảnh, video thời gian thực, điều khiển robot |

Chi tiết: [`docs/problem_statement.md`](docs/problem_statement.md).

## 2. Bộ dữ liệu TrashCan 1.0

- **7.212 ảnh** thu thập từ môi trường dưới nước (khung hình video), có annotation dạng COCO cho instance segmentation và bounding box.
- Hai phiên bản nhãn: **Instance** (theo loại đối tượng — nhóm sử dụng) và **Material** (theo chất liệu — chỉ dùng để đối chiếu khi cần).
- Nguồn: [UMN Data Repository](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7) · Hong, Fulton & Sattar, *TrashCan: A Semantically-Segmented Dataset towards Visual Detection of Marine Debris*, [arXiv:2007.08097](https://arxiv.org/abs/2007.08097).
- Dữ liệu **không được commit** lên GitHub; lưu trên Google Drive của nhóm.

### 2.1. Phạm vi lớp (ADR-001 — Option D)

CNN phân loại đủ 22 lớp TrashCan-Instance để tận dụng toàn bộ dữ liệu và không ép đối tượng ngoài phạm vi (ROV, sinh vật, thực vật) vào lớp rác. Kết quả được báo cáo ở hai cấp:

1. **Tổng thể 22 lớp** — Macro F1 là chỉ số chính.
2. **Nhóm rác liên quan nhựa** — phân tích riêng 6 lớp dưới đây. Tên nhãn là loại đối tượng, *không* phải bằng chứng chất liệu.

| Nhãn TrashCan-Instance | Số annotation (train + val gốc) | Ghi chú |
|---|---:|---|
| `trash_bag` | 908 | Lớp lớn nhất trong nhóm |
| `trash_net` | 127 | |
| `trash_bottle` | 126 | Chai có thể là nhựa hoặc thủy tinh |
| `trash_tarp` | 121 | |
| `trash_rope` | 117 | Không có mẫu ở val gốc → được chia lại |
| `trash_snack_wrapper` | 84 | Ít mẫu, kết luận thận trọng |

| Thông số | Giá trị |
|---|---|
| Kích thước ảnh | 128 × 128 RGB |
| Chia dữ liệu | Gộp train + val gốc, chia lại theo video 70 / 15 / 15, seed 42 |
| Crop | bbox + 10% lề, bỏ box cạnh < 8 px |
| Mất cân bằng | Giữ nguyên các lớp lớn (`rov`, `trash_unknown_instance`), dùng class weights |

Số liệu annotation lấy từ Dataset Audit nội bộ; số crop thực tế từng tập do `src/preprocessing.py` tạo và ghi trong `docs/dataset_report.md`. Phân tích đầy đủ: [`docs/class_scope_analysis.md`](docs/class_scope_analysis.md).

## 3. Quy trình học máy

```text
TrashCan-Instance (COCO JSON + ảnh)
  → Dataset Audit & EDA            notebooks/01 · src/eda.py
  → Chọn lớp + crop đối tượng       notebooks/02 · src/preprocessing.py
  → Chia Train/Val/Test theo video (tránh data leakage)
  → Huấn luyện CNN                  notebooks/03 · src/train.py
  → Đánh giá trên Validation → chọn mô hình → đánh giá Test 1 lần
                                    notebooks/04 · src/evaluate.py
  → Dự đoán ảnh mới (demo)          notebooks/05 · src/predict.py · main.py
```

Mọi thông số trao đổi giữa các module (kích thước ảnh, chuẩn hóa, `class_mapping`, định dạng checkpoint) được khóa trong [`configs/data_contract.yaml`](configs/data_contract.yaml).

## 4. Cấu trúc thư mục

```text
ocean-plastic-ml-starter/
├── .github/
│   ├── ISSUE_TEMPLATE/task.md
│   ├── pull_request_template.md
│   └── CODEOWNERS
├── configs/
│   ├── data_contract.yaml
│   ├── class_mapping.json
│   └── experiments/
├── data/                            # KHÔNG commit dữ liệu
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── problem_statement.md
│   ├── class_scope_analysis.md
│   ├── dataset_report.md
│   ├── evaluation_plan.md
│   ├── experiment_plan.md
│   ├── experiment_log.md
│   ├── colab_setup.md
│   ├── demo_spec.md
│   ├── sprint_plan.md
│   ├── decisions/
│   └── reports/
├── notebooks/
├── src/
├── tests/
├── models/                          # KHÔNG commit trọng số
├── results/
│   ├── figures/
│   └── metrics/
├── scripts/github/
├── main.py
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

Quy tắc đặt tên file, nhánh, commit, PR: xem [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 5. Môi trường làm việc: Google Colab + GitHub + VS Code

| Công cụ | Dùng cho |
|---|---|
| **VS Code** | Viết `src/`, `tests/`, tài liệu; tạo nhánh, commit, mở PR |
| **Google Colab** (GPU) | Chạy notebook EDA, tiền xử lý, huấn luyện, đánh giá |
| **Google Drive** | Lưu dữ liệu, checkpoint, log huấn luyện |
| **GitHub** | Mã nguồn, issue, PR, review |

Thư mục Drive chuẩn của nhóm:

```text
MyDrive/ocean-plastic-ml/
├── data/raw/trashcan/
├── data/processed/<DATA_VERSION>/
└── runs/<EXP_ID>_<YYYYMMDD-HHMM>/
    ├── config.json
    ├── history.csv
    ├── checkpoint (.keras hoặc .pt)
    └── metrics_val.json
```

Hướng dẫn chi tiết: [`docs/colab_setup.md`](docs/colab_setup.md).

Chạy nhanh trên máy cá nhân:

```bash
git clone https://github.com/phanyen06ym-ien/ocean-plastic-ml-starter.git
cd ocean-plastic-ml-starter
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## 6. Nhóm thực hiện

| Thành viên | GitHub | Vai trò | Khu vực |
|---|---|---|---|
| Phan Hải Yến | [@phanyen06ym-ien](https://github.com/phanyen06ym-ien) | Trưởng nhóm · Kiến trúc · Quản lý dự án · Reviewer duy nhất | `area:arch` |
| Hùng | [@tranngochung126hqv-stack](https://github.com/tranngochung126hqv-stack) | Dataset Audit · EDA · Tiền xử lý | `area:data` |
| Hoàng Bách | [@phannguyenhoangbach2](https://github.com/phannguyenhoangbach2) | Kiến trúc CNN · Huấn luyện | `area:cnn` |
| Nguyễn Hoàng Ân | *(cập nhật)* | Đánh giá · So sánh thực nghiệm · Phân tích lỗi | `area:eval` |
| Ngọc Nguyên | [@hiang0910](https://github.com/hiang0910) | Inference · Kiểm thử · Demo | `area:demo` |

Yến là người phụ trách review, phê duyệt và merge Pull Request theo quy ước nhóm.

## 7. Lộ trình

| Sprint | Thời gian | Mục tiêu | Milestone |
|---|---|---|---|
| Sprint 1 — Foundation | 23/10 – 03/11/2026 | Bài toán, kiến trúc, ADR-001, khung code DATA/CNN/EVAL/DEMO | `Sprint 1 — Foundation` |
| Sprint 2 — Data & Baseline | 04/11/2026 | Dữ liệu đã xử lý, huấn luyện E01/E02, đánh giá Validation, demo với checkpoint thật | `Sprint 2 — Data & Baseline` |
| Sprint 3 — Improvement & Midterm | 05/11 – 18/11/2026 | Cải tiến mô hình, đánh giá, demo, báo cáo giữa kỳ | `Sprint 3 — Improvement & Midterm` |

Chi tiết: [`docs/sprint_plan.md`](docs/sprint_plan.md).

**Định hướng tiếp theo:** xác nhận checkpoint cuối; hoàn thiện đánh giá Test sau khi chốt mô hình; cải thiện dữ liệu và nghiên cứu Object Detection, phục hồi ảnh dưới nước, mô hình nhẹ cho robot thu gom.

## 8. Kết quả thực nghiệm

Các số liệu dưới đây là trên **Validation** với `v1_crop128` (1.782 ảnh, 22 lớp). Macro F1 nhóm nhựa là kết quả trên 6 lớp liên quan nhựa. Dấu `—` nghĩa là số liệu chưa có trong bảng kết quả đang dùng. **Tập Test chưa được dùng để chọn mô hình.**

| EXP_ID | Mô tả | Macro F1 (val) | Accuracy (val) | Macro F1 nhóm nhựa (val) |
|---|---|---:|---:|---:|
| E01 | CNN baseline + class weights | 0,0757 | 0,2190 | 0,0000 |
| E02 | E01 + Data Augmentation | 0,0648 | 0,1610 | 0,0165 |
| E03 | MobileNetV2, 2 pha | 0,2034 | 0,4052 | 0,1228 |
| E03a | E03 + dropout 0,6 | 0,2099 | — | 0,1377 |
| E03c | E03 + learning rate pha 1 = 3e-4 | 0,1955 | — | 0,1083 |
| E03d | E03 + learning rate backbone pha 2 = 3e-5 | 0,2034 | — | 0,1228 |
| E03f | E03 với seed 43 | 0,2122 | — | 0,1168 |
| E03d-F1 | E03d, checkpoint chọn theo Macro F1 | 0,2398 | 0,4994 | 0,1076 |
| E03f-F1 | E03f, checkpoint chọn theo Macro F1 | 0,2372 | 0,4826 | 0,1357 |
| E03G | E03 với lịch huấn luyện mở rộng | 0,2034 | 0,4052 | 0,1228 |

### 8.1. Nhận xét

- E02 không cải thiện so với E01; giả thuyết H2 không được xác nhận trong các run hiện có.
- E03 cải thiện Macro F1 Validation so với E01 (0,2034 so với 0,0757), cho thấy transfer learning có lợi trong thí nghiệm này.
- Các thay đổi dropout và learning rate chưa cho thấy cải thiện ổn định. Hai run E03 và E03f khác seed chênh 0,009 Macro F1, vì vậy cần thận trọng với khác biệt nhỏ.
- E03d-F1 và E03f-F1 có Macro F1 cao hơn checkpoint chọn theo `val_loss`, nhưng đó là kết quả theo **tiêu chí checkpoint khác**. Reviewer cần xác nhận tiêu chí chính thức trước khi chốt mô hình.
- E03G kéo dài thời gian huấn luyện nhưng checkpoint cuối theo `val_loss` có kết quả tương đương E03. Trong fine-tuning, training loss giảm trong khi validation loss nhìn chung tăng; log pha 2 có Macro F1 tối đa 0,236 tại epoch 9, nhưng epoch này không phải epoch có `val_loss` tốt nhất.
- Không diễn giải các kết quả Validation thành kết quả Test. Không dùng Test để chọn checkpoint.

Chi tiết cấu hình, lịch sử và giới hạn của từng run: [`docs/experiment_plan.md`](docs/experiment_plan.md) và [`docs/experiment_log.md`](docs/experiment_log.md).

### 8.2. Kiểm thử phần mềm

Lần chạy test được báo cáo: `48 passed, 50 warnings in 88.05s`. Đây là kết quả kiểm thử phần mềm (`pytest`), **không phải** chỉ số đánh giá mô hình. Các warning không làm test thất bại; cần xem log nếu muốn xử lý riêng.

## 9. Trích dẫn

```bibtex
@article{hong2020trashcan,
  title   = {TrashCan: A Semantically-Segmented Dataset towards Visual Detection of Marine Debris},
  author  = {Hong, Jungseok and Fulton, Michael and Sattar, Junaed},
  journal = {arXiv preprint arXiv:2007.08097},
  year    = {2020}
}
```
