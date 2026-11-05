# Ocean Plastic Waste Classification and Monitoring

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phanyen06ym-ien/ocean-plastic-ml-starter/blob/main/notebooks/00_pipeline_colab.ipynb)

**Phân loại và giám sát rác thải nhựa đại dương** — đồ án môn Học máy và Khai phá dữ liệu.

Dự án xây dựng mô hình mạng nơ-ron tích chập (CNN) phân loại **ảnh thực tế dưới nước** theo **loại đối tượng** của bộ dữ liệu **TrashCan 1.0 — phiên bản Instance** (22 lớp), trong đó trọng tâm phân tích là **nhóm 6 lớp rác liên quan đến nhựa** (túi, chai, vỏ bao bì, lưới, dây, bạt). Nhóm không phân loại theo chất liệu nhựa.

> **Trạng thái:** Sprint 1–2 (03–04/11/2026). Phạm vi đã chốt tại [ADR-001](docs/decisions/ADR-001-class-scope.md). Chưa có kết quả huấn luyện chính thức. Mọi số liệu trong README chỉ lấy từ Dataset Audit nội bộ và sẽ được cập nhật khi các PR liên quan được merge.

---

## 1. Bài toán

| Mục | Nội dung |
|---|---|
| Loại bài toán | Supervised multi-class image classification |
| Đầu vào | Ảnh RGB **một đối tượng** 128×128 đã được cắt (crop) từ ảnh dưới nước theo annotation của TrashCan-Instance |
| Đầu ra | Một trong **22 lớp** TrashCan-Instance: tên lớp, `class_index` và vector điểm softmax |
| Mô hình | CNN baseline tự xây dựng (TensorFlow/Keras), sau đó so sánh với phương án cải tiến |
| Chỉ số chính | Macro F1-score trên 22 lớp + Macro F1 nhóm 6 lớp liên quan nhựa; kèm Accuracy, Precision, Recall, Confusion Matrix |
| Ngoài phạm vi giữa kỳ | Object Detection, GAN phục hồi ảnh, video thời gian thực, điều khiển robot |

Chi tiết: [`docs/problem_statement.md`](docs/problem_statement.md).

## 2. Bộ dữ liệu TrashCan 1.0

- **7.212 ảnh** thu thập từ môi trường dưới nước (khung hình video), có annotation dạng COCO cho instance segmentation và bounding box.
- Hai phiên bản nhãn: **Instance** (theo loại đối tượng — nhóm sử dụng) và **Material** (theo chất liệu — chỉ dùng để đối chiếu khi cần).
- Nguồn: [UMN Data Repository](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7) · Bài báo: Hong, Fulton & Sattar, *TrashCan: A Semantically-Segmented Dataset towards Visual Detection of Marine Debris*, [arXiv:2007.08097](https://arxiv.org/abs/2007.08097).
- Dữ liệu **không được commit** lên GitHub; lưu trên Google Drive của nhóm (xem mục 5).

### 2.1. Phạm vi lớp (ADR-001 — Option D)

CNN phân loại **đủ 22 lớp** của TrashCan-Instance để tận dụng toàn bộ dữ liệu và không ép đối tượng ngoài phạm vi (ROV, sinh vật, thực vật) vào lớp rác. Kết quả được báo cáo ở hai cấp:

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
| Kích thước ảnh | 128 × 128 RGB, pixel / 255 |
| Chia dữ liệu | Gộp train + val gốc, chia lại **theo video** 70 / 15 / 15, seed 42 |
| Crop | bbox + 10% lề, bỏ box cạnh < 8 px |
| Mất cân bằng | Giữ nguyên các lớp lớn (`rov`, `trash_unknown_instance`), dùng class weights |

Số liệu annotation lấy từ Dataset Audit nội bộ; số crop thực tế từng tập do `src/preprocessing.py` tạo và ghi trong `docs/dataset_report.md`. Phân tích đầy đủ: [`docs/class_scope_analysis.md`](docs/class_scope_analysis.md).

## 3. Quy trình học máy

```
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

```
ocean-plastic-ml-starter/
├── .github/
│   ├── ISSUE_TEMPLATE/task.md       # Mẫu issue duy nhất cho mọi task
│   ├── pull_request_template.md     # Mẫu PR bắt buộc
│   └── CODEOWNERS                   # Yến là reviewer bắt buộc
├── configs/
│   ├── data_contract.yaml           # Hợp đồng dữ liệu giữa các module
│   ├── class_mapping.json           # category_id → class_name → class_index (DATA tạo)
│   └── experiments/                 # E01_baseline.yaml, E02_augment.yaml, ...
├── data/                            # KHÔNG commit (chỉ .gitkeep)
│   ├── raw/                         # TrashCan gốc
│   ├── interim/                     # crop, manifest tạm
│   └── processed/                   # dữ liệu đã chia train/val/test
├── docs/
│   ├── problem_statement.md
│   ├── class_scope_analysis.md
│   ├── dataset_report.md
│   ├── evaluation_plan.md
│   ├── colab_setup.md
│   ├── demo_spec.md
│   ├── sprint_plan.md               # Phân công + timeline
│   ├── decisions/                   # ADR-xxx: quyết định đã chốt
│   └── reports/                     # Báo cáo sprint, báo cáo giữa kỳ
├── notebooks/                       # NN_<khu-vực>_<mô-tả>.ipynb, chạy trên Colab
├── src/                             # Code dùng chung, notebook chỉ gọi lại
│   ├── eda.py · preprocessing.py · train.py · evaluate.py · predict.py
├── tests/                           # test_<module>.py (pytest)
├── models/                          # KHÔNG commit trọng số (lưu Drive)
├── results/
│   ├── figures/                     # hình < 1 MB dùng cho báo cáo
│   └── metrics/                     # <EXP_ID>_<split>.json / .csv
├── scripts/github/                  # Script quản lý issue, label, milestone
├── main.py                          # CLI demo dự đoán
├── requirements.txt
├── CONTRIBUTING.md                  # Quy ước làm việc (bắt buộc đọc)
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

```
MyDrive/ocean-plastic-ml/
├── data/raw/trashcan/               # dataset gốc
├── data/processed/<DATA_VERSION>/   # ví dụ v1_crop224
└── runs/<EXP_ID>_<YYYYMMDD-HHMM>/   # config.json, history.csv, best.keras, metrics_val.json
```

Hướng dẫn chi tiết (mount Drive, clone repo bằng token trong Colab Secrets, push từ Colab): [`docs/colab_setup.md`](docs/colab_setup.md).

Chạy nhanh trên máy cá nhân:

```bash
git clone https://github.com/phanyen06ym-ien/ocean-plastic-ml-starter.git
cd ocean-plastic-ml-starter
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

## 6. Nhóm thực hiện

| Thành viên | GitHub | Vai trò | Khu vực |
|---|---|---|---|
| Phan Hải Yến | [@phanyen06ym-ien](https://github.com/phanyen06ym-ien) | Trưởng nhóm · Kiến trúc · Quản lý dự án · **Reviewer duy nhất** | `area:arch` |
| Hùng | [@tranngochung126hqv-stack](https://github.com/tranngochung126hqv-stack) | Dataset Audit · EDA · Tiền xử lý | `area:data` |
| Hoàng Bách | [@phannguyenhoangbach2](https://github.com/phannguyenhoangbach2) | Kiến trúc CNN · Huấn luyện | `area:cnn` |
| Nguyễn Hoàng Ân | *(cập nhật)* | Đánh giá · So sánh thực nghiệm · Phân tích lỗi | `area:eval` |
| Ngọc Nguyên | [@hiang0910](https://github.com/hiang0910) | Inference · Kiểm thử · Demo | `area:demo` |

Yến là **người phụ trách phụ** của mọi issue và là người review, phê duyệt, merge mọi Pull Request.

## 7. Lộ trình

| Sprint | Thời gian | Mục tiêu | Milestone |
|---|---|---|---|
| Sprint 1 — Foundation | 23/10 – 03/11/2026 | Bài toán, kiến trúc, ADR-001, khung code DATA/CNN/EVAL/DEMO | `Sprint 1 — Foundation` |
| Sprint 2 — Data & Baseline | 04/11/2026 | Dữ liệu đã xử lý, huấn luyện E01/E02, đánh giá Validation, demo với checkpoint thật | `Sprint 2 — Data & Baseline` |
| Sprint 3 — Improvement & Midterm | 05/11 – 18/11/2026 | Cải tiến mô hình, đánh giá Test, demo, báo cáo giữa kỳ | `Sprint 3 — Improvement & Midterm` |

Chi tiết phân công và các mốc: [`docs/sprint_plan.md`](docs/sprint_plan.md).

**Định hướng sau giữa kỳ:** phát hiện và định vị rác (Object Detection), cải thiện ảnh dưới nước, mô hình nhẹ cho robot thu gom.

## 8. Kết quả

Sprint 2 (05/11/2026, Colab T4, dữ liệu `v1_crop128`, Validation 1.782 crop). Nguồn: `results/metrics/`, phân tích: [`docs/evaluation_report.md`](docs/evaluation_report.md).

| EXP_ID | Mô tả | Macro F1 (val) | Accuracy (val) | Macro F1 nhóm nhựa (val) | Macro F1 (test) |
|---|---|---:|---:|---:|---:|
| E01 | CNN baseline + class weights | **0,076** | 0,219 | 0,000 | — |
| E02 | E01 + Data Augmentation | 0,065 | 0,161 | 0,017 | — |

Cả hai mô hình mới cao hơn mức ngẫu nhiên (≈ 0,045) một chút: mô hình học bối cảnh của từng video và phân bố lớp giữa Train và Validation lệch mạnh do chia theo video. Sprint 3 thử học chuyển giao (E03) và phiên bản chia dữ liệu mới. Tập Test chưa được dùng.

## 9. Trích dẫn

```bibtex
@article{hong2020trashcan,
  title   = {TrashCan: A Semantically-Segmented Dataset towards Visual Detection of Marine Debris},
  author  = {Hong, Jungseok and Fulton, Michael and Sattar, Junaed},
  journal = {arXiv preprint arXiv:2007.08097},
  year    = {2020}
}
```
