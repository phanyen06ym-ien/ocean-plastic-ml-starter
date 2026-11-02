# Ocean Plastic Waste Classification and Monitoring

**Phân loại và giám sát rác thải nhựa đại dương** — đồ án môn Học máy và Khai phá dữ liệu.

Dự án xây dựng mô hình mạng nơ-ron tích chập (CNN) phân loại **ảnh thực tế dưới nước** các đối tượng rác thải liên quan đến nhựa, sử dụng bộ dữ liệu **TrashCan 1.0 — phiên bản Instance**. Nhóm phân loại theo **loại đối tượng** (túi, chai, lưới, dây, bạt…), không phân loại theo chất liệu nhựa.

> **Trạng thái:** Sprint 1 — Foundation (đang thực hiện). Chưa có kết quả huấn luyện chính thức. Mọi số liệu trong README chỉ lấy từ Dataset Audit nội bộ và sẽ được cập nhật khi các PR liên quan được merge.

---

## 1. Bài toán

| Mục | Nội dung |
|---|---|
| Loại bài toán | Supervised multi-class image classification |
| Đầu vào | Ảnh RGB **một đối tượng** đã được cắt (crop) từ ảnh dưới nước theo annotation của TrashCan-Instance |
| Đầu ra | Tên lớp, `class_index` và vector điểm softmax |
| Mô hình | CNN baseline tự xây dựng (TensorFlow/Keras), sau đó so sánh với phương án cải tiến |
| Chỉ số chính | Macro F1-score (dữ liệu mất cân bằng); kèm Accuracy, Precision, Recall, Confusion Matrix |
| Ngoài phạm vi giữa kỳ | Object Detection, GAN phục hồi ảnh, video thời gian thực, điều khiển robot |

Chi tiết: [`docs/problem_statement.md`](docs/problem_statement.md).

## 2. Bộ dữ liệu TrashCan 1.0

- **7.212 ảnh** thu thập từ môi trường dưới nước (khung hình video), có annotation dạng COCO cho instance segmentation và bounding box.
- Hai phiên bản nhãn: **Instance** (theo loại đối tượng — nhóm sử dụng) và **Material** (theo chất liệu — chỉ dùng để đối chiếu khi cần).
- Nguồn: [UMN Data Repository](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7) · Bài báo: Hong, Fulton & Sattar, *TrashCan: A Semantically-Segmented Dataset towards Visual Detection of Marine Debris*, [arXiv:2007.08097](https://arxiv.org/abs/2007.08097).
- Dữ liệu **không được commit** lên GitHub; lưu trên Google Drive của nhóm (xem mục 5).

### 2.1. Nguyên tắc chọn nhãn

Nhóm khảo sát TrashCan-Instance và **chỉ giữ các nhãn rác liên quan đến nhựa có đủ số mẫu để huấn luyện**:

1. **Liên quan đến nhựa:** đối tượng thường được làm từ nhựa hoặc vật liệu tổng hợp. Tên nhãn là loại đối tượng, *không* phải bằng chứng chất liệu — báo cáo không tuyên bố một mẫu là nhựa khi chưa kiểm tra.
2. **Đủ mẫu:** số crop hợp lệ (sau khi lọc box quá nhỏ/lỗi) đạt ngưỡng tối thiểu do nhóm chốt.
3. **Đánh giá được:** sau khi chia lại Train/Validation/Test theo nhóm video, mỗi lớp phải có mẫu ở cả ba tập.

Lớp ứng viên theo Dataset Audit nội bộ (số annotation, train + val gốc):

| Nhãn TrashCan-Instance | Số annotation | Ghi chú |
|---|---:|---|
| `trash_bag` | 908 | Lớp lớn nhất trong nhóm ứng viên |
| `trash_net` | 127 | |
| `trash_bottle` | 126 | Chai có thể là nhựa hoặc thủy tinh |
| `trash_tarp` | 121 | |
| `trash_rope` | 117 | Không có mẫu ở val gốc → cần chia lại |
| `trash_snack_wrapper` | 84 | Có nguy cơ dưới ngưỡng |

> Danh sách lớp, ngưỡng mẫu tối thiểu và việc có thêm lớp `other` hay không **chưa chốt**. Quyết định cuối cùng được ghi tại `docs/decisions/ADR-001-class-scope.md` (Issue #4). Phân tích đầy đủ 22 lớp: [`docs/class_scope_analysis.md`](docs/class_scope_analysis.md).

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
| Sprint 1 — Foundation | 23/10 – 08/11/2026 | Bài toán, kiến trúc, Dataset Audit, khung code CNN/Eval/Demo | `Sprint 1 — Foundation` |
| Sprint 2 — Data & Baseline | 09/11 – 22/11/2026 | Chốt lớp, dữ liệu đã xử lý, huấn luyện E01/E02, đánh giá Validation | `Sprint 2 — Data & Baseline` |
| Sprint 3 — Improvement & Midterm | 23/11 – 06/12/2026 | Cải tiến mô hình, đánh giá Test, demo, báo cáo giữa kỳ | `Sprint 3 — Improvement & Midterm` |

Chi tiết phân công và các mốc: [`docs/sprint_plan.md`](docs/sprint_plan.md).

**Định hướng sau giữa kỳ:** phát hiện và định vị rác (Object Detection), cải thiện ảnh dưới nước, mô hình nhẹ cho robot thu gom.

## 8. Kết quả

Chưa có. Bảng kết quả sẽ được cập nhật từ `results/metrics/` sau khi các thí nghiệm được review và merge.

| EXP_ID | Mô tả | Macro F1 (val) | Accuracy (val) | Macro F1 (test) |
|---|---|---:|---:|---:|
| E01 | CNN baseline | — | — | — |
| E02 | E01 + Data Augmentation | — | — | — |

## 9. Trích dẫn

```bibtex
@article{hong2020trashcan,
  title   = {TrashCan: A Semantically-Segmented Dataset towards Visual Detection of Marine Debris},
  author  = {Hong, Jungseok and Fulton, Michael and Sattar, Junaed},
  journal = {arXiv preprint arXiv:2007.08097},
  year    = {2020}
}
```
