# Đặc tả demo dự đoán (S1/S2-DEMO)

Tài liệu này mô tả hành vi **hiện tại** của `src/predict.py` và `main.py`. Khi code thay đổi phải cập nhật tài liệu này trong cùng PR.

| Mục | Nội dung |
|---|---|
| Phụ trách | Ngọc Nguyên (dự phòng: Yến) |
| Code | `src/predict.py` (logic), `main.py` (CLI) |
| Test | `tests/test_predict.py` (dùng mô hình giả, không cần checkpoint thật) |
| Căn cứ | ADR-001, `configs/data_contract.yaml` v1 |

## 1. Phạm vi

- Đầu vào là **ảnh của một đối tượng đã được crop sẵn** (giống ảnh trong `data/processed/<DATA_VERSION>/<split>/<class_name>/`). Demo không phát hiện hay định vị đối tượng trong ảnh toàn cảnh. Đưa vào ảnh toàn cảnh thì vẫn có kết quả, nhưng kết quả đó không có ý nghĩa.
- Đầu ra là lớp TrashCan-Instance (22 lớp) kèm confidence. Cờ `plastic_related` chỉ cho biết lớp đó thuộc nhóm 6 lớp *liên quan nhựa* (ADR-001 D2); đây **không** phải kết luận về chất liệu.
- Ngoài phạm vi: video, thời gian thực, giao diện web (để Sprint 3).

## 2. Luồng xử lý

```
ảnh (.jpg/.jpeg/.png/.bmp)
  → load_image      : kiểm tra tồn tại, đuôi file, đọc được; chuyển về RGB (ảnh xám / RGBA / palette đều được)
  → preprocess      : resize thẳng (bilinear) về input_size × input_size, float32, giá trị 0..255, shape (1, H, W, 3)
  → model.predict   : lớp Rescaling(1/255) NẰM TRONG mô hình → không chia 255 ở ngoài
  → kiểm tra softmax: tổng xác suất ≈ 1 (sai số 1e-3)
  → top_k           : sắp xếp giảm dần, gắn class_name / plastic_related theo class mapping
```

- `input_size` được đọc từ shape đầu vào của checkpoint (128 theo data contract v1), không viết cứng trong code.
- Thứ tự lớp được lấy từ `class_mapping.json`, sắp theo `class_index` (0–21, `category_id` tăng dần). Khi nạp, code kiểm tra số đầu ra của mô hình phải bằng số lớp trong mapping.

## 3. Đầu vào

| Tham số CLI | Bắt buộc | Mặc định | Ý nghĩa |
|---|---|---|---|
| `--image <path>` | một trong hai | — | Một ảnh |
| `--folder <dir>` | một trong hai | — | Thư mục ảnh; quét **đệ quy**, lấy các file có đuôi được hỗ trợ, sắp theo tên |
| `--model <path>` | có | — | Checkpoint `.keras` (`runs/<EXP_ID>_<YYYYMMDD-HHMM>/best.keras` trên Drive) |
| `--mapping <path>` | không | `configs/class_mapping.json` | Class mapping; nên dùng `class_mapping.json` nằm cùng run để chắc chắn khớp với checkpoint |
| `--top-k <int>` | không | `3` | Số lớp in ra; phải ≥ 1 (xem mục 6) |
| `--out <path>` | không | — | Chỉ dùng với `--folder`: ghi CSV kết quả top-1 |

Định dạng ảnh hỗ trợ: `.jpg`, `.jpeg`, `.png`, `.bmp` (không phân biệt hoa thường).

## 4. Đầu ra

### 4.1. CLI `--image`

```
Ảnh: <đường dẫn>
  1. <class_name>               <confidence %> [liên quan nhựa]
  2. ...
```
Nhãn `[liên quan nhựa]` chỉ hiện khi `plastic_related = true`.

### 4.2. CLI `--folder`

Mỗi ảnh in một dòng: `<đường dẫn>\t<top1 class_name>\t<top1 confidence, 3 chữ số thập phân>`.
Có `--out` thì ghi thêm CSV với các cột:

| Cột | Kiểu |
|---|---|
| `image` | đường dẫn ảnh |
| `top1_class_index` | int 0–21 |
| `top1_class_name` | str |
| `top1_confidence` | float 0–1 |
| `top1_plastic_related` | bool |

### 4.3. API Python (notebook, giao diện Sprint 3)

```python
from src.predict import Predictor
pred = Predictor("runs/E01_.../best.keras", "runs/E01_.../class_mapping.json")
r = pred.predict_path("crop.jpg", k=3)
# r = {"image": "crop.jpg",
#      "top1":  {"class_index": 11, "class_name": "trash_bag", "confidence": 0.81, "plastic_related": True},
#      "top_k": [ {...}, {...}, {...} ]}           # các giá trị trên chỉ minh hoạ định dạng
rows = pred.predict_folder("crops/", k=3)          # list các dict như trên
preds = pred.predict_image(pil_image, k=3)         # list[Prediction] (dataclass, cùng 4 trường)
```

Mô hình chỉ được nạp **một lần** trong `Predictor.__init__`. Nếu đã có mô hình trong bộ nhớ (ví dụ trong notebook), truyền vào bằng `Predictor(None, mapping, model=model)`.

## 5. Lỗi và mã thoát

`main.py` bắt `InferenceError`, in `Lỗi: <thông báo>` ra **stderr** rồi thoát với **mã 1**. Thành công thì thoát với mã 0. Nếu sai cú pháp tham số, argparse in hướng dẫn sử dụng và thoát với mã 2.

| Tình huống | Thông báo (stderr) | Mã |
|---|---|---|
| File ảnh không tồn tại | `Lỗi: Không tìm thấy file ảnh: <path>` | 1 |
| Đuôi file không hỗ trợ | `Lỗi: Định dạng không hỗ trợ (<ext>). Dùng: .bmp, .jpeg, .jpg, .png` | 1 |
| File có đuôi ảnh nhưng hỏng / không phải ảnh | `Lỗi: File không phải ảnh hợp lệ: <path> (<chi tiết PIL>)` | 1 |
| Checkpoint không tồn tại | `Lỗi: Không tìm thấy checkpoint: <path>` | 1 |
| Số đầu ra mô hình ≠ số lớp trong mapping | `Lỗi: Mô hình có <n> đầu ra nhưng class mapping có <m> lớp` | 1 |
| Đầu ra mô hình không phải softmax | `Lỗi: Đầu ra mô hình không phải phân phối softmax` | 1 |
| `--folder` không có ảnh nào (kể cả thư mục không tồn tại) | `Lỗi: Thư mục không có ảnh: <dir>` | 1 |
| Thiếu `--model`, hoặc dùng cùng lúc `--image` và `--folder` | thông báo `usage` của argparse | 2 |

## 6. Hạn chế đã biết (kiểm chứng ngày 04/11/2026 trên nhánh `feat/s2-pipeline-end-to-end` @ daf3846)

Các trường hợp dưới đây **chưa** được chuyển thành `InferenceError`, nên người dùng sẽ thấy traceback Python. Đề xuất cho S2-DEMO: bắt thêm các lỗi này và thêm test tương ứng.

| Tình huống | Hành vi hiện tại |
|---|---|
| File mapping không tồn tại (gồm cả mặc định `configs/class_mapping.json` khi DATA chưa commit file này) | Traceback `FileNotFoundError` |
| File mapping sai cấu trúc / `class_index` không liên tục | Traceback `KeyError` / `ValueError` |
| `--model` trỏ tới file tồn tại nhưng không phải `.keras` / `.h5` hợp lệ | Traceback `ValueError` của Keras |
| `--top-k 0` hoặc số âm | Traceback `IndexError` |
| `--folder` có một ảnh hỏng | Dừng cả lượt chạy ở ảnh đó, các ảnh khác không có kết quả |
| `--out` dùng với `--image` | Bị bỏ qua, không có cảnh báo |

`configs/class_mapping.json` chưa có trên `main`; file này được sinh bởi `python -m src.preprocessing ... --mapping-out configs/class_mapping.json` (S2-DATA). Từ giờ đến lúc đó, luôn truyền `--mapping runs/<run>/class_mapping.json`.

## 7. Cách chạy

### 7.1. Máy cá nhân

```bash
pip install -r requirements.txt
python main.py --image path/to/crop.jpg \
               --model runs/E01_YYYYMMDD-HHMM/best.keras \
               --mapping runs/E01_YYYYMMDD-HHMM/class_mapping.json
python main.py --folder data/processed/v1_crop128/val/trash_bag \
               --model runs/E01_YYYYMMDD-HHMM/best.keras \
               --mapping runs/E01_YYYYMMDD-HHMM/class_mapping.json \
               --top-k 3 --out predictions.csv
```

Tải `best.keras` và `class_mapping.json` của run cần demo từ Drive `MyDrive/ocean-plastic-ml/runs/<run>/`. **Không commit** checkpoint hay file `predictions.csv`.

### 7.2. Google Colab

Sau cell cài đặt của notebook (mount Drive, `os.chdir(REPO)`):

```python
RUN = DRIVE / "runs/E01_YYYYMMDD-HHMM"
!python main.py --image /content/processed/v1_crop128/val/trash_bag/<file>.jpg --model {RUN}/best.keras --mapping {RUN}/class_mapping.json
```

Hoặc dùng trực tiếp API `Predictor` như trong mục 9 của `notebooks/00_pipeline_colab.ipynb`.

### 7.3. Kiểm thử

```bash
pytest -q tests/test_predict.py
```

Các test dùng mô hình giả để kiểm tra lỗi đầu vào, shape/khoảng giá trị sau tiền xử lý (ảnh RGB, L, RGBA), thứ tự top-k theo mapping, lỗi lệch số lớp, lỗi thiếu checkpoint, và mã thoát của CLI khi gặp lỗi.

## 8. Tiêu chí demo với checkpoint thật (S2-DEMO, mốc 15/11)

- [ ] Chạy `--image` và `--folder` bằng checkpoint E01 thật trên ảnh **Validation** (không dùng Test).
- [ ] Ghi đường dẫn run trên Drive và `DATA_VERSION` vào PR.
- [ ] Ảnh chụp output đặt trong PR; không ghi số liệu vào tài liệu khi chưa có `results/metrics/` tương ứng được merge.
