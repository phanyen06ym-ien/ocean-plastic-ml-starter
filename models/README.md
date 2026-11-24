# models/

Trọng số các mô hình đã huấn luyện (E01–E04).

> **Không commit trọng số lên GitHub.** `.gitignore` chặn `*.keras`, `*.h5`, `*.pt`, `*.pth`; thư mục này trên GitHub chỉ có README.
> Bản gốc của mọi checkpoint nằm trên Google Drive của nhóm: `MyDrive/ocean-plastic-ml/runs/<EXP_ID>_<YYYYMMDD-HHMM>/`.

## Checkpoint của từng thí nghiệm

| EXP_ID | Mô hình | Dữ liệu | Run trên Drive | File | Cần kèm |
|---|---|---|---|---|---|
| E01 | CNN tự thiết kế (Keras) | `v1_crop128` | `runs/E01_20261105-1513/` | `best.keras` | — |
| E02 | E01 + augmentation (Keras) | `v1_crop128` | `runs/E02_20261105-1515/` | `best.keras` | — |
| E03 | MobileNetV2 (PyTorch) | `v1_crop128` | `runs/E03_20261108-1459/` (máy của Bách; **không có trên Drive**, thư mục `runs/E03_20261112-1613/` trống) | `best.pt` (9,2 MB) | `config.json` cùng thư mục |
| E04 | EfficientNet-B0 (PyTorch), 18 lớp, lấy mẫu 1/số mẫu | `v2_crop224` | `runs/E04_20261122-0853/` | `best.pt` (16,4 MB) | `config.json`, `class_mapping.json` cùng thư mục |
| E04 bản đầu (thay thế 22/11) | như trên, lấy mẫu √(1/số mẫu) | `v2_crop224` | `runs/E04_20261121-1102/` | `best.pt` | chỉ để đối chiếu với `results/metrics/E04v1_*` |

Mỗi thư mục run còn có `history.csv`, `config.json`, `class_mapping.json`, `<EXP_ID>_val.json` và file dự đoán (`E04_test_predictions.csv` với hai run E04). Bản sao các file dự đoán nằm trong [`results/predictions/`](../results/predictions/).

Các run thử nghiệm, run bị ngắt và thư mục `exports/` cũ đã được dọn khỏi Drive ngày 24/11/2026.
Thứ tự lớp đầu ra: E01–E03 theo [`configs/class_mapping.json`](../configs/class_mapping.json) (22 lớp); E04 theo [`configs/class_mapping_v2.json`](../configs/class_mapping_v2.json) (18 lớp). `main.py` tự dùng `class_mapping.json` nằm cạnh checkpoint.

## Dùng checkpoint

Tải thư mục run từ Drive về `runs/<tên run>/` trong repo (giữ `config.json` và `class_mapping.json` cạnh checkpoint; `runs/` đã có trong `.gitignore`), rồi:

```bash
# PyTorch (E03 MobileNetV2, E04 EfficientNet-B0)
python main.py --image path/to/crop.jpg --model runs/E04_20261122-0853/best.pt

# CNN tự thiết kế (E01–E02)
python main.py --image path/to/crop.jpg --model runs/E01_20261105-1513/best.keras
```

Đánh giá lại một checkpoint PyTorch:

```bash
python -m src.evaluate --run runs/E04_20261122-0853 --ckpt best.pt --processed-dir data/processed/v2_crop224 --split val
```
