# models/

Trọng số các mô hình đã huấn luyện (E01–E04).

> **Không commit trọng số lên GitHub.** `.gitignore` chặn `*.keras`, `*.h5`, `*.pt`, `*.pth`; thư mục này trên GitHub chỉ có README.
> Bản gốc của mọi checkpoint nằm trên Google Drive của nhóm: `MyDrive/ocean-plastic-ml/runs/<EXP_ID>_<YYYYMMDD-HHMM>/`.

## Checkpoint của từng thí nghiệm

| EXP_ID | Mô hình | Dữ liệu | Run trên Drive | File | Cần kèm |
|---|---|---|---|---|---|
| E01 | CNN tự thiết kế (Keras) | `v1_crop128` | `runs/E01_20261105-1513/` | `best.keras` | — |
| E02 | E01 + augmentation (Keras) | `v1_crop128` | `runs/E02_20261105-1515/` | `best.keras` | — |
| E03 | MobileNetV2 (PyTorch) | `v1_crop128` | `runs/E03_20261108-1459/` (máy của Bách; bản chạy lại trên Drive: `runs/E03_20261112-1613/`) | `best.pt` (9,2 MB) | `config.json` cùng thư mục |
| E04 | EfficientNet-B0 (PyTorch), 18 lớp | `v2_crop224` | `runs/E04_20261121-1102/` | `best.pt` (16,4 MB) | `config.json`, `class_mapping.json` cùng thư mục |

Mỗi thư mục run còn có `history.csv`, `config.json`, `class_mapping.json` và `<EXP_ID>_val.json`.
Thứ tự lớp đầu ra: E01–E03 theo [`configs/class_mapping.json`](../configs/class_mapping.json) (22 lớp); E04 theo [`configs/class_mapping_v2.json`](../configs/class_mapping_v2.json) (18 lớp). `main.py` tự dùng `class_mapping.json` nằm cạnh checkpoint.

## Dùng checkpoint

Tải thư mục run từ Drive về `models/<EXP_ID>/` (giữ `config.json` cạnh `best.pt`), rồi:

```bash
# MobileNetV2 (E03–E04)
python main.py --image path/to/crop.jpg --model models/E04/best.pt

# CNN tự thiết kế (E01–E02)
python main.py --image path/to/crop.jpg --model models/E01/best.keras
```

Đánh giá lại một checkpoint PyTorch:

```bash
python -m src.evaluate --run models/E04 --ckpt best.pt --processed-dir data/processed/v2_crop224 --split val
```
