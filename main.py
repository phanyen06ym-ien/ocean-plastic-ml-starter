"""CLI demo: phân loại ảnh một đối tượng dưới nước (TrashCan-Instance, 22 lớp).

Hỗ trợ checkpoint Keras (E01–E02, CNN tự thiết kế) và PyTorch (E03–E04, MobileNetV2).
Ảnh đầu vào là ảnh đã cắt quanh một đối tượng (crop), giống dữ liệu lúc huấn luyện.

Ví dụ:
    # Một ảnh, mô hình MobileNetV2 (config.json nằm cùng thư mục run)
    python main.py --image path/to/crop.jpg --model runs/E03_20261108-1459/best.pt

    # Một ảnh, mô hình Keras
    python main.py --image path/to/crop.jpg --model runs/E01_20261105-1513/best.keras

    # Cả thư mục ảnh, ghi kết quả ra CSV
    python main.py --folder path/to/crops --model runs/E03_20261108-1459/best.pt --out predictions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing import DEFAULT_MAPPING
from src.predict import InferenceError, Predictor

PLASTIC_TAG = " [liên quan nhựa]"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python main.py",
        description="Phân loại ảnh một đối tượng dưới nước (TrashCan-Instance, 22 lớp) bằng CNN / MobileNetV2.",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Đường dẫn một ảnh JPG/PNG/BMP")
    src.add_argument("--folder", help="Thư mục ảnh để dự đoán hàng loạt (đọc cả thư mục con)")
    ap.add_argument("--model", required=True,
                    help="Checkpoint .keras/.h5 (E01–E02) hoặc .pt/.pth (E03–E04)")
    ap.add_argument("--config", default=None,
                    help="config.json của run, chỉ cần cho .pt/.pth khi không nằm cùng thư mục checkpoint")
    ap.add_argument("--mapping", default=str(DEFAULT_MAPPING), help="class_mapping.json (mặc định: configs/)")
    ap.add_argument("--top-k", type=int, default=3, help="Số lớp có xác suất cao nhất cần in (mặc định 3)")
    ap.add_argument("--device", default=None, help="cpu hoặc cuda (chỉ dùng cho PyTorch; mặc định tự chọn)")
    ap.add_argument("--out", help="Ghi kết quả --folder ra file CSV")
    return ap


def print_single(result: dict) -> None:
    print(f"Ảnh: {result['image']}")
    for i, p in enumerate(result["top_k"], 1):
        tag = PLASTIC_TAG if p["plastic_related"] else ""
        print(f"  {i}. {p['class_name']:<24} {p['confidence']:6.1%}{tag}")


def print_folder(rows: list[dict]) -> None:
    for r in rows:
        t = r["top1"]
        tag = PLASTIC_TAG if t["plastic_related"] else ""
        print(f"{r['image']}\t{t['class_name']}\t{t['confidence']:.3f}{tag}")
    n_plastic = sum(r["top1"]["plastic_related"] for r in rows)
    print(f"Tổng: {len(rows)} ảnh, {n_plastic} ảnh được dự đoán thuộc nhóm liên quan nhựa")


def write_csv(rows: list[dict], out: str | Path) -> None:
    import pandas as pd

    records = []
    for r in rows:
        rec = {"image": r["image"]}
        for i, p in enumerate(r["top_k"], 1):
            rec[f"top{i}_class_name"] = p["class_name"]
            rec[f"top{i}_class_index"] = p["class_index"]
            rec[f"top{i}_confidence"] = round(p["confidence"], 6)
        rec["top1_plastic_related"] = r["top1"]["plastic_related"]
        records.append(rec)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False, encoding="utf-8")
    print(f"Đã ghi {out}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.top_k < 1:
            raise InferenceError("--top-k phải lớn hơn hoặc bằng 1")
        if args.out and not args.folder:
            raise InferenceError("--out chỉ dùng cùng --folder")
        predictor = Predictor(args.model, args.mapping, config=args.config, device=args.device)
        k = min(args.top_k, len(predictor.mapping))
        if args.image:
            print_single(predictor.predict_path(args.image, k))
        else:
            rows = predictor.predict_folder(args.folder, k)
            print_folder(rows)
            if args.out:
                write_csv(rows, args.out)
    except InferenceError as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())