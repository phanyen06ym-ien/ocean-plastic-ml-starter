"""CLI demo dự đoán đối tượng dưới nước (TrashCan-Instance, 22 lớp).

Ví dụ:
    python main.py --image path/to/crop.jpg --model runs/E01_x/best.keras
    python main.py --folder path/to/crops --model runs/E01_x/best.keras --out predictions.csv
"""
from __future__ import annotations

import argparse
import sys

from src.common import DEFAULT_MAPPING
from src.predict import InferenceError, Predictor


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phân loại ảnh một đối tượng dưới nước bằng CNN")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Đường dẫn một ảnh JPG/PNG")
    src.add_argument("--folder", help="Thư mục ảnh để dự đoán hàng loạt")
    ap.add_argument("--model", required=True, help="Checkpoint .keras (vd runs/E01_*/best.keras)")
    ap.add_argument("--mapping", default=str(DEFAULT_MAPPING), help="class_mapping.json")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out", help="Ghi kết quả --folder ra CSV")
    args = ap.parse_args(argv)

    try:
        predictor = Predictor(args.model, args.mapping)
        if args.image:
            r = predictor.predict_path(args.image, args.top_k)
            print(f"Ảnh: {r['image']}")
            for i, p in enumerate(r["top_k"], 1):
                tag = " [liên quan nhựa]" if p["plastic_related"] else ""
                print(f"  {i}. {p['class_name']:<24} {p['confidence']:6.1%}{tag}")
        else:
            rows = predictor.predict_folder(args.folder, args.top_k)
            for r in rows:
                print(f"{r['image']}\t{r['top1']['class_name']}\t{r['top1']['confidence']:.3f}")
            if args.out:
                import pandas as pd
                pd.DataFrame([{"image": r["image"], **{f"top1_{k}": v for k, v in r["top1"].items()}} for r in rows]
                             ).to_csv(args.out, index=False)
                print(f"Đã ghi {args.out}")
    except InferenceError as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
