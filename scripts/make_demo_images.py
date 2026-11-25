"""Chọn 5 ảnh cho buổi demo: 3 ảnh mô hình E04 đoán đúng, 2 ảnh đoán sai (lỗi điển hình của nhóm nhựa).

Ảnh lấy từ tập Kiểm tra của v2_crop224 dựa trên results/predictions/E04_test_predictions.csv. Nếu có checkpoint,
mô hình được chạy lại để lấy độ tin cậy và chọn ảnh rõ nhất trong mỗi nhóm (đúng: tin cậy cao nhất; sai: mô hình
tự tin nhất vào nhãn sai). Kết quả chép vào demo_images/ kèm README.md mô tả từng ảnh.

Chạy:
    python scripts/make_demo_images.py
    python scripts/make_demo_images.py --processed-dir data/processed/v2_crop224 --model runs/E04_20261122-0853/best.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREDICTIONS = ROOT / "results" / "predictions" / "E04_test_predictions.csv"
MAPPING = ROOT / "configs" / "class_mapping_v2.json"
PROCESSED = ROOT / "data" / "processed" / "v2_crop224"
MODEL = ROOT / "runs" / "E04_20261122-0853" / "best.pt"
OUT = ROOT / "demo_images"

# (tên file, lớp thật, lớp dự đoán mong đợi, ghi chú cho người thuyết trình)
PLAN = [
    ("1_dung_sao_bien.jpg", "animal_starfish", "animal_starfish", "Lớp có F1 cao nhất (0,89): hình dạng đặc trưng."),
    ("2_dung_tui_nilon.jpg", "trash_bag", "trash_bag", "Lớp nhựa mô hình nhận dạng tốt nhất (F1 0,34)."),
    ("3_dung_lon.jpg", "trash_can", "trash_can", "Rác kim loại, hình dạng rõ (F1 0,49)."),
    ("4_sai_vo_bao_bi_thanh_tui_nilon.jpg", "trash_snack_wrapper", "trash_bag",
     "Cả 10 ảnh vỏ bao bì trong tập Kiểm tra đều bị đoán thành túi nilon: hai lớp cùng là màng nhựa mỏng, sáng màu."),
    ("5_sai_chai_thanh_rac_khac.jpg", "trash_bottle", "trash_other",
     "Lỗi phổ biến nhất của nhóm nhựa: lớp gộp trash_other hút nhầm (11/17 ảnh chai)."),
]


def load_names(mapping: Path = MAPPING) -> dict[int, str]:
    """class_index → class_name của v2 (18 lớp)."""
    data = json.loads(mapping.read_text(encoding="utf-8"))
    return {c["class_index"]: c["class_name"] for c in data["classes"]}


def candidates(pred_csv: Path = PREDICTIONS, mapping: Path = MAPPING) -> pd.DataFrame:
    """Bảng dự đoán Test của E04 với tên lớp thật/dự đoán."""
    df = pd.read_csv(pred_csv)
    names = load_names(mapping)
    df["true"] = df["y_true"].map(names)
    df["pred"] = df["y_pred"].map(names)
    return df


def choose(df: pd.DataFrame, true: str, pred: str, scores: dict[str, float] | None = None) -> str:
    """Đường dẫn (tương đối) của ảnh có lớp thật `true` và dự đoán `pred`; ưu tiên độ tin cậy cao nhất nếu có."""
    rows = df[(df["true"] == true) & (df["pred"] == pred)]
    if rows.empty:
        raise SystemExit(f"Không có ảnh Test nào: thật={true}, dự đoán={pred}")
    paths = sorted(rows["path"])
    if scores:
        paths = sorted(paths, key=lambda p: -scores.get(p, 0.0))
    return paths[0]


def main(argv: list[str] | None = None) -> None:
    """Chọn, xác nhận bằng mô hình (nếu có) và chép 5 ảnh demo vào demo_images/."""
    ap = argparse.ArgumentParser(description="Chọn 5 ảnh demo (3 đúng, 2 sai) từ tập Kiểm tra của E04")
    ap.add_argument("--processed-dir", default=str(PROCESSED))
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    processed, out, model = Path(args.processed_dir), Path(args.out), Path(args.model)
    if not processed.exists():
        raise SystemExit(f"Không thấy {processed}. Tải v2_crop224.zip từ Drive (data/processed/) và giải nén vào data/processed/.")

    df = candidates()
    predictor = None
    if model.exists():
        from src.predict import Predictor
        predictor = Predictor(model)
    else:
        print(f"Không thấy checkpoint {model}: chọn theo thứ tự tên file, không có độ tin cậy.")

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fname, true, pred, note in PLAN:
        scores = None
        if predictor is not None:   # chấm điểm mọi ứng viên, giữ ảnh mô hình tự tin nhất
            pool = df[(df["true"] == true) & (df["pred"] == pred)]["path"]
            scores = {p: predictor.predict_image(_open(processed / p), k=1)[0].confidence for p in pool}
        rel = choose(df, true, pred, scores)
        shutil.copy(processed / rel, out / fname)
        conf = ""
        if predictor is not None:
            top = predictor.predict_image(_open(out / fname), k=1)[0]
            if top.class_name != pred:
                raise SystemExit(f"{fname}: mô hình dự đoán {top.class_name}, khác {pred} trong file dự đoán đã lưu")
            conf = f"{top.confidence * 100:.1f}".replace(".", ",") + "%"
        rows.append((fname, true, pred, conf, rel, note))
        print(f"{'ĐÚNG' if true == pred else 'SAI '}  {fname:40s} thật={true:20s} đoán={pred:16s} {conf}")

    lines = ["# Ảnh demo (tập Kiểm tra, mô hình E04)", "",
             "Tạo bằng `python scripts/make_demo_images.py`. Ba ảnh đầu mô hình đoán đúng, hai ảnh sau đoán sai.", "",
             "| Ảnh | Lớp thật | Mô hình đoán | Độ tin cậy | Ảnh gốc | Ghi chú |", "|---|---|---|---|---|---|"]
    lines += [f"| `{f}` | {t} | {p} | {c or '—'} | `{r}` | {n} |" for f, t, p, c, r, n in rows]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã chép {len(rows)} ảnh vào {out}")


def _open(path: Path):
    from src.predict import load_image
    return load_image(path)


if __name__ == "__main__":
    main()
