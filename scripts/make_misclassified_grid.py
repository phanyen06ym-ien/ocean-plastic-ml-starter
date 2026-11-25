"""Lưới ảnh bị đoán sai (nhóm nhựa) cho báo cáo/slide, kèm độ tin cậy.

Chạy trên Colab sau khi giải nén dữ liệu v2 và có checkpoint E04:

    python scripts/make_misclassified_grid.py \
        --predictions results/predictions/E04_test_predictions.csv \
        --processed-dir /content/processed/v2_crop224 \
        --checkpoint /content/drive/MyDrive/ocean-plastic-ml/runs/E04_20261122-0853/best.pt \
        --out results/figures/E04_test_misclassified_plastic.png

Chọn các mẫu nhựa bị đoán sai có độ tin cậy cao nhất, tối đa `--per-class` ảnh mỗi lớp thật.
Ghi thêm <out>.csv (path, nhãn thật, nhãn đoán, độ tin cậy) để tra cứu.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def select_errors(pred: pd.DataFrame, mapping: list[dict], per_class: int, limit: int, conf_fn) -> pd.DataFrame:
    """Lỗi của các lớp nhựa, sắp theo độ tin cậy giảm dần; conf_fn(path) -> confidence của nhãn đoán."""
    plastic = {m["class_index"] for m in mapping if m.get("plastic_related")}
    err = pred[(pred["y_true"] != pred["y_pred"]) & pred["y_true"].isin(plastic)].copy()
    err["confidence"] = [conf_fn(p) for p in err["path"]]
    err = err.sort_values("confidence", ascending=False).groupby("y_true", group_keys=False).head(per_class)
    names = {m["class_index"]: m["class_name"] for m in mapping}
    err["true_name"] = err["y_true"].map(names)
    err["pred_name"] = err["y_pred"].map(names)
    return err.sort_values(["true_name", "confidence"], ascending=[True, False]).head(limit)


def draw(err: pd.DataFrame, processed_dir: Path, out: Path, cols: int = 4) -> Path:
    """Vẽ lưới ảnh lỗi (thật / đoán / độ tin cậy) ra `out` và ghi danh sách kèm theo ra <out>.csv."""
    rows = max(1, -(-len(err) // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows), layout="constrained")
    for ax in getattr(axes, "flat", [axes]):
        ax.axis("off")
    for ax, r in zip(getattr(axes, "flat", [axes]), err.itertuples()):
        with Image.open(processed_dir / r.path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"Thật: {r.true_name.replace('trash_', '')}\nĐoán: {r.pred_name} ({r.confidence:.0%})",
                     fontsize=8, color="#b91c1c")
    fig.suptitle("E04 — mẫu nhựa bị đoán sai trên Test (độ tin cậy cao nhất mỗi lớp)", fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    err[["path", "true_name", "pred_name", "confidence"]].to_csv(out.with_suffix(".csv"), index=False)
    return out


def main() -> None:
    """Dòng lệnh: nạp checkpoint, chọn lỗi nhóm nhựa có độ tin cậy cao nhất và vẽ lưới ảnh."""
    from src.predict import Predictor
    from src.preprocessing import ensure_utf8_stdio

    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", default=str(ROOT / "results" / "predictions" / "E04_test_predictions.csv"))
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "E04_test_misclassified_plastic.png"))
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--limit", type=int, default=16)
    a = ap.parse_args()
    proc = Path(a.processed_dir)
    predictor = Predictor(a.checkpoint)
    names = {m["class_index"]: m["class_name"] for m in predictor.mapping}
    pred = pd.read_csv(a.predictions)

    def conf(path: str) -> float:
        """Xác suất mô hình gán cho nhãn đã đoán (y_pred trong file dự đoán) của ảnh `path`."""
        top = predictor.predict_path(proc / path, k=len(names))["top_k"]
        y_pred = int(pred.loc[pred["path"] == path, "y_pred"].iloc[0])
        return next(p["confidence"] for p in top if p["class_index"] == y_pred)

    err = select_errors(pred, predictor.mapping, a.per_class, a.limit, conf)
    print(draw(err, proc, Path(a.out)))
    print(json.dumps(err[["true_name", "pred_name", "confidence"]].round(3).to_dict(orient="records"), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
