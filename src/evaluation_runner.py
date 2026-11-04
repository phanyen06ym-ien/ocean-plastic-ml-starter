"""Chạy đánh giá một checkpoint trên một tập (S2-EVAL), dùng các hàm của src/evaluate.py (Ân).

Kết quả theo quy ước tên file:
    results/metrics/<EXP_ID>_<split>.json
    results/metrics/<EXP_ID>_<split>_predictions.csv
    results/figures/<EXP_ID>_<split>_confusion_matrix.png
    results/figures/<EXP_ID>_learning_curves.png
    results/figures/<EXP_ID>_<split>_misclassified.png

Báo cáo hai cấp theo ADR-001: tổng thể 22 lớp và nhóm lớp liên quan nhựa.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

try:
    from src import evaluate as ev
    from src.common import class_names, load_class_mapping, plastic_indices
except ImportError:  # pragma: no cover
    import evaluate as ev
    from common import class_names, load_class_mapping, plastic_indices


def group_metrics(y_true, y_pred, indices: list[int]) -> dict[str, Any]:
    """Macro P/R/F1 chỉ trên các lớp trong `indices` (mô hình vẫn dự đoán 22 lớp).

    Lớp không có mẫu thật trong tập đánh giá bị loại khỏi trung bình và được liệt kê.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = [i for i in indices if (y_true == i).any()]
    if not present:
        return {"classes": indices, "classes_evaluated": [], "macro_precision": None, "macro_recall": None, "macro_f1": None}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=present, zero_division=0)
    return {
        "classes": indices,
        "classes_evaluated": present,
        "classes_without_samples": [i for i in indices if i not in present],
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "support": int(s.sum()),
    }


def summarize(y_true, y_pred, probs, mapping: list[dict[str, Any]], exp_id: str, split: str) -> dict[str, Any]:
    names = class_names(mapping)
    labels = list(range(len(names)))
    overall = ev.compute_classification_metrics(y_true, y_pred, labels=labels, target_names=names)
    present = sorted(set(np.asarray(y_true).tolist()))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, zero_division=0)
    return {
        "exp_id": exp_id,
        "split": split,
        "num_samples": int(len(y_true)),
        "num_classes": len(names),
        "overall": overall,
        "macro_f1_present_classes": float(f.mean()),   # chỉ tính lớp có mẫu trong tập này
        "plastic_related": group_metrics(y_true, y_pred, plastic_indices(mapping)),
        "note": "Kết quả trên TrashCan-Instance; lớp có support nhỏ cần diễn giải thận trọng.",
    }


def predict_split(model, processed_dir: str | Path, split: str, mapping: list[dict[str, Any]],
                  input_size: int = 128, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Dự đoán toàn bộ ảnh của một tập theo thứ tự cố định. Trả (y_true, probs, file_paths)."""
    import tensorflow as tf
    names = class_names(mapping)
    ds = tf.keras.utils.image_dataset_from_directory(
        str(Path(processed_dir) / split), labels="inferred", label_mode="int", class_names=names, color_mode="rgb",
        image_size=(input_size, input_size), batch_size=batch_size, shuffle=False)
    paths = list(ds.file_paths)
    y_true = np.concatenate([y.numpy() for _, y in ds]) if paths else np.array([], dtype=int)
    probs = model.predict(ds, verbose=0) if paths else np.zeros((0, len(names)))
    return y_true, probs, paths


def evaluate_run(run_dir: str | Path, processed_dir: str | Path, split: str = "val",
                 results_dir: str | Path = "results", checkpoint: str = "best.keras") -> dict[str, Any]:
    """Đánh giá checkpoint trong run_dir trên `split`; ghi file kết quả; trả dict chỉ số."""
    import tensorflow as tf
    if split == "test":
        print("[CHÚ Ý] Đang đánh giá tập TEST — chỉ chạy MỘT lần cho mô hình cuối (ADR mô hình cuối).")
    run_dir, results_dir = Path(run_dir), Path(results_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    exp_id = cfg["exp_id"]
    mapping = load_class_mapping(run_dir / "class_mapping.json")
    names = class_names(mapping)
    model = tf.keras.models.load_model(str(run_dir / checkpoint))

    y_true, probs, paths = predict_split(model, processed_dir, split, mapping, cfg["input_shape"][0])
    y_pred = probs.argmax(axis=1)
    result = summarize(y_true, y_pred, probs, mapping, exp_id, split)
    result.update({"run_dir": str(run_dir), "checkpoint": checkpoint, "data_version": cfg.get("data_version")})

    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics" / f"{exp_id}_{split}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    pd.DataFrame({"file_path": [Path(p).name for p in paths], "y_true": y_true, "y_pred": y_pred,
                  "true_name": [names[i] for i in y_true], "pred_name": [names[i] for i in y_pred],
                  "confidence": probs.max(axis=1) if len(probs) else []}
                 ).to_csv(results_dir / "metrics" / f"{exp_id}_{split}_predictions.csv", index=False)

    cm = ev.compute_confusion_matrix(y_true, y_pred, labels=list(range(len(names))), normalize="true")
    ev.plot_confusion_matrix(np.nan_to_num(cm), target_names=names, normalize=True, figsize=(14, 12),
                             title=f"{exp_id} — {split} (normalized)",
                             save_path=str(results_dir / "figures" / f"{exp_id}_{split}_confusion_matrix.png"))
    hist_path = run_dir / "history.csv"
    if hist_path.exists():
        ev.plot_learning_curves(pd.read_csv(hist_path).to_dict(orient="list"), title_prefix=exp_id,
                                save_path=str(results_dir / "figures" / f"{exp_id}_learning_curves.png"))
    wrong = ev.identify_misclassified_samples(y_true, y_pred, probabilities=probs, target_names=names)
    if wrong:
        from PIL import Image
        wrong = sorted(wrong, key=lambda w: -w.get("confidence", 0))[:16]
        imgs = [np.asarray(Image.open(paths[w["sample_index"]]).convert("RGB")) for w in wrong]
        ev.plot_misclassified_samples(imgs, [{**w, "sample_index": i} for i, w in enumerate(wrong)], target_names=names,
                                      max_samples=16, figsize=(12, 12),
                                      save_path=str(results_dir / "figures" / f"{exp_id}_{split}_misclassified.png"))
    return result


def comparison_table(metric_files: list[str | Path]) -> pd.DataFrame:
    """Bảng so sánh các thí nghiệm từ file results/metrics/<EXP_ID>_<split>.json."""
    rows = []
    for p in metric_files:
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        o, g = r["overall"], r["plastic_related"]
        rows.append({"exp_id": r["exp_id"], "split": r["split"], "n": r["num_samples"],
                     "accuracy": o["accuracy"], "macro_f1": o["macro_f1"], "weighted_f1": o.get("weighted_f1"),
                     "macro_f1_present": r["macro_f1_present_classes"], "plastic_macro_f1": g["macro_f1"]})
    return pd.DataFrame(rows).round(4)
