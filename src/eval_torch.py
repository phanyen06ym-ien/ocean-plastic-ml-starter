"""Evaluate one saved checkpoint of a run on the Validation set (never Test).

Use it for checkpoints that train.py did not evaluate itself, for example bestf1_phase2.pt
(best Macro F1 epoch of phase 2) or best_phase2.pt.

    python -m src.eval_torch --run runs/E03d_20261110-1530 --ckpt bestf1_phase2.pt --processed-dir $DATA

Reads <run>/config.json (written by src/train.py) and writes, next to the checkpoint:
    <ckpt>_val.json                  Macro F1, plastic-6 Macro F1, per-class scores, size, CPU/GPU time
    <ckpt>_val_confusion_counts.csv  confusion matrix counts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from src.common import class_names, load_class_mapping, plastic_indices
    from src.train import (build_model, build_transform, compute_metrics, find_class_mapping, measure_efficiency,
                           ordered_image_folder, predict)
except ImportError:  # pragma: no cover
    from common import class_names, load_class_mapping, plastic_indices
    from train import (build_model, build_transform, compute_metrics, find_class_mapping, measure_efficiency,
                       ordered_image_folder, predict)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the Validation set")
    ap.add_argument("--run", required=True, help="run folder, e.g. runs/E03d_20261110-1530")
    ap.add_argument("--ckpt", required=True, help="checkpoint file inside the run folder, e.g. bestf1_phase2.pt")
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    mapping = load_class_mapping(find_class_mapping(a.processed_dir, cfg["data"].get("class_mapping_path", "class_mapping.json")))
    names, plastic = class_names(mapping), plastic_indices(mapping)
    if names != list(cfg["data"].get("class_names", names)):
        raise ValueError("class order differs from the one the run was trained with")

    val_ds = ordered_image_folder(Path(a.processed_dir) / "val", names, build_transform(cfg))   # val only
    loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=0)

    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(torch.load(run / a.ckpt, map_location=device))
    val_loss, y, p = predict(model, loader, device)
    metrics = compute_metrics(y, p, names, plastic)
    metrics.update({"exp_id": cfg["exp_id"], "checkpoint": a.ckpt, "split": "val", "num_samples": int(len(y)),
                    "val_loss_unweighted": val_loss, "efficiency": measure_efficiency(model, cfg, device)})

    stem = Path(a.ckpt).stem
    (run / f"{stem}_val.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, p, labels=list(range(len(names))))
    pd.DataFrame(cm, index=names, columns=names).to_csv(run / f"{stem}_val_confusion_counts.csv")

    o = metrics["overall"]
    print(f"{cfg['exp_id']} / {a.ckpt}: macro_f1 {o['macro_f1']:.4f}  plastic6 {metrics['plastic_related']['macro_f1']:.4f}  "
          f"acc {o['accuracy']:.4f}  val_loss {val_loss:.4f}")
    print(f"written: {run / (stem + '_val.json')}")


if __name__ == "__main__":
    main()
