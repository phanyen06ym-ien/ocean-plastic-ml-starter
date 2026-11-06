"""Training pipeline (PyTorch): E03 transfer learning with torchvision MobileNetV2, 2 phases (S3-CNN).

Replaces the Keras version used for E01/E02 (still available in git history, commit 7520c2b).
Follows configs/experiments/E03_*.yaml and mirrors the E01 Keras pipeline
so the comparison stays fair:
- same data folders <processed_dir>/{train,val}/<class_name>/*.jpg, same class order (class_mapping.json);
- same class weights (compute_class_weights from src/train.py, "balanced");
- Keras-style class weighting: per-sample loss * weight, plain mean over the batch;
- validation loss is the UNWEIGHTED mean cross-entropy (Keras ignores class_weight in validation);
- checkpoint / early stopping on val_loss (patience 6), ReduceLROnPlateau (factor 0.5, patience 3);
- Adam with eps=1e-7 (Keras default), no weight decay;
- the Test set is never read.

Phase 1: backbone frozen (kept in eval mode), train the classifier head.
Phase 2: unfreeze model.features[k:], BatchNorm kept in eval mode, new optimizer with small learning rates.

Usage (Colab):
    python -m src.train --config configs/experiments/E03_transfer.yaml \
        --processed-dir /content/processed/v1_crop128 \
        --runs-root /content/drive/MyDrive/ocean-plastic-ml/runs
    add --smoke for a 1-epoch-per-phase pipeline check.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # config / class-weight helpers below keep working on machines without PyTorch
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms
except ImportError:  # pragma: no cover
    torch = None

try:
    from src.common import class_names, load_class_mapping, load_yaml, plastic_indices
except ImportError:  # pragma: no cover
    from common import class_names, load_class_mapping, load_yaml, plastic_indices

# Legacy E01/E02 (Keras) defaults, kept so existing tests and notebooks can still import them.
DEFAULT_CONFIG: dict[str, Any] = {
    "exp_id": "E01", "description": "CNN baseline + class weights", "seed": 42, "data_version": "v1_crop128",
    "input_shape": [128, 128, 3], "num_classes": 22, "filters": [32, 64, 128, 256], "batch_norm": True,
    "dense_units": 128, "dropout": 0.4, "label_mode": "int", "optimizer": "adam", "learning_rate": 1e-3,
    "batch_size": 64, "epochs": 40, "early_stopping_patience": 6, "reduce_lr_patience": 3,
    "class_weight": "balanced", "augmentation": None, "normalization": "rescale_1_255_in_model",
}


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch and torchvision are required: pip install torch torchvision")


# --------------------------------------------------------------------------- setup
def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Read configs/experiments/<EXP_ID>_*.yaml as is (no defaults are merged in)."""
    return load_yaml(path) or {}


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def make_run_dir(runs_root: str | Path, exp_id: str) -> Path:
    d = Path(runs_root) / f"{exp_id}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed: int, repro: dict[str, Any] | None = None) -> None:
    repro = repro or {}
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(repro.get("cudnn_deterministic", True))
    torch.backends.cudnn.benchmark = bool(repro.get("cudnn_benchmark", False))


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# --------------------------------------------------------------------------- class weights
def count_images_per_class(split_dir: str | Path, names: list[str]) -> np.ndarray:
    exts = {".jpg", ".jpeg", ".png"}
    return np.array([
        sum(1 for p in (Path(split_dir) / n).iterdir() if p.suffix.lower() in exts) if (Path(split_dir) / n).exists() else 0
        for n in names
    ])


def compute_class_weights(counts: np.ndarray | list[int], mode: str | None = "balanced", max_weight: float = 50.0) -> dict[int, float] | None:
    """sklearn-style 'balanced' weights: n_samples / (n_classes_with_samples * count). Same as the E01 code.

    Classes with no training samples get weight 0; weights are capped at max_weight.
    """
    if not mode:
        return None
    counts = np.asarray(counts, dtype=float)
    present = counts > 0
    if not present.any():
        raise ValueError("Tập train không có ảnh nào")
    w = np.zeros_like(counts)
    w[present] = counts[present].sum() / (present.sum() * counts[present])
    w = np.minimum(w, max_weight)
    return {i: float(v) for i, v in enumerate(w)}


# --------------------------------------------------------------------------- data
def ordered_image_folder(root, names: list[str], transform=None):
    """ImageFolder whose label index follows class_mapping.json instead of alphabetical order."""
    class _Ordered(datasets.ImageFolder):
        def find_classes(self, directory):
            return list(names), {n: i for i, n in enumerate(names)}
    return _Ordered(root, transform=transform)


def build_transform(cfg: dict[str, Any]):
    _require_torch()
    """ToTensor -> ImageNet Normalize. Crops are already 128x128; no augmentation (E01 had none)."""
    norm = cfg["input"]["normalize"]
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm["mean"], norm["std"])])


def make_loaders(cfg: dict[str, Any], processed_dir: str | Path, names: list[str]):
    tf = build_transform(cfg)
    train_ds = ordered_image_folder(Path(processed_dir) / "train", names, tf)
    val_ds = ordered_image_folder(Path(processed_dir) / "val", names, tf)   # never open "test"
    g = torch.Generator()
    g.manual_seed(int(cfg["seed"]))
    nw = int(cfg["data"].get("num_workers", 2))
    common = dict(batch_size=int(cfg["training"]["batch_size"]), num_workers=nw,
                  pin_memory=torch.cuda.is_available(),
                  worker_init_fn=_seed_worker if nw > 0 else None, persistent_workers=nw > 0)
    train_loader = DataLoader(train_ds, shuffle=True, generator=g, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_ds, val_ds, train_loader, val_loader


# --------------------------------------------------------------------------- model
def build_model(cfg: dict[str, Any], pretrained: bool = True) -> nn.Module:
    _require_torch()
    m = cfg["model"]
    assert m["backbone"] == "mobilenet_v2", "this script is written for torchvision mobilenet_v2"
    assert float(m.get("width_mult", 1.0)) == 1.0, "pretrained weights exist only for width_mult=1.0"
    weights = models.MobileNet_V2_Weights[m["pretrained_weights"]] if pretrained and m.get("pretrained_weights") else None
    net = models.mobilenet_v2(weights=weights)
    net.classifier = nn.Sequential(nn.Dropout(float(m["head"]["dropout"])),
                                   nn.Linear(int(m["head"]["in_features"]), int(cfg["data"]["num_classes"])))
    return net


def configure_phase(model: nn.Module, phase: dict[str, Any]) -> None:
    """Set requires_grad for the phase: phase 1 = head only; phase 2 = head + features[k:]."""
    for p in model.features.parameters():
        p.requires_grad = False
    if not phase["freeze_backbone"]:
        for block in model.features[int(phase["unfreeze_from_block"]):]:
            for p in block.parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def set_train_mode(model: nn.Module, phase: dict[str, Any]) -> None:
    """model.train() re-enables BatchNorm updates, so re-apply the eval() rules every epoch."""
    model.train()
    if phase["freeze_backbone"]:
        model.features.eval()                       # frozen backbone: BN statistics must not change
    elif phase.get("keep_batchnorm_eval", True):
        for mod in model.features.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()


def make_optimizer(model: nn.Module, phase: dict[str, Any], tcfg: dict[str, Any]):
    assert tcfg["optimizer"] == "adam"
    lr = phase["learning_rate"]
    if isinstance(lr, dict):
        feats = [p for p in model.features.parameters() if p.requires_grad]
        groups = [{"params": feats, "lr": float(lr["features"]), "name": "features"},
                  {"params": list(model.classifier.parameters()), "lr": float(lr["classifier"]), "name": "classifier"}]
    else:
        groups = [{"params": list(model.classifier.parameters()), "lr": float(lr), "name": "classifier"}]
    return torch.optim.Adam(groups, eps=float(tcfg.get("adam_eps", 1e-7)), weight_decay=float(tcfg.get("weight_decay", 0.0)))


def make_scheduler(opt, phase: dict[str, Any], tcfg: dict[str, Any]):
    """Keras ReduceLROnPlateau(patience=P) cuts after P bad epochs; torch cuts after P+1, so use P-1."""
    s = tcfg["lr_scheduler"]
    min_lr = phase.get("min_lr", s["min_lr"])       # list order = optimizer param groups
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=float(s["factor"]), patience=max(int(s["patience"]) - 1, 0),
        threshold=1e-4, threshold_mode="abs", min_lr=min_lr)


def weighted_ce(logits: torch.Tensor, y: torch.Tensor, class_w: torch.Tensor) -> torch.Tensor:
    """Keras class_weight semantics: mean over the batch of (loss_i * weight[y_i])."""
    return (F.cross_entropy(logits, y, reduction="none") * class_w[y]).mean()


# --------------------------------------------------------------------------- train / eval loops
def train_epoch(model, loader, opt, class_w, device, phase) -> float:
    set_train_mode(model, phase)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = weighted_ce(model(x), y, class_w)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def predict(model, loader, device):
    """Return (unweighted val loss, y_true, y_pred). Validation only."""
    model.eval()
    ys, ps, loss_sum = [], [], 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            ps.append(logits.argmax(1).cpu())
            ys.append(y.cpu())
    y_true, y_pred = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return loss_sum / len(y_true), y_true, y_pred


def compute_metrics(y_true, y_pred, names: list[str], plastic: list[int]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    labels = list(range(len(names)))
    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    wpr, wrc, wf1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_class = {n: {"label": i, "precision": float(pr[i]), "recall": float(rc[i]),
                     "f1_score": float(f1[i]), "support": int(sup[i])} for i, n in enumerate(names)}
    pl = [i for i in plastic if sup[i] > 0]
    return {
        "overall": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(pr.mean()), "macro_recall": float(rc.mean()), "macro_f1": float(f1.mean()),
            "weighted_precision": float(wpr), "weighted_recall": float(wrc), "weighted_f1": float(wf1),
            "per_class": per_class, "total_samples": int(len(y_true)), "target_names": names,
        },
        "plastic_related": {
            "classes": plastic, "classes_evaluated": pl,
            "macro_precision": float(pr[pl].mean()) if pl else 0.0,
            "macro_recall": float(rc[pl].mean()) if pl else 0.0,
            "macro_f1": float(f1[pl].mean()) if pl else 0.0,
            "support": int(sup[plastic].sum()),
        },
    }


def run_phase(model, phase: dict[str, Any], tcfg, loaders, class_w, device, run_dir: Path, ckpt_name: str,
              history: list[dict[str, Any]], names, plastic) -> dict[str, Any]:
    train_loader, val_loader = loaders
    configure_phase(model, phase)
    opt = make_optimizer(model, phase, tcfg)
    sched = make_scheduler(opt, phase, tcfg)
    es = phase["early_stopping"]
    assert es["monitor"] == "val_loss" and es["mode"] == "min", "E03 keeps E01's val_loss selection rule"
    best, best_epoch, bad, best_state = float("inf"), 0, 0, None
    for epoch in range(1, int(phase["epochs"]) + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, opt, class_w, device, phase)
        val_loss, y, p = predict(model, val_loader, device)
        m = compute_metrics(y, p, names, plastic)["overall"]
        lrs = {g["name"]: g["lr"] for g in opt.param_groups}
        sched.step(val_loss)
        history.append({"phase": phase["name"], "epoch": epoch, "loss": tr_loss, "val_loss": val_loss,
                        "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"],
                        "lr_classifier": lrs["classifier"], "lr_features": lrs.get("features", 0.0),
                        "seconds": round(time.time() - t0, 1)})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        improved = val_loss < best
        if improved:
            best, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / ckpt_name)
        else:
            bad += 1
        print(f"[{phase['name']}] epoch {epoch:02d} loss {tr_loss:.4f} val_loss {val_loss:.4f} "
              f"val_acc {m['accuracy']:.3f} val_macroF1 {m['macro_f1']:.3f}{'  *best*' if improved else ''}")
        if bad >= int(es["patience"]):
            print(f"[{phase['name']}] early stopping at epoch {epoch} (best epoch {best_epoch})")
            break
    if es.get("restore_best_weights", True) and best_state is not None:
        model.load_state_dict(best_state)
    return {"phase": phase["name"], "best_val_loss": best, "best_epoch": best_epoch, "checkpoint": ckpt_name}


# --------------------------------------------------------------------------- efficiency
def count_params(model: nn.Module) -> dict[str, int]:
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable_phase2": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def file_size_mb(model: nn.Module) -> float:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.pt"
        torch.save(model.state_dict(), path)
        return os.path.getsize(path) / 1e6


def inference_ms(model: nn.Module, device: str, size, warmup: int = 20, iters: int = 100) -> float:
    model = model.to(device).eval()
    x = torch.randn(1, 3, *size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def measure_efficiency(model, cfg, device) -> dict[str, Any]:
    ec = cfg["evaluation"]["efficiency"]["inference_time"]
    size = cfg["input"]["image_size"]
    out = {"params": count_params(model), "file_size_mb": round(file_size_mb(model), 3),
           "input_size": size, "torch_threads": torch.get_num_threads()}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        out[f"inference_ms_{dev}_bs1"] = round(inference_ms(model, dev, size, ec["warmup_iters"], ec["measure_iters"]), 3)
    model.to(device)
    return out


# --------------------------------------------------------------------------- main entry
def run_training(cfg: dict[str, Any], processed_dir: str | Path, runs_root: str | Path,
                 pretrained: bool = True, smoke: bool = False, device: str | None = None):
    _require_torch()
    set_seed(int(cfg["seed"]), cfg.get("reproducibility"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(runs_root, cfg["exp_id"])

    mp = Path(cfg["data"].get("class_mapping_path", "class_mapping.json"))
    mapping = load_class_mapping(mp if mp.is_absolute() else Path(processed_dir) / mp)
    names, plastic = class_names(mapping), plastic_indices(mapping)
    if names != list(cfg["data"].get("class_names", names)):
        raise ValueError("class_mapping.json order differs from data.class_names in the config")
    if len(names) != int(cfg["data"]["num_classes"]):
        raise ValueError(f"config has {cfg['data']['num_classes']} classes, mapping has {len(names)}")

    train_ds, val_ds, train_loader, val_loader = make_loaders(cfg, processed_dir, names)
    counts = np.bincount(train_ds.targets, minlength=len(names))
    wcfg = cfg["training"]["class_weights"]
    cw = compute_class_weights(counts, wcfg["mode"], float(wcfg.get("max_weight", 50.0)))
    class_w = torch.tensor([cw[i] for i in range(len(names))], dtype=torch.float32, device=device)

    cfg_out = {**cfg, "train_counts": dict(zip(names, map(int, counts))), "class_weights": cw,
               "processed_dir": str(processed_dir), "device": device, "torch": torch.__version__, "pretrained": pretrained}
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, ensure_ascii=False, indent=2)
    with open(run_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"classes": mapping}, f, ensure_ascii=False, indent=2)

    tcfg = cfg["training"]
    p1, p2 = dict(tcfg["phase1"]), dict(tcfg["phase2"])
    if smoke:
        p1["epochs"] = p2["epochs"] = 1
    model = build_model(cfg, pretrained).to(device)
    history: list[dict[str, Any]] = []
    loaders = (train_loader, val_loader)

    r1 = run_phase(model, p1, tcfg, loaders, class_w, device, run_dir, "best_phase1.pt", history, names, plastic)
    r2 = run_phase(model, p2, tcfg, loaders, class_w, device, run_dir, "best_phase2.pt", history, names, plastic)

    # Final checkpoint = phase with the lower val_loss (same rule as E01's best.keras)
    chosen = r2 if r2["best_val_loss"] <= r1["best_val_loss"] else r1
    state = torch.load(run_dir / chosen["checkpoint"], map_location=device)
    model.load_state_dict(state)
    torch.save(state, run_dir / "best.pt")

    val_loss, y, p = predict(model, val_loader, device)
    metrics = compute_metrics(y, p, names, plastic)
    metrics.update({"exp_id": cfg["exp_id"], "split": "val", "num_samples": int(len(y)), "num_classes": len(names),
                    "val_loss_unweighted": val_loss, "selected_phase": chosen["phase"], "best_epoch": chosen["best_epoch"],
                    "framework": "pytorch", "data_version": cfg["data"]["data_version"], "run_dir": str(run_dir),
                    "checkpoint": "best.pt", "baseline_exp_id": cfg.get("baseline_exp_id"),
                    "known_extra_differences": cfg.get("known_extra_differences")})
    with open(run_dir / f"{cfg['exp_id']}_val.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, p, labels=list(range(len(names))))
    pd.DataFrame(cm, index=names, columns=names).to_csv(run_dir / f"{cfg['exp_id']}_val_confusion_counts.csv")
    pd.DataFrame({"path": [Path(s[0]).relative_to(processed_dir).as_posix() for s in val_ds.samples],
                  "y_true": y, "y_pred": p}).to_csv(run_dir / f"{cfg['exp_id']}_val_predictions.csv", index=False)

    eff = measure_efficiency(model, cfg, device)
    summary = {"exp_id": cfg["exp_id"], "run_dir": str(run_dir), "phase1": r1, "phase2": r2, "selected": chosen,
               "macro_f1_val": metrics["overall"]["macro_f1"], "accuracy_val": metrics["overall"]["accuracy"],
               "macro_f1_plastic6_val": metrics["plastic_related"]["macro_f1"], "efficiency": eff,
               "baseline_reference": cfg.get("baseline_reference")}
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in ("macro_f1_val", "accuracy_val", "macro_f1_plastic6_val", "efficiency")}, indent=2))
    return model, history, run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="E03 transfer learning (PyTorch)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--processed-dir", default=None, help="default: data.processed_dir in the config")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline smoke tests only)")
    ap.add_argument("--smoke", action="store_true", help="1 epoch per phase to check the pipeline")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = load_experiment_config(a.config)
    run_training(cfg, a.processed_dir or cfg["data"]["processed_dir"], a.runs_root,
                 pretrained=not a.no_pretrained, smoke=a.smoke, device=a.device)


if __name__ == "__main__":
    main()
