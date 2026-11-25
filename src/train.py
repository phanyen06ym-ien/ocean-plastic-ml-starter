"""Huấn luyện mọi thí nghiệm E01–E04 (S1–S3-CNN).

Hai pipeline trong một module, chọn tự động theo file cấu hình (run_training / dòng lệnh):
- E01–E02: CNN tự thiết kế, TensorFlow/Keras (cấu hình phẳng: filters, dense_units, ...).
  Chuẩn hoá /255 nằm trong mô hình; augmentation (E02) chỉ chạy khi train; class weights "balanced";
  thư mục run runs/<EXP_ID>_<YYYYMMDD-HHMM>/, tiếp tục huấn luyện được khi Colab ngắt (resume).
- E03–E04: học chuyển giao ImageNet, PyTorch, 2 pha (cấu hình có khoá `model.backbone`).
  E03 = MobileNetV2, dữ liệu v1_crop128 (22 lớp), class weights, chọn checkpoint theo val_loss.
  E04 = EfficientNet-B0, dữ liệu v2_crop224 (18 lớp, configs/data_contract_v2.yaml), augmentation chỉ trên Train,
  lấy mẫu bù lớp hiếm (training.sampler) + focal loss (training.loss: focal), chọn checkpoint theo val Macro F1.

TensorFlow và PyTorch chỉ được import khi cần, nên phần cấu hình / class weights chạy được ở mọi máy.

Pipeline PyTorch bám theo pipeline Keras của E01 để so sánh công bằng:
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
    python -m src.train --config configs/experiments/E01_baseline.yaml \
        --processed-dir /content/processed/v1_crop128 \
        --runs-root /content/drive/MyDrive/ocean-plastic-ml/runs
    python -m src.train --config configs/experiments/E03_transfer_mobilenetv2.yaml \
        --processed-dir /content/processed/v1_crop128 \
        --runs-root /content/drive/MyDrive/ocean-plastic-ml/runs
    python -m src.train --config configs/experiments/E04_efficientnet_b0_data_v2.yaml \
        --processed-dir /content/processed/v2_crop224 \
        --runs-root /content/drive/MyDrive/ocean-plastic-ml/runs
    add --smoke for a 1-epoch-per-phase pipeline check.

Tuning / new runs without editing the YAML (every key must already exist in the config):
    python -m src.train --config ... --exp-id E03a --set model.head.dropout=0.6
    python -m src.train --config ... --exp-id E03d --set training.phase2.learning_rate.features=3e-5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
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
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, models, transforms
except ImportError:  # pragma: no cover
    torch = None

try:
    from src.preprocessing import (DEFAULT_MAPPING, class_names, ensure_utf8_stdio, load_class_mapping, load_yaml,
                                   plastic_indices)
except ImportError:  # pragma: no cover
    from preprocessing import (DEFAULT_MAPPING, class_names, ensure_utf8_stdio, load_class_mapping, load_yaml,
                               plastic_indices)

# Mặc định cho cấu hình Keras (E01–E02); file configs/experiments/E01_*.yaml, E02_*.yaml ghi đè lên.
DEFAULT_CONFIG: dict[str, Any] = {
    "exp_id": "E01", "description": "CNN baseline + class weights", "seed": 42, "data_version": "v1_crop128",
    "input_shape": [128, 128, 3], "num_classes": 22, "filters": [32, 64, 128, 256], "batch_norm": True,
    "dense_units": 128, "dropout": 0.4, "label_mode": "int", "optimizer": "adam", "learning_rate": 1e-3,
    "batch_size": 64, "epochs": 40, "early_stopping_patience": 6, "reduce_lr_patience": 3,
    "class_weight": "balanced", "augmentation": None, "normalization": "rescale_1_255_in_model",
}

BACKBONES = {  # tên trong config → (hàm torchvision, enum trọng số)
    "mobilenet_v2": ("mobilenet_v2", "MobileNet_V2_Weights"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
}


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch and torchvision are required: pip install torch torchvision")


# --------------------------------------------------------------------------- setup
def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Read configs/experiments/<EXP_ID>_*.yaml (no defaults are merged in).

    A tuning config may set `base_config: <file in the same folder>`; its keys are deep-merged
    over that file, so a tuning run only states what it changes (S3-CNN tuning, E03a-f).
    """
    path = Path(path)
    cfg = load_yaml(path) or {}
    base = cfg.pop("base_config", None)
    if base:
        cfg = _deep_merge(load_experiment_config(path.parent / base), cfg)
    if is_keras_config(cfg):
        cfg = {**DEFAULT_CONFIG, **cfg}
    return cfg


def is_keras_config(cfg: dict[str, Any]) -> bool:
    """E01–E02 (CNN tự thiết kế, Keras) dùng cấu hình phẳng; E03–E04 (MobileNetV2 / EfficientNet-B0) có khoá model.backbone."""
    return not (isinstance(cfg.get("model"), dict) and "backbone" in cfg["model"])


def _parse_value(text: str) -> Any:
    """'0.6' -> 0.6, '3e-5' -> 3e-05, 'true' -> True, '[1, 2]' -> list; anything else stays a string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply 'a.b.c=value' strings to the nested config. Unknown keys raise, so typos cannot pass silently."""
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got: {item}")
        parts, node = key.split("."), cfg
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"unknown config key: {key}")
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise KeyError(f"unknown config key: {key}")
        node[parts[-1]] = _parse_value(raw)
    return cfg


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
    if "tensorflow" in sys.modules:          # pipeline Keras đã import TensorFlow
        sys.modules["tensorflow"].random.set_seed(seed)
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


def find_class_mapping(processed_dir: str | Path, name: str = "class_mapping.json") -> Path:
    """Look next to the processed data first, then fall back to configs/class_mapping.json in the repo."""
    mp = Path(name)
    candidates = [mp] if mp.is_absolute() else [Path(processed_dir) / mp, Path(processed_dir).parent / mp]
    candidates.append(Path(DEFAULT_MAPPING))
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("class_mapping.json not found; tried: " + ", ".join(map(str, candidates)))


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


def _image_hw(cfg: dict[str, Any]) -> tuple[int, int]:
    size = cfg["input"]["image_size"]
    return (int(size), int(size)) if isinstance(size, int) else (int(size[0]), int(size[1]))


def build_transform(cfg: dict[str, Any], train: bool = False):
    """Validation / inference: ToTensor -> ImageNet Normalize (crops are already image_size, e.g. 128 or 224).

    Training with `augmentation.enabled: true` (E04) adds, before ToTensor:
    RandomResizedCrop(image_size, scale) -> RandomHorizontalFlip -> RandomRotation -> ColorJitter.
    E03 has augmentation disabled, so its transform is unchanged.
    """
    _require_torch()
    norm = cfg["input"]["normalize"]
    tail = [transforms.ToTensor(), transforms.Normalize(norm["mean"], norm["std"])]
    aug = cfg.get("augmentation") or {}
    if not (train and aug.get("enabled")):
        return transforms.Compose(tail)
    ops = []
    if aug.get("random_resized_crop_scale"):
        lo, hi = aug["random_resized_crop_scale"]
        ops.append(transforms.RandomResizedCrop(_image_hw(cfg), scale=(float(lo), float(hi)), antialias=True))
    if aug.get("horizontal_flip_p"):
        ops.append(transforms.RandomHorizontalFlip(float(aug["horizontal_flip_p"])))
    if aug.get("rotation_degrees"):
        ops.append(transforms.RandomRotation(float(aug["rotation_degrees"])))
    cj = aug.get("color_jitter") or {}
    if cj:
        ops.append(transforms.ColorJitter(**{k: float(v) for k, v in cj.items()}))
    return transforms.Compose(ops + tail)


def make_loaders(cfg: dict[str, Any], processed_dir: str | Path, names: list[str]):
    train_ds = ordered_image_folder(Path(processed_dir) / "train", names, build_transform(cfg, train=True))
    val_ds = ordered_image_folder(Path(processed_dir) / "val", names, build_transform(cfg))   # never open "test"
    g = torch.Generator()
    g.manual_seed(int(cfg["seed"]))
    nw = int(cfg["data"].get("num_workers", 2))
    common = dict(batch_size=int(cfg["training"]["batch_size"]), num_workers=nw,
                  pin_memory=torch.cuda.is_available(),
                  worker_init_fn=_seed_worker if nw > 0 else None, persistent_workers=nw > 0)
    sampler = make_sampler(train_ds.targets, cfg["training"].get("sampler"), len(names), g)
    if sampler is not None:
        train_loader = DataLoader(train_ds, sampler=sampler, **common)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, generator=g, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_ds, val_ds, train_loader, val_loader


def sample_weights(targets, num_classes: int, mode: str) -> np.ndarray:
    """Trọng số lấy mẫu cho từng ảnh Train. sqrt_inverse: p(lớp) ∝ sqrt(số mẫu) (lớp hiếm được bù, không quá tay);
    inverse: mọi lớp xuất hiện đều nhau."""
    targets = np.asarray(targets)
    counts = np.bincount(targets, minlength=num_classes).astype(float)
    power = {"sqrt_inverse": 0.5, "inverse": 1.0}[mode]
    per_class = np.zeros(num_classes)
    per_class[counts > 0] = counts[counts > 0] ** -power
    return per_class[targets]


def make_sampler(targets, sampler_cfg: dict[str, Any] | None, num_classes: int, generator):
    """WeightedRandomSampler (lấy mẫu có hoàn lại) khi training.sampler.mode được đặt; None = xáo trộn thường."""
    if not sampler_cfg or not sampler_cfg.get("mode"):
        return None
    w = sample_weights(targets, num_classes, sampler_cfg["mode"])
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(w), replacement=True,
                                 generator=generator)


# --------------------------------------------------------------------------- model
def build_model(cfg: dict[str, Any], pretrained: bool = True) -> nn.Module:
    """MobileNetV2 (E03) hoặc EfficientNet-B0 (E04), tiền huấn luyện ImageNet; thay classifier bằng Dropout → Linear."""
    _require_torch()
    m = cfg["model"]
    assert m["backbone"] in BACKBONES, f"backbone không hỗ trợ: {m['backbone']} (có: {sorted(BACKBONES)})"
    assert float(m.get("width_mult", 1.0)) == 1.0, "pretrained weights exist only for width_mult=1.0"
    fn_name, weights_enum = BACKBONES[m["backbone"]]
    weights = getattr(models, weights_enum)[m["pretrained_weights"]] if pretrained and m.get("pretrained_weights") else None
    net = getattr(models, fn_name)(weights=weights)
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


def make_scheduler(opt, phase: dict[str, Any], tcfg: dict[str, Any], mode: str = "min"):
    """Keras ReduceLROnPlateau(patience=P) cuts after P bad epochs; torch cuts after P+1, so use P-1.
    mode="max" khi theo dõi val Macro F1 (E04)."""
    s = tcfg["lr_scheduler"]
    min_lr = phase.get("min_lr", s["min_lr"])       # list order = optimizer param groups
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode=mode, factor=float(s["factor"]), patience=max(int(s["patience"]) - 1, 0),
        threshold=1e-4, threshold_mode="abs", min_lr=min_lr)


def weighted_ce(logits: torch.Tensor, y: torch.Tensor, class_w: torch.Tensor, gamma: float = 0.0) -> torch.Tensor:
    """Keras class_weight semantics: mean over the batch of (loss_i * weight[y_i]).

    gamma > 0 → focal loss (Lin et al., 2017): loss_i * (1 - p_i)^gamma, giảm trọng số mẫu dễ để mô hình
    tập trung vào mẫu khó / lớp hiếm. gamma = 0 là cross-entropy thường.
    """
    ce = F.cross_entropy(logits, y, reduction="none")
    if gamma:
        ce = ce * (1.0 - torch.exp(-ce)).pow(gamma)
    return (ce * class_w[y]).mean()


# --------------------------------------------------------------------------- train / eval loops
def train_epoch(model, loader, opt, class_w, device, phase, gamma: float = 0.0) -> float:
    set_train_mode(model, phase)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = weighted_ce(model(x), y, class_w, gamma)
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
    es = phase["early_stopping"]
    monitor = es.get("monitor", "val_loss")
    assert (monitor, es.get("mode", "min")) in {("val_loss", "min"), ("val_macro_f1", "max")}, \
        "early_stopping: val_loss/min (E03) hoặc val_macro_f1/max (E04)"
    sched = make_scheduler(opt, phase, tcfg, mode="max" if monitor == "val_macro_f1" else "min")
    gamma = float(tcfg.get("focal_gamma", 0.0)) if tcfg.get("loss") == "focal" else 0.0
    sign = -1.0 if monitor == "val_macro_f1" else 1.0      # so sánh "nhỏ hơn là tốt hơn" cho cả hai chỉ số
    best, best_epoch, bad, best_state, best_val_loss = float("inf"), 0, 0, None, float("nan")
    best_f1, best_f1_epoch, best_f1_val_loss = -1.0, 0, float("nan")   # extra information, not the selection rule
    for epoch in range(1, int(phase["epochs"]) + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, opt, class_w, device, phase, gamma)
        val_loss, y, p = predict(model, val_loader, device)
        m = compute_metrics(y, p, names, plastic)["overall"]
        if m["macro_f1"] > best_f1:
            best_f1, best_f1_epoch, best_f1_val_loss = m["macro_f1"], epoch, val_loss
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       run_dir / ckpt_name.replace("best_", "bestf1_"))
        lrs = {g["name"]: g["lr"] for g in opt.param_groups}
        sched.step(m["macro_f1"] if monitor == "val_macro_f1" else val_loss)
        history.append({"phase": phase["name"], "epoch": epoch, "loss": tr_loss, "val_loss": val_loss,
                        "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"],
                        "lr_classifier": lrs["classifier"], "lr_features": lrs.get("features", 0.0),
                        "seconds": round(time.time() - t0, 1)})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        score = sign * (m["macro_f1"] if monitor == "val_macro_f1" else val_loss)
        improved = score < best
        if improved:
            best, best_epoch, bad, best_val_loss = score, epoch, 0, val_loss
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
    return {"phase": phase["name"], "monitor": monitor, "best_score": sign * best, "best_val_loss": best_val_loss,
            "best_epoch": best_epoch, "checkpoint": ckpt_name,
            "best_macro_f1": best_f1, "best_macro_f1_epoch": best_f1_epoch, "best_macro_f1_val_loss": best_f1_val_loss,
            "best_macro_f1_checkpoint": ckpt_name.replace("best_", "bestf1_")}


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


# =========================================================================== E01–E02: CNN tự thiết kế (Keras)
# --------------------------------------------------------------------------- mô hình (Keras)
def build_augmentation(aug: dict[str, Any] | None):
    from tensorflow import keras
    from tensorflow.keras import layers
    if not aug:
        return None
    ops = []
    if aug.get("flip"):
        ops.append(layers.RandomFlip(aug["flip"]))
    if aug.get("rotation"):
        ops.append(layers.RandomRotation(aug["rotation"]))
    if aug.get("zoom"):
        ops.append(layers.RandomZoom(aug["zoom"]))
    if aug.get("contrast"):
        ops.append(layers.RandomContrast(aug["contrast"]))
    return keras.Sequential(ops, name="augmentation") if ops else None


def build_cnn(input_shape=(128, 128, 3), num_classes: int = 22, filters=(32, 64, 128, 256), dense_units: int = 128,
              dropout: float = 0.4, batch_norm: bool = True, augmentation: dict[str, Any] | None = None):
    """CNN baseline: Rescaling → [Conv-BN-ReLU-MaxPool]×(N-1) → Conv-BN-ReLU → GAP → Dense → Softmax.

    Đầu vào: ảnh RGB pixel 0..255 (float hoặc uint8). Augmentation chỉ chạy khi training=True.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=tuple(input_shape), name="image")
    x = inputs
    aug = build_augmentation(augmentation)
    if aug is not None:
        x = aug(x)
    x = layers.Rescaling(1.0 / 255, name="rescale")(x)
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, 3, padding="same", use_bias=not batch_norm, name=f"conv{i + 1}")(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"bn{i + 1}")(x)
        x = layers.Activation("relu", name=f"relu{i + 1}")(x)
        if i < len(filters) - 1:
            x = layers.MaxPooling2D(2, name=f"pool{i + 1}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.Dense(dense_units, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="cnn_baseline")


def compile_model(model, cfg: dict[str, Any]):
    from tensorflow import keras
    loss = "sparse_categorical_crossentropy" if cfg["label_mode"] == "int" else "categorical_crossentropy"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]), loss=loss, metrics=["accuracy"])
    return model


def model_from_config(cfg: dict[str, Any]):
    return compile_model(build_cnn(cfg["input_shape"], cfg["num_classes"], cfg["filters"], cfg["dense_units"],
                                   cfg["dropout"], cfg.get("batch_norm", True), cfg.get("augmentation")), cfg)


def get_callbacks(run_dir: str | Path, cfg: dict[str, Any]):
    from tensorflow import keras
    run_dir = str(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    cbs = [
        keras.callbacks.ModelCheckpoint(os.path.join(run_dir, "best.keras"), monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.ModelCheckpoint(os.path.join(run_dir, "last.keras"), save_best_only=False, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["early_stopping_patience"], restore_best_weights=True),
        keras.callbacks.CSVLogger(os.path.join(run_dir, "history.csv"), append=True),
    ]
    if cfg.get("reduce_lr_patience"):
        cbs.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=cfg["reduce_lr_patience"], min_lr=1e-5))
    return cbs


# --------------------------------------------------------------------------- dữ liệu (Keras)
def load_split(processed_dir: str | Path, split: str, names: list[str], cfg: dict[str, Any], shuffle: bool):
    """Đọc <processed_dir>/<split>/<class_name>/*.jpg; nhãn = class_index theo class mapping."""
    import tensorflow as tf
    h, w, _ = cfg["input_shape"]
    ds = tf.keras.utils.image_dataset_from_directory(
        str(Path(processed_dir) / split), labels="inferred", label_mode=cfg["label_mode"], class_names=names,
        color_mode="rgb", image_size=(h, w), batch_size=cfg["batch_size"], shuffle=shuffle, seed=cfg["seed"])
    return ds


def load_datasets(processed_dir: str | Path, cfg: dict[str, Any], mapping_path: str | Path | None = None):
    import tensorflow as tf
    mapping = load_class_mapping(mapping_path or Path(processed_dir) / "class_mapping.json")
    names = class_names(mapping)
    if len(names) != cfg["num_classes"]:
        raise ValueError(f"Config có {cfg['num_classes']} lớp nhưng class mapping có {len(names)}")
    train = load_split(processed_dir, "train", names, cfg, shuffle=True).prefetch(tf.data.AUTOTUNE)
    val = load_split(processed_dir, "val", names, cfg, shuffle=False).prefetch(tf.data.AUTOTUNE)
    return train, val, mapping


# --------------------------------------------------------------------------- huấn luyện (Keras)
def run_training_keras(cfg: dict[str, Any], processed_dir: str | Path, runs_root: str | Path,
                       run_dir: str | Path | None = None, resume: bool = False):
    """Huấn luyện một thí nghiệm. Trả (model, history_dict, run_dir).

    resume=True + run_dir có sẵn → nạp last.keras và chạy tiếp từ epoch đã lưu trong history.csv.
    """
    import tensorflow as tf

    set_seed(cfg["seed"])            # TensorFlow đã được import ở trên → seed cả TF
    run_dir = Path(run_dir) if run_dir else make_run_dir(runs_root, cfg["exp_id"])
    train_ds, val_ds, mapping = load_datasets(processed_dir, cfg)
    names = class_names(mapping)

    counts = count_images_per_class(Path(processed_dir) / "train", names)
    cw = compute_class_weights(counts, cfg.get("class_weight"))
    save_config({**cfg, "train_counts": dict(zip(names, map(int, counts))), "class_weights": cw,
                 "processed_dir": str(processed_dir)}, run_dir / "config.json")
    with open(run_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"classes": mapping}, f, ensure_ascii=False, indent=2)

    initial_epoch = 0
    if resume and (run_dir / "last.keras").exists():
        model = tf.keras.models.load_model(str(run_dir / "last.keras"))
        initial_epoch = len(pd.read_csv(run_dir / "history.csv"))
        print(f"Tiếp tục từ epoch {initial_epoch}")
    else:
        model = model_from_config(cfg)

    model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"], initial_epoch=initial_epoch,
              class_weight=cw, callbacks=get_callbacks(run_dir, cfg), verbose=2)
    # Đọc lại từ CSVLogger (append=True) để history gồm cả các epoch trước khi resume
    history = pd.read_csv(run_dir / "history.csv").to_dict(orient="list")
    return model, history, run_dir


# =========================================================================== E03–E04: MobileNetV2 / EfficientNet-B0 (PyTorch)
# --------------------------------------------------------------------------- main entry
def run_training_torch(cfg: dict[str, Any], processed_dir: str | Path, runs_root: str | Path,
                       pretrained: bool = True, smoke: bool = False, device: str | None = None):
    """E03 MobileNetV2 / E04 EfficientNet-B0, 2 pha. Trả (model, history, run_dir)."""
    _require_torch()
    set_seed(int(cfg["seed"]), cfg.get("reproducibility"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(runs_root, cfg["exp_id"])

    mapping_path = find_class_mapping(processed_dir, cfg["data"].get("class_mapping_path", "class_mapping.json"))
    print(f"class mapping: {mapping_path}")
    mapping = load_class_mapping(mapping_path)
    names, plastic = class_names(mapping), plastic_indices(mapping)
    if names != list(cfg["data"].get("class_names", names)):
        raise ValueError("class_mapping.json order differs from data.class_names in the config")
    if len(names) != int(cfg["data"]["num_classes"]):
        raise ValueError(f"config has {cfg['data']['num_classes']} classes, mapping has {len(names)}")

    train_ds, val_ds, train_loader, val_loader = make_loaders(cfg, processed_dir, names)
    counts = np.bincount(train_ds.targets, minlength=len(names))
    wcfg = cfg["training"]["class_weights"]
    cw = compute_class_weights(counts, wcfg.get("mode"), float(wcfg.get("max_weight", 50.0)))
    cw = cw or {i: 1.0 for i in range(len(names))}       # mode null → không trọng số lớp (E04 dùng sampler)
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

    # Checkpoint cuối = pha có chỉ số theo dõi tốt hơn: val_loss thấp hơn (E01–E03) hoặc val Macro F1 cao hơn (E04)
    if r1["monitor"] == "val_macro_f1":
        chosen = r2 if r2["best_score"] >= r1["best_score"] else r1
    else:
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
               "baseline_reference": cfg.get("baseline_reference"), "overrides": cfg.get("overrides", [])}
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in ("macro_f1_val", "accuracy_val", "macro_f1_plastic6_val", "efficiency")}, indent=2))
    return model, history, run_dir


def run_training(cfg: dict[str, Any], processed_dir: str | Path, runs_root: str | Path, **kwargs):
    """Huấn luyện một thí nghiệm, tự chọn pipeline theo cấu hình. Trả (model, history, run_dir).

    Keras (E01–E02): kwargs run_dir, resume. PyTorch (E03–E04): kwargs pretrained, smoke, device.
    """
    if is_keras_config(cfg):
        return run_training_keras(cfg, processed_dir, runs_root, **kwargs)
    return run_training_torch(cfg, processed_dir, runs_root, **kwargs)


def main() -> None:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="Huấn luyện E01–E04 (Keras cho E01–E02, PyTorch cho E03–E04)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--processed-dir", default=None, help="default: data.processed_dir in the config")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--no-pretrained", action="store_true", help="PyTorch: skip ImageNet weights (offline smoke tests only)")
    ap.add_argument("--smoke", action="store_true", help="1 epoch (per phase) to check the pipeline")
    ap.add_argument("--resume-run", default=None, help="Keras: tiếp tục huấn luyện trong thư mục run này")
    ap.add_argument("--device", default=None)
    ap.add_argument("--exp-id", default=None, help="override exp_id, e.g. E03a (names the run folder and result files)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. model.head.dropout=0.6 (repeatable)")
    a = ap.parse_args()
    cfg = load_experiment_config(a.config)
    if a.exp_id:
        cfg["exp_id"] = a.exp_id
    apply_overrides(cfg, a.overrides)
    cfg["overrides"] = list(a.overrides)      # saved in config.json and summary.json
    if is_keras_config(cfg):
        if not a.processed_dir:
            ap.error("--processed-dir là bắt buộc với E01–E02")
        if a.smoke:
            cfg["epochs"] = 1
        run_training(cfg, a.processed_dir, a.runs_root, run_dir=a.resume_run, resume=bool(a.resume_run))
    else:
        run_training(cfg, a.processed_dir or cfg["data"]["processed_dir"], a.runs_root,
                     pretrained=not a.no_pretrained, smoke=a.smoke, device=a.device)


if __name__ == "__main__":
    main()
