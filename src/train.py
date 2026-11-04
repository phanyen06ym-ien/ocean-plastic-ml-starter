"""CNN baseline và training pipeline (S1-CNN / S2-CNN).

Dựa trên bản nháp của Hoàng Bách trong notebooks/03_cnn_training.ipynb; giữ nguyên API
(DEFAULT_CONFIG, set_seed, build_cnn, compile_model, get_callbacks, save_config) và bổ sung:
- chuẩn hoá pixel/255 nằm TRONG mô hình (lớp Rescaling) → train, eval, predict dùng chung ảnh 0..255;
- data augmentation (E02) chỉ hoạt động khi train;
- class weights "balanced" cho dữ liệu mất cân bằng (ADR-001 D6);
- nạp dữ liệu theo thứ tự lớp của configs/class_mapping.json;
- thư mục run chuẩn runs/<EXP_ID>_<YYYYMMDD-HHMM>/ và tiếp tục huấn luyện khi Colab ngắt.

TensorFlow chỉ được import trong các hàm cần nó, để phần cấu hình/class weights
kiểm thử được cả khi máy không cài TensorFlow.
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from src.common import class_names, load_class_mapping, load_yaml
except ImportError:  # pragma: no cover
    from common import class_names, load_class_mapping, load_yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "exp_id": "E01",
    "description": "CNN baseline + class weights",
    "seed": 42,
    "data_version": "v1_crop128",
    "input_shape": [128, 128, 3],
    "num_classes": 22,
    "filters": [32, 64, 128, 256],
    "batch_norm": True,
    "dense_units": 128,
    "dropout": 0.4,
    "label_mode": "int",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 40,
    "early_stopping_patience": 6,
    "reduce_lr_patience": 3,
    "class_weight": "balanced",
    "augmentation": None,          # E02: {"flip": "horizontal", "rotation": 0.1, "zoom": 0.1, "contrast": 0.2}
    "normalization": "rescale_1_255_in_model",
}


# --------------------------------------------------------------------------- cấu hình
def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """DEFAULT_CONFIG được ghi đè bởi file configs/experiments/<EXP_ID>_*.yaml."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(load_yaml(path) or {})
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def make_run_dir(runs_root: str | Path, exp_id: str) -> Path:
    d = Path(runs_root) / f"{exp_id}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- class weights
def count_images_per_class(split_dir: str | Path, names: list[str]) -> np.ndarray:
    exts = {".jpg", ".jpeg", ".png"}
    return np.array([
        sum(1 for p in (Path(split_dir) / n).iterdir() if p.suffix.lower() in exts) if (Path(split_dir) / n).exists() else 0
        for n in names
    ])


def compute_class_weights(counts: np.ndarray | list[int], mode: str | None = "balanced", max_weight: float = 50.0) -> dict[int, float] | None:
    """Trọng số lớp kiểu sklearn 'balanced': n_samples / (n_classes_có_mẫu * count).

    Lớp không có mẫu train nhận trọng số 0 (không ảnh hưởng loss); trọng số được chặn trên
    bởi max_weight để lớp rất hiếm không làm loss bùng nổ.
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


# --------------------------------------------------------------------------- mô hình
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


# --------------------------------------------------------------------------- dữ liệu
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


# --------------------------------------------------------------------------- huấn luyện
def run_training(cfg: dict[str, Any], processed_dir: str | Path, runs_root: str | Path,
                 run_dir: str | Path | None = None, resume: bool = False):
    """Huấn luyện một thí nghiệm. Trả (model, history_dict, run_dir).

    resume=True + run_dir có sẵn → nạp last.keras và chạy tiếp từ epoch đã lưu trong history.csv.
    """
    import pandas as pd
    import tensorflow as tf

    set_seed(cfg["seed"])
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

    hist = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"], initial_epoch=initial_epoch,
                     class_weight=cw, callbacks=get_callbacks(run_dir, cfg), verbose=2)
    history = pd.read_csv(run_dir / "history.csv").to_dict(orient="list")
    return model, history, run_dir
