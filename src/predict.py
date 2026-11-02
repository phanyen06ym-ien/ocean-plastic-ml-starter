"""Inference (S1-DEMO / S2-DEMO): ảnh một đối tượng → lớp TrashCan-Instance + confidence.

Luồng: load_image → preprocess (RGB, 128×128, pixel 0..255 — chuẩn hoá /255 nằm trong mô hình)
→ model.predict → softmax → top-k (class_index, class_name, confidence, plastic_related).

Logic tách khỏi giao diện: main.py (CLI) và notebook demo chỉ gọi các hàm ở đây.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    from src.common import load_class_mapping
except ImportError:  # pragma: no cover
    from common import load_class_mapping

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class InferenceError(Exception):
    """Lỗi đầu vào/cấu hình có thông báo dễ hiểu cho người dùng demo."""


@dataclass
class Prediction:
    class_index: int
    class_name: str
    confidence: float
    plastic_related: bool


def load_image(path: str | Path) -> Image.Image:
    path = Path(path)
    if not path.exists():
        raise InferenceError(f"Không tìm thấy file ảnh: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise InferenceError(f"Định dạng không hỗ trợ ({path.suffix}). Dùng: {', '.join(sorted(SUPPORTED_EXTS))}")
    try:
        with Image.open(path) as im:
            im.load()
            return im.convert("RGB")      # ảnh xám / RGBA / palette → RGB
    except (UnidentifiedImageError, OSError) as e:
        raise InferenceError(f"File không phải ảnh hợp lệ: {path} ({e})") from e


def preprocess(img: Image.Image, input_size: int = 128) -> np.ndarray:
    """Trả mảng float32 shape (1, H, W, 3), giá trị 0..255 — giống ảnh lúc huấn luyện."""
    arr = np.asarray(img.convert("RGB").resize((input_size, input_size), Image.BILINEAR), dtype=np.float32)
    return arr[None, ...]


def load_model(checkpoint: str | Path):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise InferenceError(f"Không tìm thấy checkpoint: {checkpoint}")
    import tensorflow as tf
    return tf.keras.models.load_model(str(checkpoint))


def check_compatible(model, mapping: list[dict[str, Any]]) -> int:
    """Kiểm tra số đầu ra của mô hình khớp class mapping; trả kích thước ảnh đầu vào."""
    out_shape = model.outputs[0].shape if hasattr(model, "outputs") else model.output_shape   # Keras 3 / Keras 2
    in_shape = model.inputs[0].shape if hasattr(model, "inputs") else model.input_shape
    n_out = int(out_shape[-1])
    if n_out != len(mapping):
        raise InferenceError(f"Mô hình có {n_out} đầu ra nhưng class mapping có {len(mapping)} lớp")
    return int(in_shape[1])


def top_k(probs: np.ndarray, mapping: list[dict[str, Any]], k: int = 3) -> list[Prediction]:
    probs = np.asarray(probs).reshape(-1)
    order = np.argsort(-probs)[:k]
    return [Prediction(int(i), mapping[i]["class_name"], float(probs[i]), bool(mapping[i].get("plastic_related")))
            for i in order]


class Predictor:
    """Nạp mô hình + mapping một lần, dự đoán nhiều ảnh."""

    def __init__(self, checkpoint: str | Path, mapping_path: str | Path, model=None):
        self.mapping = load_class_mapping(mapping_path)
        self.model = model if model is not None else load_model(checkpoint)
        self.input_size = check_compatible(self.model, self.mapping)

    def predict_image(self, img: Image.Image, k: int = 3) -> list[Prediction]:
        probs = self.model.predict(preprocess(img, self.input_size), verbose=0)[0]
        if not np.isclose(probs.sum(), 1.0, atol=1e-3):
            raise InferenceError("Đầu ra mô hình không phải phân phối softmax")
        return top_k(probs, self.mapping, k)

    def predict_path(self, path: str | Path, k: int = 3) -> dict[str, Any]:
        preds = self.predict_image(load_image(path), k)
        return {"image": str(path), "top1": asdict(preds[0]), "top_k": [asdict(p) for p in preds]}

    def predict_folder(self, folder: str | Path, k: int = 3) -> list[dict[str, Any]]:
        files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)
        if not files:
            raise InferenceError(f"Thư mục không có ảnh: {folder}")
        return [self.predict_path(p, k) for p in files]
