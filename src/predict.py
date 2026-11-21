"""Inference (S1-DEMO / S2-DEMO / S3-DEMO): ảnh một đối tượng → lớp TrashCan-Instance + confidence.

Luồng: load_image → preprocess (RGB, resize về kích thước đầu vào của mô hình, pixel 0..255)
→ model.predict → softmax → top-k (class_index, class_name, confidence, plastic_related).

Hỗ trợ hai loại checkpoint:
- Keras ``.keras`` / ``.h5`` (E01–E02): chuẩn hoá /255 nằm trong mô hình.
- PyTorch ``.pt`` / ``.pth`` (E03–E04, MobileNetV2): state_dict do src/train.py lưu; cần ``config.json``
  của run (mặc định nằm cùng thư mục với checkpoint). Chuẩn hoá mean/std ImageNet làm trong ``TorchModel``.

Logic tách khỏi giao diện: main.py (CLI) và notebook demo chỉ gọi các hàm ở đây.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    from src.preprocessing import load_class_mapping
except ImportError:  # pragma: no cover
    from preprocessing import load_class_mapping

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
KERAS_EXTS = {".keras", ".h5"}
TORCH_EXTS = {".pt", ".pth"}


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


class TorchModel:
    """Bọc checkpoint PyTorch MobileNetV2 để có cùng giao diện với Keras model.

    ``predict(x)`` nhận mảng (N, H, W, 3) giá trị 0..255 (đầu ra của ``preprocess``) và trả xác suất softmax
    (N, num_classes) — giống ToTensor + Normalize(ImageNet) lúc huấn luyện trong src/train.py.
    """

    def __init__(self, checkpoint: str | Path, config: dict[str, Any], device: str | None = None):
        try:
            import torch
        except ImportError as e:   # pragma: no cover - phụ thuộc môi trường
            raise InferenceError("Checkpoint PyTorch cần cài torch và torchvision (pip install -r requirements.txt)") from e
        try:
            from src.train import build_model
        except ImportError:  # pragma: no cover
            from train import build_model

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_model(config, pretrained=False)
        try:
            state = torch.load(str(checkpoint), map_location=self.device)
            self.net.load_state_dict(state)
        except Exception as e:   # sai kiến trúc, file hỏng, số lớp khác…
            raise InferenceError(f"Không nạp được checkpoint PyTorch {checkpoint}: {e}") from e
        self.net.to(self.device).eval()

        size = config["input"]["image_size"]
        h, w = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
        norm = config["input"]["normalize"]
        self.mean = np.asarray(norm["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(norm["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.input_shape = (None, h, w, 3)
        self.output_shape = (None, int(config["data"]["num_classes"]))
        self.class_names = list(config["data"].get("class_names") or [])

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        arr = np.transpose(np.asarray(x, dtype=np.float32) / 255.0, (0, 3, 1, 2))
        arr = (arr - self.mean) / self.std
        torch = self._torch
        with torch.no_grad():
            logits = self.net(torch.from_numpy(arr).to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()


def find_run_config(checkpoint: str | Path, config: str | Path | None = None) -> Path:
    """Đường dẫn config.json của run: tham số ``config`` hoặc file cùng thư mục với checkpoint."""
    path = Path(config) if config else Path(checkpoint).parent / "config.json"
    if not path.exists():
        raise InferenceError(f"Checkpoint PyTorch cần config.json của run (không thấy {path}). "
                             "Chỉ định bằng --config runs/<EXP_ID>_<thời gian>/config.json")
    return path


def load_model(checkpoint: str | Path, config: str | Path | None = None, device: str | None = None):
    """Nạp checkpoint Keras (.keras/.h5) hoặc PyTorch (.pt/.pth) theo đuôi file."""
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise InferenceError(f"Không tìm thấy checkpoint: {checkpoint}")
    ext = checkpoint.suffix.lower()
    if ext in KERAS_EXTS:
        import tensorflow as tf
        return tf.keras.models.load_model(str(checkpoint))
    if ext in TORCH_EXTS:
        cfg_path = find_run_config(checkpoint, config)
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return TorchModel(checkpoint, cfg, device)
    raise InferenceError(f"Đuôi checkpoint không hỗ trợ ({ext}). Dùng: "
                         f"{', '.join(sorted(KERAS_EXTS | TORCH_EXTS))}")


def check_compatible(model, mapping: list[dict[str, Any]]) -> int:
    """Kiểm tra số đầu ra của mô hình khớp class mapping; trả kích thước ảnh đầu vào."""
    out_shape = model.outputs[0].shape if hasattr(model, "outputs") else model.output_shape   # Keras 3 / Keras 2
    in_shape = model.inputs[0].shape if hasattr(model, "inputs") else model.input_shape
    n_out = int(out_shape[-1])
    if n_out != len(mapping):
        raise InferenceError(f"Mô hình có {n_out} đầu ra nhưng class mapping có {len(mapping)} lớp")
    trained_names = list(getattr(model, "class_names", None) or [])
    if trained_names and trained_names != [c["class_name"] for c in mapping]:
        raise InferenceError("Thứ tự lớp trong class mapping khác với lúc huấn luyện mô hình")
    return int(in_shape[1])


def top_k(probs: np.ndarray, mapping: list[dict[str, Any]], k: int = 3) -> list[Prediction]:
    probs = np.asarray(probs).reshape(-1)
    order = np.argsort(-probs)[:k]
    return [Prediction(int(i), mapping[i]["class_name"], float(probs[i]), bool(mapping[i].get("plastic_related")))
            for i in order]


class Predictor:
    """Nạp mô hình + mapping một lần, dự đoán nhiều ảnh."""

    def __init__(self, checkpoint: str | Path, mapping_path: str | Path, model=None,
                 config: str | Path | None = None, device: str | None = None):
        mapping_path = Path(mapping_path)
        if not mapping_path.exists():
            raise InferenceError(f"Không tìm thấy class mapping: {mapping_path}")
        self.mapping = load_class_mapping(mapping_path)
        self.model = model if model is not None else load_model(checkpoint, config, device)
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