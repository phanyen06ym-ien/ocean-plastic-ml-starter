"""Kiểm thử mô hình CNN — cần TensorFlow (tự bỏ qua khi máy không có; chạy trên Colab)."""
import numpy as np
import pytest

from src.train_keras import DEFAULT_CONFIG

tf = pytest.importorskip("tensorflow")
from src.train_keras import build_cnn, model_from_config  # noqa: E402


@pytest.mark.parametrize("num_classes", [2, 6, 22])
def test_forward_pass_shape_and_softmax(num_classes):
    model = build_cnn((128, 128, 3), num_classes)
    x = np.random.randint(0, 256, size=(4, 128, 128, 3)).astype("float32")
    y = model.predict(x, verbose=0)
    assert y.shape == (4, num_classes)
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5)


def test_augmentation_only_in_training():
    cfg = {**DEFAULT_CONFIG, "augmentation": {"flip": "horizontal", "rotation": 0.1}}
    model = model_from_config(cfg)
    x = np.random.randint(0, 256, size=(2, 128, 128, 3)).astype("float32")
    assert np.allclose(model(x, training=False), model(x, training=False))


def test_tiny_training_run(tmp_path):
    """Huấn luyện 1 epoch trên dữ liệu giả để kiểm tra pipeline, không đánh giá độ chính xác."""
    from PIL import Image
    import json
    from src.train_keras import run_training
    names = ["a", "b", "c"]
    for s in ("train", "val"):
        for i, n in enumerate(names):
            d = tmp_path / "proc" / s / n
            d.mkdir(parents=True)
            for k in range(4 if s == "train" else 2):
                Image.fromarray(np.full((128, 128, 3), 60 * i + k, dtype=np.uint8)).save(d / f"{k}.jpg")
    (tmp_path / "proc" / "class_mapping.json").write_text(json.dumps({"classes": [
        {"class_index": i, "category_id": i + 1, "class_name": n, "plastic_related": False} for i, n in enumerate(names)]}))
    cfg = {**DEFAULT_CONFIG, "num_classes": 3, "epochs": 1, "batch_size": 4, "exp_id": "T00"}
    _, history, run_dir = run_training(cfg, tmp_path / "proc", tmp_path / "runs")
    assert (run_dir / "best.keras").exists() and (run_dir / "config.json").exists()
    assert len(history["loss"]) == 1
