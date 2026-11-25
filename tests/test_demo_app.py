"""Kiểm thử phần logic của demo_app.py (không cần gradio hay checkpoint)."""
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from src.predict import Prediction

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demo_app", ROOT / "demo_app.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class FakePredictor:
    mapping = [{"class_index": 0, "class_name": "trash_bag"}, {"class_index": 1, "class_name": "rov"}]
    input_size = 224

    def __init__(self):
        self.seen = None

    def predict_image(self, img, k=3):
        self.seen = img
        return [Prediction(0, "trash_bag", 0.3, True), Prediction(1, "rov", 0.7, False)][:k]


def test_center_square_and_inputs():
    img = Image.new("RGB", (300, 120), "red")
    assert demo.center_square(img).size == (120, 120)
    assert demo.prepare(img, square=False).size == (300, 120)
    assert demo.prepare(np.zeros((50, 80, 3), dtype=np.uint8), square=True).size == (50, 50)
    assert demo.prepare({"composite": img.convert("RGBA"), "background": None}, square=True).mode == "RGB"
    assert demo.prepare(None, square=True) is None


def test_predict_fn_outputs_and_warning():
    fake = FakePredictor()
    shown, probs, text = demo.make_predict_fn(fake)(Image.new("RGB", (400, 200)), True)
    assert fake.seen.size == (200, 200)            # đã cắt vuông trước khi đưa vào mô hình
    assert shown.size == (224, 224)
    assert probs["Túi nilon (trash_bag)"] == 0.3
    assert "Túi nilon" in text and "nhựa" in text and "không chắc chắn" in text
    assert demo.make_predict_fn(fake)(None, True)[2] == "Hãy tải lên một ảnh."
