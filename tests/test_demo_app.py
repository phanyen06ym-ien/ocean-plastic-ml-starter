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


def test_vietnamese_error_readable_when_redirected(tmp_path):
    """Thông báo lỗi tiếng Việt đọc được khi stdout/stderr bị chuyển hướng (Windows mặc định cp1252)."""
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    env["PYTHONUTF8"] = "0"
    r = subprocess.run([sys.executable, "demo_app.py", "--model", str(tmp_path / "none.pt")], cwd=ROOT, env=env,
                       capture_output=True, timeout=300)
    assert r.returncode == 1
    assert "Không tìm thấy checkpoint" in r.stderr.decode("utf-8")


def test_busy_port_is_reported_before_loading_model():
    import socket

    import pytest
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(16)          # như server thật: không để hàng đợi đầy sau lần kiểm tra đầu
        port = s.getsockname()[1]
        assert demo.port_in_use(port)
        with pytest.raises(SystemExit, match=rf"Cổng {port} đang được dùng.*--port {port + 1}"):
            demo.main(["--port", str(port), "--model", "khong_ton_tai.pt"])   # lỗi cổng báo trước lỗi checkpoint
    assert not demo.port_in_use(port)


def test_editor_dict_with_numpy_layers():
    """gr.ImageEditor(type="numpy") trả dict chứa mảng numpy; không được dùng `or` trên mảng."""
    arr = np.zeros((60, 90, 4), dtype=np.uint8)
    assert demo.prepare({"composite": arr, "background": None}, square=True).size == (60, 60)
    assert demo.prepare({"composite": None, "background": arr[..., :3]}, square=False).size == (90, 60)
    assert demo.prepare({"composite": None, "background": None, "layers": []}, square=True) is None
