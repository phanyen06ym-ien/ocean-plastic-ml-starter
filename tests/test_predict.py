"""Kiểm thử src/predict.py và main.py bằng mock model (không cần TensorFlow)."""
import json

import numpy as np
import pytest
from PIL import Image

import main as cli
from src.predict import InferenceError, Predictor, load_image, preprocess, top_k

NAMES = ["rov", "trash_bag", "trash_net"]


class MockModel:
    """Giả lập Keras model: 3 lớp, ảnh 128×128, luôn dự đoán lớp có index = độ sáng trung bình // 86."""
    input_shape = (None, 128, 128, 3)

    def __init__(self, n_out=3):
        self.output_shape = (None, n_out)
        self.seen = None

    def predict(self, x, verbose=0):
        self.seen = x
        k = min(int(x.mean() // 86), self.output_shape[1] - 1)
        p = np.full(self.output_shape[1], 0.1 / (self.output_shape[1] - 1))
        p[k] = 0.9
        return p[None, :]


@pytest.fixture
def mapping_file(tmp_path):
    p = tmp_path / "class_mapping.json"
    p.write_text(json.dumps({"classes": [
        {"class_index": i, "category_id": i + 1, "class_name": n, "plastic_related": n != "rov"} for i, n in enumerate(NAMES)]}))
    return p


def _img(tmp_path, name="a.png", mode="RGB", value=200):
    p = tmp_path / name
    Image.new(mode, (40, 30), color=value if mode == "L" else (value,) * len(mode)).save(p)
    return p


def test_load_image_errors(tmp_path):
    with pytest.raises(InferenceError, match="Không tìm thấy"):
        load_image(tmp_path / "missing.jpg")
    bad = tmp_path / "x.jpg"
    bad.write_text("not an image", encoding="utf-8")
    with pytest.raises(InferenceError, match="không phải ảnh"):
        load_image(bad)
    txt = tmp_path / "x.txt"
    txt.write_text("hi", encoding="utf-8")
    with pytest.raises(InferenceError, match="Định dạng"):
        load_image(txt)


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_preprocess_shape_and_range(tmp_path, mode):
    arr = preprocess(load_image(_img(tmp_path, "m.png", mode)), 128)
    assert arr.shape == (1, 128, 128, 3) and arr.dtype == np.float32
    assert 0 <= arr.min() and arr.max() <= 255 and arr.max() > 1   # chưa chia 255 (mô hình tự chuẩn hoá)


def test_predictor_top_k_follows_mapping(tmp_path, mapping_file):
    pr = Predictor(None, mapping_file, model=MockModel())
    r = pr.predict_path(_img(tmp_path, value=250), k=3)
    assert r["top1"]["class_name"] == "trash_net" and r["top1"]["plastic_related"]
    assert abs(sum(p["confidence"] for p in r["top_k"]) - 1.0) < 1e-6
    assert [p["class_index"] for p in r["top_k"]][0] == 2


def test_mapping_mismatch_is_reported(mapping_file):
    with pytest.raises(InferenceError, match="đầu ra"):
        Predictor(None, mapping_file, model=MockModel(n_out=22))


def test_missing_checkpoint(mapping_file, tmp_path):
    with pytest.raises(InferenceError, match="checkpoint"):
        Predictor(tmp_path / "none.keras", mapping_file)


def test_top_k_order():
    m = [{"class_index": i, "class_name": n} for i, n in enumerate(NAMES)]
    out = top_k(np.array([0.2, 0.7, 0.1]), m, 2)
    assert [p.class_name for p in out] == ["trash_bag", "rov"]


def test_cli_reports_errors(tmp_path, mapping_file, capsys):
    code = cli.main(["--image", str(tmp_path / "nope.jpg"), "--model", str(tmp_path / "none.keras"), "--mapping", str(mapping_file)])
    assert code == 1
    assert "Lỗi" in capsys.readouterr().err


# --------------------------------------------------------------------------- S3: checkpoint PyTorch, CLI final
def test_unsupported_checkpoint_extension(mapping_file, tmp_path):
    ckpt = tmp_path / "model.onnx"
    ckpt.write_bytes(b"x")
    with pytest.raises(InferenceError, match="Đuôi checkpoint"):
        Predictor(ckpt, mapping_file)


def test_torch_checkpoint_needs_run_config(mapping_file, tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x")
    with pytest.raises(InferenceError, match="config.json"):
        Predictor(ckpt, mapping_file)


def test_class_order_mismatch_is_reported(mapping_file):
    m = MockModel()
    m.class_names = ["trash_net", "trash_bag", "rov"]
    with pytest.raises(InferenceError, match="Thứ tự lớp"):
        Predictor(None, mapping_file, model=m)


def test_cli_folder_writes_csv(tmp_path, mapping_file, monkeypatch, capsys):
    import pandas as pd
    import src.predict as sp

    monkeypatch.setattr(sp, "load_model", lambda *a, **k: MockModel())
    folder = tmp_path / "crops"
    folder.mkdir()
    _img(folder, "dark.png", value=10)
    _img(folder, "bright.png", value=250)
    out = tmp_path / "out" / "pred.csv"
    code = cli.main(["--folder", str(folder), "--model", "fake.keras", "--mapping", str(mapping_file),
                     "--top-k", "2", "--out", str(out)])
    assert code == 0
    df = pd.read_csv(out)
    assert len(df) == 2 and {"top1_class_name", "top2_class_name", "top1_plastic_related"} <= set(df.columns)
    assert set(df["top1_class_name"]) == {"rov", "trash_net"}
    assert "Tổng: 2 ảnh" in capsys.readouterr().out


def test_cli_rejects_bad_options(tmp_path, mapping_file, capsys):
    img = _img(tmp_path)
    assert cli.main(["--image", str(img), "--model", "m.keras", "--mapping", str(mapping_file), "--top-k", "0"]) == 1
    assert cli.main(["--image", str(img), "--model", "m.keras", "--mapping", str(mapping_file), "--out", "x.csv"]) == 1
    assert "--out chỉ dùng" in capsys.readouterr().err


def test_torch_model_end_to_end(tmp_path, mapping_file):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.train import build_model

    cfg = {"data": {"num_classes": 3, "class_names": NAMES},
           "model": {"backbone": "mobilenet_v2", "pretrained_weights": None, "width_mult": 1.0,
                     "head": {"dropout": 0.0, "in_features": 1280}},
           "input": {"image_size": [64, 64], "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}}
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    torch.save(build_model(cfg, pretrained=False).state_dict(), run / "best.pt")
    pr = Predictor(run / "best.pt", mapping_file, device="cpu")
    assert pr.input_size == 64
    r = pr.predict_path(_img(tmp_path), k=3)
    assert abs(sum(p["confidence"] for p in r["top_k"]) - 1.0) < 1e-5