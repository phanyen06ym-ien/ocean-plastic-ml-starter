"""evaluate_torch_checkpoint trên dữ liệu giả lập rất nhỏ (CPU, không tải trọng số ImageNet)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from src.evaluate import evaluate_torch_checkpoint  # noqa: E402
from src.train import build_model, load_experiment_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def e03_run(tmp_path):
    cfg = load_experiment_config(ROOT / "configs/experiments/E03_transfer_mobilenetv2.yaml")
    cfg["evaluation"]["efficiency"]["inference_time"].update(warmup_iters=1, measure_iters=1)
    proc = tmp_path / "proc"
    rng = np.random.default_rng(0)
    for name in cfg["data"]["class_names"]:
        (proc / "val" / name).mkdir(parents=True)
        Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(proc / "val" / name / "a.jpg")
    shutil.copy(ROOT / "configs/class_mapping.json", proc / "class_mapping.json")
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    torch.manual_seed(0)
    torch.save(build_model(cfg, pretrained=False).state_dict(), run / "best.pt")
    return run, proc


def test_trainable_params_follow_phase2(e03_run, tmp_path):
    run, proc = e03_run
    m = evaluate_torch_checkpoint(run, "best.pt", proc, "val", out_dir=tmp_path / "out", device="cpu")
    params = m["efficiency"]["params"]
    # Giống summary.json do src/train.py ghi cho E03 (results/metrics/E03_summary.json): chỉ features[14:] + classifier
    assert params["total"] == 2252054
    assert params["trainable_phase2"] == 1709526
    assert m["num_samples"] == 22
    for f in ("E03_val.json", "E03_val_confusion_counts.csv", "E03_val_predictions.csv"):
        assert (tmp_path / "out" / f).exists()
    assert not (run / "E03_val.json").exists()          # out_dir được tôn trọng, không ghi vào thư mục run


def test_wrong_processed_dir_gives_clear_error(e03_run, tmp_path):
    run, proc = e03_run
    shutil.copy(ROOT / "configs/class_mapping_v2.json", proc / "class_mapping.json")   # dữ liệu v2 (18 lớp) cho run v1
    with pytest.raises(ValueError, match=r"18 lớp.*22 lớp.*--processed-dir"):
        evaluate_torch_checkpoint(run, "best.pt", proc, "val", out_dir=tmp_path / "out", device="cpu")
