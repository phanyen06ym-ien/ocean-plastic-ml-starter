"""scripts/make_report_figures.py chạy được trên số liệu trong results/metrics."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_make_all(tmp_path):
    spec = importlib.util.spec_from_file_location("mrf", ROOT / "scripts" / "make_report_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    paths = mod.make_all(ROOT / "results" / "metrics", tmp_path)
    assert len(paths) == 7 and all(p.exists() and p.stat().st_size > 10_000 for p in paths)


def test_misclassified_grid(tmp_path):
    import numpy as np
    import pandas as pd
    from PIL import Image
    spec = importlib.util.spec_from_file_location("mmg", ROOT / "scripts" / "make_misclassified_grid.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mapping = [{"class_index": 0, "class_name": "rov", "plastic_related": False},
               {"class_index": 1, "class_name": "trash_bag", "plastic_related": True}]
    rows = []
    for i, (t, p) in enumerate([(1, 0), (1, 0), (1, 1), (0, 1)]):
        f = f"test/x/{i}.jpg"
        (tmp_path / "test/x").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((32, 32, 3), 40 * i, dtype=np.uint8)).save(tmp_path / f)
        rows.append((f, t, p))
    pred = pd.DataFrame(rows, columns=["path", "y_true", "y_pred"])
    err = mod.select_errors(pred, mapping, per_class=1, limit=8, conf_fn=lambda p: 0.5 + int(p[7]) / 10)
    assert list(err["path"]) == ["test/x/1.jpg"]          # chỉ lỗi của lớp nhựa, độ tin cậy cao nhất
    out = mod.draw(err, tmp_path, tmp_path / "grid.png")
    assert out.exists() and out.with_suffix(".csv").exists()


def test_default_metrics_path_is_repo_relative(tmp_path):
    """Chạy từ thư mục khác vẫn đọc results/metrics của repo (mặc định không phụ thuộc thư mục đang đứng)."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_report_figures.py"), "--out", str(tmp_path / "f")],
                       cwd=tmp_path, capture_output=True, timeout=300)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert len(list((tmp_path / "f").glob("*.png"))) == 7
