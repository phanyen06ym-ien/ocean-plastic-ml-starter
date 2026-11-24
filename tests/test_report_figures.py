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
