"""CLI in tiếng Việt không được crash khi stdout bị chuyển hướng (Windows dùng cp1252 cho pipe / file)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(args, cwd=ROOT):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    env["PYTHONUTF8"] = "0"                       # mô phỏng mặc định của Windows
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, timeout=300)


@pytest.mark.parametrize("args", [
    ["main.py", "--help"],
    ["-m", "src.preprocessing", "--help"],
    ["-m", "src.train", "--help"],
    ["-m", "src.evaluate", "--help"],
    ["scripts/make_report_figures.py", "--help"],
    ["scripts/make_misclassified_grid.py", "--help"],
    ["scripts/visual_inspection_prepare.py", "--help"],
    ["scripts/collect_runs.py", "--help"],
])
def test_help_with_redirected_stdout(args):
    r = _run(args)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "usage" in r.stdout.decode("utf-8")


def test_summarize_runs_vietnamese_table_redirected(tmp_path):
    run = tmp_path / "E03x_1"
    run.mkdir()
    (run / "summary.json").write_text('{"exp_id": "E03x", "macro_f1_val": 0.2, "macro_f1_plastic6_val": 0.1}',
                                      encoding="utf-8")
    r = _run([str(ROOT / "scripts" / "summarize_runs.py"), str(run)])
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "Tham số" in r.stdout.decode("utf-8")
