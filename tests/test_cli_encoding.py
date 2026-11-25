"""CLI in tiếng Việt không được crash khi stdout bị chuyển hướng (Windows dùng cp1252 cho pipe / file)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("args", [
    ["main.py", "--help"],
    ["-m", "src.preprocessing", "--help"],
    ["-m", "src.train", "--help"],
    ["-m", "src.evaluate", "--help"],
])
def test_help_with_redirected_stdout(args):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    env["PYTHONUTF8"] = "0"                       # mô phỏng mặc định của Windows
    r = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, timeout=300)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "usage" in r.stdout.decode("utf-8")
