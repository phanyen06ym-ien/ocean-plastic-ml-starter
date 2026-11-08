import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_runs import summarize_run, to_markdown  # noqa: E402


def test_summarize_and_markdown(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({
        "exp_id": "E03", "macro_f1_val": 0.2034, "macro_f1_plastic6_val": 0.1228,
        "efficiency": {"params": {"total": 2252054}, "file_size_mb": 9.23, "inference_ms_cpu_bs1": 32.391}}))
    row = summarize_run(tmp_path)
    assert row["params"] == 2252054
    md = to_markdown([row, {"exp_id": "X", "params": None, "file_size_mb": None, "inference_ms": None, "macro_f1": None, "plastic_f1": None}])
    assert "| E03 | 2,252 | 9,23 | 32,4 | 0,2034 | 0,1228 |" in md
    assert "| X | — |" in md
