"""Tổng hợp bảng ứng viên mô hình từ các thư mục run (summary.json + <EXP_ID>_val.json).

Dùng:  python scripts/summarize_runs.py runs/E03_* runs/E03a_*
In bảng Markdown: EXP_ID, tham số, kích thước file, thời gian suy luận, Macro F1 val.
Chỉ đọc Validation; không đụng tới tập Test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize_run(run_dir: str | Path) -> dict:
    run = Path(run_dir)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    exp_id = summary["exp_id"]
    val_path = run / f"{exp_id}_val.json"
    val = json.loads(val_path.read_text(encoding="utf-8")) if val_path.exists() else {}
    eff = summary.get("efficiency", {})
    return {
        "exp_id": exp_id,
        "params": eff.get("params", {}).get("total"),
        "file_size_mb": eff.get("file_size_mb"),
        "inference_ms": eff.get("inference_ms_cpu_bs1"),
        "macro_f1": summary.get("macro_f1_val", val.get("overall", {}).get("macro_f1")),
        "plastic_f1": summary.get("macro_f1_plastic6_val", val.get("plastic_related", {}).get("macro_f1")),
    }


def _fmt(v, nd=4, scale=1.0):
    return "—" if v is None else f"{v / scale:.{nd}f}".replace(".", ",")


def to_markdown(rows: list[dict]) -> str:
    out = ["| EXP_ID | Tham số (M) | Kích thước (MB) | Suy luận (ms) | Macro F1 val | Nhóm nhựa val |", "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: -(r["macro_f1"] or 0)):
        out.append(f"| {r['exp_id']} | {_fmt(r['params'], 3, 1e6)} | {_fmt(r['file_size_mb'], 2)} | "
                   f"{_fmt(r['inference_ms'], 1)} | {_fmt(r['macro_f1'])} | {_fmt(r['plastic_f1'])} |")
    return "\n".join(out)


if __name__ == "__main__":
    dirs = [Path(a) for a in sys.argv[1:]]
    if not dirs:
        sys.exit("Cách dùng: python scripts/summarize_runs.py <run_dir> [<run_dir> ...]")
    print(to_markdown([summarize_run(d) for d in dirs if (d / "summary.json").exists()]))
