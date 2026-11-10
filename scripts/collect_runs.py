"""Collect run summaries (runs/<EXP_ID>_*/summary.json) into one comparison table.

Usage:
    python scripts/collect_runs.py --runs-root runs --csv results/metrics/e03_candidates.csv

Prints a Markdown table (decimal comma, like docs/experiment_plan.md) sorted by Macro F1 val.
Paste it into the candidate section of docs/experiment_plan.md. Reads only files written by src/train.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_rows(runs_root: Path, prefix: str) -> list[dict]:
    rows = []
    for path in sorted(runs_root.glob(f"{prefix}*/summary.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        eff, sel = d.get("efficiency", {}), d.get("selected", {})
        best_f1 = max((d.get(k, {}).get("best_macro_f1", -1.0) for k in ("phase1", "phase2")), default=-1.0)
        rows.append({
            "exp_id": d["exp_id"],
            "run": path.parent.name,
            "overrides": " ".join(d.get("overrides", [])) or "-",
            "selected": f"{sel.get('phase', '?')} ep{sel.get('best_epoch', '?')}",
            "macro_f1_val": d["macro_f1_val"],
            "macro_f1_plastic6_val": d["macro_f1_plastic6_val"],
            "accuracy_val": d["accuracy_val"],
            "best_macro_f1_any_epoch": best_f1 if best_f1 >= 0 else None,
            "params": eff.get("params", {}).get("total"),
            "file_size_mb": eff.get("file_size_mb"),
            "inference_ms_cpu_bs1": eff.get("inference_ms_cpu_bs1"),
        })
    return sorted(rows, key=lambda r: r["macro_f1_val"], reverse=True)


def fmt(x, nd=4, comma=True):
    if x is None:
        return "-"
    s = f"{x:.{nd}f}" if isinstance(x, float) else f"{x:,}".replace(",", ".") if comma else str(x)
    return s.replace(".", ",") if (comma and isinstance(x, float)) else s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--prefix", default="E0", help="only folders starting with this (default: E0)")
    ap.add_argument("--csv", default=None, help="also write the table to this CSV file (decimal point)")
    a = ap.parse_args()
    rows = load_rows(Path(a.runs_root), a.prefix)
    if not rows:
        raise SystemExit(f"no summary.json found under {a.runs_root}/{a.prefix}*/")

    print("| EXP_ID | Thay đổi so với E03 | Checkpoint chọn | Macro F1 val | Macro F1 nhóm nhựa | Accuracy val "
          "| Macro F1 cao nhất (mọi epoch) | Tham số | MB | CPU ms/ảnh |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['exp_id']} | {r['overrides']} | {r['selected']} | {fmt(r['macro_f1_val'])} | "
              f"{fmt(r['macro_f1_plastic6_val'])} | {fmt(r['accuracy_val'])} | {fmt(r['best_macro_f1_any_epoch'])} | "
              f"{fmt(r['params'])} | {fmt(r['file_size_mb'], 2)} | {fmt(r['inference_ms_cpu_bs1'], 1)} |")

    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten: {a.csv}")


if __name__ == "__main__":
    main()
