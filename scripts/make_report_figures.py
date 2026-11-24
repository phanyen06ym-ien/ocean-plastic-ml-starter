"""Vẽ các hình cho báo cáo từ results/metrics (không cần Drive, GPU hay checkpoint).

    python scripts/make_report_figures.py [--metrics results/metrics] [--out results/figures]

Tạo:
- E03_learning_curves.png, E03_val_confusion_matrix.png
- comparison_e01_e04.png              Macro F1 và Macro F1 nhóm nhựa, Validation (+ Test của E04)
- E04_plastic_f1_per_class.png        F1 Test 6 lớp nhựa: E04 bản đầu so với E04
- error_bbox_size_by_class.png        Cạnh ngắn bbox (trung vị) theo lớp, dữ liệu v2
- error_recall_by_size.png            Recall theo kích thước bbox, Test, E04
- error_plastic_by_video.png          Mẫu nhựa Test theo video và recall nhựa từng video, E04
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PLASTIC = ["trash_bag", "trash_bottle", "trash_snack_wrapper", "trash_tarp", "trash_rope", "trash_net"]
C_MAIN, C_ACCENT, C_MUTED = "#2563eb", "#ea580c", "#9ca3af"


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def learning_curves(history: pd.DataFrame, title: str, path: Path) -> Path:
    x = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(x, history["loss"], color=C_MAIN, label="train")
    axes[0].plot(x, history["val_loss"], color=C_ACCENT, label="val")
    axes[0].set_title(f"{title}: loss")
    axes[0].legend()
    axes[1].plot(x, history["val_accuracy"], color=C_MAIN)
    axes[1].set_title(f"{title}: val accuracy")
    axes[2].plot(x, history["val_macro_f1"], color=C_MAIN)
    axes[2].set_title(f"{title}: val Macro F1")
    n1 = int((history["phase"] == history["phase"].iloc[0]).sum())
    best = int(history["val_loss"].idxmin()) + 1
    for ax in axes:
        ax.axvline(n1 + 0.5, color="gray", ls="--", lw=1)
        ax.axvline(best, color=C_ACCENT, ls=":", lw=1)
        ax.set_xlabel("epoch (pha 1 | pha 2); chấm cam = min val_loss")
    return _save(fig, path)


def confusion(cm: pd.DataFrame, title: str, path: Path) -> Path:
    cmn = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(cmn.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm)))
    ax.set_xticklabels(cm.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cm)))
    ax.set_yticklabels(cm.index, fontsize=8)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    fig.colorbar(im)
    return _save(fig, path)


def comparison(metrics: Path, path: Path) -> Path:
    rows = []
    for label, f in [("E01", "E01_val"), ("E02", "E02_val"), ("E03", "E03_val"),
                     ("E04 bản đầu", "E04v1_val"), ("E04", "E04_val")]:
        d = json.loads((metrics / f"{f}.json").read_text())
        rows.append((label, d["overall"]["macro_f1"], d["plastic_related"]["macro_f1"]))
    t = json.loads((metrics / "E04_test.json").read_text())
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - 0.18, [r[1] for r in rows], 0.36, color=C_MAIN, label="Macro F1 (Validation)")
    b2 = ax.bar(x + 0.18, [r[2] for r in rows], 0.36, color=C_ACCENT, label="Macro F1 nhóm nhựa (Validation)")
    ax.scatter([x[-1] - 0.18], [t["overall"]["macro_f1"]], marker="D", color="black", zorder=3,
               label=f"E04 Test: {t['overall']['macro_f1']:.3f} / nhựa {t['plastic_related']['macro_f1']:.3f}")
    ax.scatter([x[-1] + 0.18], [t["plastic_related"]["macro_f1"]], marker="D", color="black", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("Macro F1")
    ax.set_title("So sánh E01–E04 (E01–E03: 22 lớp, v1_crop128; E04: 18 lớp, v2_crop224)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, max(r[1] for r in rows) * 1.25)
    return _save(fig, path)


def plastic_per_class(metrics: Path, path: Path) -> Path:
    old = json.loads((metrics / "E04v1_test.json").read_text())["overall"]["per_class"]
    new = json.loads((metrics / "E04_test.json").read_text())["overall"]["per_class"]
    x = np.arange(len(PLASTIC))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.18, [old[c]["f1_score"] for c in PLASTIC], 0.36, color=C_MUTED, label="E04 bản đầu (√(1/n))")
    ax.bar(x + 0.18, [new[c]["f1_score"] for c in PLASTIC], 0.36, color=C_MAIN, label="E04 (1/n)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c.replace('trash_', '')}\n(n={new[c]['support']})" for c in PLASTIC])
    ax.set_ylabel("F1 (Test)")
    ax.set_title("F1 từng lớp nhựa trên Test")
    ax.legend()
    return _save(fig, path)


def bbox_by_class(metrics: Path, path: Path) -> Path:
    d = pd.read_csv(metrics / "bbox_stats_by_class_v2.csv").sort_values("50%")
    colors = [C_ACCENT if c in PLASTIC else C_MUTED for c in d["class_name"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(d["class_name"], d["50%"], color=colors)
    ax.axvline(32, color="gray", ls="--", lw=1)
    ax.set_xlabel("Cạnh ngắn bbox, trung vị (px); đường đứt = 32 px")
    ax.set_title("Kích thước vật thể theo lớp (cam = lớp nhựa)")
    return _save(fig, path)


def recall_by_size(metrics: Path, path: Path) -> Path:
    d = pd.read_csv(metrics / "E04_test_by_size.csv")
    bins = list(dict.fromkeys(d["size_bin"]))
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(8, 4))
    for off, grp, col, lab in [(-0.18, "all", C_MAIN, "Mọi lớp"), (0.18, "plastic6", C_ACCENT, "Nhóm nhựa")]:
        g = d[d["group"] == grp].set_index("size_bin").reindex(bins)
        bars = ax.bar(x + off, g["recall"].fillna(0), 0.36, color=col, label=lab)
        for b, n in zip(bars, g["num_samples"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"n={int(n)}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel("Cạnh ngắn bbox")
    ax.set_ylabel("Recall (Test, E04)")
    ax.set_title("Recall theo kích thước vật thể")
    ax.legend()
    return _save(fig, path)


def plastic_by_video(metrics: Path, path: Path) -> Path:
    d = pd.read_csv(metrics / "E04_test_by_video.csv")
    d = d[d["num_plastic"] > 0].sort_values("num_plastic", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(d["video_id"], d["num_plastic"], color=C_MAIN)
    ax.set_ylabel("Số mẫu nhựa (Test)", color=C_MAIN)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax2 = ax.twinx()
    ax2.plot(d["video_id"], d["plastic_recall"], "o", color=C_ACCENT)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Recall nhựa trong video (E04)", color=C_ACCENT)
    total = int(d["num_plastic"].sum())
    ax.set_title(f"Mẫu nhựa Test theo video ({len(d)} video, {total} mẫu; "
                 f"video lớn nhất = {d['num_plastic'].iloc[0] / total:.0%})")
    return _save(fig, path)


def make_all(metrics: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    return [
        learning_curves(pd.read_csv(metrics / "E03_history.csv"), "E03", out / "E03_learning_curves.png"),
        confusion(pd.read_csv(metrics / "E03_val_confusion_counts.csv", index_col=0),
                  "E03 — Confusion matrix (Validation, chuẩn hoá theo hàng)", out / "E03_val_confusion_matrix.png"),
        comparison(metrics, out / "comparison_e01_e04.png"),
        plastic_per_class(metrics, out / "E04_plastic_f1_per_class.png"),
        bbox_by_class(metrics, out / "error_bbox_size_by_class.png"),
        recall_by_size(metrics, out / "error_recall_by_size.png"),
        plastic_by_video(metrics, out / "error_plastic_by_video.png"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", default="results/metrics")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    for p in make_all(Path(a.metrics), Path(a.out)):
        print(p)


if __name__ == "__main__":
    main()
