"""EDA cho TrashCan-Instance: thống kê lớp, kích thước bbox, mất cân bằng, ảnh mẫu.

Đầu vào là bảng đối tượng từ `src.preprocessing.build_object_table` hoặc
`split_manifest.csv` sau khi tiền xử lý.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def class_distribution(df: pd.DataFrame, by: str = "source_split") -> pd.DataFrame:
    """Số annotation theo lớp × (train/val gốc hoặc split mới), kèm số ảnh và số video có lớp đó."""
    table = df.pivot_table(index="class_name", columns=by, values="uid", aggfunc="count", fill_value=0)
    table["total"] = table.sum(axis=1)
    table["num_images"] = df.groupby("class_name")["image_id"].nunique()
    table["num_videos"] = df.groupby("class_name")["video_id"].nunique()
    table["share_pct"] = (100 * table["total"] / table["total"].sum()).round(2)
    return table.sort_values("total", ascending=False)


def imbalance_ratio(df: pd.DataFrame) -> float:
    """Tỉ lệ số mẫu lớp lớn nhất / lớp nhỏ nhất (lớp có ít nhất 1 mẫu)."""
    counts = df["class_name"].value_counts()
    return float(counts.max() / counts.min())


def bbox_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Thống kê cạnh ngắn của bbox theo lớp (pixel) và tỉ lệ box < 32 px."""
    short = np.minimum(df["bbox_w"], df["bbox_h"])
    s = df.assign(short_side=short).groupby("class_name")["short_side"]
    out = s.describe()[["count", "min", "25%", "50%", "75%", "max"]].round(1)
    out["pct_lt_32px"] = (df.assign(small=short < 32).groupby("class_name")["small"].mean() * 100).round(1)
    return out.sort_values("count", ascending=False)


def plot_class_distribution(table: pd.DataFrame, highlight: list[str] | None = None,
                            save_path: str | Path | None = None, title: str = "Phân bố annotation theo lớp"):
    highlight = set(highlight or [])
    t = table.sort_values("total")
    colors = ["#d95f02" if c in highlight else "#7570b3" for c in t.index]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(t))))
    ax.barh(t.index, t["total"], color=colors)
    for y, v in enumerate(t["total"]):
        ax.text(v, y, f" {v}", va="center", fontsize=8)
    ax.set_xlabel("Số annotation")
    ax.set_title(title + (" (cam = lớp liên quan nhựa)" if highlight else ""))
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig, ax


def plot_split_counts(split_table: pd.DataFrame, save_path: str | Path | None = None):
    cols = [c for c in ("train", "val", "test") if c in split_table.columns]
    t = split_table[cols].sort_values(cols[0])
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(t))))
    left = np.zeros(len(t))
    for c in cols:
        ax.barh(t.index, t[c], left=left, label=c)
        left += t[c].to_numpy()
    ax.set_xlim(0, max(1, left.max()) * 1.05)
    ax.set_xlabel("Số crop hợp lệ")
    ax.set_title("Số crop theo lớp và tập (chia theo video)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig, ax


def plot_sample_crops(manifest: pd.DataFrame, processed_dir: str | Path, class_names: list[str],
                      per_class: int = 4, seed: int = 42, save_path: str | Path | None = None):
    """Lưới ảnh crop mẫu: mỗi hàng một lớp."""
    ok = manifest[manifest["crop_status"] == "ok"]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(class_names), per_class, figsize=(per_class * 1.6, len(class_names) * 1.6))
    axes = np.atleast_2d(axes)
    for r, c in enumerate(class_names):
        rows = ok[ok["class_name"] == c]
        pick = rows.iloc[rng.permutation(len(rows))[:per_class]] if len(rows) else rows
        for k in range(per_class):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(pick):
                ax.imshow(Image.open(Path(processed_dir) / pick.iloc[k]["crop_path"]))
        axes[r, 0].set_title(c, fontsize=7, loc="left")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=110)
    return fig, axes
