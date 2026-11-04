"""Tiền xử lý TrashCan 1.0 — Instance theo ADR-001 / configs/data_contract.yaml.

Quy trình (một lần chạy tạo lại toàn bộ dữ liệu):
    1. Tìm 2 file annotation COCO (train, val) trong thư mục dataset gốc.
    2. Sinh class mapping 22 lớp (class_index theo category_id tăng dần).
    3. Lập bảng đối tượng: mỗi dòng là một annotation (bbox) + ảnh nguồn + video id.
    4. Gộp train+val gốc, chia lại Train/Val/Test THEO VIDEO (chia trước khi crop).
    5. Crop bbox (+ lề), bỏ box quá nhỏ, resize 128x128, lưu
       <out_dir>/<split>/<class_name>/<ann_id>.jpg
    6. Ghi split_manifest.csv, class_mapping.json, preprocessing_summary.json.

Chạy từ dòng lệnh:
    python -m src.preprocessing --raw-root /path/to/trashcan --out-dir /path/to/processed/v1_crop128
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:  # cho phép chạy cả `python -m src.preprocessing` lẫn import từ notebook
    from src.common import DEFAULT_CONTRACT, load_contract
except ImportError:  # pragma: no cover
    from common import DEFAULT_CONTRACT, load_contract

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- 1. Tìm file
def find_annotation_files(raw_root: str | Path, names: dict[str, str] | None = None) -> dict[str, Path]:
    """Tìm file annotation COCO của bản Instance.

    Ưu tiên tên trong data contract (instances_train_trashcan.json, ...). Nếu có cả
    bản material và instance, chỉ lấy file nằm trong thư mục chứa chữ 'instance'.
    """
    names = names or {"train": "instances_train_trashcan.json", "val": "instances_val_trashcan.json"}
    raw_root = Path(raw_root)
    found: dict[str, Path] = {}
    for split, fname in names.items():
        hits = sorted(raw_root.rglob(fname))
        inst = [h for h in hits if "instance" in str(h.parent).lower()]
        chosen = inst or hits
        if not chosen:
            raise FileNotFoundError(f"Không tìm thấy {fname} trong {raw_root}")
        found[split] = chosen[0]
    return found


def load_coco(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- 2. Class mapping
def build_class_mapping(categories: list[dict[str, Any]], plastic_related: list[str] | None = None) -> list[dict[str, Any]]:
    """category_id → class_name → class_index (0..K-1 theo category_id tăng dần)."""
    plastic = set(plastic_related or [])
    cats = sorted(categories, key=lambda c: c["id"])
    names = [c["name"] for c in cats]
    if len(set(names)) != len(names):
        raise ValueError("Tên category bị trùng")
    return [
        {"class_index": i, "category_id": c["id"], "class_name": c["name"], "plastic_related": c["name"] in plastic}
        for i, c in enumerate(cats)
    ]


def save_class_mapping(mapping: list[dict[str, Any]], path: str | Path, source: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "TrashCan 1.0 — Instance",
        "num_classes": len(mapping),
        "index_order": "category_id_ascending",
        "source_annotations": source,
        "classes": mapping,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- 3. Bảng đối tượng
def extract_video_id(file_name: str, pattern: str = r"vid_(\d+)") -> str:
    """Lấy video id từ tên file ảnh. Không khớp → dùng tên file (mỗi ảnh một nhóm)."""
    m = re.search(pattern, os.path.basename(file_name))
    return f"vid_{m.group(1)}" if m else f"img_{Path(file_name).stem}"


def index_images(raw_root: str | Path) -> dict[str, list[Path]]:
    """basename → danh sách đường dẫn ảnh dưới raw_root (để tìm ảnh khi cấu trúc thư mục khác nhau)."""
    idx: dict[str, list[Path]] = defaultdict(list)
    for dirpath, _, files in os.walk(raw_root):
        for fn in files:
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                idx[fn].append(Path(dirpath) / fn)
    return idx


def _resolve_image(file_name: str, ann_path: Path, source_split: str, idx: dict[str, list[Path]]) -> Path | None:
    for cand in (ann_path.parent / source_split / file_name, ann_path.parent / file_name, ann_path.parent / "images" / file_name):
        if cand.exists():
            return cand
    hits = idx.get(os.path.basename(file_name), [])
    if not hits:
        return None
    preferred = [h for h in hits if f"{os.sep}{source_split}{os.sep}" in str(h) and "instance" in str(h).lower()]
    return (preferred or hits)[0]


def build_object_table(raw_root: str | Path, contract: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Đọc 2 file COCO → (bảng annotation, class mapping)."""
    ann_files = find_annotation_files(raw_root, contract["dataset"].get("annotation_files"))
    cocos = {s: load_coco(p) for s, p in ann_files.items()}

    cat_sets = [tuple(sorted((c["id"], c["name"]) for c in coco["categories"])) for coco in cocos.values()]
    if len(set(cat_sets)) != 1:
        raise ValueError("Danh sách category giữa train và val gốc không giống nhau")
    mapping = build_class_mapping(next(iter(cocos.values()))["categories"],
                                  contract["classes"].get("plastic_related_candidates"))
    by_cat = {m["category_id"]: m for m in mapping}

    idx = index_images(raw_root)
    pattern = contract["split"].get("video_id_regex", r"vid_(\d+)")
    rows = []
    for source_split, coco in cocos.items():
        images = {im["id"]: im for im in coco["images"]}
        for ann in coco["annotations"]:
            im = images.get(ann["image_id"])
            if im is None or ann["category_id"] not in by_cat:
                continue
            m = by_cat[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            path = _resolve_image(im["file_name"], ann_files[source_split], source_split, idx)
            rows.append({
                "uid": f"{source_split}_{ann['id']}",
                "source_split": source_split,
                "image_id": f"{source_split}_{im['id']}",
                "file_name": im["file_name"],
                "image_path": str(path) if path else "",
                "video_id": extract_video_id(im["file_name"], pattern),
                "ann_id": ann["id"],
                "category_id": ann["category_id"],
                "class_name": m["class_name"],
                "class_index": m["class_index"],
                "bbox_x": float(x), "bbox_y": float(y), "bbox_w": float(w), "bbox_h": float(h),
                "img_w": im.get("width"), "img_h": im.get("height"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Không đọc được annotation nào")
    return df, mapping


# --------------------------------------------------------------------------- 4. Chia theo video
def _assign_groups(group_sizes: pd.Series, ratios: dict[str, float], rng: np.random.Generator) -> dict[str, str]:
    groups = list(group_sizes.index)
    rng.shuffle(groups)
    total = group_sizes.sum()
    targets = {s: ratios[s] * total for s in SPLITS}
    filled = {s: 0 for s in SPLITS}
    assign = {}
    for g in groups:
        # đưa nhóm vào tập đang thiếu nhiều nhất so với chỉ tiêu (tương đối)
        split = max(SPLITS, key=lambda s: (targets[s] - filled[s]) / max(targets[s], 1e-9))
        assign[g] = split
        filled[split] += group_sizes[g]
    return assign


def grouped_split(df: pd.DataFrame, ratios: dict[str, float], seed: int = 42, attempts: int = 300) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Chia theo video, tỉ lệ tính trên số ảnh nguồn.

    Thử nhiều hoán vị (xác định bởi seed) và chọn phương án có
    (1) nhiều lớp có mặt ở cả val và test nhất, (2) tỉ lệ ảnh gần mục tiêu nhất.
    """
    if abs(sum(ratios[s] for s in SPLITS) - 1.0) > 1e-6:
        raise ValueError("Tổng tỉ lệ chia phải bằng 1")
    imgs_per_video = df.groupby("video_id")["image_id"].nunique()
    classes = sorted(df["class_index"].unique())
    rng = np.random.default_rng(seed)
    best, best_key, best_info = None, None, None
    for attempt in range(attempts):
        assign = _assign_groups(imgs_per_video, ratios, rng)
        split = df["video_id"].map(assign)
        present = df.assign(split=split).groupby("split")["class_index"].apply(set)
        covered = sum(1 for c in classes if all(c in present.get(s, set()) for s in ("val", "test")))
        img_frac = df.assign(split=split).groupby("split")["image_id"].nunique() / df["image_id"].nunique()
        dev = sum(abs(img_frac.get(s, 0) - ratios[s]) for s in SPLITS)
        key = (covered, -dev)
        if best_key is None or key > best_key:
            best, best_key = split, key
            best_info = {"attempt": attempt, "classes_in_val_and_test": covered, "num_classes": len(classes),
                         "image_fraction": {s: round(float(img_frac.get(s, 0)), 4) for s in SPLITS}}
        if covered == len(classes) and dev < 0.02:
            break
    out = df.copy()
    out["split"] = best
    best_info.update({"seed": seed, "num_videos": int(imgs_per_video.size)})
    return out, best_info


def check_no_leakage(df: pd.DataFrame) -> None:
    """Báo lỗi nếu một video hoặc một ảnh nguồn nằm ở hơn một tập."""
    for col in ("video_id", "image_id"):
        n = df.groupby(col)["split"].nunique()
        bad = n[n > 1]
        if len(bad):
            raise AssertionError(f"Rò rỉ dữ liệu: {len(bad)} {col} xuất hiện ở nhiều tập, ví dụ {list(bad.index[:3])}")


# --------------------------------------------------------------------------- 5. Crop
def crop_box(img: Image.Image, bbox: tuple[float, float, float, float], padding_ratio: float,
             min_side: float, size: int) -> Image.Image | None:
    """Crop bbox COCO (x, y, w, h) có lề, trả ảnh RGB size×size hoặc None nếu box không hợp lệ."""
    x, y, w, h = bbox
    if w < min_side or h < min_side:
        return None
    W, H = img.size
    px, py = w * padding_ratio, h * padding_ratio
    x0, y0 = max(0, int(round(x - px))), max(0, int(round(y - py)))
    x1, y1 = min(W, int(round(x + w + px))), min(H, int(round(y + h + py)))
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return None
    return img.crop((x0, y0, x1, y1)).convert("RGB").resize((size, size), Image.BILINEAR)


def export_crops(df: pd.DataFrame, out_dir: str | Path, class_names: list[str], contract: dict[str, Any],
                 verbose: bool = True) -> pd.DataFrame:
    """Lưu crop vào <out_dir>/<split>/<class_name>/<uid>.jpg. Mọi thư mục lớp được tạo ở mọi tập."""
    out_dir = Path(out_dir)
    size = int(contract["image"]["input_size"])
    pad = float(contract["crop"]["padding_ratio"])
    min_side = float(contract["crop"]["min_box_side_px"])
    for s in SPLITS:
        for c in class_names:
            (out_dir / s / c).mkdir(parents=True, exist_ok=True)

    status, paths = {}, {}
    groups = df.groupby("image_path")
    for i, (img_path, g) in enumerate(groups):
        if not img_path:
            for uid in g["uid"]:
                status[uid], paths[uid] = "missing_image", ""
            continue
        try:
            with Image.open(img_path) as im:
                im.load()
                for r in g.itertuples():
                    crop = crop_box(im, (r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h), pad, min_side, size)
                    if crop is None:
                        status[r.uid], paths[r.uid] = "too_small", ""
                        continue
                    rel = Path(r.split) / r.class_name / f"{r.uid}.jpg"
                    crop.save(out_dir / rel, quality=95)
                    status[r.uid], paths[r.uid] = "ok", str(rel)
        except (OSError, ValueError) as e:
            for uid in g["uid"]:
                status[uid], paths[uid] = f"unreadable:{type(e).__name__}", ""
        if verbose and (i + 1) % 500 == 0:
            print(f"  đã xử lý {i + 1}/{len(groups)} ảnh nguồn")
    out = df.copy()
    out["crop_status"] = out["uid"].map(status)
    out["crop_path"] = out["uid"].map(paths)
    return out


# --------------------------------------------------------------------------- 6. Tóm tắt
def split_summary(df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """Bảng số crop hợp lệ theo lớp × tập (đủ 22 lớp, kể cả lớp 0 mẫu)."""
    ok = df[df["crop_status"] == "ok"] if "crop_status" in df else df
    table = ok.pivot_table(index="class_name", columns="split", values="uid", aggfunc="count", fill_value=0)
    table = table.reindex(index=class_names, columns=list(SPLITS), fill_value=0)
    table["total"] = table.sum(axis=1)
    return table.astype(int)


def run_preprocessing(raw_root: str | Path, out_dir: str | Path, contract_path: str | Path = DEFAULT_CONTRACT,
                      mapping_out: str | Path | None = None, verbose: bool = True) -> dict[str, Any]:
    """Chạy toàn bộ quy trình. Trả dict tóm tắt (cũng ghi ra preprocessing_summary.json)."""
    contract = load_contract(contract_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, mapping = build_object_table(raw_root, contract)
    if verbose:
        print(f"Annotation: {len(df)} · ảnh nguồn: {df['image_id'].nunique()} · video: {df['video_id'].nunique()} · lớp: {len(mapping)}")
        missing = (df["image_path"] == "").sum()
        if missing:
            print(f"  [CẢNH BÁO] {missing} annotation không tìm thấy file ảnh")
    expected = contract["classes"].get("num_classes")
    if expected and len(mapping) != expected:
        raise ValueError(f"Data contract yêu cầu {expected} lớp nhưng dataset có {len(mapping)}")

    df, split_info = grouped_split(df, contract["split"]["ratios"], int(contract["split"]["seed"]))
    check_no_leakage(df)
    names = [m["class_name"] for m in mapping]
    df = export_crops(df, out_dir, names, contract, verbose=verbose)

    df.to_csv(out_dir / contract["split"].get("manifest", "split_manifest.csv"), index=False)
    save_class_mapping(mapping, out_dir / "class_mapping.json", source="instances_train/val_trashcan.json")
    if mapping_out:
        save_class_mapping(mapping, mapping_out, source="instances_train/val_trashcan.json")
    table = split_summary(df, names)
    table.to_csv(out_dir / "class_split_counts.csv")

    summary = {
        "data_version": contract["dataset"].get("data_version"),
        "num_annotations": int(len(df)),
        "num_source_images": int(df["image_id"].nunique()),
        "num_videos": int(df["video_id"].nunique()),
        "crop_status": {k: int(v) for k, v in df["crop_status"].value_counts().items()},
        "split": split_info,
        "crops_per_split": {s: int(table[s].sum()) for s in SPLITS},
        "classes_missing_in": {s: [c for c in names if table.loc[c, s] == 0] for s in SPLITS},
    }
    with open(out_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if verbose:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Tiền xử lý TrashCan-Instance theo data contract")
    ap.add_argument("--raw-root", required=True, help="Thư mục chứa dataset TrashCan đã giải nén")
    ap.add_argument("--out-dir", required=True, help="Thư mục đầu ra, ví dụ .../data/processed/v1_crop128")
    ap.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    ap.add_argument("--mapping-out", default=None, help="Ghi thêm class_mapping.json vào đường dẫn này (vd configs/)")
    args = ap.parse_args()
    run_preprocessing(args.raw_root, args.out_dir, args.contract, args.mapping_out)


if __name__ == "__main__":
    main()
