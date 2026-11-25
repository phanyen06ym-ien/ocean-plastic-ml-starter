"""Tiền xử lý TrashCan 1.0 — Instance theo data contract (configs/data_contract.yaml, configs/data_contract_v2.yaml).

Quy trình (một lần chạy tạo lại toàn bộ dữ liệu):
    1. Tìm 2 file annotation COCO (train, val) trong thư mục dataset gốc.
    2. Sinh class mapping 22 lớp (class_index theo category_id tăng dần).
    3. Lập bảng đối tượng: mỗi dòng là một annotation (bbox) + ảnh nguồn + video id.
    4. Gộp train+val gốc, chia lại Train/Val/Test THEO VIDEO (chia trước khi crop).
    5. Crop bbox (+ lề), bỏ box quá nhỏ, resize về input_size, lưu
       <out_dir>/<split>/<class_name>/<ann_id>.jpg
    6. Ghi split_manifest.csv, class_mapping.json, preprocessing_summary.json.

Hai phiên bản dữ liệu (chọn bằng --contract):
    v1_crop128 (E01–E03): crop theo bbox + 10% lề, resize thẳng 128x128, bỏ box < 8 px,
                          chia theo video bằng cách thử ngẫu nhiên nhiều hoán vị.
    v2_crop224 (E04):     mở bbox thành hình vuông + 15% lề, 224x224, bỏ box < 16 px,
                          chia theo video, tìm phân công giữ tỉ lệ 70/15/15 cho từng lớp.

Module này cũng chứa các tiện ích dùng chung cho cả dự án (đọc data contract, class mapping)
và hàm tải TrashCan 1.0 từ UMN Data Repository (download_trashcan).

Chạy từ dòng lệnh:
    python -m src.preprocessing --raw-root /path/to/trashcan --out-dir /path/to/processed/v1_crop128
    python -m src.preprocessing --raw-root /path/to/trashcan --out-dir /path/to/processed/v2_crop224 \
        --contract configs/data_contract_v2.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image


# --------------------------------------------------------------------------- 0. Tiện ích dùng chung
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "configs" / "data_contract.yaml"
DEFAULT_MAPPING = PROJECT_ROOT / "configs" / "class_mapping.json"


def ensure_utf8_stdio() -> None:
    """In tiếng Việt an toàn trên Windows: khi stdout/stderr bị chuyển hướng (> file, pipe), Python dùng cp1252
    và báo UnicodeEncodeError. Chuyển cả hai sang UTF-8; bỏ qua nếu luồng không hỗ trợ reconfigure."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Đọc file YAML (UTF-8)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_contract(path: str | Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    """Đọc configs/data_contract.yaml."""
    return load_yaml(path)


def load_class_mapping(path: str | Path = DEFAULT_MAPPING) -> list[dict[str, Any]]:
    """Đọc class mapping, trả danh sách sắp xếp theo class_index.

    Mỗi phần tử: {class_index, category_id, class_name, plastic_related}.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    classes = data["classes"] if isinstance(data, dict) else data
    classes = sorted(classes, key=lambda c: c["class_index"])
    for expected, c in enumerate(classes):
        if c["class_index"] != expected:
            raise ValueError(f"class_index không liên tục tại {c}")
    return classes


def class_names(mapping: list[dict[str, Any]]) -> list[str]:
    return [c["class_name"] for c in mapping]


def plastic_indices(mapping: list[dict[str, Any]]) -> list[int]:
    return [c["class_index"] for c in mapping if c.get("plastic_related")]


# --------------------------------------------------------------------------- 0b. Tải dữ liệu gốc
# Tải TrashCan 1.0 (dataset.zip) trực tiếp từ UMN Data Repository (DSpace 7 REST API).
# Dùng trên Colab khi Drive `data/raw/` chưa có file .zip.
UMN_API = "https://conservancy.umn.edu/server/api/core"
TRASHCAN_ITEM_UUID = "6dd6a960-c44a-4510-a679-efb8c82ebfb7"


def _get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def pick_bitstream(bitstreams: list[dict[str, Any]], name: str = "dataset.zip") -> dict[str, Any]:
    """Chọn bitstream theo tên (không phân biệt hoa thường); nếu không có thì lấy file .zip lớn nhất."""
    exact = [b for b in bitstreams if b.get("name", "").lower() == name.lower()]
    if exact:
        return exact[0]
    zips = [b for b in bitstreams if b.get("name", "").lower().endswith(".zip")]
    if not zips:
        raise FileNotFoundError(f"Không thấy file .zip trong item; có: {[b.get('name') for b in bitstreams]}")
    return max(zips, key=lambda b: b.get("sizeBytes", 0))


def list_bitstreams(item_uuid: str = TRASHCAN_ITEM_UUID, api: str = UMN_API) -> list[dict[str, Any]]:
    bundles = _get_json(f"{api}/items/{item_uuid}/bundles")["_embedded"]["bundles"]
    out: list[dict[str, Any]] = []
    for b in bundles:
        if b.get("name") != "ORIGINAL":
            continue
        data = _get_json(b["_links"]["bitstreams"]["href"])
        out.extend(data["_embedded"]["bitstreams"])
    return out


def download_trashcan(dest_dir: str | Path, name: str = "dataset.zip") -> Path:
    """Tải dataset.zip vào dest_dir (bỏ qua nếu đã có file cùng tên, đúng dung lượng)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    bs = pick_bitstream(list_bitstreams(), name)
    target = dest_dir / bs["name"]
    size = bs.get("sizeBytes")
    if target.exists() and (size is None or target.stat().st_size == size):
        print("Đã có", target)
        return target
    url = bs["_links"]["content"]["href"]
    print(f"Tải {bs['name']} ({(size or 0) / 1e9:.2f} GB) từ {url}")
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=16 << 20)
    tmp.rename(target)
    return target

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
def build_class_mapping(categories: list[dict[str, Any]], plastic_related: list[str] | None = None,
                        merge: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """category_id → class_name → class_index.

    Không gộp (v1): 22 lớp, class_index 0..21 theo category_id tăng dần.
    merge = {tên_gốc: tên_mới} (v2): các category cùng tên mới thành một lớp; thứ tự lớp theo
    category_id nhỏ nhất của nhóm. Mỗi phần tử có thêm `category_ids` và `source_names`.
    """
    plastic = set(plastic_related or [])
    merge = merge or {}
    cats = sorted(categories, key=lambda c: c["id"])
    names = [c["name"] for c in cats]
    if len(set(names)) != len(names):
        raise ValueError("Tên category bị trùng")
    unknown = set(merge) - set(names)
    if unknown:
        raise ValueError(f"classes.merge có tên không tồn tại trong dataset: {sorted(unknown)}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in cats:                                   # dict giữ thứ tự xuất hiện → theo category_id nhỏ nhất
        groups.setdefault(merge.get(c["name"], c["name"]), []).append(c)
    out = []
    for i, (name, members) in enumerate(groups.items()):
        entry = {"class_index": i, "category_id": members[0]["id"], "class_name": name, "plastic_related": name in plastic}
        if merge:
            entry["category_ids"] = [m["id"] for m in members]
            entry["source_names"] = [m["name"] for m in members]
        out.append(entry)
    return out


def save_class_mapping(mapping: list[dict[str, Any]], path: str | Path, source: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "TrashCan 1.0 — Instance",
        "num_classes": len(mapping),
        "index_order": "min_category_id_of_group" if any("category_ids" in m for m in mapping) else "category_id_ascending",
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
                                  contract["classes"].get("plastic_related_candidates"),
                                  contract["classes"].get("merge"))
    by_cat = {cid: m for m in mapping for cid in m.get("category_ids", [m["category_id"]])}

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


def _split_key(df: pd.DataFrame, split: pd.Series, ratios: dict[str, float], classes: list[int]) -> tuple[tuple, dict]:
    """Điểm của một phương án chia: (số lớp có ở cả val và test, số mẫu ít nhất của một lớp ở val/test, -độ lệch tỉ lệ)."""
    tmp = df.assign(split=split.values)
    counts = tmp.groupby(["split", "class_index"]).size()
    covered = sum(1 for c in classes if all(counts.get((s, c), 0) > 0 for s in ("val", "test")))
    min_eval = min(int(counts.get((s, c), 0)) for c in classes for s in ("val", "test"))
    img_frac = tmp.groupby("split")["image_id"].nunique() / tmp["image_id"].nunique()
    dev = sum(abs(float(img_frac.get(s, 0)) - ratios[s]) for s in SPLITS)
    info = {"classes_in_val_and_test": covered, "num_classes": len(classes), "min_class_count_val_test": min_eval,
            "image_fraction": {s: round(float(img_frac.get(s, 0)), 4) for s in SPLITS}}
    return (covered, min(min_eval, 5), -round(dev, 4)), info


def _split_loss(M: np.ndarray, I: np.ndarray, T: np.ndarray, r: np.ndarray) -> float:
    """Độ lệch của một phương án chia so với tỉ lệ mục tiêu.

    M: số mẫu (3 tập x C lớp), I: số ảnh nguồn mỗi tập, T: tổng mẫu mỗi lớp, r: tỉ lệ mục tiêu.
    Mỗi lớp có trọng số như nhau (phù hợp Macro F1); phạt thêm khi một lớp trống ở một tập.
    """
    share = M / np.maximum(T, 1)[None, :]
    loss = float((((share - r[:, None]) ** 2) / r[:, None]).sum())
    loss += 0.5 * M.shape[1] * float((((I / max(I.sum(), 1)) - r) ** 2 / r).sum())
    loss += 2.0 * float((M == 0).sum())
    return loss


def stratified_grouped_split(df: pd.DataFrame, ratios: dict[str, float], seed: int = 42, n_restarts: int = 30,
                             min_box_side: float = 0.0, max_passes: int = 40) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Chia theo video, giữ tỉ lệ 70/15/15 cho TỪNG lớp (v2, E04).

    Mọi annotation của một video nằm trong cùng một tập (không rò rỉ). Mục tiêu: với mỗi lớp,
    tỉ lệ mẫu ở Train/Val/Test gần ratios nhất, đồng thời tỉ lệ ảnh nguồn gần ratios.
    Cách làm: bắt đầu từ một phân công ngẫu nhiên các video, lần lượt thử chuyển từng video sang
    tập khác nếu làm giảm độ lệch (_split_loss), lặp đến khi không cải thiện được; làm lại với
    n_restarts khởi tạo và giữ phương án tốt nhất. Kết quả xác định hoàn toàn bởi seed.
    Chỉ các box đủ lớn (cạnh >= min_box_side, sẽ được crop) được tính vào số mẫu.
    """
    if abs(sum(ratios[s] for s in SPLITS) - 1.0) > 1e-6:
        raise ValueError("Tổng tỉ lệ chia phải bằng 1")
    df = df.reset_index(drop=True)
    classes = sorted(df["class_index"].unique())
    eligible = df
    if min_box_side and {"bbox_w", "bbox_h"} <= set(df.columns):
        eligible = df[(df["bbox_w"] >= min_box_side) & (df["bbox_h"] >= min_box_side)]
    videos = sorted(df["video_id"].unique())
    counts = (eligible.pivot_table(index="video_id", columns="class_index", values="image_id", aggfunc="count", fill_value=0)
              .reindex(index=videos, columns=classes, fill_value=0))
    A = counts.to_numpy(dtype=float)
    img = df.groupby("video_id")["image_id"].nunique().reindex(videos).to_numpy(dtype=float)
    T, r = A.sum(0), np.array([ratios[s] for s in SPLITS], dtype=float)
    V = len(videos)

    rng = np.random.default_rng(seed)
    best_loss, best_assign = None, None
    for _ in range(n_restarts):
        a = rng.choice(3, size=V, p=r)
        M = np.stack([A[a == k].sum(0) for k in range(3)])
        I = np.array([img[a == k].sum() for k in range(3)])
        cur = _split_loss(M, I, T, r)
        for _ in range(max_passes):
            improved = False
            for v in rng.permutation(V):
                k0 = a[v]
                for k1 in range(3):
                    if k1 == k0:
                        continue
                    M[k0] -= A[v]; M[k1] += A[v]; I[k0] -= img[v]; I[k1] += img[v]
                    new = _split_loss(M, I, T, r)
                    if new < cur - 1e-12:
                        cur, a[v], k0, improved = new, k1, k1, True
                    else:
                        M[k1] -= A[v]; M[k0] += A[v]; I[k1] -= img[v]; I[k0] += img[v]
            if not improved:
                break
        if best_loss is None or cur < best_loss:
            best_loss, best_assign = cur, a.copy()

    assign = {vid: SPLITS[k] for vid, k in zip(videos, best_assign)}
    out = df.copy()
    out["split"] = out["video_id"].map(assign)
    M = np.stack([A[best_assign == k].sum(0) for k in range(3)])
    img_frac = out.groupby("split")["image_id"].nunique() / out["image_id"].nunique()
    info = {"method": "stratified_video_search", "seed": seed, "n_restarts": n_restarts, "loss": round(best_loss, 4),
            "classes_in_val_and_test": int(sum(1 for j in range(len(classes)) if M[1, j] > 0 and M[2, j] > 0)),
            "num_classes": len(classes), "min_class_count_val_test": int(M[1:].min()),
            "max_class_share_deviation": round(float(np.abs(M / np.maximum(T, 1) - r[:, None]).max()), 4),
            "image_fraction": {s: round(float(img_frac.get(s, 0)), 4) for s in SPLITS},
            "num_videos": int(V)}
    return out, info


def split_by_contract(df: pd.DataFrame, split_cfg: dict[str, Any], min_box_side: float = 0.0) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Chọn cách chia theo data contract: split.method = random_group_search (v1) | stratified_video_search (v2)."""
    method = split_cfg.get("method", "random_group_search")
    ratios, seed = split_cfg["ratios"], int(split_cfg["seed"])
    if method == "stratified_video_search":
        return stratified_grouped_split(df, ratios, seed, min_box_side=min_box_side)
    if method == "random_group_search":
        out, info = grouped_split(df, ratios, seed)
        info["method"] = method
        return out, info
    raise ValueError(f"split.method không hỗ trợ: {method}")


def check_no_leakage(df: pd.DataFrame) -> None:
    """Báo lỗi nếu một video hoặc một ảnh nguồn nằm ở hơn một tập."""
    for col in ("video_id", "image_id"):
        n = df.groupby(col)["split"].nunique()
        bad = n[n > 1]
        if len(bad):
            raise AssertionError(f"Rò rỉ dữ liệu: {len(bad)} {col} xuất hiện ở nhiều tập, ví dụ {list(bad.index[:3])}")


# --------------------------------------------------------------------------- 5. Crop
def _square_window(x: float, y: float, w: float, h: float, padding_ratio: float, W: int, H: int) -> tuple[int, int, int, int]:
    """Cửa sổ vuông quanh tâm bbox, cạnh = max(w, h) * (1 + 2 * lề), dời vào trong ảnh khi chạm mép."""
    side = min(max(w, h) * (1 + 2 * padding_ratio), W, H)
    cx, cy = x + w / 2, y + h / 2
    x0 = min(max(cx - side / 2, 0), W - side)
    y0 = min(max(cy - side / 2, 0), H - side)
    return int(round(x0)), int(round(y0)), int(round(x0 + side)), int(round(y0 + side))


def crop_box(img: Image.Image, bbox: tuple[float, float, float, float], padding_ratio: float,
             min_side: float, size: int, square: bool = False) -> Image.Image | None:
    """Crop bbox COCO (x, y, w, h) có lề, trả ảnh RGB size×size hoặc None nếu box không hợp lệ.

    square=False (v1): lề theo từng chiều rồi resize thẳng (đối tượng có thể bị méo).
    square=True  (v2): mở rộng thành cửa sổ vuông quanh tâm bbox, giữ đúng tỉ lệ đối tượng.
    """
    x, y, w, h = bbox
    if w < min_side or h < min_side:
        return None
    W, H = img.size
    if square:
        x0, y0, x1, y1 = _square_window(x, y, w, h, padding_ratio, W, H)
    else:
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
    square = bool(contract["crop"].get("square", False))
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
                    crop = crop_box(im, (r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h), pad, min_side, size, square)
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

    df, split_info = split_by_contract(df, contract["split"], float(contract["crop"]["min_box_side_px"]))
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
        "dropped_fraction": round(float((df["crop_status"] != "ok").mean()), 4),
    }
    if verbose and summary["dropped_fraction"] > 0.05:
        print(f"  [CẢNH BÁO] {summary['dropped_fraction']:.1%} annotation bị loại (box nhỏ / thiếu ảnh) — "
              "cân nhắc giảm crop.min_box_side_px")
    with open(out_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if verbose:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="Tiền xử lý TrashCan-Instance theo data contract")
    ap.add_argument("--raw-root", required=True, help="Thư mục chứa dataset TrashCan đã giải nén")
    ap.add_argument("--out-dir", required=True, help="Thư mục đầu ra, ví dụ .../data/processed/v1_crop128")
    ap.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    ap.add_argument("--mapping-out", default=None, help="Ghi thêm class_mapping.json vào đường dẫn này (vd configs/)")
    args = ap.parse_args()
    run_preprocessing(args.raw_root, args.out_dir, args.contract, args.mapping_out)


if __name__ == "__main__":
    main()
