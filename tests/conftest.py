"""Fixture dùng chung: bộ dữ liệu giả lập có cấu trúc giống TrashCan 1.0 — Instance.

Chỉ dùng để kiểm thử code; KHÔNG phải dữ liệu thật.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATEGORY_NAMES = [
    "rov", "plant", "animal_fish", "animal_starfish", "animal_shells", "animal_crab", "animal_eel",
    "animal_etc", "trash_clothing", "trash_pipe", "trash_bottle", "trash_bag", "trash_snack_wrapper",
    "trash_can", "trash_cup", "trash_container", "trash_unknown_instance", "trash_branch",
    "trash_wreckage", "trash_tarp", "trash_rope", "trash_net",
]


def make_fake_trashcan(root: Path, n_videos: int = 40, frames: int = 3, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    base = root / "dataset" / "instance_version"
    cats = [{"id": i + 1, "name": n, "supercategory": ""} for i, n in enumerate(CATEGORY_NAMES)]
    coco = {s: {"images": [], "annotations": [], "categories": cats} for s in ("train", "val")}
    img_id = ann_id = 0
    for v in range(n_videos):
        split = "val" if v % 5 == 0 else "train"
        (base / split).mkdir(parents=True, exist_ok=True)
        for f in range(frames):
            img_id += 1
            fn = f"vid_{v:06d}_frame{f:07d}.jpg"
            arr = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
            Image.fromarray(arr).save(base / split / fn)
            coco[split]["images"].append({"id": img_id, "file_name": fn, "width": 160, "height": 120})
            for k in range(2):
                ann_id += 1
                cat = (v * 2 + k) % len(CATEGORY_NAMES) + 1
                w, h = (4, 4) if (v + f + k) % 17 == 0 else (30 + k * 5, 25)
                coco[split]["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": cat,
                                                   "bbox": [10 + k * 40, 20, w, h], "area": w * h, "iscrowd": 0})
    for s in ("train", "val"):
        with open(base / f"instances_{s}_trashcan.json", "w") as fh:
            json.dump(coco[s], fh)
    return root / "dataset"


@pytest.fixture
def fake_trashcan(tmp_path):
    return make_fake_trashcan(tmp_path / "raw")
