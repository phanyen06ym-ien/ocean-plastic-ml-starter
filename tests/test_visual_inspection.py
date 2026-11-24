"""scripts/visual_inspection_prepare.py: chọn mẫu tái lập được, đúng chỉ tiêu, không dùng Test."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vip", ROOT / "scripts" / "visual_inspection_prepare.py")
vip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vip)


def _manifest(n_val_rope=2):
    rows, i = [], 0
    for ci, cls in enumerate(vip.PLASTIC + ["rov"]):
        for split, n in (("train", 30), ("val", n_val_rope if cls == "trash_rope" else 8), ("test", 8)):
            for _ in range(n):
                i += 1
                rows.append({"uid": f"train_{i}", "source_split": "train", "ann_id": i, "image_id": f"train_{i // 3}",
                             "video_id": f"vid_{i // 10}", "class_name": cls, "class_index": ci, "split": split,
                             "crop_status": "ok", "crop_path": f"{split}/{cls}/{i}.jpg", "file_name": f"vid_{i // 10}_f{i}.jpg",
                             "bbox_x": 5, "bbox_y": 5, "bbox_w": 20, "bbox_h": 20})
    return pd.DataFrame(rows)


def test_selection_quota_and_reproducible():
    m = _manifest()
    a, b = vip.select_samples(m), vip.select_samples(m)
    assert a.equals(b)
    main = a[a["set"] == "main"]
    assert len(main) == 120 and len(a[a["set"] == "pilot"]) == 10
    assert set(a["split"]) <= {"train", "val"} and "rov" not in set(a["class_name"])
    assert a["annotation_id"].is_unique
    per = main.groupby(["class_name", "split"]).size()
    assert per[("trash_bag", "train")] == 15 and per[("trash_bag", "val")] == 5
    assert per[("trash_rope", "val")] == 2 and per[("trash_rope", "train")] == 18   # thiếu val -> bù train
    assert list(a["sample_no"][:2]) == ["P01", "P02"] and main["sample_no"].iloc[0] == "001"


def test_contact_pages(tmp_path):
    m = _manifest()
    s = vip.select_samples(m).head(12)
    for p in s["crop_path"]:
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((224, 224, 3), 120, dtype=np.uint8)).save(tmp_path / p)
    src = tmp_path / "src.jpg"
    Image.fromarray(np.full((240, 320, 3), 60, dtype=np.uint8)).save(src)
    sources = {u: (src if k % 2 else None) for k, u in enumerate(s["annotation_id"])}
    pages = vip.contact_pages(s, tmp_path, sources)
    assert len(pages) >= 3 and all(p.width == 1240 for p in pages)
    t = vip.review_template(s)
    assert list(t.columns[-5:]) == vip.REVIEW_COLS and (t["visibility"] == "").all()
