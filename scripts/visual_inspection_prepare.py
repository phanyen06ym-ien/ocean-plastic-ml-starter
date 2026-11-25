"""Chuẩn bị bộ kiểm tra ảnh bằng mắt (docs/visual_inspection_protocol.md) trên dữ liệu v2_crop224.

    python scripts/visual_inspection_prepare.py \
        --processed-dir /content/processed/v2_crop224 --raw-root /content/raw \
        --out /content/drive/MyDrive/ocean-plastic-ml/audits/visual_inspection_v2

Chọn mẫu (tái lập được, không xem ảnh trước khi chọn):
- Chính thức: 6 lớp nhựa × 20 annotation = 120; mỗi lớp 15 Train + 5 Validation, thiếu thì bù từ split còn lại.
  Không dùng tập Test. random.Random(42); duyệt lớp theo class_index tăng dần, train trước val;
  ứng viên sắp theo (source_split, ann_id) rồi rng.sample(...).
- Chấm thử (pilot): 10 annotation lấy ngẫu nhiên (cùng rng, sau mẫu chính thức) từ phần còn lại của 6 lớp,
  Train + Validation, không trùng mẫu chính thức. Không tính vào kết quả.

Ghi ra --out:
- visual_inspection_sample_manifest.csv   danh sách mẫu (sample_no P01–P10, 001–120)
- visual_inspection_contact_sheets.pdf    mỗi mẫu: ảnh crop mô hình thấy + ảnh gốc có khung bbox
- visual_inspection_review_template.csv   bảng trống để mỗi người chấm điền
- visual_inspection_provenance.json       seed, thuật toán, SHA-256 manifest nguồn và manifest mẫu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SEED = 42
PER_CLASS, TRAIN_K, VAL_K, PILOT_N = 20, 15, 5, 10
PLASTIC = ["trash_bag", "trash_bottle", "trash_snack_wrapper", "trash_net", "trash_rope", "trash_tarp"]
REVIEW_COLS = ["visibility", "label_match", "quality", "material_evidence", "note"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(row) -> tuple:
    return (row["source_split"], int(row["ann_id"]))


def select_samples(manifest: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Trả về mẫu pilot + chính thức theo quy tắc ở docstring; manifest là split_manifest.csv của v2."""
    df = manifest[(manifest["crop_status"] == "ok") & manifest["split"].isin(["train", "val"])
                  & manifest["class_name"].isin(PLASTIC)].copy()
    df["_k"] = df.apply(_key, axis=1)
    rng = random.Random(seed)
    chosen = []
    order = df.drop_duplicates("class_name").sort_values("class_index")["class_name"]
    for cls in order:
        pools = {s: sorted(df[(df["class_name"] == cls) & (df["split"] == s)]["_k"]) for s in ("train", "val")}
        want = {"train": min(TRAIN_K, len(pools["train"])), "val": min(VAL_K, len(pools["val"]))}
        short = PER_CLASS - sum(want.values())
        for s, o in (("train", "val"), ("val", "train")):          # bù từ split còn lại nếu thiếu
            extra = min(short, len(pools[s]) - want[s])
            if extra > 0:
                want[s] += extra
                short -= extra
        for s in ("train", "val"):
            chosen += [("main", k) for k in rng.sample(pools[s], want[s])]
    taken = {k for _, k in chosen}
    rest = sorted(k for k in df["_k"] if k not in taken)
    chosen = [("pilot", k) for k in rng.sample(rest, min(PILOT_N, len(rest)))] + chosen
    pos = {k: i for i, k in enumerate(df["_k"])}
    rows = []
    n_main = n_pilot = 0
    for kind, k in chosen:
        r = df.iloc[pos[k]]
        if kind == "pilot":
            n_pilot += 1
            no = f"P{n_pilot:02d}"
        else:
            n_main += 1
            no = f"{n_main:03d}"
        rows.append({"sample_no": no, "set": kind, "annotation_id": r["uid"], "ann_id": int(r["ann_id"]),
                     "image_id": r["image_id"], "video_id": r["video_id"], "class_name": r["class_name"],
                     "split": r["split"], "crop_path": r["crop_path"], "file_name": r["file_name"],
                     "source_split": r["source_split"], "bbox_x": r["bbox_x"], "bbox_y": r["bbox_y"],
                     "bbox_w": r["bbox_w"], "bbox_h": r["bbox_h"]})
    return pd.DataFrame(rows)


def find_source(raw_root: Path, file_name: str, source_split: str, index: dict[str, list[Path]]) -> Path | None:
    hits = index.get(Path(file_name).name, [])
    pref = [h for h in hits if source_split in h.parts]
    return (pref or hits or [None])[0]


def _font(size: int):
    cands = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    try:
        from matplotlib import font_manager
        cands.insert(0, font_manager.findfont("DejaVu Sans", fallback_to_default=True))
    except Exception:  # pragma: no cover
        pass
    for f in cands + ["DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sample_panel(row, processed_dir: Path, source: Path | None, h: int = 300) -> Image.Image:
    crop = Image.open(processed_dir / row["crop_path"]).convert("RGB").resize((h, h))
    if source is not None and source.exists():
        src = Image.open(source).convert("RGB")
        d = ImageDraw.Draw(src)
        x, y, w, hh = row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]
        d.rectangle([x, y, x + w, y + hh], outline=(255, 40, 40), width=max(2, src.width // 200))
        ctx = src.resize((int(src.width * h / src.height), h))
    else:
        ctx = Image.new("RGB", (int(h * 1.33), h), (230, 230, 230))
        ImageDraw.Draw(ctx).text((10, h // 2), "không tìm thấy ảnh gốc", fill=(80, 80, 80), font=_font(14))
    label_w = 190
    panel = Image.new("RGB", (label_w + h + 8 + ctx.width, h + 8), "white")
    d = ImageDraw.Draw(panel)
    d.text((8, 10), row["sample_no"], fill=(0, 0, 0), font=_font(40))
    d.text((8, 64), "Nhãn:", fill=(90, 90, 90), font=_font(18))
    d.text((8, 88), row["class_name"].replace("trash_", ""), fill=(30, 64, 175), font=_font(24))
    d.text((8, 124), row["split"], fill=(90, 90, 90), font=_font(18))
    panel.paste(crop, (label_w, 4))
    panel.paste(ctx, (label_w + h + 8, 4))
    return panel


def contact_pages(samples: pd.DataFrame, processed_dir: Path, sources: dict[str, Path | None],
                  per_page: int = 4, width: int = 1240) -> list[Image.Image]:
    pages = []
    group = samples["class_name"].where(samples["set"] == "main", "pilot")      # chấm thử: một nhóm chung
    for (kind, cls), g in samples.groupby([samples["set"], group], sort=False):
        title = "CHẤM THỬ (không tính kết quả)" if kind == "pilot" else f"Lớp: {cls}"
        rows = list(g.iterrows())
        for i in range(0, len(rows), per_page):
            chunk = rows[i:i + per_page]
            panels = [sample_panel(r, processed_dir, sources.get(r["annotation_id"])) for _, r in chunk]
            page = Image.new("RGB", (width, 75 + sum(p.height + 10 for p in panels)), "white")
            d = ImageDraw.Draw(page)
            d.text((20, 12), f"{title} · trang {i // per_page + 1}", fill=(0, 0, 0), font=_font(24))
            d.text((20, 40), "Trái: ảnh crop mô hình thấy · Phải: ảnh gốc, khung đỏ = vật thể được gán nhãn",
                   fill=(90, 90, 90), font=_font(16))
            y = 70
            for p in panels:
                if p.width > width - 20:
                    p = p.resize((width - 20, int(p.height * (width - 20) / p.width)))
                page.paste(p, (10, y))
                y += p.height + 10
            pages.append(page)
    return pages


def review_template(samples: pd.DataFrame) -> pd.DataFrame:
    t = samples[["sample_no", "set", "annotation_id", "class_name", "split"]].copy()
    for c in REVIEW_COLS:
        t[c] = ""
    return t


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.preprocessing import ensure_utf8_stdio
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--raw-root", required=True, help="thư mục đã giải nén dataset.zip (TrashCan gốc)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    proc, raw, out = Path(a.processed_dir), Path(a.raw_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man_path = proc / "split_manifest.csv"
    samples = select_samples(pd.read_csv(man_path))
    index: dict[str, list[Path]] = {}
    for p in raw.rglob("*.jpg"):
        index.setdefault(p.name, []).append(p)
    sources = {r["annotation_id"]: find_source(raw, r["file_name"], r["source_split"], index) for _, r in samples.iterrows()}
    samples["source_found"] = samples["annotation_id"].map(lambda u: sources[u] is not None)
    sm = out / "visual_inspection_sample_manifest.csv"
    samples.to_csv(sm, index=False)
    review_template(samples).to_csv(out / "visual_inspection_review_template.csv", index=False)
    pages = contact_pages(samples, proc, sources)
    pages[0].save(out / "visual_inspection_contact_sheets.pdf", save_all=True, append_images=pages[1:], resolution=110)
    prov = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "prng": "random.Random(42), Python stdlib",
            "python": platform.python_version(), "data_version": "v2_crop224",
            "rule": "per class (class_index asc): train then val; candidates sorted by (source_split, ann_id); "
                    "15 train + 5 val, top up from other split; then 10 pilot from the rest; test split excluded",
            "source_manifest_sha256": sha256(man_path), "sample_manifest_sha256": sha256(sm),
            "counts": {"main": int((samples["set"] == "main").sum()), "pilot": int((samples["set"] == "pilot").sum()),
                       "unique_images": int(samples["image_id"].nunique()), "unique_videos": int(samples["video_id"].nunique()),
                       "source_images_found": int(samples["source_found"].sum())},
            "per_class_split": samples[samples["set"] == "main"].groupby(["class_name", "split"]).size()
                                      .unstack(fill_value=0).to_dict(orient="index"),
            "pages": len(pages)}
    (out / "visual_inspection_provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(prov, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
