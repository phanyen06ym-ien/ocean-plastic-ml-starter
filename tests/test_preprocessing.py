"""Kiểm thử src/preprocessing.py và src/eda.py trên dữ liệu giả lập."""
import json

import pandas as pd
import pytest
from PIL import Image

from src import eda
from src.common import load_class_mapping, load_contract
from src.preprocessing import (
    build_class_mapping,
    build_object_table,
    check_no_leakage,
    crop_box,
    extract_video_id,
    grouped_split,
    run_preprocessing,
)
from tests.conftest import CATEGORY_NAMES


def test_extract_video_id():
    assert extract_video_id("vid_000123_frame0000045.jpg") == "vid_000123"
    assert extract_video_id("folder/vid_7_frame1.png") == "vid_7"
    assert extract_video_id("random.jpg") == "img_random"


def test_class_mapping_sorted_by_category_id():
    cats = [{"id": 3, "name": "c"}, {"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    m = build_class_mapping(cats, plastic_related=["b"])
    assert [x["class_name"] for x in m] == ["a", "b", "c"]
    assert [x["class_index"] for x in m] == [0, 1, 2]
    assert [x["plastic_related"] for x in m] == [False, True, False]


def test_crop_box_padding_and_min_side():
    img = Image.new("RGB", (100, 80))
    assert crop_box(img, (10, 10, 5, 30), 0.1, 8, 128) is None          # quá nhỏ
    out = crop_box(img, (90, 70, 20, 20), 0.1, 8, 128)                  # tràn mép → bị cắt vào ảnh
    assert out is not None and out.size == (128, 128) and out.mode == "RGB"


def test_object_table_reads_both_original_splits(fake_trashcan):
    df, mapping = build_object_table(fake_trashcan, load_contract())
    assert len(mapping) == 22
    assert set(df["source_split"]) == {"train", "val"}
    assert (df["image_path"] != "").all()
    assert df["video_id"].str.startswith("vid_").all()


def test_grouped_split_has_no_leakage_and_close_ratios(fake_trashcan):
    df, _ = build_object_table(fake_trashcan, load_contract())
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    out, info = grouped_split(df, ratios, seed=42)
    check_no_leakage(out)
    frac = out.groupby("split")["image_id"].nunique() / out["image_id"].nunique()
    for s, r in ratios.items():
        assert abs(frac[s] - r) < 0.08
    # cùng seed → cùng kết quả
    out2, _ = grouped_split(df, ratios, seed=42)
    assert (out["split"] == out2["split"]).all()


def test_check_no_leakage_detects_leak():
    df = pd.DataFrame({"video_id": ["v1", "v1"], "image_id": ["a", "b"], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        check_no_leakage(df)


def test_run_preprocessing_end_to_end(fake_trashcan, tmp_path):
    out_dir = tmp_path / "processed" / "v1_crop128"
    mapping_out = tmp_path / "configs" / "class_mapping.json"
    summary = run_preprocessing(fake_trashcan, out_dir, mapping_out=mapping_out, verbose=False)

    mapping = load_class_mapping(mapping_out)
    assert [m["class_name"] for m in mapping] == CATEGORY_NAMES
    assert sum(m["plastic_related"] for m in mapping) == 6

    for s in ("train", "val", "test"):                       # đủ 22 thư mục lớp ở mọi tập
        assert len([p for p in (out_dir / s).iterdir() if p.is_dir()]) == 22

    manifest = pd.read_csv(out_dir / "split_manifest.csv")
    assert (manifest["crop_status"] == "too_small").sum() > 0  # box nhỏ bị loại
    ok = manifest[manifest["crop_status"] == "ok"]
    assert all((out_dir / p).exists() for p in ok["crop_path"])
    assert Image.open(out_dir / ok.iloc[0]["crop_path"]).size == (128, 128)
    assert summary["crops_per_split"]["train"] > summary["crops_per_split"]["val"]
    assert json.loads((out_dir / "preprocessing_summary.json").read_text())["num_videos"] == 40


def test_eda_tables(fake_trashcan):
    df, _ = build_object_table(fake_trashcan, load_contract())
    table = eda.class_distribution(df)
    assert table["total"].sum() == len(df)
    assert eda.imbalance_ratio(df) >= 1
    assert "pct_lt_32px" in eda.bbox_stats(df).columns
