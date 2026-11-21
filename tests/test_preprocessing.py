"""Kiểm thử src/preprocessing.py và src/eda.py trên dữ liệu giả lập."""
import json

import pandas as pd
import pytest
from PIL import Image

from src import eda
from src.preprocessing import load_class_mapping, load_contract
from src.preprocessing import (
    build_class_mapping,
    build_object_table,
    check_no_leakage,
    crop_box,
    extract_video_id,
    grouped_split,
    run_preprocessing,
)
from conftest import CATEGORY_NAMES  # thư mục tests/ nằm trong sys.path khi pytest chạy (tránh trùng gói "tests" khác trên Colab)


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


# --------------------------------------------------------------------------- v2_crop224 (E04)
def _v2_contract():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "configs" / "data_contract_v2.yaml"


def test_square_crop_keeps_aspect_and_stays_inside_image():
    from src.preprocessing import _square_window
    img = Image.new("RGB", (160, 120))
    x0, y0, x1, y1 = _square_window(20, 30, 60, 20, 0.15, 160, 120)       # box dẹt 60x20
    assert x1 - x0 == y1 - y0 == 78                                       # cạnh = 60 * 1.3
    assert x0 <= 20 and x1 >= 80                                          # chứa trọn box
    x0, y0, x1, y1 = _square_window(140, 100, 20, 20, 0.15, 160, 120)     # sát góc dưới phải
    assert 0 <= x0 and x1 <= 160 and 0 <= y0 and y1 <= 120 and x1 - x0 == y1 - y0
    out = crop_box(img, (20, 30, 60, 20), 0.15, 16, 224, square=True)
    assert out.size == (224, 224)
    assert crop_box(img, (20, 30, 60, 10), 0.15, 16, 224, square=True) is None   # cạnh < 16 px


def test_stratified_split_no_leakage_deterministic_and_covers_classes(fake_trashcan):
    from src.preprocessing import stratified_grouped_split
    df, _ = build_object_table(fake_trashcan, load_contract(_v2_contract()))
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    out, info = stratified_grouped_split(df, ratios, seed=42, n_restarts=3)
    check_no_leakage(out)
    assert info["method"] == "stratified_video_search"
    assert set(out["split"]) == {"train", "val", "test"}
    frac = out.groupby("split")["image_id"].nunique() / out["image_id"].nunique()
    assert frac["train"] > frac["val"] and frac["train"] > frac["test"]
    out2, info2 = stratified_grouped_split(df, ratios, seed=42, n_restarts=3)
    assert (out["split"] == out2["split"]).all() and info == info2


def test_run_preprocessing_v2_end_to_end(fake_trashcan, tmp_path):
    out_dir = tmp_path / "processed" / "v2_crop224"
    summary = run_preprocessing(fake_trashcan, out_dir, contract_path=_v2_contract(), verbose=False)
    assert summary["data_version"] == "v2_crop224"
    assert summary["split"]["method"] == "stratified_video_search"
    manifest = pd.read_csv(out_dir / "split_manifest.csv")
    check_no_leakage(manifest)
    ok = manifest[manifest["crop_status"] == "ok"]
    assert Image.open(out_dir / ok.iloc[0]["crop_path"]).size == (224, 224)
    assert 0 < summary["dropped_fraction"] < 1


def test_class_merge_keeps_plastic_classes_and_orders_by_category_id():
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(CATEGORY_NAMES)]
    contract = load_contract(_v2_contract())
    m = build_class_mapping(cats, contract["classes"]["plastic_related_candidates"], contract["classes"]["merge"])
    names = [x["class_name"] for x in m]
    assert len(m) == contract["classes"]["num_classes"] == 18
    assert "trash_other" in names and "trash_unknown_instance" not in names
    assert sum(x["plastic_related"] for x in m) == 6
    other = next(x for x in m if x["class_name"] == "trash_other")
    assert sorted(other["source_names"]) == ["trash_clothing", "trash_cup", "trash_unknown_instance", "trash_wreckage"]
    assert [x["class_index"] for x in m] == list(range(18))
    with pytest.raises(ValueError):
        build_class_mapping(cats, [], {"not_a_class": "x"})


def test_sample_weights_rebalance_classes():
    import numpy as np
    from src.train import sample_weights
    targets = [0] * 100 + [1] * 4
    w = sample_weights(targets, 3, "sqrt_inverse")
    share_rare = w[100:].sum() / w.sum()
    assert 4 / 104 < share_rare < 0.5                        # lớp hiếm được bù nhưng không vượt lớp lớn
    assert np.isclose(sample_weights(targets, 3, "inverse")[100:].sum() / sample_weights(targets, 3, "inverse").sum(), 0.5)
