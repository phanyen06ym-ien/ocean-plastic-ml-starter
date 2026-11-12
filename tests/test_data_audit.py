"""Synthetic data only: exercise audit failures without touching TrashCan."""
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.common import DEFAULT_MAPPING, load_class_mapping
from src.data_audit import PENDING, SPLITS, audit_dataset, run_audit


def write_manifest(dataset):
    with (dataset["root"] / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dataset["rows"][0]))
        writer.writeheader()
        writer.writerows(dataset["rows"])


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "v1_crop128"
    root.mkdir()
    shutil.copy(DEFAULT_MAPPING, root / "class_mapping.json")
    rows = []
    for split_idx, split in enumerate(SPLITS):
        for c in load_class_mapping():
            index = split_idx * 22 + c["class_index"]
            rel = f"{split}/{c['class_name']}/{index}.png"
            path = root / rel
            path.parent.mkdir(parents=True)
            pixels = np.random.default_rng(index).integers(0, 256, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            rows.append({"uid": str(index), "split": split, "class_name": c["class_name"],
                         "class_index": c["class_index"], "category_id": c["category_id"],
                         "video_id": f"vid_{index}", "image_id": f"image_{index}",
                         "crop_status": "ok", "crop_path": rel})
    data = {"root": root, "rows": rows, "expected": {s: 22 for s in SPLITS}}
    write_manifest(data)
    return data


def audit(dataset):
    return audit_dataset(dataset["root"], expected_counts=dataset["expected"])


def checks(report):
    return {e["check"] for e in report["errors"]}


def snapshot(root):
    return {p.relative_to(root).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_repeatable_read_only_and_never_full_pass(dataset, tmp_path):
    before = snapshot(dataset["root"])
    report = audit(dataset)
    assert report == audit(dataset)
    assert report["status"] == "CONDITIONAL"
    assert report["decision"] == "KEEP_V1_WITH_CONDITIONS"
    assert report["automated_checks_passed"] is True
    assert report["pending_verifications"] == PENDING
    assert set(PENDING.values()) == {"UNVERIFIED"}
    assert report["split_counts"] == {s: 22 for s in SPLITS}
    assert report["manifest_video_count"] == 66
    assert report["train_imbalance_ratio"] == 1
    assert [c["class_index"] for c in report["per_class"]] == list(range(22))
    assert all(c["train_videos"] == 1 for c in report["per_class"])
    # The real CLI/run_audit defaults must reject fixture counts as non-v1.
    written = run_audit(dataset["root"], tmp_path / "audit")
    assert "split_counts" in checks(written)
    assert json.loads((tmp_path / "audit/audit_summary.json").read_text())["status"] == "FAIL"
    assert (tmp_path / "audit/audit_class_counts.csv").exists()
    assert snapshot(dataset["root"]) == before


def test_identical_decoded_pixels_with_different_encoding_are_not_removed(dataset):
    row = dict(dataset["rows"][0], uid="duplicate", crop_path="train/rov/duplicate.png")
    source = dataset["root"] / dataset["rows"][0]["crop_path"]
    target = dataset["root"] / row["crop_path"]
    with Image.open(source) as im:
        im.save(target, compress_level=0)
    assert source.read_bytes() != target.read_bytes()
    dataset["rows"].append(row)
    dataset["expected"]["train"] += 1
    write_manifest(dataset)
    report = audit(dataset)
    assert report["automated_checks_passed"]
    assert len(report["exact_duplicate_groups"]) == 1
    assert report["cross_split_exact_duplicate_groups"] == 0
    assert report["train_imbalance_ratio"] == 2
    assert report["per_class"][0]["train_videos"] == 1
    assert target.exists()


@pytest.mark.parametrize("target_index,expected_check", [(22, "cross_split_exact_pixels"),
                                                         (1, "conflicting_duplicate_labels")])
def test_dangerous_exact_duplicates(dataset, target_index, expected_check):
    shutil.copyfile(dataset["root"] / dataset["rows"][0]["crop_path"],
                    dataset["root"] / dataset["rows"][target_index]["crop_path"])
    report = audit(dataset)
    assert expected_check in checks(report)
    assert report["decision"] == "DO_NOT_CONTINUE"


@pytest.mark.parametrize("field,check", [("video_id", "cross_split_video"),
                                         ("image_id", "cross_split_source_image")])
def test_group_leakage(dataset, field, check):
    dataset["rows"][22][field] = dataset["rows"][0][field]
    write_manifest(dataset)
    assert check in checks(audit(dataset))


@pytest.mark.parametrize("field,value,check", [
    ("class_index", 21, "manifest_class_id"),
    ("category_id", 22, "manifest_class_id"),
    ("video_id", "img_fallback", "unverified_group_id"),
    ("image_id", "", "unverified_group_id"),
    ("split", "test", "manifest_folder_mismatch"),
    ("crop_path", "../../outside.png", "unsafe_crop_path"),
    ("crop_path", "train/rov/missing.png", "missing_crop"),
])
def test_manifest_validation(dataset, field, value, check):
    dataset["rows"][0][field] = value
    write_manifest(dataset)
    assert check in checks(audit(dataset))


def test_manifest_duplicate_orphan_and_dropped_crop(dataset):
    dataset["rows"].append(dict(dataset["rows"][0]))
    dataset["rows"][1]["crop_status"] = "too_small"
    dataset["rows"][1]["crop_path"] = ""
    write_manifest(dataset)
    assert {"manifest_uid", "duplicate_manifest_path", "orphan_crop"} <= checks(audit(dataset))


def test_missing_required_metadata_is_not_no_leakage(dataset):
    for row in dataset["rows"]:
        del row["video_id"]
    write_manifest(dataset)
    with pytest.raises(ValueError, match="missing columns"):
        audit(dataset)


def test_unreadable_wrong_size_wrong_mode_and_missing_crop(dataset):
    root, rows = dataset["root"], dataset["rows"]
    (root / rows[0]["crop_path"]).write_bytes(b"broken image")
    Image.new("L", (12, 20)).save(root / rows[1]["crop_path"])
    (root / rows[2]["crop_path"]).unlink()
    report = audit(dataset)
    assert {"unreadable_images", "image_format", "missing_crop", "empty_class"} <= checks(report)
    assert len(report["unreadable_images"]) == len(report["invalid_images"]) == 1
    assert report["train_imbalance_ratio"] is None


def test_unknown_and_missing_class_folders(dataset):
    (dataset["root"] / "train/extra_class").mkdir()
    shutil.rmtree(dataset["root"] / "val/plant")
    assert "class_directories" in checks(audit(dataset))


def test_changed_local_mapping_is_rejected(dataset):
    path = dataset["root"] / "class_mapping.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    mapping["classes"][0]["category_id"] = 22
    path.write_text(json.dumps(mapping))
    assert "dataset_mapping" in checks(audit(dataset))
    with pytest.raises(ValueError, match="22 Instance"):
        audit_dataset(dataset["root"], mapping_path=path)


@pytest.mark.parametrize("relative", [".", "audit", "train/rov"])
def test_no_outputs_inside_dataset(dataset, relative):
    before = snapshot(dataset["root"])
    with pytest.raises(ValueError, match="outside processed_dir"):
        run_audit(dataset["root"], dataset["root"] / relative)
    assert snapshot(dataset["root"]) == before


def test_cli_failure_writes_report_and_exits_nonzero(dataset, tmp_path):
    result = subprocess.run([sys.executable, "-m", "src.data_audit", "--processed-dir", str(dataset["root"]),
                             "--output-dir", str(tmp_path / "output")], capture_output=True, text=True)
    assert result.returncode == 1
    assert "FAIL: DO_NOT_CONTINUE" in result.stdout
    assert "UNVERIFIED" in result.stdout


def test_cli_success_stays_conditional(dataset, tmp_path, monkeypatch, capsys):
    from src import data_audit

    # Only the test fixture's expected counts change, never the v1 contract on disk.
    monkeypatch.setattr(data_audit, "V1_COUNTS", dataset["expected"])
    monkeypatch.setattr(sys, "argv", ["audit", "--processed-dir", str(dataset["root"]),
                                    "--output-dir", str(tmp_path / "conditional")])
    assert data_audit.main() == 0
    assert "CONDITIONAL: KEEP_V1_WITH_CONDITIONS" in capsys.readouterr().out
    report = json.loads((tmp_path / "conditional/audit_summary.json").read_text())
    assert report["pending_verifications"] == PENDING
