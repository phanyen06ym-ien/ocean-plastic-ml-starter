"""Synthetic review evidence; these tests do not certify the real dataset."""
import ast
import csv
import json
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

from src import audit_review, data_audit
from src.audit_review import (RARE_CLASSES, collect_review, difference_hash, read_csv,
                              reconcile_videos, reviewed_rows, summarize_review)
from test_data_audit import dataset, snapshot, write_manifest  # shared full 22-class fixture


@pytest.fixture
def evidence(dataset, tmp_path, monkeypatch):
    cocos = {s: {"images": [], "annotations": [], "categories": []} for s in ("train", "val")}
    for i, row in enumerate(dataset["rows"]):
        source = "train" if i < 44 else "val"
        row.update(uid=f"{source}_{i}", image_id=f"{source}_{i}", source_split=source,
                   ann_id=str(i), file_name=f"vid_{i}_frame.jpg", image_path=f"/raw/{source}/vid_{i}_frame.jpg",
                   bbox_x="0", bbox_y="1", bbox_w="20", bbox_h="30")
        coco = cocos[source]
        coco["images"].append({"id": i, "file_name": row["file_name"]})
        coco["annotations"].append({"id": i, "image_id": i, "category_id": row["category_id"]})
    for source, video in (("train", 900), ("val", 901)):
        cocos[source]["images"].append({"id": video, "file_name": f"vid_{video}_empty.jpg"})
        cocos[source]["categories"] = [{"id": r["category_id"]} for r in dataset["rows"][:22]]
    sources = {}
    for source, coco in cocos.items():
        sources[source] = tmp_path / f"{source}.json"
        sources[source].write_text(json.dumps(coco), encoding="utf-8")
    write_manifest(dataset)
    summary = dataset["root"] / "preprocessing_summary.json"
    summary.write_text(json.dumps({"data_version": "v1_crop128", "num_videos": 66,
        "num_annotations": 66, "num_source_images": 66, "crop_status": {"ok": 66},
        "crops_per_split": dataset["expected"], "split": {"num_videos": 66}}))
    monkeypatch.setattr(data_audit, "V1_COUNTS", dataset["expected"])
    return {**dataset, "sources": sources, "summary": summary, "out": tmp_path / "evidence"}


def collect(evidence, **kwargs):
    return collect_review(evidence["root"], evidence["out"], evidence["sources"],
                          expected_source_videos=68, max_pair_images=2, **kwargs)


def complete_reviews(path, target, decision="PASS"):
    rows = read_csv(path)
    with path.open(encoding="utf-8", newline="") as handle:
        fields = next(csv.reader(handle))
    for row in rows:
        row.update(decision=decision, reviewer="fixture reviewer", evidence="fixture-source.json#annotation",
                   notes="Synthetic review only; no real-dataset claim")
    audit_review.write_csv(target, rows, fields)


def approve(evidence, report):
    out = evidence["out"]
    decisions = json.loads((out / "condition_decisions.template.json").read_text())
    assert all(not c["decision"] for c in decisions["conditions"].values())
    for row in decisions["conditions"].values():
        row.update(decision="PASS", reviewer="fixture reviewer", evidence="fixture evidence",
                   notes="Scope and limitations reviewed on synthetic fixtures")
    dp, pp, lp = out / "decisions.json", out / "pairs_reviewed.csv", out / "labels_reviewed.csv"
    audit_review.write_json(dp, decisions)
    complete_reviews(out / "near_duplicate_candidates.csv", pp)
    complete_reviews(out / "rare_class_review.csv", lp)
    return dp, pp, lp


def test_reconcile_explicit_ids_without_guessing(evidence):
    report = reconcile_videos(evidence["rows"], evidence["summary"], evidence["sources"], 68)
    assert report["source_without_annotations"] == ["vid_900", "vid_901"]
    assert report["counts"] == {"source_metadata": 68, "annotated_source": 66, "manifest": 66, "valid_crops": 66}
    assert report["ready_for_human_review"] and report["status"] == "UNVERIFIED"
    assert not report["conflicts"] and not report["missing_evidence"]
    assert all(v["observed_difference"] == "no_annotations_in_supplied_metadata"
               for v in report["videos"] if v["video_id"] in {"vid_900", "vid_901"})


@pytest.mark.parametrize("sources", [None, {}, {"train": "not-found.json"}])
def test_missing_metadata_never_invents_two_ids(evidence, sources):
    report = reconcile_videos(evidence["rows"], evidence["summary"], sources)
    assert report["source_without_annotations"] is None
    assert report["status"] == "UNVERIFIED"
    assert report["missing_evidence"] and not report["ready_for_human_review"]


def test_partial_metadata_does_not_explain_gap(evidence):
    report = reconcile_videos(evidence["rows"], None, {"train": evidence["sources"]["train"]})
    assert report["source_without_annotations"] is None
    assert len(report["missing_evidence"]) == 2


@pytest.mark.parametrize("change", ["summary_count", "summary_crop_status", "filename", "source_annotation", "unresolved_video", "prior_count"])
def test_reconciliation_conflicts_block_acceptance(evidence, change):
    expected = 68
    if change.startswith("summary"):
        summary = json.loads(evidence["summary"].read_text())
        summary["num_videos" if change == "summary_count" else "crop_status"] = 999
        evidence["summary"].write_text(json.dumps(summary))
    elif change == "filename":
        evidence["rows"][0]["file_name"] = "vid_999_frame.jpg"
    elif change == "source_annotation":
        path = evidence["sources"]["train"]
        coco = json.loads(path.read_text())
        coco["annotations"].pop()
        path.write_text(json.dumps(coco))
    elif change == "unresolved_video":
        evidence["rows"][0]["video_id"] = "img_fallback"
    else:
        expected = 312
    report = reconcile_videos(evidence["rows"], evidence["summary"], evidence["sources"], expected)
    assert report["conflicts"] or report["missing_evidence"]
    assert not report["ready_for_human_review"]


def test_read_only_collect_never_automatically_passes(evidence):
    before = snapshot(evidence["root"])
    source_before = {s: p.read_bytes() for s, p in evidence["sources"].items()}
    report = collect(evidence)
    assert report["decision"] == "KEEP_V1_WITH_CONDITIONS"
    assert report["automated_checks_passed"]
    assert {c["status"] for c in report["conditions"].values()} == {"UNVERIFIED"}
    assert snapshot(evidence["root"]) == before
    assert source_before == {s: p.read_bytes() for s, p in evidence["sources"].items()}
    assert read_csv(evidence["out"] / "rare_class_review.csv")
    assert all(not r["decision"] for r in read_csv(evidence["out"] / "rare_class_review.csv"))
    assert all(not r["decision"] for r in read_csv(evidence["out"] / "near_duplicate_candidates.csv"))
    assert (evidence["out"] / "audit_review_summary.json").is_file()


def test_dhash_known_gradient_and_encoding_invariance(tmp_path):
    decreasing = Image.new("L", (9, 8))
    decreasing.putdata([255 - x * 20 for y in range(8) for x in range(9)])
    increasing = Image.new("L", (9, 8))
    increasing.putdata([x * 20 for y in range(8) for x in range(9)])
    assert difference_hash(decreasing) == (1 << 64) - 1
    assert difference_hash(increasing) == 0
    decreasing.save(tmp_path / "gradient.png")
    with Image.open(tmp_path / "gradient.png") as im:
        assert difference_hash(im) == difference_hash(decreasing.convert("RGB"))


def test_scan_all_split_pairs_threshold_statistics_and_examples(evidence):
    # Same content in all three splits must be candidates; same-split pairs excluded.
    for index in (22, 44):
        shutil.copyfile(evidence["root"] / evidence["rows"][0]["crop_path"],
                        evidence["root"] / evidence["rows"][index]["crop_path"])
    report = collect(evidence, threshold=0)
    scan = report["near_duplicate_scan"]
    pairs = read_csv(evidence["out"] / "near_duplicate_candidates.csv")
    hashes = read_csv(evidence["out"] / "image_hashes.csv")
    expected = {(a["uid"], b["uid"]) for sa, sb in (("train", "val"), ("train", "test"), ("val", "test"))
                for a in hashes for b in hashes if a["split"] == sa and b["split"] == sb and a["dhash"] == b["dhash"]}
    assert {(r["uid_a"], r["uid_b"]) for r in pairs} == expected
    assert scan["pairs_compared"] == 3 * 22 * 22
    assert scan["candidate_count"] == len(pairs) >= 3
    assert scan["example_count"] == 2 and scan["examples_omitted"] == len(pairs) - 2
    assert sum(v["candidates"] for v in scan["split_pairs"].values()) == len(pairs)
    for row in pairs[:2]:
        with Image.open(evidence["out"] / row["example"]) as im:
            assert im.size == (560, 210)
    # Pixel integrity failure is separate from hash suspicion.
    assert report["decision"] == "DO_NOT_CONTINUE"
    assert report["conditions"]["cross_split_near_duplicates"]["status"] == "UNVERIFIED"


def test_configurable_hash_and_inclusive_distance(evidence):
    report = collect(evidence, hash_size=2, threshold=4)
    assert report["near_duplicate_scan"]["candidate_count"] == 3 * 22 * 22
    assert report["near_duplicate_scan"]["hash_bits"] == 4
    assert max(int(r["distance"]) for r in read_csv(evidence["out"] / "near_duplicate_candidates.csv")) <= 4


@pytest.mark.parametrize("kwargs", [{"hash_size": 1}, {"threshold": -1}, {"threshold": 65}, {"max_pair_images": -1}])
def test_bad_hash_parameters(evidence, kwargs):
    with pytest.raises(ValueError):
        audit_review.scan_near_duplicates(evidence["root"], evidence["rows"], evidence["out"], **kwargs)


def test_label_contact_sheet_covers_every_selected_crop_with_source_keys(evidence):
    report = collect(evidence, page_size=2)
    rows = read_csv(evidence["out"] / "rare_class_review.csv")
    assert {r["uid"] for r in rows} == {r["uid"] for r in evidence["rows"] if r["class_name"] in RARE_CLASSES}
    assert len(rows) == 9 and report["rare_class_scope"]["review_rows"] == 9
    assert {r["split"] for r in rows} == set(data_audit.SPLITS)
    for row in rows:
        assert row["ann_id"] and row["bbox_w"] and row["source_split"]
        with Image.open(evidence["out"] / row["contact_sheet"]) as im:
            assert im.width == 1120 and im.height == 210
        assert 1 <= int(row["tile"]) <= 2


@pytest.mark.parametrize("target", [".", "review"])
def test_outputs_cannot_be_inside_dataset(evidence, target):
    with pytest.raises(ValueError, match="outside"):
        collect_review(evidence["root"], evidence["root"] / target)


def test_existing_evidence_is_not_overwritten(evidence):
    collect(evidence)
    before = snapshot(evidence["out"])
    with pytest.raises(ValueError, match="fresh output_dir"):
        collect(evidence)
    assert snapshot(evidence["out"]) == before


def test_outside_crop_path_and_orphans_rejected(evidence):
    evidence["rows"][0]["crop_path"] = "../../outside.png"
    write_manifest(evidence)
    with pytest.raises(ValueError, match="Unsafe"):
        collect(evidence)
    evidence["rows"][0]["crop_path"] = "train/rov/0.png"
    write_manifest(evidence)
    Image.new("RGB", (128, 128)).save(evidence["root"] / "train/rov/orphan.png")
    with pytest.raises(ValueError, match="coverage"):
        collect(evidence)


def test_complete_human_evidence_can_change_verdict(evidence):
    report = collect(evidence)
    dp, pp, lp = approve(evidence, report)
    reviewed = summarize_review(evidence["out"], dp, pp, lp)
    assert reviewed["decision"] == "KEEP_V1" and reviewed["status"] == "PASS"
    assert all(c["status"] == "PASS" for c in reviewed["conditions"].values())


@pytest.mark.parametrize("change", ["no_attestation", "missing_source", "uncertain", "blank_evidence", "dropped_row", "duplicate", "changed_uid", "changed_metadata"])
def test_incomplete_evidence_never_passes(evidence, change):
    if change == "missing_source":
        evidence["sources"].clear()
    report = collect(evidence)
    dp, pp, lp = approve(evidence, report)
    if change == "no_attestation":
        dp = None
    elif change != "missing_source":
        rows = read_csv(lp)
        if change == "uncertain":
            rows[0]["decision"] = "UNCERTAIN"
        elif change == "blank_evidence":
            rows[0]["evidence"] = ""
        elif change == "dropped_row":
            rows.pop()
        elif change == "duplicate":
            rows.append(rows[0])
        elif change == "changed_uid":
            rows[0]["uid"] = "unknown"
        else:
            rows[0]["video_id"] = "vid_9999"
        lp.unlink()
        audit_review.write_csv(lp, rows, list(rows[0]))
    reviewed = summarize_review(evidence["out"], dp, pp, lp)
    assert reviewed["status"] == "CONDITIONAL"
    assert reviewed["decision"] == "KEEP_V1_WITH_CONDITIONS"


def test_missing_pair_reviews_block_even_with_attestation(evidence):
    report = collect(evidence, hash_size=2, threshold=4)
    dp, pp, lp = approve(evidence, report)
    reviewed = summarize_review(evidence["out"], dp, None, lp)
    assert reviewed["conditions"]["cross_split_near_duplicates"]["status"] == "UNVERIFIED"


def test_manual_fail_blocks_and_wrong_run_is_rejected(evidence):
    report = collect(evidence)
    dp, pp, lp = approve(evidence, report)
    lp.unlink()
    complete_reviews(evidence["out"] / "rare_class_review.csv", lp, "FAIL")
    assert summarize_review(evidence["out"], dp, pp, lp)["decision"] == "DO_NOT_CONTINUE"
    decisions = json.loads(dp.read_text())
    decisions["run_id"] = "another-run"
    dp.write_text(json.dumps(decisions))
    with pytest.raises(ValueError, match="different evidence run_id"):
        summarize_review(evidence["out"], dp, pp, lp)


def test_null_attestation_fields_and_truncated_csv_cannot_pass(evidence):
    report = collect(evidence, threshold=0)
    dp, pp, lp = approve(evidence, report)
    decisions = json.loads(dp.read_text())
    decisions["conditions"]["video_count_310_vs_312"]["evidence"] = None
    dp.write_text(json.dumps(decisions))
    reviewed = summarize_review(evidence["out"], dp, pp, lp)
    assert reviewed["conditions"]["video_count_310_vs_312"]["status"] == "UNVERIFIED"
    # A CSV editor may leave the final reviewer fields absent rather than empty.
    rows = read_csv(lp)
    with lp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(rows[0]))
        writer.writerow(list(rows[0].values())[:-4])
    result = reviewed_rows(evidence["out"] / "rare_class_review.csv", lp, "uid")
    assert not result["complete"] and result["missing_evidence"]


def test_modified_immutable_artifacts_rejected(evidence):
    collect(evidence)
    path = evidence["out"] / "near_duplicate_scan.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Evidence artifact changed"):
        summarize_review(evidence["out"])


def test_cli_and_refresh_do_not_write_into_dataset(evidence, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["review", "collect", "--processed-dir", str(evidence["root"]),
        "--output-dir", str(evidence["out"]), "--max-pair-images", "0"])
    assert audit_review.main() == 0
    output = capsys.readouterr().out
    assert "KEEP_V1_WITH_CONDITIONS" in output and "UNVERIFIED" in output
    monkeypatch.setattr(sys, "argv", ["review", "summarize", "--evidence-dir", str(evidence["out"]),
        "--output", str(evidence["root"] / "review.json")])
    with pytest.raises(SystemExit) as exc:
        audit_review.main()
    assert exc.value.code == 2
    assert not (evidence["root"] / "review.json").exists()


def test_audit_notebook_code_is_valid_and_collects_before_summarizing():
    path = Path(__file__).resolve().parents[1] / "notebooks/05_data_audit_review.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    ast.parse(code)
    assert code.index("collect_review(") < code.index("summarize_review(")
    assert "run_training" not in code and "run_preprocessing" not in code
    assert "audit_review_summary.json" in code and 'pytest' in code
    assert '"fix/dataset-audit-mobilenetv2"' in code


def test_notebook_collection_verification_and_summary_cells_execute_in_order(evidence, tmp_path):
    """Execute real cells with local frozen fixture paths, no Drive/GPU/training."""
    from datetime import datetime, timezone
    import subprocess

    path = Path(__file__).resolve().parents[1] / "notebooks/05_data_audit_review.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    paths = {"/content/drive/MyDrive/ocean-plastic-ml": tmp_path / "drive",
             "/content/processed/v1_crop128": evidence["root"]}
    namespace = {"Path": lambda value: paths.get(value, Path(value)), "datetime": datetime,
                 "timezone": timezone, "subprocess": subprocess, "sys": sys, "json": json}
    for index in (3, 5, 8):
        exec(compile("".join(notebook["cells"][index]["source"]), str(path), "exec"), namespace)
    assert namespace["reviewed"]["decision"] == "KEEP_V1_WITH_CONDITIONS"
    assert namespace["REVIEWED_OUTPUT"].is_file()
    assert namespace["report"]["video_reconciliation"]["source_without_annotations"] is None
