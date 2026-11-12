"""Read-only evidence collection and human review for frozen v1_crop128.

Hash matches are candidates, never a leakage verdict. No command certifies PASS
without complete, run-bound human decisions. Outputs must be outside the dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageOps

from src.common import DEFAULT_MAPPING
from src.data_audit import IMAGE_EXTS, PENDING, SPLITS, _file_hash, run_audit

RARE_CLASSES = ("trash_cup", "trash_snack_wrapper", "trash_bottle")
REVIEW_FIELDS = ("decision", "reviewer", "evidence", "notes")


def nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def crop_file(root, row):
    rel = row["crop_path"].replace("\\", "/")
    path = (root / rel).resolve()
    if (not rel or Path(rel).is_absolute() or not path.is_relative_to(root)
            or Path(rel).parts[:2] != (row["split"], row["class_name"])):
        raise ValueError(f"Unsafe/mismatched crop path: {rel}")
    return path


def load_manifest(root):
    rows = read_csv(root / "split_manifest.csv")
    required = {"uid", "video_id", "image_id", "split", "class_name", "crop_status", "crop_path"}
    if not rows or not required <= rows[0].keys():
        raise ValueError("Empty manifest or missing review columns")
    if len({r["uid"] for r in rows}) != len(rows) or any(not r["uid"] for r in rows):
        raise ValueError("Manifest UIDs must be nonempty and unique")
    for row in rows:
        if row["split"] not in SPLITS:
            raise ValueError("Invalid split")
        if row["crop_status"] == "ok":
            crop_file(root, row)
    return rows


def reconcile_videos(rows, summary_path, source_cocos=None, expected_source_videos=312):
    """COCO images = source metadata scope; annotations are joined by source split.

    Explicit train/val paths avoid guessing which COCO export was used. Image
    filenames follow the frozen preprocessing regex, with no fallback video IDs.
    No claim is made about raw files absent from the supplied COCO images lists.
    """
    source_cocos = source_cocos or {}
    missing, conflicts = [], []
    sets = {k: set() for k in ("source_metadata", "annotated_source", "manifest", "valid_crops")}
    evidence = {}
    source_anns = {}
    for split in ("train", "val"):
        path = Path(source_cocos[split]) if split in source_cocos else None
        if path is None or not path.is_file():
            missing.append(f"source COCO {split}: images + annotations + categories required")
            continue
        evidence[split] = {"path": str(path.resolve()), "sha256": _file_hash(path)}
        coco = json.loads(path.read_text(encoding="utf-8"))
        categories = {str(c["id"]) for c in coco["categories"]}
        images = {}
        for im in coco["images"]:
            key = str(im["id"])
            if key in images:
                conflicts.append(f"duplicate source image: {split}_{key}")
            match = re.search(r"vid_(\d+)", im["file_name"].replace("\\", "/").split("/")[-1])
            video = f"vid_{match.group(1)}" if match else ""
            images[key] = {"video_id": video, "file_name": im["file_name"]}
            if video:
                sets["source_metadata"].add(video)
            else:
                missing.append(f"unresolved source video: {split}_{key}")
        for ann in coco["annotations"]:
            uid = f"{split}_{ann['id']}"
            if uid in source_anns:
                conflicts.append(f"duplicate source annotation: {uid}")
            image = images.get(str(ann["image_id"]))
            if image is None or str(ann["category_id"]) not in categories:
                conflicts.append(f"invalid source annotation: {uid}")
                continue
            source_anns[uid] = {"image_id": f"{split}_{ann['image_id']}",
                                "category_id": str(ann["category_id"]), **image}
            if image["video_id"]:
                sets["annotated_source"].add(image["video_id"])
    for row in rows:
        video = row["video_id"]
        if not re.fullmatch(r"vid_\d+", video):
            missing.append(f"unresolved manifest video: {row['uid']}")
            continue
        sets["manifest"].add(video)
        if row["crop_status"] == "ok":
            sets["valid_crops"].add(video)
        if len(evidence) == 2:
            ann = source_anns.get(row["uid"])
            if ann is None or any(str(row.get(k)) != ann[k] for k in ("image_id", "category_id", "video_id")):
                conflicts.append(f"manifest/source mismatch: {row['uid']}")
            if ann and row.get("file_name") and row["file_name"] != ann["file_name"]:
                conflicts.append(f"manifest/source filename mismatch: {row['uid']}")
    if len(evidence) == 2 and set(source_anns) != {r["uid"] for r in rows}:
        conflicts.append("source annotation UID set differs from manifest")
    summary = None
    if summary_path and Path(summary_path).is_file():
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        evidence["preprocessing_summary"] = {"path": str(Path(summary_path).resolve()),
                                               "sha256": _file_hash(Path(summary_path))}
        observed = {"num_videos": len(sets["manifest"]), "num_annotations": len(rows),
                    "num_source_images": len({r["image_id"] for r in rows}),
                    "crop_status": dict(Counter(r["crop_status"] for r in rows)),
                    "crops_per_split": {s: sum(r["split"] == s and r["crop_status"] == "ok" for r in rows)
                                        for s in SPLITS}}
        for key, value in observed.items():
            if summary.get(key) != value:
                conflicts.append(f"preprocessing summary mismatch: {key}")
        if summary.get("data_version") != "v1_crop128":
            conflicts.append("preprocessing summary data_version mismatch")
        if summary.get("split", {}).get("num_videos") != len(sets["manifest"]):
            conflicts.append("preprocessing split.num_videos mismatch")
    else:
        missing.append("preprocessing_summary.json")
    if len(evidence.keys() & {"train", "val"}) == 2:
        if len(sets["source_metadata"]) != expected_source_videos:
            conflicts.append(f"source metadata count does not substantiate prior claim {expected_source_videos}")
        if sets["annotated_source"] != sets["manifest"]:
            conflicts.append("annotated source and manifest video ID sets differ")
    union = set().union(*sets.values())
    statuses = defaultdict(Counter)
    for row in rows:
        statuses[row["video_id"]][row["crop_status"]] += 1
    videos = [{"video_id": v, **{k: v in values for k, values in sets.items()},
               "manifest_crop_status_counts": json.dumps(dict(sorted(statuses[v].items()))),
               "observed_difference": ("no_annotations_in_supplied_metadata"
                                       if len(evidence.keys() & {"train", "val"}) == 2
                                       and v in sets["source_metadata"] - sets["annotated_source"]
                                       else "no_valid_crops" if v in sets["manifest"] - sets["valid_crops"] else "")}
              for v in sorted(union)]
    return {"status": "UNVERIFIED", "ready_for_human_review": not missing and not conflicts,
            "scope": "All images/annotations in explicitly supplied train/val COCO metadata, not raw disk inventory",
            "prior_source_video_claim": expected_source_videos,
            "counts": {k: len(v) for k, v in sets.items()},
            "video_ids": {k: sorted(v) for k, v in sets.items()},
            "source_without_annotations": sorted(sets["source_metadata"] - sets["annotated_source"])
                if len(evidence.keys() & {"train", "val"}) == 2 else None,
            "missing_evidence": missing, "conflicts": conflicts, "provenance": evidence,
            "videos": videos}


def difference_hash(image, hash_size=8):
    """Horizontal dHash: grayscale LANCZOS resize, strict left > right, row-major."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for y in range(hash_size):
        for x in range(hash_size):
            i = y * (hash_size + 1) + x
            value = (value << 1) | int(pixels[i] > pixels[i + 1])
    return value


def contact_sheet(root, rows, output, caption_fields, columns=4):
    """Contact sheets are references only; source annotations must be reviewed too."""
    tile_w, tile_h = 280, 210
    sheet = Image.new("RGB", (columns * tile_w, ((len(rows) + columns - 1) // columns) * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x, y = index % columns * tile_w, index // columns * tile_h
        with Image.open(crop_file(root, row)) as im:
            thumb = ImageOps.contain(im.convert("RGB"), (128, 128))
            sheet.paste(thumb, (x, y))
        # ASCII escaping keeps default fonts portable; full metadata remains in CSV.
        text = "\n".join(f"{k}: {row.get(k, '')}" for k in caption_fields)
        text = text.encode("ascii", "backslashreplace").decode("ascii")
        draw.multiline_text((x, y + 130), text, fill="black")
    sheet.save(output)


def scan_near_duplicates(root, rows, out, threshold=6, hash_size=8, max_pair_images=50):
    if not 2 <= hash_size <= 16 or not 0 <= threshold <= hash_size * hash_size or max_pair_images < 0:
        raise ValueError("hash_size must be 2..16, threshold 0..hash bits, max_pair_images >= 0")
    ok = sorted((r for r in rows if r["crop_status"] == "ok"), key=lambda r: r["uid"])
    paths = {crop_file(root, r) for r in ok}
    physical = {p.resolve() for s in SPLITS for p in (root / s).rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    if paths != physical or len(paths) != len(ok):
        raise ValueError("Manifest crop coverage differs from physical dataset; inspect base audit")
    by_split = defaultdict(list)
    hashes = []
    for row in ok:
        path = crop_file(root, row)
        with Image.open(path) as im:
            value = difference_hash(im, hash_size)
        item = {k: row[k] for k in ("uid", "split", "class_name", "video_id", "image_id", "crop_path")}
        item.update(dhash=f"{value:0{(hash_size * hash_size + 3) // 4}x}", crop_sha256=_file_hash(path))
        hashes.append(item)
        by_split[row["split"]].append((row, value))
    write_csv(out / "image_hashes.csv", hashes, ["uid", "split", "class_name", "video_id", "image_id", "crop_path", "dhash", "crop_sha256"])
    fields = ["pair_id", "uid_a", "uid_b", "split_a", "split_b", "class_a", "class_b",
              "video_a", "video_b", "path_a", "path_b", "hash_a", "hash_b", "distance", "example", *REVIEW_FIELDS]
    stats, examples, candidate_count, compared = {}, 0, 0, 0
    involved = {s: set() for s in SPLITS}
    with (out / "near_duplicate_candidates.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sa, sb in itertools.combinations(SPLITS, 2):
            count = 0
            for a, ha in by_split[sa]:
                for b, hb in by_split[sb]:
                    compared += 1
                    distance = (ha ^ hb).bit_count()
                    if distance > threshold:
                        continue
                    pair_id = hashlib.sha256(json.dumps([a["uid"], b["uid"]]).encode()).hexdigest()
                    example = ""
                    if examples < max_pair_images:
                        example = f"pair_examples/{pair_id}.png"
                        (out / "pair_examples").mkdir(exist_ok=True)
                        contact_sheet(root, [a, b], out / example, ("uid", "split", "class_name", "video_id"), columns=2)
                        examples += 1
                    writer.writerow(dict(zip(fields, [pair_id, a["uid"], b["uid"], sa, sb,
                        a["class_name"], b["class_name"], a["video_id"], b["video_id"],
                        a["crop_path"], b["crop_path"], f"{ha:x}", f"{hb:x}", distance, example, "", "", "", ""])))
                    involved[sa].add(a["uid"])
                    involved[sb].add(b["uid"])
                    count += 1
            stats[f"{sa}:{sb}"] = {"compared": len(by_split[sa]) * len(by_split[sb]), "candidates": count}
            candidate_count += count
    return {"status": "UNVERIFIED", "method": "horizontal_dhash_grayscale_LANCZOS_strict_gt",
            "hash_size": hash_size, "hash_bits": hash_size * hash_size, "max_hamming_distance": threshold,
            "complete_cross_split_scan": True, "images_hashed": len(hashes), "pairs_compared": compared,
            "candidate_count": candidate_count, "split_pairs": stats,
            "candidate_images_per_split": {s: len(v) for s, v in involved.items()},
            "images_per_split": {s: len(by_split[s]) for s in SPLITS},
            "example_count": examples, "examples_omitted": candidate_count - examples,
            "limitations": "dHash may miss rotations/crop changes and collide on low-texture images. "
                           "Threshold is a screening parameter, not proof of leakage or absence of leakage."}


def export_label_review(root, rows, out, classes=RARE_CLASSES, page_size=24):
    if page_size < 1 or page_size > 100:
        raise ValueError("page_size must be 1..100")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Select unique rare classes")
    selected, pages, counts = [], [], {}
    for cls in classes:
        members = sorted((r for r in rows if r["class_name"] == cls and r["crop_status"] == "ok"),
                         key=lambda r: (SPLITS.index(r["split"]), r["uid"]))
        if not members:
            raise ValueError(f"No valid crops for selected class: {cls}")
        counts[cls] = {s: sum(r["split"] == s for r in members) for s in SPLITS}
        for start in range(0, len(members), page_size):
            # Numeric names avoid placing untrusted class names in output paths.
            page = f"label_contact_sheets/page_{len(pages) + 1:04d}.png"
            (out / "label_contact_sheets").mkdir(exist_ok=True)
            chunk = members[start:start + page_size]
            contact_sheet(root, chunk, out / page, ("uid", "split", "class_name", "video_id"))
            pages.append(page)
            for tile, row in enumerate(chunk, 1):
                selected.append({**row, "contact_sheet": page, "tile": tile,
                                 **{k: "" for k in REVIEW_FIELDS}})
    fields = ["uid", "split", "class_name", "category_id", "class_index", "video_id", "image_id",
              "crop_path", "source_split", "ann_id", "file_name", "image_path",
              "bbox_x", "bbox_y", "bbox_w", "bbox_h", "contact_sheet", "tile", *REVIEW_FIELDS]
    selected = [{k: r.get(k, "") for k in fields} for r in selected]
    write_csv(out / "rare_class_review.csv", selected, fields)
    return {"status": "UNVERIFIED", "classes": list(classes), "scope": "all valid crops in train/val/test",
            "counts": counts, "review_rows": len(selected), "contact_sheets": pages,
            "required_evidence": "Inspect crop AND original image/annotation bbox/category. "
                                 "Record source evidence reference, reviewer, decision and notes; do not relabel v1."}


def reviewed_rows(template, reviewed, key):
    """Require exact scope/immutable metadata; rows cannot silently disappear."""
    original = read_csv(template)
    if reviewed is None:
        return {"complete": False, "counts": {}, "missing_evidence": ["reviewed CSV not supplied"]}
    actual = read_csv(reviewed)
    expected = {r[key]: r for r in original}
    seen, counts, issues = set(), Counter(), []
    for row in actual:
        identity = row.get(key)
        if identity not in expected or identity in seen:
            issues.append(f"extra/duplicate review row: {identity}")
            continue
        seen.add(identity)
        if any(row.get(k) != v for k, v in expected[identity].items() if k not in REVIEW_FIELDS):
            issues.append(f"changed review metadata: {identity}")
        decision = (row.get("decision") or "").strip()
        counts[decision or "PENDING"] += 1
        if decision not in {"PASS", "FAIL", "UNCERTAIN"} or any(not nonblank(row.get(k)) for k in REVIEW_FIELDS):
            issues.append(f"incomplete review: {identity}")
    if seen != expected.keys():
        issues.append("review scope is incomplete")
    return {"complete": not issues, "counts": dict(counts), "missing_evidence": issues,
            "reviewed_csv": str(Path(reviewed).resolve()), "sha256": _file_hash(Path(reviewed))}


def summarize_review(out, decisions_path=None, pair_review=None, label_review=None):
    """Rebuild summary from immutable artifacts; never trusts edited summary status."""
    out = Path(out).resolve(strict=True)
    run = json.loads((out / "evidence_index.json").read_text(encoding="utf-8"))
    for name, digest in run["artifacts"].items():
        if _file_hash(out / name) != digest:
            raise ValueError(f"Evidence artifact changed: {name}; collect a fresh run")
    video = json.loads((out / "video_reconciliation.json").read_text(encoding="utf-8"))
    near = json.loads((out / "near_duplicate_scan.json").read_text(encoding="utf-8"))
    labels = json.loads((out / "label_review_scope.json").read_text(encoding="utf-8"))
    base = json.loads((out / "audit_summary.json").read_text(encoding="utf-8"))
    pair_rows = reviewed_rows(out / "near_duplicate_candidates.csv", pair_review, "pair_id")
    label_rows = reviewed_rows(out / "rare_class_review.csv", label_review, "uid")
    # Empty candidate sets still require explicit scan/limitations attestation.
    if near["candidate_count"] == 0 and pair_review is None:
        pair_rows = {"complete": True, "counts": {}, "missing_evidence": []}
    decisions = json.loads(Path(decisions_path).read_text(encoding="utf-8")) if decisions_path else {}
    if decisions and decisions.get("run_id") != run["run_id"]:
        raise ValueError("Decisions refer to a different evidence run_id")
    gates = {
        "video_count_310_vs_312": (video["ready_for_human_review"], video["missing_evidence"] + video["conflicts"]),
        "cross_split_near_duplicates": (near["complete_cross_split_scan"] and pair_rows["complete"], pair_rows["missing_evidence"]),
        "rare_class_label_quality": (label_rows["complete"], label_rows["missing_evidence"]),
    }
    condition_artifacts = {
        "video_count_310_vs_312": ["video_reconciliation.json", "video_reconciliation.csv"],
        "cross_split_near_duplicates": ["near_duplicate_scan.json", "near_duplicate_candidates.csv", "image_hashes.csv"],
        "rare_class_label_quality": ["label_review_scope.json", "rare_class_review.csv"],
    }
    conditions = {}
    for key, (ready, missing) in gates.items():
        human = decisions.get("conditions", {}).get(key, {})
        valid = human.get("decision") in {"PASS", "FAIL", "UNCERTAIN"} and all(
            nonblank(human.get(k)) for k in ("reviewer", "evidence", "notes"))
        row_review = pair_rows if key == "cross_split_near_duplicates" else label_rows if key == "rare_class_label_quality" else None
        row_fail = row_review and row_review["counts"].get("FAIL", 0) > 0
        all_rows_pass = row_review is None or not any(v for k, v in row_review["counts"].items() if k != "PASS")
        status = "UNVERIFIED"
        if (valid and human["decision"] == "FAIL") or row_fail:
            status = "FAIL"
        elif ready and valid and human["decision"] == "PASS" and all_rows_pass:
            status = "PASS"
        conditions[key] = {"status": status, "evidence_ready": ready, "missing_evidence": list(missing),
                           "artifacts": condition_artifacts[key], "human_decision": human if valid else None}
        if not valid:
            conditions[key]["missing_evidence"] = [*missing, "run-bound human attestation with reviewer/evidence/notes"]
        elif human["decision"] == "UNCERTAIN":
            conditions[key]["missing_evidence"].append("human condition review remains UNCERTAIN")
        if row_review and not all_rows_pass:
            conditions[key]["missing_evidence"].append("row decisions are not all PASS; inspect review counts")
    failed = not base["automated_checks_passed"] or any(c["status"] == "FAIL" for c in conditions.values())
    all_pass = all(c["status"] == "PASS" for c in conditions.values())
    return {"schema_version": 1, "data_version": "v1_crop128", "run_id": run["run_id"],
            "status": "FAIL" if failed else "PASS" if all_pass else "CONDITIONAL",
            "decision": "DO_NOT_CONTINUE" if failed else "KEEP_V1" if all_pass else "KEEP_V1_WITH_CONDITIONS",
            "automated_checks_passed": base["automated_checks_passed"],
            "errors": base["errors"], "base_integrity_artifact": "audit_summary.json",
            "conditions": conditions, "video_reconciliation": video, "near_duplicate_scan": near,
            "rare_class_scope": labels, "pair_review": pair_rows, "label_review": label_rows,
            "provenance": run,
            "decisions_provenance": {"path": str(Path(decisions_path).resolve()), "sha256": _file_hash(Path(decisions_path))}
                if decisions_path else None}


def collect_review(processed_dir, output_dir, source_cocos=None, summary_path=None,
                   threshold=6, hash_size=8, max_pair_images=50, page_size=24,
                   expected_source_videos=312):
    root, out = Path(processed_dir).resolve(strict=True), Path(output_dir).resolve()
    if out.is_relative_to(root):
        raise ValueError("output_dir must be outside processed_dir")
    if out.exists():
        raise ValueError("Use a fresh output_dir; existing evidence/reviewer work must not be overwritten")
    rows = load_manifest(root)
    out.mkdir(parents=True, exist_ok=False)
    run_audit(root, out, DEFAULT_MAPPING)
    video = reconcile_videos(rows, summary_path or root / "preprocessing_summary.json", source_cocos, expected_source_videos)
    write_json(out / "video_reconciliation.json", video)
    write_csv(out / "video_reconciliation.csv", video["videos"],
              ["video_id", "source_metadata", "annotated_source", "manifest", "valid_crops",
               "manifest_crop_status_counts", "observed_difference"])
    near = scan_near_duplicates(root, rows, out, threshold, hash_size, max_pair_images)
    write_json(out / "near_duplicate_scan.json", near)
    labels = export_label_review(root, rows, out, page_size=page_size)
    write_json(out / "label_review_scope.json", labels)
    repo = Path(__file__).resolve().parents[1]
    git = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
    try:
        commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run([*git, "status", "--porcelain", "--untracked-files=normal"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit = "UNVERIFIED"
        dirty = None
    artifacts = {p.relative_to(out).as_posix(): _file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}
    provenance = {"artifacts": artifacts, "code_commit": commit, "code_dirty": dirty,
                  "code_sha256": {name: _file_hash(repo / "src" / name)
                                  for name in ("audit_review.py", "data_audit.py", "common.py")},
                  "python": platform.python_version(),
                  "pillow": PIL.__version__, "processed_dir": str(root),
                  "manifest_sha256": _file_hash(root / "split_manifest.csv"),
                  "mapping_sha256": _file_hash(root / "class_mapping.json")}
    provenance["run_id"] = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    write_json(out / "evidence_index.json", provenance)
    write_json(out / "condition_decisions.template.json", {"run_id": provenance["run_id"], "conditions": {
        key: {"decision": "", "reviewer": "", "evidence": "", "notes": ""} for key in PENDING}})
    report = summarize_review(out)
    write_json(out / "audit_review_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--processed-dir", required=True)
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--source-train-coco")
    collect.add_argument("--source-val-coco")
    collect.add_argument("--preprocessing-summary")
    collect.add_argument("--expected-source-videos", type=int, default=312)
    collect.add_argument("--threshold", type=int, default=6)
    collect.add_argument("--hash-size", type=int, default=8)
    collect.add_argument("--max-pair-images", type=int, default=50)
    collect.add_argument("--page-size", type=int, default=24)
    review = sub.add_parser("summarize")
    review.add_argument("--evidence-dir", required=True)
    review.add_argument("--decisions")
    review.add_argument("--pair-review")
    review.add_argument("--label-review")
    review.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "collect":
            sources = {s: p for s, p in (("train", args.source_train_coco), ("val", args.source_val_coco)) if p}
            report = collect_review(args.processed_dir, args.output_dir, sources, args.preprocessing_summary,
                                    args.threshold, args.hash_size, args.max_pair_images, args.page_size,
                                    args.expected_source_videos)
        else:
            report = summarize_review(args.evidence_dir, args.decisions, args.pair_review, args.label_review)
            target = Path(args.output).resolve()
            root = Path(report["provenance"]["processed_dir"]).resolve()
            protected = [Path(p).resolve() for p in (args.decisions, args.pair_review, args.label_review) if p]
            if target.is_relative_to(root) or target in protected or target.is_symlink():
                raise ValueError("Summary output must not overwrite dataset/review inputs")
            write_json(target, report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Review input error: {exc}\n")
    print(report["status"], report["decision"])
    print(json.dumps({k: v["status"] for k, v in report["conditions"].items()}))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
