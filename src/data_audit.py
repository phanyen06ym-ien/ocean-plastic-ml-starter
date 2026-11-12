"""Read-only audit of the frozen v1_crop128 folders and split manifest.

No crops, labels, splits, mappings or checkpoints are written. Exact duplicates
refer to decoded RGB pixels (including dimensions), not compressed file bytes.
Near duplicates and human label review are deliberately NOT certified here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from src.common import DEFAULT_MAPPING, class_names, load_class_mapping, validate_instance_mapping

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
V1_COUNTS = {"train": 8283, "val": 1782, "test": 2009}
PENDING = {
    "video_count_310_vs_312": "UNVERIFIED",
    "cross_split_near_duplicates": "UNVERIFIED",
    "rare_class_label_quality": "UNVERIFIED",
}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _leaks(groups: dict[str, set[str]]) -> dict[str, list[str]]:
    return {key: sorted(splits) for key, splits in sorted(groups.items()) if len(splits) > 1}


def audit_dataset(processed_dir: str | Path, mapping_path: str | Path = DEFAULT_MAPPING,
                  manifest_path: str | Path | None = None,
                  expected_counts: dict[str, int] | None = None) -> dict[str, Any]:
    """Inspect all three frozen splits, including Test for data integrity only.

    The default expected counts are the confirmed v1 counts. Tests may supply
    fixture counts; this never changes the dataset. Missing metadata is an error,
    rather than evidence that there is no leakage.
    """
    root = Path(processed_dir).resolve(strict=True)
    mapping_path = Path(mapping_path).resolve(strict=True)
    manifest_path = Path(manifest_path or root / "split_manifest.csv").resolve(strict=True)
    mapping = load_class_mapping(mapping_path)
    validate_instance_mapping(mapping)
    names = class_names(mapping)
    by_name = {c["class_name"]: c for c in mapping}
    expected_counts = V1_COUNTS if expected_counts is None else expected_counts
    errors: list[dict[str, Any]] = []

    def error(check: str, **details: Any) -> None:
        errors.append({"check": check, **details})

    local_mapping = root / "class_mapping.json"
    if local_mapping.exists():
        try:
            validate_instance_mapping(load_class_mapping(local_mapping))
        except (ValueError, KeyError, TypeError) as exc:
            error("dataset_mapping", message=str(exc))
    else:
        error("dataset_mapping", message="class_mapping.json is missing")

    files: dict[str, dict[str, Any]] = {}
    pixel_groups: dict[str, list[str]] = defaultdict(list)
    counts = {s: Counter() for s in SPLITS}
    unreadable, invalid_images = [], []
    for split in SPLITS:
        folder = root / split
        if not folder.is_dir():
            error("missing_split", split=split)
            continue
        found = {p.name for p in folder.iterdir() if p.is_dir()}
        if found != set(names):
            error("class_directories", split=split, missing=sorted(set(names) - found),
                  unexpected=sorted(found - set(names)))
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = path.relative_to(root).as_posix()
            parts = path.relative_to(root).parts
            if not path.resolve().is_relative_to(root):
                error("unsafe_image_path", path=rel)
                continue
            if len(parts) < 3 or parts[1] not in by_name:
                error("unexpected_image", path=rel)
                continue
            label = parts[1]
            counts[split][label] += 1
            files[rel] = {"split": split, "class_name": label}
            try:
                with Image.open(path) as im:
                    im.load()
                    if im.size != (128, 128) or im.mode != "RGB":
                        invalid_images.append({"path": rel, "size": list(im.size), "mode": im.mode})
                    rgb = im.convert("RGB")
                    digest = hashlib.sha256(f"RGB:{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
                    pixel_groups[digest].append(rel)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                unreadable.append({"path": rel, "error": type(exc).__name__})
    if unreadable:
        error("unreadable_images", count=len(unreadable))
    if invalid_images:
        error("image_format", count=len(invalid_images))

    videos: dict[str, set[str]] = defaultdict(set)
    images: dict[str, set[str]] = defaultdict(set)
    train_videos: dict[str, set[str]] = defaultdict(set)
    valid_videos: set[str] = set()
    manifest_paths: set[str] = set()
    seen_uids: set[str] = set()
    row_count = 0
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"uid", "split", "class_name", "class_index", "category_id", "video_id",
                    "image_id", "crop_status", "crop_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            row_count += 1
            split, label = row["split"], row["class_name"]
            if split not in SPLITS or label not in by_name:
                error("manifest_label_or_split", line=line)
                continue
            if not row["uid"] or row["uid"] in seen_uids:
                error("manifest_uid", line=line, uid=row["uid"])
            seen_uids.add(row["uid"])
            for field in ("class_index", "category_id"):
                if row[field] != str(by_name[label][field]):
                    error("manifest_class_id", line=line, field=field, value=row[field])
            for field, groups in (("video_id", videos), ("image_id", images)):
                value = row[field].strip()
                # The preprocessing fallback img_* is NOT a verified video ID.
                if not value or (field == "video_id" and not value.startswith("vid_")):
                    error("unverified_group_id", line=line, field=field)
                else:
                    groups[value].add(split)
            if row["crop_status"] != "ok":
                continue
            rel = row["crop_path"].replace("\\", "/")
            target = (root / rel).resolve()
            if not rel or not target.is_relative_to(root) or Path(rel).is_absolute():
                error("unsafe_crop_path", line=line, path=rel)
                continue
            if rel in manifest_paths:
                error("duplicate_manifest_path", line=line, path=rel)
            manifest_paths.add(rel)
            if rel not in files:
                error("missing_crop", line=line, path=rel)
            elif files[rel] != {"split": split, "class_name": label}:
                error("manifest_folder_mismatch", line=line, path=rel)
            else:
                if row["video_id"].startswith("vid_"):
                    valid_videos.add(row["video_id"])
                    if split == "train":
                        train_videos[label].add(row["video_id"])
    for rel in sorted(set(files) - manifest_paths):
        error("orphan_crop", path=rel)
    video_leaks, image_leaks = _leaks(videos), _leaks(images)
    if video_leaks:
        error("cross_split_video", count=len(video_leaks))
    if image_leaks:
        error("cross_split_source_image", count=len(image_leaks))

    duplicates = []
    for digest, paths in sorted(pixel_groups.items()):
        if len(paths) < 2:
            continue
        splits = sorted({files[p]["split"] for p in paths})
        labels = sorted({files[p]["class_name"] for p in paths})
        duplicates.append({"pixel_sha256": digest, "paths": paths, "splits": splits, "classes": labels})
        if len(splits) > 1:
            error("cross_split_exact_pixels", pixel_sha256=digest)
        if len(labels) > 1:
            error("conflicting_duplicate_labels", pixel_sha256=digest)
    split_counts = {s: sum(counts[s].values()) for s in SPLITS}
    if split_counts != expected_counts:
        error("split_counts", expected=expected_counts, observed=split_counts)
    per_class = [{"class_index": c["class_index"], "category_id": c["category_id"],
                  "class_name": c["class_name"], **{s: counts[s][c["class_name"]] for s in SPLITS},
                  "train_videos": len(train_videos[c["class_name"]])} for c in mapping]
    for row in per_class:
        for split in SPLITS:
            if row[split] == 0:
                error("empty_class", split=split, class_name=row["class_name"])
    train_counts = [r["train"] for r in per_class]
    ratio = max(train_counts) / min(train_counts) if min(train_counts) else None
    return {
        "schema_version": 1, "data_version": "v1_crop128", "dataset": "TrashCan Instance",
        "status": "FAIL" if errors else "CONDITIONAL",
        "decision": "DO_NOT_CONTINUE" if errors else "KEEP_V1_WITH_CONDITIONS",
        "automated_checks_passed": not errors,
        "pending_verifications": dict(PENDING),
        "provenance": {"manifest_sha256": _file_hash(manifest_path),
                       "mapping_sha256": _file_hash(mapping_path),
                       "dataset_mapping_sha256": _file_hash(local_mapping) if local_mapping.exists() else None},
        "num_classes": len(mapping), "split_counts": split_counts, "manifest_rows": row_count,
        "manifest_video_count": len(videos), "valid_crop_video_count": len(valid_videos),
        "unreadable_images": unreadable, "invalid_images": invalid_images,
        "exact_duplicate_groups": duplicates,
        "cross_split_exact_duplicate_groups": sum(len(g["splits"]) > 1 for g in duplicates),
        "cross_split_video_ids": video_leaks, "cross_split_image_ids": image_leaks,
        "train_imbalance_ratio": ratio, "per_class": per_class, "errors": errors,
    }


def run_audit(processed_dir: str | Path, output_dir: str | Path,
              mapping_path: str | Path = DEFAULT_MAPPING,
              manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Write only JSON/CSV audit artifacts outside the original dataset."""
    root, out = Path(processed_dir).resolve(strict=True), Path(output_dir).resolve()
    if out.is_relative_to(root):
        raise ValueError("output_dir must be outside processed_dir; the dataset is read-only")
    inputs = {Path(mapping_path).resolve(), Path(manifest_path or root / "split_manifest.csv").resolve()}
    targets = [out / "audit_summary.json", out / "audit_class_counts.csv"]
    if any(p.resolve().is_relative_to(root) or p.resolve() in inputs for p in targets):
        raise ValueError("audit output must not overwrite source files or follow symlinks into the dataset")
    report = audit_dataset(root, mapping_path, manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with targets[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["per_class"][0]))
        writer.writeheader()
        writer.writerows(report["per_class"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    try:
        report = run_audit(args.processed_dir, args.output_dir, args.mapping, args.manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Audit input error: {exc}\n")
    print(f"{report['status']}: {report['decision']} ({len(report['errors'])} errors)")
    print("UNVERIFIED: " + ", ".join(report["pending_verifications"]))
    return 0 if report["automated_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
