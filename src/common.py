"""Tiện ích dùng chung: đọc data contract và class mapping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "configs" / "data_contract.yaml"
DEFAULT_MAPPING = PROJECT_ROOT / "configs" / "class_mapping.json"


def load_yaml(path: str | Path) -> dict[str, Any]:
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


def validate_instance_mapping(mapping: list[dict[str, Any]]) -> None:
    """Require the frozen 22 Instance class IDs, names and plastic flags.

    Compare against the repository mapping, never ImageFolder alphabetical order
    or a possibly modified mapping shipped with the processed dataset.
    """
    canonical = load_class_mapping(DEFAULT_MAPPING)
    fields = ("class_index", "category_id", "class_name", "plastic_related")
    if len(mapping) != 22 or any(
        tuple(c.get(k) for k in fields) != tuple(ref.get(k) for k in fields)
        for c, ref in zip(mapping, canonical)
    ):
        raise ValueError("class mapping must match the frozen 22 Instance class IDs (0..21)")


def plastic_indices(mapping: list[dict[str, Any]]) -> list[int]:
    return [c["class_index"] for c in mapping if c.get("plastic_related")]
