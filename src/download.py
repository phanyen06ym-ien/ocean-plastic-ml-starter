"""Tải TrashCan 1.0 (dataset.zip) trực tiếp từ UMN Data Repository (DSpace 7 REST API).

Dùng trên Colab khi Drive `data/raw/` chưa có file .zip, tránh phải tải về máy rồi tải lên Drive.
"""
from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

UMN_API = "https://conservancy.umn.edu/server/api/core"
TRASHCAN_ITEM_UUID = "6dd6a960-c44a-4510-a679-efb8c82ebfb7"


def _get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def pick_bitstream(bitstreams: list[dict[str, Any]], name: str = "dataset.zip") -> dict[str, Any]:
    """Chọn bitstream theo tên (không phân biệt hoa thường); nếu không có thì lấy file .zip lớn nhất."""
    exact = [b for b in bitstreams if b.get("name", "").lower() == name.lower()]
    if exact:
        return exact[0]
    zips = [b for b in bitstreams if b.get("name", "").lower().endswith(".zip")]
    if not zips:
        raise FileNotFoundError(f"Không thấy file .zip trong item; có: {[b.get('name') for b in bitstreams]}")
    return max(zips, key=lambda b: b.get("sizeBytes", 0))


def list_bitstreams(item_uuid: str = TRASHCAN_ITEM_UUID, api: str = UMN_API) -> list[dict[str, Any]]:
    bundles = _get_json(f"{api}/items/{item_uuid}/bundles")["_embedded"]["bundles"]
    out: list[dict[str, Any]] = []
    for b in bundles:
        if b.get("name") != "ORIGINAL":
            continue
        data = _get_json(b["_links"]["bitstreams"]["href"])
        out.extend(data["_embedded"]["bitstreams"])
    return out


def download_trashcan(dest_dir: str | Path, name: str = "dataset.zip") -> Path:
    """Tải dataset.zip vào dest_dir (bỏ qua nếu đã có file cùng tên, đúng dung lượng)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    bs = pick_bitstream(list_bitstreams(), name)
    target = dest_dir / bs["name"]
    size = bs.get("sizeBytes")
    if target.exists() and (size is None or target.stat().st_size == size):
        print("Đã có", target)
        return target
    url = bs["_links"]["content"]["href"]
    print(f"Tải {bs['name']} ({(size or 0) / 1e9:.2f} GB) từ {url}")
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=16 << 20)
    tmp.rename(target)
    return target
