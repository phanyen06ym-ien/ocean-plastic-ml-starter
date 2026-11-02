#!/usr/bin/env python3
"""Tạo labels, milestones, issue Sprint 1–3 và chuẩn hoá Pull Request theo quy ước nhóm.

Yêu cầu: GitHub CLI (`gh`) đã đăng nhập bằng tài khoản có quyền admin repo (`gh auth login`).

Cách dùng:
    python scripts/github/setup_sprints.py --dry-run          # xem trước, không ghi gì
    python scripts/github/setup_sprints.py --an <github-cua-An>
    python scripts/github/setup_sprints.py --an <github-cua-An> --close-superseded

Script chạy lại nhiều lần an toàn: label/milestone đã có thì cập nhật, issue đã có
(theo số hoặc theo tiền tố tiêu đề [S2-DATA]...) thì sửa thay vì tạo trùng.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "phanyen06ym-ien/ocean-plastic-ml-starter"
LEAD = "phanyen06ym-ien"
HERE = Path(__file__).resolve().parent

LABELS = [
    # name, color, description
    ("area:arch", "5319e7", "Kiến trúc, quản lý dự án, tài liệu chung"),
    ("area:data", "0e8a16", "Dataset Audit, EDA, tiền xử lý"),
    ("area:cnn", "1d76db", "Kiến trúc và huấn luyện CNN"),
    ("area:eval", "fbca04", "Đánh giá, so sánh, phân tích lỗi"),
    ("area:demo", "c5def5", "Inference, demo, kiểm thử"),
    ("type:feat", "a2eeef", "Tính năng / code mới"),
    ("type:docs", "0075ca", "Tài liệu"),
    ("type:exp", "d4c5f9", "Thí nghiệm huấn luyện / đánh giá"),
    ("type:chore", "ededed", "Cấu hình, quy trình"),
    ("type:bug", "d73a4a", "Lỗi cần sửa"),
    ("P0-critical", "b60205", "Đường găng — trễ là chặn người khác"),
    ("P1-high", "d93f0b", "Ưu tiên cao"),
    ("P2-medium", "fef2c0", "Ưu tiên trung bình"),
    ("status:blocked", "000000", "Đang chờ issue khác"),
    ("status:needs-changes", "e99695", "Sai quy ước hoặc cần sửa theo review"),
    ("status:superseded", "cfd3d7", "Đã bị PR/issue khác thay thế"),
]

MILESTONES = [
    ("Sprint 1 — Foundation", "2026-11-08T16:59:59Z",
     "23/10 – 08/11/2026 · Bài toán, kiến trúc, Dataset Audit, khung code CNN/Eval/Demo"),
    ("Sprint 2 — Data & Baseline", "2026-11-22T16:59:59Z",
     "09/11 – 22/11/2026 · Chốt lớp, dữ liệu processed, E01/E02, đánh giá Validation"),
    ("Sprint 3 — Improvement & Midterm", "2026-12-06T16:59:59Z",
     "23/11 – 06/12/2026 · Cải tiến, đánh giá Test, demo, báo cáo giữa kỳ, release v0.1-midterm"),
]

ISSUE_ORDER = [
    "S1-ARCH", "S1-DATA", "S1-CNN", "S1-EVAL", "S1-DEMO",
    "S2-ARCH", "S2-DATA", "S2-CNN", "S2-EVAL", "S2-DEMO",
    "S3-ARCH", "S3-DATA", "S3-CNN", "S3-EVAL", "S3-DEMO",
]
PR_ORDER = ["PR-CONVENTIONS", "PR-9", "PR-1", "PR-2"]

DRY = False


# ---------------------------------------------------------------- helpers
def gh(*args: str, payload: dict | None = None, check: bool = True) -> object:
    cmd = ["gh", *args]
    res = subprocess.run(cmd, input=json.dumps(payload) if payload is not None else None,
                         capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        if check:
            sys.exit(f"[LỖI] {' '.join(cmd)}\n{res.stderr.strip() or res.stdout.strip()}")
        return None
    out = res.stdout.strip()
    try:
        return json.loads(out) if out else None
    except json.JSONDecodeError:
        return out


def api(method: str, path: str, payload: dict | None = None, check: bool = True):
    if DRY and method != "GET":
        print(f"    (dry-run) {method} {path}")
        return {"number": 0}
    args = ["api", "-X", method, f"repos/{REPO}/{path}"]
    if payload is not None:
        args += ["--input", "-"]
    return gh(*args, payload=payload, check=check)


def api_all(path: str) -> list:
    sep = "&" if "?" in path else "?"
    return gh("api", "--paginate", f"repos/{REPO}/{path}{sep}per_page=100") or []


def parse_spec(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        sys.exit(f"[LỖI] {path} thiếu phần header ---")
    meta = {}
    for line in m.group(1).splitlines():
        key, _, val = line.partition(":")
        meta[key.strip()] = val.strip()
    return meta, m.group(2)


def as_list(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


# ---------------------------------------------------------------- steps
def setup_labels() -> None:
    print("== Labels")
    existing = {l["name"] for l in api_all("labels")}
    for name, color, desc in LABELS:
        body = {"name": name, "color": color, "description": desc}
        if name in existing:
            api("PATCH", f"labels/{name.replace(':', '%3A')}", {"new_name": name, "color": color, "description": desc})
            print(f"  cập nhật {name}")
        else:
            api("POST", "labels", body)
            print(f"  tạo {name}")


def setup_milestones() -> dict[str, int]:
    print("== Milestones")
    existing = {m["title"]: m["number"] for m in api_all("milestones?state=all")}
    out = {}
    for title, due, desc in MILESTONES:
        body = {"title": title, "due_on": due, "description": desc}
        if title in existing:
            api("PATCH", f"milestones/{existing[title]}", body)
            out[title] = existing[title]
            print(f"  cập nhật {title}")
        else:
            res = api("POST", "milestones", body)
            out[title] = res["number"]
            print(f"  tạo {title}")
    return out


def assignable(users: list[str]) -> list[str]:
    ok = []
    for u in users:
        res = subprocess.run(["gh", "api", f"repos/{REPO}/assignees/{u}"], capture_output=True)
        if res.returncode == 0:
            ok.append(u)
        else:
            print(f"  [CẢNH BÁO] @{u} chưa là collaborator → bỏ qua khi gán")
    return ok


def render(body: str, numbers: dict[str, int], an: str | None) -> str:
    body = body.replace("{{AN_MENTION}}", f"(@{an})" if an else "(GitHub: chưa cập nhật)")
    for key, num in numbers.items():
        body = body.replace("{{" + key + "}}", f"#{num}" if num else key)
    return body


def setup_issues(milestones: dict[str, int], an: str | None) -> dict[str, int]:
    print("== Issues")
    specs = {k: parse_spec(HERE / "issues" / f"{k}.md") for k in ISSUE_ORDER}
    all_issues = [i for i in api_all("issues?state=all") if "pull_request" not in i]

    numbers: dict[str, int] = {}
    for key, (meta, _) in specs.items():
        if meta.get("number"):
            numbers[key] = int(meta["number"])
            continue
        prefix = f"[{key}]"
        found = next((i["number"] for i in all_issues if i["title"].startswith(prefix)), None)
        numbers[key] = found or 0

    # Pass 1: tạo issue còn thiếu để có số
    for key in ISSUE_ORDER:
        meta, body = specs[key]
        if numbers[key]:
            continue
        res = api("POST", "issues", {"title": meta["title"], "body": "(đang khởi tạo)"})
        numbers[key] = res["number"]
        print(f"  tạo {meta['title']} → #{numbers[key]}")

    # Pass 2: ghi nội dung, labels, milestone, assignees
    for key in ISSUE_ORDER:
        meta, body = specs[key]
        users = [an if u == "{{AN}}" else u for u in as_list(meta.get("assignees"))]
        users = [u for u in users if u]
        if LEAD not in users:
            users.append(LEAD)
        payload = {
            "title": meta["title"],
            "body": render(body, numbers, an),
            "labels": as_list(meta.get("labels")),
            "milestone": milestones.get(meta["milestone"]),
            "assignees": assignable(users) if not DRY else users,
        }
        if meta.get("state") == "open":
            payload["state"] = "open"
        api("PATCH", f"issues/{numbers[key]}", payload)
        print(f"  chuẩn hoá #{numbers[key]} {meta['title']}  assignees={payload['assignees']}")
    return numbers


def setup_prs(milestones: dict[str, int], issue_numbers: dict[str, int], close_superseded: bool) -> None:
    print("== Pull Requests")
    pr_numbers: dict[str, int] = {}
    open_prs = api_all("pulls?state=open")
    for key in PR_ORDER:
        meta, body = parse_spec(HERE / "prs" / f"{key}.md")
        number = int(meta["number"]) if meta.get("number") else None
        if number is None:
            head = meta["head"]
            number = next((p["number"] for p in open_prs if p["head"]["ref"] == head), None)
            if number is None:
                branch = gh("api", f"repos/{REPO}/branches/{head}", check=False)
                if not branch:
                    print(f"  [BỎ QUA] chưa có nhánh {head} trên GitHub → push nhánh rồi chạy lại")
                    continue
                res = api("POST", "pulls", {"title": meta["title"], "head": head,
                                            "base": meta.get("base", "main"), "body": "(đang khởi tạo)",
                                            "draft": False})
                number = res["number"]
                print(f"  tạo PR #{number} từ {head}")
        pr_numbers[meta.get("key", key)] = number

    for key in PR_ORDER:
        meta, body = parse_spec(HERE / "prs" / f"{key}.md")
        pkey = meta.get("key", key)
        number = int(meta["number"]) if meta.get("number") else pr_numbers.get(pkey)
        if not number:
            continue
        text = render(body, {**issue_numbers, **pr_numbers}, None)
        api("PATCH", f"pulls/{number}", {"title": meta["title"], "body": text})
        api("PATCH", f"issues/{number}", {
            "labels": as_list(meta.get("labels")),
            "milestone": milestones.get(meta["milestone"]),
            "assignees": as_list(meta.get("assignees")),
        })
        pr = api("GET", f"pulls/{number}")
        author = pr["user"]["login"] if isinstance(pr, dict) and "user" in pr else LEAD
        reviewers = [r for r in [LEAD, *as_list(meta.get("reviewers"))] if r != author]
        reviewers = assignable(reviewers) if not DRY else reviewers
        if reviewers:
            api("POST", f"pulls/{number}/requested_reviewers", {"reviewers": reviewers}, check=False)
        print(f"  chuẩn hoá PR #{number}: {meta['title']}  reviewers={reviewers}")

        sup = meta.get("superseded_by")
        if sup and close_superseded:
            target = pr_numbers.get(sup) or (int(sup) if sup.isdigit() else None)
            if target:
                api("POST", f"issues/{number}/comments",
                    {"body": f"Đóng vì đã được thay thế bởi #{target} (quy ước repo Sprint 1). — Yến"})
                api("PATCH", f"pulls/{number}", {"state": "closed"})
                print(f"  đóng PR #{number} (superseded by #{target})")


def main() -> None:
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Chỉ in ra, không ghi lên GitHub")
    ap.add_argument("--an", default="", help="GitHub username của Ân (để gán issue EVAL)")
    ap.add_argument("--close-superseded", action="store_true", help="Đóng PR #1, #2 đã bị thay thế")
    ap.add_argument("--skip-prs", action="store_true")
    args = ap.parse_args()
    DRY = args.dry_run

    setup_labels()
    ms = setup_milestones()
    nums = setup_issues(ms, args.an.strip() or None)
    if not args.skip_prs:
        setup_prs(ms, nums, args.close_superseded)
    print("\nXong. Kiểm tra: https://github.com/" + REPO + "/issues")


if __name__ == "__main__":
    main()
