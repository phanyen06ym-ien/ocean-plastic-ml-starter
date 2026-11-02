---
key: CONVENTIONS
head: chore/s1-arch-repo-conventions
base: main
title: chore(s1-arch): add repository conventions, README v2 and sprint plan
labels: area:arch, type:chore, P0-critical
milestone: Sprint 1 — Foundation
assignees: phanyen06ym-ien
reviewers: tranngochung126hqv-stack, phannguyenhoangbach2, hiang0910
---
## 1. Tóm tắt

Thiết lập quy chuẩn chung cho cả nhóm: cấu trúc thư mục, quy ước nhánh/commit/PR/issue, mẫu issue và PR, CODEOWNERS (Yến là reviewer bắt buộc), README v2 và kế hoạch Sprint 1–3.

## 2. Issue liên quan

Refs #4

## 3. Thay đổi

| File | Thay đổi |
|---|---|
| `README.md` | README v2: bài toán, TrashCan-Instance, nguyên tắc chọn nhãn nhựa, cấu trúc thư mục, Colab + GitHub + VS Code, nhóm, lộ trình |
| `CONTRIBUTING.md` | Quy trình, đặt tên nhánh/commit/PR/file, review & merge, dữ liệu/bí mật, Definition of Done |
| `.github/ISSUE_TEMPLATE/task.md`, `config.yml` | Mẫu issue duy nhất, tắt issue trống |
| `.github/pull_request_template.md` | Mẫu PR 7 mục |
| `.github/CODEOWNERS` | `* @phanyen06ym-ien` |
| `configs/data_contract.yaml` | Khung hợp đồng dữ liệu (giá trị `TBD` chốt tại #4) |
| `docs/sprint_plan.md` | Phân công, timeline, đường găng Sprint 1–3 |
| `.gitignore` | Giữ khung thư mục `data/*`, chặn dữ liệu, checkpoint, `runs/` |
| `data/`, `results/`, `configs/experiments/`, `docs/decisions/`, `docs/reports/`, `tests/` | Khung thư mục chuẩn (`.gitkeep`) |
| `scripts/github/` | Script tạo labels, milestones, issue Sprint 1–3 và chuẩn hoá PR |

## 4. Cách kiểm tra

```bash
python scripts/github/setup_sprints.py --dry-run
```
Mở thử "New issue" và "New pull request" sau khi merge để thấy template.

## 5. Checklist của tác giả

- [x] Tên nhánh và tiêu đề PR đúng quy ước.
- [x] Tạo từ `main` mới nhất, không xung đột.
- [x] Không commit dữ liệu, checkpoint, token.
- [x] Không sửa `src/`, notebook, `requirements.txt` để tránh xung đột với nhánh của thành viên.
- [x] Đã gắn labels, milestone.

## 6. Bằng chứng / kết quả

Không áp dụng (cấu hình + tài liệu).

## 7. Ghi chú cho reviewer

PR do Yến tạo → mỗi chủ khu vực đọc `CONTRIBUTING.md` và xác nhận bằng approve/bình luận. Sau khi merge, Yến bật **Settings → Branches → `main`**: *Require a pull request*, *Require approvals (1)*, *Require review from Code Owners*.

PR này thay thế #2 (README v1.0).

---
**Review (Yến):** ☐ Đúng quy ước ☐ Đủ DoD của issue ☐ Approve
