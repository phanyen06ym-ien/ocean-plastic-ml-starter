# Quy ước làm việc của nhóm

Áp dụng cho **mọi thành viên, mọi sprint**. PR không tuân thủ sẽ bị gắn `status:needs-changes` và không được merge.

Người quản lý quy trình, review và merge: **Phan Hải Yến (@phanyen06ym-ien)**.

---

## 1. Quy trình chuẩn

```
Issue (Yến tạo, giao) → Nhánh riêng → Commit → Pull Request (Draft)
→ Ready for review → Yến review → Sửa theo nhận xét → Yến approve → Squash & merge → Xoá nhánh → Đóng issue
```

- **Không ai commit trực tiếp vào `main`**, kể cả từ Colab ("Created using Colab"), kể cả trưởng nhóm.
- Mỗi PR phải gắn với ít nhất một issue.
- Issue chỉ được đóng khi **PR bàn giao đã được merge** và mọi mục Definition of Done đã tick.

## 2. Issue

- Dùng duy nhất mẫu `.github/ISSUE_TEMPLATE/task.md`. Không tạo issue trống.
- Tiêu đề: `[S<sprint>-<AREA>] <Tên nhiệm vụ>` — ví dụ `[S2-DATA] Processed Dataset, Class Mapping & Grouped Split`.
- **Assignees:** người phụ trách chính + **@phanyen06ym-ien** (người phụ trách phụ).
- **Bắt buộc có:** milestone (sprint), 1 label `area:*`, 1 label `type:*`, 1 label ưu tiên `P0`/`P1`/`P2`.
- Mỗi đầu việc trong checklist có **hạn chót (dd/mm)**. Trễ hạn → người phụ trách bình luận lý do + hạn mới, Yến xác nhận.
- Bị chặn bởi việc của người khác → gắn `status:blocked` và ghi rõ đang chờ issue nào.
- Báo tiến độ: bình luận trong issue **mỗi thứ Hai và thứ Năm** (đã làm / sẽ làm / vướng mắc).

## 3. Nhánh (branch)

```
<type>/s<sprint>-<area>-<mo-ta-ngan>
```

| Thành phần | Giá trị cho phép |
|---|---|
| `type` | `feat` `fix` `docs` `exp` `test` `refactor` `chore` |
| `area` | `arch` `data` `cnn` `eval` `demo` |
| `mo-ta-ngan` | chữ thường, không dấu, nối bằng `-`, tối đa 5 từ |

Ví dụ: `feat/s2-data-grouped-split`, `exp/s2-cnn-e01-baseline`, `docs/s1-arch-class-scope`.

## 4. Commit

Theo [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(s<sprint>-<area>): <mô tả ngắn, động từ mệnh lệnh, ≤ 72 ký tự>
```

- `feat(s2-data): crop objects from TrashCan-Instance annotations`
- `fix(s2-cnn): use sparse categorical loss for int labels`
- `exp(s2-cnn): add E02 augmentation config`
- `docs(s1-arch): record class scope decision ADR-001`

Một commit làm một việc. Không commit file sinh tự động, dữ liệu, checkpoint.

## 5. Pull Request

**Tiêu đề** giống định dạng commit: `<type>(s<sprint>-<area>): <mô tả>`.

**Nội dung:** bắt buộc dùng `.github/pull_request_template.md`, điền đủ 7 mục.

**Liên kết issue:**
- `Closes #<n>` — PR hoàn thành toàn bộ issue.
- `Refs #<n>` — PR chỉ hoàn thành một phần.

**Trước khi bấm "Ready for review":**
- [ ] Đã rebase/merge `main` mới nhất, không xung đột.
- [ ] `pytest -q` chạy qua (nếu PR có code).
- [ ] Notebook chạy lại được từ đầu trên Colab; không chứa token, mật khẩu, đường dẫn cá nhân.
- [ ] Không có file trong `data/`, `models/`, file > 5 MB.
- [ ] Gắn đủ labels + milestone giống issue; reviewer: **@phanyen06ym-ien**.

**Kích thước:** ưu tiên PR nhỏ (≤ ~400 dòng code thay đổi, không tính output notebook). Việc lớn → tách nhiều PR `Refs #n`.

**Draft:** mở PR ở trạng thái Draft ngay khi có commit đầu tiên để Yến theo dõi.

## 6. Review và merge

| Quy định | Chi tiết |
|---|---|
| Reviewer | **Yến review tất cả PR.** CODEOWNERS bắt buộc Yến phê duyệt. |
| PR do Yến tạo | GitHub không cho tự approve → chủ khu vực liên quan review nội dung chuyên môn; Yến kiểm tra checklist và merge. |
| Thời hạn | Yến review trong **48 giờ** kể từ "Ready for review"; tác giả xử lý nhận xét trong **48 giờ**. |
| Hạn mở PR | Chậm nhất **2 ngày trước hạn chót** của issue. |
| Mức nhận xét | `[BLOCKER]` phải sửa trước merge · `[SHOULD]` nên sửa · `[NIT]` tuỳ chọn. |
| Merge | Chỉ Yến merge, dùng **Squash and merge**, xoá nhánh sau merge. |
| Từ chối | PR thiếu template, thiếu issue, sai tên nhánh/tiêu đề → gắn `status:needs-changes`, không review nội dung. |

## 7. Đặt tên file

| Loại | Quy tắc | Ví dụ |
|---|---|---|
| Notebook | `NN_<area>_<mo_ta>.ipynb` (NN tăng dần) | `01_trashcan_eda.ipynb`, `05_inference_demo.ipynb` |
| Module Python | `snake_case.py` trong `src/` | `src/preprocessing.py` |
| Test | `tests/test_<module>.py` | `tests/test_evaluate.py` |
| Tài liệu | `docs/<snake_case>.md` | `docs/dataset_report.md` |
| Quyết định | `docs/decisions/ADR-<NNN>-<chu-de>.md` | `ADR-001-class-scope.md` |
| Cấu hình thí nghiệm | `configs/experiments/<EXP_ID>_<mo_ta>.yaml` | `E01_baseline.yaml` |
| Kết quả số liệu | `results/metrics/<EXP_ID>_<split>.json` | `E01_val.json` |
| Hình | `results/figures/<EXP_ID>_<loai>.png` | `E01_confusion_matrix.png` |
| Run trên Drive | `runs/<EXP_ID>_<YYYYMMDD-HHMM>/` | `runs/E01_20261114-0930/` |

## 8. Dữ liệu, mô hình và bí mật

- **Không commit:** dữ liệu TrashCan, ảnh crop, checkpoint (`*.keras`, `*.h5`, `*.pt`), token, file `.env`.
- Mọi thứ lớn để trên Drive theo cấu trúc trong README; PR chỉ ghi **đường dẫn Drive** và mã phiên bản (`DATA_VERSION`, `EXP_ID`).
- Token GitHub dùng trong Colab đặt trong **Colab Secrets** (`GH_TOKEN`), không in ra màn hình, không lưu trong output notebook.

## 9. Thực nghiệm và báo cáo số liệu

- Mỗi lần huấn luyện có `EXP_ID`, seed, `DATA_VERSION`, cấu hình đầy đủ lưu cùng run.
- Chọn mô hình **chỉ dựa trên Validation**. Tập Test chỉ chạy **một lần** cho mô hình cuối, do người phụ trách EVAL thực hiện và Yến xác nhận.
- Không ghi số liệu nào vào tài liệu/README khi chưa có run tương ứng trên Drive và file `results/metrics/` được merge.
- Dữ liệu giả dùng để kiểm thử code không được trình bày như kết quả trên TrashCan.

## 10. Definition of Done chung

Một issue được đóng khi:

1. Mọi mục trong checklist đã tick, sản phẩm bàn giao có trên `main`.
2. PR liên quan được Yến approve và merge.
3. Tài liệu liên quan (README, `docs/`) được cập nhật.
4. Người nhận bàn giao (ghi trong mục Phụ thuộc) xác nhận trong issue.
