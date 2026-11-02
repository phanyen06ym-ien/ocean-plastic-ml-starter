---
number: 9
title: docs(s1-arch): define problem statement, class scope and visual inspection protocol
labels: area:arch, type:docs, P1-high
milestone: Sprint 1 — Foundation
assignees: phanyen06ym-ien
reviewers: tranngochung126hqv-stack, phannguyenhoangbach2, hiang0910
---
## 1. Tóm tắt

Bổ sung bộ tài liệu phân tích bài toán, phạm vi lớp và thiết kế kiểm tra trực quan cho Sprint 1. PR đề xuất **Option D** (một CNN 22 lớp + phân tích chuyên sâu 6 lớp ứng viên liên quan nhựa). **Phạm vi lớp chưa được phê duyệt.**

## 2. Issue liên quan

Refs #4

## 3. Thay đổi

| File | Thay đổi |
|---|---|
| `docs/problem_statement.md` | Bài toán, đầu vào/đầu ra CNN (ảnh crop 224×224), ranh giới phạm vi, tiêu chí nghiệm thu |
| `docs/class_scope_analysis.md` | Thống kê 22 lớp TrashCan-Instance, so sánh Option A–D, đề xuất Option D |
| `docs/visual_inspection_protocol.md` | Quy trình lấy mẫu tái lập, kiểm tra trực quan độc lập, xử lý bất đồng |

## 4. Cách kiểm tra

Đọc 3 tài liệu; đối chiếu bảng thống kê lớp với số liệu Dataset Audit của Hùng (`grouped_class_distribution.csv`).

## 5. Checklist của tác giả

- [x] Tên nhánh `docs/s1-arch-…` và tiêu đề PR đúng quy ước.
- [x] Đã cập nhật từ `main`, không xung đột.
- [ ] Code/test chạy qua — không áp dụng (chỉ tài liệu).
- [x] Không commit dữ liệu, checkpoint, token.
- [ ] Số liệu báo cáo có nguồn — chờ Hùng xác nhận bảng 22 lớp.
- [ ] Tài liệu liên quan đã cập nhật — README cập nhật trong PR quy ước repo.
- [x] Đã gắn labels, milestone.

## 6. Bằng chứng / kết quả

Không áp dụng (tài liệu).

## 7. Ghi chú cho reviewer

PR do Yến tạo nên **chủ khu vực review chuyên môn**, Yến kiểm tra checklist và merge:

- [ ] **Hùng (DATA):** xác minh số lớp, số annotation, quy tắc chia theo video.
- [ ] **Hoàng Bách (CNN):** xác nhận đầu vào 224×224 (notebook 03 hiện dùng 128) và số lớp đầu ra khả thi.
- [ ] **Ân (EVAL):** duyệt phương pháp đánh giá tổng thể + phân tích nhóm lớp nhựa, quy trình Visual Inspection.
- [ ] **Ngọc Nguyên (DEMO):** xác nhận nhãn và ánh xạ đầu ra thống nhất.

**Quyết định cần chốt trước khi merge:** định hướng nhóm là *chỉ giữ các nhãn rác liên quan nhựa có đủ mẫu để huấn luyện* (gần Option C), trong khi PR đề xuất Option D (22 lớp). Nhóm chọn một phương án, ghi vào `docs/decisions/ADR-001-class-scope.md` và cập nhật tài liệu tương ứng.

**Điều kiện merge:** reviewer chuyên môn đã xác nhận; nhận xét `[BLOCKER]` đã xử lý; ADR-001 đã chốt và ghi trong Issue #4.

---
**Review (Yến):** ☐ Đúng quy ước ☐ Đủ DoD của issue ☐ Approve
