---
key: S1-ARCH
number: 4
title: [S1-ARCH] Problem Analysis, Architecture & Experiment Design
assignees: phanyen06ym-ien
labels: area:arch, type:docs, P1-high
milestone: Sprint 1 — Foundation
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 1 — Foundation |
| Mã nhiệm vụ | `S1-ARCH` |
| Người phụ trách chính | Phan Hải Yến (@phanyen06ym-ien) |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) — trưởng nhóm; nội dung chuyên môn do chủ từng khu vực review chéo |
| Vai trò | Team Leader · Problem Analysis · ML System Design · Project Management |
| Ưu tiên | P1 — High |
| Nhánh | `docs/s1-arch-<mo-ta>` (PR #9: `docs/s1-arch-option-d-review`) |
| Thời gian | 23/10 → 08/11/2026 |
| Hạn mở PR | 05/11 |

## 2. Mục tiêu

Phân tích bài toán **phân loại ảnh đối tượng rác thải liên quan đến nhựa dưới nước bằng CNN** trên **TrashCan 1.0 — Instance**; thiết kế kiến trúc dự án, ML pipeline, hợp đồng dữ liệu và kế hoạch thực nghiệm; thiết lập quy trình làm việc thống nhất cho 5 thành viên.

**Kết quả mong đợi:** phạm vi lớp được chốt bằng văn bản (ADR-001), `configs/data_contract.yaml` không còn mục `TBD` quan trọng, mọi thành viên có issue riêng với tiêu chí nghiệm thu, quy trình Branch → PR → Review được áp dụng.

## 3. Phạm vi công việc

### A. Phân tích bài toán — hạn 05/11
- [x] Xác định chủ đề và mục tiêu nghiên cứu.
- [x] Lựa chọn bộ dữ liệu TrashCan 1.0, phiên bản Instance (phân loại theo loại đối tượng).
- [x] Xác định bài toán giữa kỳ là CNN Image Classification trên ảnh đối tượng đã crop.
- [x] Soạn Problem Statement (PR #1, cập nhật trong PR #9).
- [ ] Đối chiếu đề xuất **Option D (22 lớp)** trong PR #9 với định hướng **chỉ giữ nhãn rác liên quan nhựa có đủ mẫu**; chọn một phương án và ghi lý do.
- [ ] Chốt danh sách lớp, ngưỡng số crop tối thiểu, có/không lớp `other` → `docs/decisions/ADR-001-class-scope.md`.
- [ ] Cập nhật `docs/problem_statement.md` theo quyết định đã chốt.

### B. Kiến trúc và hợp đồng dữ liệu — hạn 05/11
- [x] Khởi tạo GitHub repository.
- [x] Khởi tạo cấu trúc Colab notebooks (PR #3 — đã merge).
- [ ] Áp dụng quy ước thư mục, README, CONTRIBUTING, issue/PR template (PR `chore(s1-arch): repository conventions`).
- [ ] Chốt `configs/data_contract.yaml`: kích thước ảnh (224 trong problem statement ≠ 128 trong notebook 03), chuẩn hoá, cách crop, tỉ lệ split theo video, seed.
- [ ] Hoàn thiện `docs/colab_setup.md` (gộp nội dung từ nhánh của Ân).

### C. Thiết kế thực nghiệm — hạn 07/11
- [ ] Giả thuyết và mục tiêu thực nghiệm.
- [ ] Danh sách thí nghiệm: E01 CNN baseline, E02 = E01 + Data Augmentation (các yếu tố khác giữ cố định).
- [ ] Thống nhất với S1-EVAL chỉ số chính (Macro F1) và quy tắc dùng Validation/Test.
- [ ] Tham số bắt buộc ghi nhận mỗi run: `EXP_ID`, `DATA_VERSION`, epochs, batch size, learning rate, optimizer, seed.

### D. Điều phối — hạn 08/11
- [x] Tạo issue Sprint 1 cho 5 thành viên (#4–#8).
- [ ] Bật branch protection cho `main`: bắt buộc PR, 1 approval, review từ Code Owners.
- [ ] Mời GitHub của Ân làm collaborator và gán vào issue #7.
- [ ] Review PR #1, #2, #9 và các PR S1 của thành viên.
- [ ] Tổ chức Sprint 1 Review, ghi biên bản `docs/reports/sprint1_review.md`, chuyển việc dở sang Sprint 2.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Problem Statement | `docs/problem_statement.md` | 05/11 |
| Class Scope Analysis + quyết định | `docs/class_scope_analysis.md`, `docs/decisions/ADR-001-class-scope.md` | 05/11 |
| Data Contract | `configs/data_contract.yaml` | 05/11 |
| Quy ước repo | `README.md`, `CONTRIBUTING.md`, `.github/` | 04/11 |
| Kế hoạch thực nghiệm | `docs/experiment_plan.md` | 07/11 |
| Sprint 1 Review | `docs/reports/sprint1_review.md` | 08/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] ADR-001 được 4 chủ khu vực xác nhận bằng bình luận trong issue này.
- [ ] `data_contract.yaml` không còn `TBD` ở các mục image, split, classes.
- [ ] README, CONTRIBUTING, templates đã merge; branch protection đã bật.
- [ ] Mọi thành viên có issue Sprint 2 với hạn chót và tiêu chí nghiệm thu.
- [ ] PR #1, #2, #9 được merge hoặc đóng có lý do (superseded).
- [ ] Biên bản Sprint 1 Review đã merge.

## 6. Ngoài phạm vi

- Lập trình CNN và huấn luyện — Hoàng Bách (#6).
- Dataset Audit, EDA, tiền xử lý — Hùng (#5).
- Cài đặt đánh giá và phân tích lỗi — Ân (#7).
- Chương trình dự đoán và demo — Ngọc Nguyên (#8).
- GAN, Object Detection, robot.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | #5 | Thống kê lớp, kết quả Visual Inspection để chốt ADR-001 |
| Bàn giao cho | #5, #6, #7, #8 | ADR-001, data contract, kế hoạch thực nghiệm |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 03/11 | Mở PR quy ước repo; bật branch protection; mời Ân | Yến |
| 04/11 | Gửi PR #9 cho Hùng, Bách, Ân, Nguyên review chuyên môn | Yến |
| 05/11 | Chốt ADR-001 + data contract | Cả nhóm |
| 07/11 | Hoàn tất review mọi PR Sprint 1 | Yến |
| 08/11 | Sprint 1 Review, mở Sprint 2 | Yến |

## 9. Quy định review

- PR do Yến tạo: chủ khu vực liên quan review chuyên môn (GitHub không cho tác giả tự approve); Yến kiểm tra checklist và merge.
- Tiêu đề PR `docs(s1-arch): ...`, nội dung theo PR template, ghi `Refs #4` hoặc `Closes #4`.

## 10. Ghi chú

- Nhãn TrashCan-Instance là loại đối tượng; không mặc định mọi `trash_bottle`, `trash_rope`… là nhựa.
- Không ghi số liệu hiệu năng khi chưa có run thực tế.
