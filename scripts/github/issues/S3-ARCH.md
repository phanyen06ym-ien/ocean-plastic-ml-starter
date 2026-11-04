---
key: S3-ARCH
title: [S3-ARCH] Midterm Report, Reproducibility Audit & Release v0.1-midterm
assignees: phanyen06ym-ien
labels: area:arch, type:docs, P0-critical
milestone: Sprint 3 — Improvement & Midterm
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 3 — Improvement & Midterm |
| Mã nhiệm vụ | `S3-ARCH` |
| Người phụ trách chính | Phan Hải Yến (@phanyen06ym-ien) |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) — chủ khu vực review chéo PR do Yến tạo |
| Vai trò | Team Leader · Report · Release |
| Ưu tiên | P0 — Critical |
| Nhánh | `docs/s3-arch-midterm-report` |
| Thời gian | 05/11 → 18/11/2026 |
| Hạn mở PR | 16/11 |

## 2. Mục tiêu

Tổng hợp toàn bộ kết quả thành báo cáo và slide giữa kỳ, kiểm tra khả năng tái lập của dự án và phát hành phiên bản `v0.1-midterm`.

**Kết quả mong đợi:** báo cáo, slide, README cập nhật đầy đủ kết quả đã kiểm chứng; một người ngoài nhóm có thể chạy lại pipeline theo hướng dẫn.

## 3. Phạm vi công việc

### A. Chốt mô hình — hạn 12/11
- [ ] Họp chọn mô hình cuối dựa trên Macro F1 **Validation** (E01/E02/E03); ghi `docs/decisions/ADR-00x-final-model.md`.
- [ ] Cho phép S3-EVAL chạy Test **một lần** sau khi ADR được merge.

### B. Báo cáo và slide — hạn 16/11
- [ ] `docs/reports/midterm_report.md`: bài toán, dữ liệu, chọn lớp, phương pháp, thí nghiệm, kết quả, phân tích lỗi, hạn chế, hướng phát triển.
- [ ] Slide trình bày (15 phút) + phân công phần nói cho 5 thành viên.
- [ ] Mọi số liệu trong báo cáo đối chiếu với `results/metrics/` (Yến tick từng bảng).

### C. Kiểm tra tái lập — hạn 15/11
- [ ] Clone sạch → làm theo README/colab_setup → chạy notebook 01→05 với `DATA_VERSION` và checkpoint cuối.
- [ ] Ghi các lỗi gặp phải thành issue `type:bug` và giao người sửa.

### D. Phát hành — hạn 18/11
- [ ] Cập nhật README: kết quả, hướng dẫn chạy demo, trích dẫn.
- [ ] Tag `v0.1-midterm` + GitHub Release (ghi chú phiên bản, đường dẫn Drive checkpoint).
- [ ] Biên bản `docs/reports/sprint3_review.md` + backlog sau giữa kỳ.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Quyết định mô hình cuối | `docs/decisions/ADR-00x-final-model.md` | 12/11 |
| Báo cáo giữa kỳ | `docs/reports/midterm_report.md` | 16/11 |
| Slide | Link trong báo cáo | 16/11 |
| Biên bản kiểm tra tái lập | `docs/reports/reproducibility_check.md` | 15/11 |
| Release | Tag `v0.1-midterm` | 18/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] Mô hình cuối chọn trên Validation, có ADR trước khi chạy Test.
- [ ] Không có số liệu nào trong báo cáo/README thiếu nguồn.
- [ ] Kiểm tra tái lập thành công hoặc các lỗi đã có issue và được sửa.
- [ ] Release đã phát hành.

## 6. Ngoài phạm vi

- Object Detection, GAN, robot — backlog sau giữa kỳ.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | {{S3-CNN}} | E03 và mô hình ứng viên |
| Nhận từ | {{S3-EVAL}} | Kết quả Test cuối, phân tích lỗi |
| Nhận từ | {{S3-DATA}} | Dataset card |
| Nhận từ | {{S3-DEMO}} | Demo, video/ảnh chụp demo |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 05/11 | Sprint 3 Planning | Cả nhóm |
| 09/11 | Checkpoint giữa sprint | Cả nhóm |
| 12/11 | Chốt mô hình cuối (ADR) | Cả nhóm |
| 13/11 | Đánh giá Test một lần | Ân → Yến |
| 15/11 | Kiểm tra tái lập | Yến |
| 16/11 | Báo cáo + slide hoàn chỉnh; tập dượt | Cả nhóm |
| 18/11 | Release `v0.1-midterm`, Sprint 3 Review | Yến |

## 9. Quy định review

- PR do Yến tạo: chủ khu vực liên quan review chuyên môn; Yến kiểm tra checklist và merge.
- Tiêu đề `docs(s3-arch): ...`; `Refs`/`Closes` issue này.

## 10. Ghi chú

- Lịch Sprint 3 bám theo hạn nộp giữa kỳ của môn; nếu hạn nộp khác 18/11, Yến dời milestone và các mốc tương ứng.
