---
key: S2-ARCH
title: [S2-ARCH] Class Scope Lock, Experiment Registry & Sprint 2 Integration
assignees: phanyen06ym-ien
labels: area:arch, type:docs, P0-critical
milestone: Sprint 2 — Data & Baseline
---
## 1. Thông tin nhiệm vụ

| Mục | Nội dung |
|---|---|
| Project | Ocean Plastic Waste Classification and Monitoring |
| Sprint | Sprint 2 — Data & Baseline |
| Mã nhiệm vụ | `S2-ARCH` |
| Người phụ trách chính | Phan Hải Yến (@phanyen06ym-ien) |
| Người phụ trách phụ & review | Phan Hải Yến (@phanyen06ym-ien) — chủ khu vực review chéo PR do Yến tạo |
| Vai trò | Team Leader · Integration · Experiment Management |
| Ưu tiên | P0 — Critical |
| Nhánh | `docs/s2-arch-<mo-ta>` |
| Thời gian | 04/11/2026 |
| Hạn mở PR | 04/11 |

## 2. Mục tiêu

Khoá phạm vi lớp và hợp đồng dữ liệu trước khi huấn luyện, quản lý danh sách thí nghiệm, tích hợp kết quả DATA → CNN → EVAL → DEMO và kiểm soát tiến độ Sprint 2.

**Kết quả mong đợi:** dữ liệu, mô hình, đánh giá, demo dùng **cùng một** class mapping và data contract; E01 và E02 có đầy đủ run, cấu hình và kết quả Validation được ghi trong sổ thí nghiệm.

## 3. Phạm vi công việc

### A. Khoá phạm vi — hạn 04/11
- [ ] Xác nhận ADR-001 không đổi sau khi S2-DATA có số crop hợp lệ thực tế; nếu đổi → ADR-002.
- [ ] Phê duyệt `configs/class_mapping.json` và `configs/data_contract.yaml` phiên bản `v1` (không còn `TBD`).

### B. Quản lý thí nghiệm — hạn 04/11
- [ ] `docs/experiment_plan.md`: bảng E01, E02 (giả thuyết, biến thay đổi, biến cố định, seed, DATA_VERSION).
- [ ] `configs/experiments/E01_baseline.yaml`, `E02_augment.yaml` (phối hợp Bách).
- [ ] Sổ thí nghiệm `docs/experiment_log.md`: mỗi run một dòng (EXP_ID, ngày, người chạy, đường dẫn Drive, Macro F1 val).

### C. Tích hợp và kiểm soát — hạn 04/11
- [ ] Kiểm tra end-to-end: dữ liệu processed → train E01 → evaluate → predict ảnh mẫu, cùng class mapping.
- [ ] Review mọi PR Sprint 2 trong 48 giờ.
- [ ] Theo dõi đường găng: S2-DATA bàn giao dữ liệu **04/11** — trễ thì điều chỉnh lịch CNN/EVAL ngay.
- [ ] Cập nhật README mục Kết quả từ `results/metrics/` đã merge.

### D. Sprint Review — hạn 04/11
- [ ] Biên bản `docs/reports/sprint2_review.md`: hoàn thành, chưa hoàn thành, rủi ro, việc chuyển Sprint 3.
- [ ] Dàn ý báo cáo giữa kỳ `docs/reports/midterm_report_outline.md`.

## 4. Sản phẩm bàn giao

| Sản phẩm | File / Bằng chứng | Hạn |
|---|---|---|
| Class mapping + data contract v1 đã phê duyệt | `configs/class_mapping.json`, `configs/data_contract.yaml` | 04/11 |
| Kế hoạch + cấu hình thí nghiệm | `docs/experiment_plan.md`, `configs/experiments/` | 04/11 |
| Sổ thí nghiệm | `docs/experiment_log.md` | 04/11 |
| Biên bản Sprint 2 | `docs/reports/sprint2_review.md` | 04/11 |
| Dàn ý báo cáo giữa kỳ | `docs/reports/midterm_report_outline.md` | 04/11 |

## 5. Tiêu chí nghiệm thu (Definition of Done)

- [ ] Không còn `TBD` trong data contract.
- [ ] E01, E02 có run trên Drive + file `results/metrics/E0x_val.json` đã merge.
- [ ] Kiểm tra end-to-end thành công, ghi lại trong biên bản.
- [ ] Mọi issue Sprint 2 đã đóng hoặc được chuyển Sprint 3 có lý do.

## 6. Ngoài phạm vi

- Trực tiếp xử lý dữ liệu, huấn luyện, đánh giá, demo — các thành viên tương ứng.

## 7. Phụ thuộc

| Hướng | Issue | Nội dung |
|---|---|---|
| Nhận từ | {{S2-DATA}} | Số crop thực tế, class mapping, split manifest |
| Nhận từ | {{S2-CNN}} | Run E01, E02 |
| Nhận từ | {{S2-EVAL}} | Báo cáo Validation |
| Nhận từ | {{S2-DEMO}} | Demo với checkpoint thật |

## 8. Mốc thời gian

| Ngày | Mốc | Người xác nhận |
|---|---|---|
| 04/11 | Sprint 2 Planning (họp 30 phút) | Cả nhóm |
| 04/11 | Khoá class mapping + data contract v1 | Yến |
| 04/11 | Dữ liệu processed sẵn sàng trên Drive | Hùng → Yến |
| 04/11 | Checkpoint giữa sprint | Cả nhóm |
| 04/11 | Hạn mở PR toàn nhóm | Yến |
| 04/11 | Merge xong | Yến |
| 04/11 | Sprint 2 Review | Cả nhóm |

## 9. Quy định review

- PR do Yến tạo: chủ khu vực liên quan review chuyên môn; Yến kiểm tra checklist và merge.
- Tiêu đề `docs(s2-arch): ...`; `Refs`/`Closes` issue này.

## 10. Ghi chú

- **Lịch rút gọn (03/11/2026):** Sprint 1 kết thúc 03/11, Sprint 2 hoàn thành trong ngày 04/11, Sprint 3 chạy 05/11 → 18/11. Việc không hoàn thành đúng hạn → Yến (dự phòng) làm thay và ghi chú tại đây. Phạm vi: [ADR-001](../blob/main/docs/decisions/ADR-001-class-scope.md) — 22 lớp, 128×128, chia theo video 70/15/15, class weights.
- Mọi thay đổi class mapping sau 04/11 phải có ADR mới và thông báo cho cả 4 khu vực.
