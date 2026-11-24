# ADR-002 — Chọn mô hình cuối: E04 (EfficientNet-B0, 18 lớp, dữ liệu `v2_crop224`)

| Mục | Nội dung |
|---|---|
| Trạng thái | **Approved** |
| Ngày | 22/11/2026 (bản đầu của E04 được chốt 21/11/2026) |
| Người quyết định | Phan Hải Yến (trưởng nhóm) |
| Thay thế | [ADR-001](ADR-001-class-scope.md), các mục D1, D3, D5, D6 (chỉ đối với mô hình cuối; E01–E03 vẫn theo ADR-001) |
| Tài liệu liên quan | `README.md` mục 6, `docs/dataset_report.md` mục 8–9, `configs/data_contract_v2.yaml`, `configs/class_mapping_v2.json`, `configs/experiments/E04_efficientnet_b0_data_v2.yaml`, `models/README.md` |
| Checkpoint | Drive `MyDrive/ocean-plastic-ml/runs/E04_20261122-0853/best.pt` (16,4 MB) |

## 1. Bối cảnh

Sau Sprint 3, nhóm có bốn thí nghiệm. E01–E02 (CNN tự thiết kế, Keras) đạt Macro F1 Validation dưới 0,08. E03 (MobileNetV2, `v1_crop128`) đạt 0,203. Phân tích lỗi E03 cho thấy nguyên nhân chính nằm ở dữ liệu: ảnh 128×128 nhỏ hơn kích thước tiền huấn luyện, cách chia theo video làm một số lớp chỉ có 1–4 mẫu ở Val/Test, lớp `trash_unknown_instance` mơ hồ hút nhầm, nhiều lớp quá ít mẫu. Cần chốt một mô hình cuối để đánh giá Test, viết báo cáo và làm demo.

## 2. Các phương án đã xem xét

Mọi số liệu là Macro F1 trên Validation, chọn checkpoint chỉ bằng Validation.

| Phương án | Mô tả | Macro F1 Val | Macro F1 nhóm nhựa Val | Quyết định |
|---|---|---:|---:|---|
| E01 | CNN tự thiết kế, `v1_crop128`, 22 lớp, class weights | 0,076 | 0,000 | Không chọn |
| E02 | E01 + augmentation | 0,065 | 0,017 | Không chọn |
| E03 | MobileNetV2, `v1_crop128`, 22 lớp | 0,203 | 0,123 | Không chọn (mô hình cuối tạm thời từ 13/11) |
| E04 thử nghiệm | MobileNetV2 trên dữ liệu làm lại, giữ 22 lớp | 0,219 | — | Không chọn: cải thiện nhỏ |
| E04 bản đầu | EfficientNet-B0, `v2_crop224`, 18 lớp, lấy mẫu √(1/n) + focal loss | 0,339 | 0,153 | Thay thế ngày 22/11 |
| **E04** | Như bản đầu, lấy mẫu **1/n** | **0,348** | **0,164** | **Chọn** |

## 3. Quyết định

| # | Quyết định |
|---|---|
| D1 | Mô hình cuối là **E04**: EfficientNet-B0 tiền huấn luyện ImageNet (khoảng 4,0 triệu tham số), `Dropout(0.3) → Linear(1280, 18)`, huấn luyện hai pha (đóng băng backbone; mở khoá `features[4:]`). |
| D2 | Dữ liệu `v2_crop224`: **18 lớp** — gộp `trash_unknown_instance`, `trash_cup`, `trash_clothing`, `trash_wreckage` → `trash_other`; `animal_shells`, `animal_etc` → `animal_other`; giữ nguyên 6 lớp liên quan nhựa. |
| D3 | Ảnh 224×224, cửa sổ vuông quanh bbox, lề 15%, bỏ box có cạnh < 16 px (7,1% annotation); chuẩn hoá mean/std ImageNet; augmentation chỉ trên Train. |
| D4 | Chia lại theo video, giữ tỉ lệ 70/15/15 cho từng lớp (tìm kiếm phân công video, 30 lần khởi tạo, seed 42); mọi lớp có ít nhất 10 mẫu ở Val và Test. |
| D5 | Bù mất cân bằng bằng `WeightedRandomSampler` lấy mẫu theo **1/số mẫu** + focal loss (γ = 2), không dùng class weights. |
| D6 | Dừng sớm và chọn checkpoint theo **Macro F1 Validation** (không theo `val_loss`). Checkpoint chọn: pha 2, epoch 4. |
| D7 | Chỉ số báo cáo giữ như ADR-001 D2: Macro F1 tổng thể (chỉ số chính) và Macro F1 nhóm 6 lớp liên quan nhựa. |

## 4. Kết quả trên Test

| Chỉ số | E04 bản đầu (Test lần 1, 21/11) | **E04 (Test lần 2, 22/11)** |
|---|---:|---:|
| Macro F1 (18 lớp) | 0,327 | **0,329** |
| Accuracy | 0,535 | 0,507 |
| Weighted F1 | 0,497 | 0,486 |
| Macro F1 nhóm nhựa | 0,119 | **0,132** |
| Recall nhóm nhựa | 0,099 | **0,151** |

Tập Test có 1.751 crop. Suy luận: khoảng 59 ms/ảnh trên CPU (1 luồng), 8 ms/ảnh trên GPU T4.

## 5. Ngoại lệ với ADR-001 D7 (Test chỉ chạy một lần)

Test đã được chạy **hai lần**. Lần 1 cho E04 bản đầu (21/11). Phân tích lỗi của lần đó (nhóm nhựa bị hút vào `trash_other`, recall nhựa 0,10) dẫn đến đổi cách lấy mẫu; E04 hiện tại được chọn bằng Validation rồi mới chạy Test lần 2. Vì quyết định thay đổi có tham khảo kết quả Test lần 1, kết quả Test lần 2 không hoàn toàn độc lập. Hai kết quả đều được giữ trong repo (`results/metrics/E04v1_test.json`, `E04_test.json`) và README ghi rõ điều này. Không chạy thêm lần Test nào cho dự án này.

## 6. Hệ quả

- Mô hình cuối không so sánh trực tiếp được với E01–E03 (khác số lớp và khác cách chia); README ghi chú điều này.
- Hạn chế còn lại: `trash_bottle`, `trash_snack_wrapper` F1 = 0 trên Test; nhiều mẫu nhựa vẫn bị đoán thành `trash_other`; mẫu nhựa tập trung trong vài video nên số đo nhóm nhựa không ổn định (`docs/dataset_report.md` mục 9).
- Demo (`main.py`) dùng checkpoint E04 với `class_mapping.json` nằm cạnh checkpoint.
- Hướng phát triển (ngoài phạm vi môn học): bỏ `trash_unknown_instance` khỏi huấn luyện hoặc phân loại hai tầng; thêm dữ liệu nhựa từ video khác; chuyển sang object detection cho robot.
- Repo được gắn tag `v1.0-final` tại commit `ede61ab` chứa mô hình và kết quả này. Các commit sau tag chỉ sửa tài liệu và lỗi đọc file UTF-8 trên Windows, không đổi kết quả.
