# Data Audit trước MobileNetV2 — v1_crop128

**Quyết định: `KEEP_V1_WITH_CONDITIONS`. Trạng thái: `CONDITIONAL`, chưa PASS hoàn toàn.**

Phạm vi: TrashCan Instance, 22 lớp; kiểm tra tính toàn vẹn của dataset đã đóng băng.
Giữ nguyên Train/Validation/Test, `configs/class_mapping.json`, cấu hình và checkpoint E03.
Không chia lại, deduplicate, sửa nhãn, crop lại hay huấn luyện lại trong bước audit này.
Liên quan #16 (S3-DATA), #17 (S3-CNN).

## Kết quả đã xác nhận trên Colab

Nguồn: kết quả Data Audit do người dùng xác nhận và cung cấp trong yêu cầu này.
Số crop được đối chiếu với `results/metrics/preprocessing_summary.json` và
`results/metrics/dataset_split_counts.csv` đã có trong repository.
Các kết quả dưới đây không phải số đo mới trên máy phát triển. Checkout này không
có ảnh dataset thật; chưa có log Colab gốc, danh sách nhóm trùng và hash của lần
audit đó để gắn vào báo cáo. Không suy diễn những bằng chứng còn thiếu.

| Hạng mục | Kết quả Colab đã xác nhận |
|---|---:|
| Train / Validation / Test | 8.283 / 1.782 / 2.009 |
| Tổng ảnh crop | 12.074 |
| Lớp / class_index | 22 / 0–21 |
| Ảnh lỗi | 0 |
| Nhóm ảnh trùng pixel chính xác | 3 |
| Nhãn và split trong từng nhóm trùng | Cùng nhãn, cùng split |
| Nhóm trùng pixel chính xác giữa split | 0 |
| Mất cân bằng **Train** | 2.130 / 14 = 152,14 lần |

Ba nhóm trùng được giữ nguyên để bảo toàn bản v1 và khả năng so sánh E01/E02/E03.
Ba **nhóm** không đồng nghĩa ba ảnh; không có danh sách chi tiết để khẳng định
số ảnh trùng. Không phát hiện trùng pixel chính xác không loại trừ ảnh gần trùng.

| Lớp ít mẫu | Ảnh Train | Video Train |
|---|---:|---:|
| `trash_cup` | 14 | 1 |
| `trash_snack_wrapper` | 41 | 3 |
| `trash_bottle` | 95 | 4 |

Các crop trong cùng video có tương quan; số ảnh không thể thay thế số video độc lập.
Class weights hiện có không bổ sung sự đa dạng hoặc xác minh tính đúng của nhãn.
Báo cáo kết quả theo lớp phải kèm support và tránh kết luận mạnh ở lớp ít mẫu.

## Ba điều kiện chưa nghiệm thu

| Điều kiện | Trạng thái | Bằng chứng cần có để nghiệm thu |
|---|---|---|
| 310/312 video | **UNVERIFIED** | Đối chiếu danh sách ID video từ toàn bộ ảnh gốc, ảnh có annotation và crop hợp lệ; chỉ rõ hai ID chênh lệch cùng lý do, không chỉ so sánh tổng số. |
| Ảnh gần trùng giữa split | **UNVERIFIED** | Chạy tìm ứng viên gần trùng có ghi phương pháp/ngưỡng (ví dụ perceptual hash), lưu cặp ảnh và split; người review xác nhận ứng viên, tổng hợp false positive và rủi ro còn lại. |
| Nhãn các lớp ít mẫu | **UNVERIFIED** | Review ảnh và annotation nguồn của `trash_cup`, `trash_snack_wrapper`, `trash_bottle`, ưu tiên toàn bộ Train; ghi UID, video, người review, nhãn nghi vấn và quyết định. Không tự sửa nhãn gốc. |

`preprocessing_summary.json` ghi 310 video trong 7.083 ảnh nguồn có annotation;
audit trước đây ghi 312 video. Việc ảnh không có annotation có thể giải thích
chênh lệch là **giả thuyết**, chưa phải kết luận đã được xác minh.
Script mới ghi cả số video trong manifest và số video có crop hợp lệ nhưng không
đọc toàn bộ ảnh gốc, nên không tự giải quyết chênh lệch này.

Không đổi các điều kiện thành đạt từ một lần chạy unit test, một lần hash pixel
hay một phép so sánh tổng số. Chỉ tiếp tục thí nghiệm trên v1 với các hạn chế
được ghi rõ; chưa đủ cơ sở chứng nhận chất lượng dataset hoặc đóng toàn bộ #16.

## Chạy lại audit mà không sửa dataset

Checkout nhánh PR, dùng bản `v1_crop128` đã giải nén có `split_manifest.csv` và
`class_mapping.json`. Không chạy lại `src.preprocessing` để tái tạo split.
Audit chỉ cần Pillow và PyYAML; không cần GPU, TensorFlow hoặc PyTorch.

```bash
python -m pip install "pillow>=10.0" "pyyaml>=6.0"
python -m src.data_audit \
  --processed-dir /content/processed/v1_crop128 \
  --output-dir /content/drive/MyDrive/ocean-plastic-ml/runs/data_audit_v1_crop128
```

`--mapping` mặc định là mapping chuẩn trong repository. Có thể chỉ định
`--manifest` nếu manifest lưu riêng. Mapping được so với bản chuẩn, bao gồm
`class_index`, `category_id`, tên lớp và cờ nhựa. Mapping đi kèm dataset cũng
phải khớp. Script không chấp nhận thư mục output nằm trong dataset hoặc symlink
output trỏ vào dataset.

Output chỉ gồm `audit_summary.json` và `audit_class_counts.csv`, lưu ngoài dữ
liệu gốc. JSON có hash SHA-256 của manifest/mapping, lỗi chi tiết, nhóm ảnh trùng
và các điều kiện còn mở. CSV có đủ 22 lớp theo class_index, số ảnh từng split và
số video Train. Khi chạy thật, lưu thêm commit code và môi trường cùng hai file
này trên Drive để đối chiếu lần chạy. Không commit ảnh, ZIP, manifest gốc hay checkpoint.

Script thực hiện:

- Kiểm tra thư mục đủ 22 lớp ở mỗi split, số ảnh đúng 8.283 / 1.782 / 2.009,
  decode ảnh, kích thước 128×128, mode RGB; phát hiện lớp rỗng.
- Đối chiếu manifest với file trên đĩa: nhãn/ID, đường dẫn, UID, file thiếu,
  file dư và đường dẫn trùng; không coi thiếu metadata là không có rò rỉ.
- Kiểm tra video và ảnh nguồn xuất hiện giữa split bằng manifest (kể cả các
  annotation không tạo crop); đánh dấu video ID fallback `img_*` chưa xác minh.
- Hash **pixel RGB đã decode kèm kích thước**, không hash byte JPEG để tìm ảnh
  giống nhau. Cùng nhãn/cùng split được ghi nhận; trùng giữa split hoặc xung đột
  nhãn là lỗi chặn tiếp tục. Không xoá file nào.
- Tính mất cân bằng từ **Train**; nếu lớp Train rỗng, ratio là `null` và báo lỗi.

Exit code `0`: kiểm tra tự động đạt, **vẫn CONDITIONAL** với ba mục UNVERIFIED.
Exit code `1`: phát hiện lỗi toàn vẹn, quyết định `DO_NOT_CONTINUE` và đã ghi
báo cáo. Exit code `2`: đầu vào/CLI không hợp lệ, chưa thể hoàn thành audit.
Script không tự cập nhật báo cáo Colab ở phần trên và không tự cấp trạng thái PASS.

Đọc Test trong audit chỉ để kiểm tra toàn vẹn/leakage. Không tính metric mô hình,
không chọn checkpoint, không chọn hyperparameter trên Test.

## E01/E02 và MobileNetV2

Ở `main` nền của PR (`4ce3978`), import thực thi trong notebook 00 và 03 đã
được chuyển sang `src.train_keras`. Hướng dẫn đầu notebook 03 và test cấu hình
E01/E02 vẫn trỏ sang `src.train`; PR sửa hai điểm này, ghi rõ module của hai
backend và thêm test thực thi import lấy trực tiếp từ cell notebook. Notebook
00/03 có thêm bước audit trước huấn luyện. Trong thời gian PR chưa merge, đặt
`BRANCH = "fix/dataset-audit-mobilenetv2"` ở cell setup để clone đúng code.
Nếu runtime đang giữ module cũ sau khi đổi checkout, restart runtime rồi chạy
lại từ đầu; không chép các hàm Keras vào `src.train.py`.

E01/E02 dùng pixel RGB 0..255, lớp Rescaling /255 bên trong mô hình Keras.
E03 dùng torchvision MobileNetV2: `ToTensor` đưa pixel về 0..1 rồi Normalize
với mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`, theo
[tài liệu torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html).
Giữ crop 128×128 của E03 để so sánh baseline; không áp dụng resize/center crop
224 của preset suy luận ImageNet và không đổi kích thước v1.

Pipeline chặn config lệch 22 lớp, thứ tự tên lớp, data version, kích thước/mode
đầu vào, cờ dùng Test và mean/std. Mapping dataset phải khớp ID/tên/cờ của
mapping chuẩn; ImageFolder chặn lớp thiếu/dư và gán ID theo mapping, không theo
alphabet. Head vẫn có 22 logits. Không đổi kiến trúc, trọng số/checkpoint E03,
learning rate, số epoch hoặc các YAML thí nghiệm.

Cài `requirements-mobilenet.txt` để kiểm tra pipeline PyTorch độc lập với
TensorFlow. `make_loaders` chỉ mở Train và Validation.

## Kiểm thử và giới hạn kiểm chứng

```bash
python -m pytest -q tests/test_data_audit.py tests/test_mobilenet_pipeline.py tests/test_notebook_imports.py
python -m pytest -q
```

Các fixture là ảnh tổng hợp, không phải TrashCan. Kiểm thử bao phủ audit lặp
lại không đổi byte/mtime của đầu vào, nhóm trùng pixel có encoding khác nhau,
trùng giữa split, xung đột nhãn, group leakage, ảnh lỗi, kích thước/mode sai,
manifest sai/thiếu, mapping đổi, output ngoài dataset, CLI lỗi, chuẩn hoá tensor
thực tế, head 22 đầu ra và nhãn ImageFolder. Import notebook được thực thi
riêng; không mount Drive hoặc huấn luyện trong test.

Kiểm chứng local ngày 12/11/2026: **85 passed, 1 skipped** (toàn bộ suite).
Môi trường Python 3.11, pytest 9.2.0, torch 2.12.1+cpu, torchvision 0.27.1+cpu;
module kiểm thử TensorFlow được skip vì chưa cài TensorFlow. Lần chạy cuối dùng
thư mục tạm trên ổ D: vì ổ C: hết dung lượng; lỗi ghi file tạm ở lần chạy trước
không được tính là kết quả đạt.

Chưa chạy lại toàn bộ notebook trên Colab hoặc audit trên ảnh thật ở checkout
này. Các kiểm tra chưa chạy được cũng được ghi trong mô tả Draft PR.
