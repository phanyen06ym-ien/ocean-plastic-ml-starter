# Thu thập bằng chứng và review ba điều kiện của v1_crop128

Quyết định hiện tại: **KEEP_V1_WITH_CONDITIONS**. Theo kết quả Colab người dùng
xác nhận: `automated_checks_passed=True`, `errors=0`, pytest **93 passed** trước
đợt bổ sung công cụ này. Đây không phải kết quả chạy mới của các công cụ dưới đây.
Ba điều kiện vẫn **UNVERIFIED**; chưa có metadata nguồn, kết quả quét ảnh thật
hoặc phiếu review hoàn tất trong checkout. Không xác định hay suy đoán hai ID thiếu.

| Điều kiện | Trạng thái hiện tại | Bằng chứng còn thiếu |
|---|---|---|
| 310/312 video | UNVERIFIED | Hai COCO nguồn đầy đủ, tập ID đối chiếu, xác minh nguồn gốc số 312 và giải thích chênh lệch có căn cứ |
| Near-duplicate xuyên split | UNVERIFIED | Scan ảnh thật, phương pháp/ngưỡng, mọi cặp nghi vấn được review cùng bối cảnh nguồn |
| Nhãn lớp ít mẫu | UNVERIFIED | Review đủ phạm vi ba lớp, ảnh/annotation gốc, reviewer/quyết định/bằng chứng/ghi chú |

Các lệnh chỉ đọc dataset, metadata, mapping; chỉ ghi bằng chứng vào thư mục mới
ngoài dataset. Không đổi split, nhãn, mapping, checkpoint E03; không huấn luyện.
Đọc Test chỉ để kiểm tra dữ liệu. Không chọn ngưỡng theo metric mô hình trên Test.

## Chạy từ trên xuống trên Colab

Mở **`notebooks/05_data_audit_review.ipynb`** từ nhánh
`fix/dataset-audit-mobilenetv2` của Draft PR #35, chọn CPU và chạy lần lượt các cell.
Notebook này độc lập với các notebook huấn luyện 00/03 và không gọi huấn luyện.

1. Mount Drive, clone/cập nhật đúng nhánh, in SHA, cài `requirements-mobilenet.txt`.
   Nếu vừa đổi code ở runtime đã import module cũ, restart runtime rồi chạy lại.
2. Khôi phục ZIP đã đóng băng từ
   `MyDrive/ocean-plastic-ml/data/processed/v1_crop128.zip` vào
   `/content/processed/v1_crop128`. ZIP phải chứa thư mục `v1_crop128/`.
   Thiếu ZIP thì khôi phục bản lưu; không chạy preprocessing để tạo lại v1.
3. Nếu có metadata nguồn, đặt **cả** `SOURCE_TRAIN_COCO` và `SOURCE_VAL_COCO`
   thành đường dẫn chính xác tới COCO Instance train/val dùng cho manifest v1.
   Giữ `None` khi chưa có. Không dùng COCO Material hoặc chọn file bằng suy đoán.
4. Chạy toàn bộ pytest, thu thập bằng chứng, kiểm chứng các artifact và xem contact
   sheet. Cell kiểm chứng yêu cầu kiểm tra tự động đạt; ba điều kiện vẫn UNVERIFIED.
5. Cell tổng hợp cuối chạy được ngay cả khi chưa có quyết định review. Sau khi
   reviewer hoàn tất, chạy lại cell cuối để xuất báo cáo mới, giữ bản ban đầu.

Mỗi lần thu thập dùng thư mục Drive mới:

```text
MyDrive/ocean-plastic-ml/runs/data_audit_v1_crop128/review/<UTC timestamp>/
```

Notebook không upload GitHub và không sửa kết quả Colab 93 tests của lần trước.
Suite mới có thêm tests; lưu log lần chạy mới cùng SHA để phân biệt.

## CLI tương đương

Công cụ chỉ cần Pillow và PyYAML để thu thập; môi trường đầy đủ cho suite dùng
`requirements-mobilenet.txt`. Không cần TensorFlow để audit; module test yêu cầu
TensorFlow có thể bị skip nếu môi trường thiếu thư viện này.

```bash
python -m pip install 'pillow>=10.0' 'pyyaml>=6.0'
python -m src.audit_review collect \
  --processed-dir /content/processed/v1_crop128 \
  --output-dir /content/drive/MyDrive/ocean-plastic-ml/runs/data_audit_v1_crop128/review/RUN_UTC \
  --hash-size 8 --threshold 6 --max-pair-images 50 --page-size 24
```

Thêm hai cờ sau khi có metadata nguồn. Đường dẫn do reviewer xác định từ bộ
nguồn đã dùng; không có đường dẫn mặc định để tránh chọn nhầm bản export:

```text
--source-train-coco /path/to/instances_train_trashcan.json
--source-val-coco /path/to/instances_val_trashcan.json
```

`--preprocessing-summary` mặc định lấy file trong dataset, không lấy summary của
một lần preprocessing khác. `--expected-source-videos` mặc định **312**, là số
được báo cáo trước đây cần đối chiếu, không phải số đã được công cụ xác nhận.
Không chỉnh cờ này để làm đẹp kết quả nghiệm thu dữ liệu thật.

Thư mục output phải chưa tồn tại. Công cụ không ghi đè evidence hay công việc
review trước đó; nếu chạy lỗi, đọc lỗi và dùng tên thư mục mới khi chạy lại.
Exit `0`: hoàn tất, có thể vẫn CONDITIONAL; `1`: FAIL/DO_NOT_CONTINUE;
`2`: đầu vào/CLI không hợp lệ hoặc scan không hoàn tất. Không coi exit 0 là PASS.

## Bằng chứng 310/312 video

`video_reconciliation.json/.csv` chứa tập ID và số lượng của bốn phạm vi:

- Toàn bộ `images` trong hai COCO nguồn, kể cả ảnh không có annotation.
- Video của ảnh có annotation trong metadata nguồn.
- Video trong mọi dòng manifest, kể cả crop bị loại.
- Video có ít nhất một crop `ok` trong manifest.

Join annotation bằng UID `source_split_ann_id`, image bằng
`source_split_image_id`, đối chiếu category, filename và video. Regex video đúng
với preprocessing v1: `vid_(\d+)` từ basename; fallback `img_*` không được xem
là video đã xác minh. Kiểm tra tập UID annotation, các tổng số summary,
crop status và số crop mỗi split. Báo lỗi đối chiếu và bằng chứng thiếu riêng.

Chỉ ghi `source_without_annotations` khi có đủ hai metadata; thiếu nguồn thì
giá trị là `null`, không bịa hai ID. `no_annotations_in_supplied_metadata` là sự
kiện quan sát trong metadata đã cung cấp. Nó không chứng minh bộ file raw đầy đủ
hoặc lý do nghiệp vụ của việc loại video. Công cụ không kiểm kê raw disk và không
thể phát hiện video không xuất hiện trong bất kỳ COCO nào. Reviewer phải kiểm tra
nguồn gốc, tính đầy đủ của export và bằng chứng ban đầu cho số 312; nếu số 312
đến từ raw inventory ngoài COCO, đính kèm inventory đó và kiểm tra thủ công.
Không nghiệm thu chỉ vì hai tổng số trùng kỳ vọng.

## Ảnh gần trùng xuyên split

`image_hashes.csv` ghi UID, split, lớp, video, image ID, đường dẫn, SHA-256 byte
ảnh và dHash. Thuật toán **horizontal difference hash (dHash)**: grayscale,
resize LANCZOS về `(hash_size+1, hash_size)`, bit `left > right`, row-major.
Mặc định 8×8 = 64 bit. So sánh Hamming **toàn bộ** train–val, train–test, val–test;
không lấy mẫu, không lọc theo nhãn. Ngưỡng **bao gồm** (`distance <= threshold`),
cấu hình 0..số bit. `--hash-size` cho phép 2..16. Đây là perceptual hashing,
không phải pHash DCT; không so trực tiếp ngưỡng giữa các thuật toán/kích thước hash.

`near_duplicate_candidates.csv` chứa mọi cặp nghi vấn, ID cặp ổn định, hai UID,
split/lớp/video/path/hash/distance. `near_duplicate_scan.json` ghi số ảnh mỗi split,
số cặp so sánh/ứng viên mỗi cặp split và số ảnh liên quan. Quét đủ không có nghĩa
không thể bỏ lọt ảnh gần trùng: dHash có thể bỏ sót xoay/crop, va chạm ở ảnh ít
texture. Đặc biệt, distance 0 cũng không chứng minh ảnh trùng pixel hay leakage.

`pair_examples/*.png` minh họa tối đa `--max-pair-images` cặp theo thứ tự ổn định;
giới hạn này **chỉ** giới hạn PNG, không giới hạn CSV/quét. Đặt `0` để tắt PNG.
Số minh họa bị lược được ghi rõ. Reviewer mở ảnh theo CSV cho mọi cặp chưa có PNG,
đối chiếu ảnh gốc, chuỗi frame/video và bối cảnh. Không tự xóa/deduplicate/split lại.

## Chất lượng nhãn lớp ít mẫu

`rare_class_review.csv` và `label_contact_sheets/*.png` bao phủ **toàn bộ crop
hợp lệ Train/Val/Test** của `trash_cup`, `trash_snack_wrapper`, `trash_bottle`.
Train được xếp trước, nhưng cần review đủ phạm vi đã xuất để nghiệm thu. Mỗi trang
có tối đa `--page-size` crop (mặc định 24, giới hạn 1..100), tối đa bốn cột; CSV
ghi trang và tile theo thứ tự từ trái sang phải, trên xuống dưới.

CSV có UID, split, lớp/ID, video/image ID, đường dẫn crop, source split, annotation
ID, filename/image_path và bbox nếu các trường này có trong manifest. `image_path`
có thể là đường dẫn từ runtime preprocessing cũ; dùng filename/source split/ann ID
để tìm ảnh gốc thay vì coi đường dẫn cũ là bằng chứng đã kiểm tra.

Phiếu được tạo với **decision trống**. Reviewer phải xem crop **và** ảnh/annotation
gốc, ghi bằng chứng cụ thể. Thiếu ảnh/annotation hoặc không rõ nhãn: UNCERTAIN,
không đánh PASS. Không sửa nhãn hay ảnh dataset trong đợt audit này.

## Ghi review và xuất báo cáo tổng hợp

Giữ nguyên artifact gốc. Sao chép hai CSV thành
`near_duplicate_candidates_reviewed.csv`, `rare_class_review_reviewed.csv` rồi điền
bốn cột `decision`, `reviewer`, `evidence`, `notes` cho **mọi** dòng:

| Giá trị | Review cặp ảnh | Review nhãn |
|---|---|---|
| PASS | Đã xem bối cảnh, không xác nhận leakage cho cặp | Nhãn/bbox được bằng chứng nguồn hỗ trợ |
| FAIL | Review xác nhận rủi ro cần chặn | Nhãn/bbox sai cần xử lý ngoài tác vụ này |
| UNCERTAIN | Thiếu bằng chứng hoặc chưa kết luận | Thiếu bằng chứng hoặc nhãn không rõ |

Không xóa/thêm dòng, không đổi UID hay metadata. Evidence nên trỏ file nguồn,
annotation ID, ảnh minh họa hoặc biên bản có thể truy xuất; notes giải thích quyết
định. Công cụ kiểm tra đủ dòng, metadata bất biến, quyết định hợp lệ, reviewer và
các trường bằng chứng/ghi chú không trống; không tự xác minh nội dung lời xác nhận.

Sao chép `condition_decisions.template.json` thành `condition_decisions.json`.
Giữ `run_id`, điền `decision`, `reviewer`, `evidence`, `notes` cho từng điều kiện.
Ngay cả khi không có cặp nghi vấn, điều kiện hash vẫn cần người xác nhận phạm vi,
ngưỡng, hạn chế và rủi ro còn lại. Đối chiếu video cần ready_for_human_review=True
và xác nhận của reviewer, không chỉ đếm được 310/312.

```bash
python -m src.audit_review summarize \
  --evidence-dir /path/to/RUN_UTC \
  --decisions /path/to/RUN_UTC/condition_decisions.json \
  --pair-review /path/to/RUN_UTC/near_duplicate_candidates_reviewed.csv \
  --label-review /path/to/RUN_UTC/rare_class_review_reviewed.csv \
  --output /path/to/RUN_UTC/audit_review_summary_reviewed_UTC.json
```

Omit các cờ review chưa có để báo UNVERIFIED. File output mới phải chưa tồn tại.
Không ghi vào dataset hoặc đè input. `evidence_index.json` ghi SHA artifact,
manifest/mapping, code commit và môi trường; báo cáo chứa provenance và hash các
file reviewer đã dùng. Thay đổi artifact gốc bị từ chối; phải thu thập run mới.
Quyết định của run khác bị từ chối. Không dùng file index như chữ ký bảo mật.

`audit_review_summary.json` là báo cáo ban đầu, chứa trạng thái riêng ba điều kiện
và bằng chứng; báo cáo `*_reviewed_*.json` chứa lần tổng hợp sau review.
Chỉ khi kiểm tra nền đạt, đối chiếu video không còn thiếu/mâu thuẫn, mọi dòng
review PASS và ba xác nhận có đủ bằng chứng thì mới có PASS / KEEP_V1. FAIL được
ghi riêng và chặn bằng DO_NOT_CONTINUE. Các trường hợp còn lại giữ
KEEP_V1_WITH_CONDITIONS / CONDITIONAL. Không dùng kết quả unit tests làm bằng
chứng nghiệm thu ảnh thật. Bằng chứng đủ hay chưa vẫn cần reviewer đánh giá.

## Kiểm thử

```bash
python -m pytest -q tests/test_audit_review.py
python -m pytest -q
```

Fixture là ảnh tổng hợp; kiểm tra đối chiếu ID có/không có nguồn, mâu thuẫn summary,
hash/ngưỡng/ba cặp split, ảnh minh họa và phạm vi contact sheets, bất biến đầu vào,
chặn thiếu hoặc thay đổi phiếu review, run_id sai và các cổng quyết định. Các cell
thu thập/kiểm chứng/tổng hợp notebook được thực thi với fixture, không mount Drive.
Chưa thay thế việc chạy lại notebook trên ảnh thật của nhóm trên Colab.

Kiểm chứng local ngày 12/11/2026: **128 passed, 1 skipped**, toàn bộ suite trong
**176,64 giây**. Skip module TensorFlow vì môi trường local chưa cài TensorFlow.
Các cell cấu hình dữ liệu, thu thập/kiểm chứng và tổng hợp của notebook 05 được
thực thi theo thứ tự trên fixture. Đã xem contact sheet tổng hợp, kiểm tra CLI
và diff. Chưa chạy các công cụ mới trên ảnh thật/Drive; ba điều kiện vẫn UNVERIFIED.
