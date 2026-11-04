# Visual Inspection Protocol — TrashCan 1.0

**Project:** Ocean Plastic Waste Classification and Monitoring  
**Document:** `docs/visual_inspection_protocol.md`  
**Issue:** S1-ARCH #4  
**Version:** 1.1  
**Owner:** Phan Hải Yến  
**Dataset:** TrashCan 1.0 — Instance  
**Scope alignment:** `docs/class_scope_analysis.md` v1.2 — Option D (proposed)  
**Status:** PROPOSED — READY FOR TEAM REVIEW (NOT APPROVED)

---

## 1. Mục tiêu

Quy trình này quy định cách kiểm tra trực quan ảnh đối tượng dưới nước nhằm:

1. Đánh giá sự phù hợp giữa nhãn annotation và đối tượng trong ảnh.
2. Xác định chất lượng ảnh có đủ để phục vụ bài toán CNN Image Classification.
3. Phát hiện các trường hợp ảnh mờ, đối tượng khó nhìn, crop quá nhỏ hoặc bị che khuất.
4. Đánh giá mức độ liên quan của các lớp ứng viên với mục tiêu nghiên cứu rác thải nhựa.
5. Thu thập bằng chứng để hỗ trợ quyết định phạm vi lớp CNN.
6. Ghi nhận các giới hạn của dữ liệu crop trước khi huấn luyện baseline 22 lớp.

Visual Inspection là bước kiểm tra chất lượng và phạm vi dữ liệu, **không phải phương pháp xác minh thành phần vật liệu bằng phân tích hóa học hoặc cảm biến chuyên dụng**.

## 2. Phạm vi

Kiểm tra sáu lớp ứng viên:

| Class | Train annotations | Val annotations | Target sample |
|---|---:|---:|---:|
| `trash_bag` | 786 | 122 | 20 |
| `trash_bottle` | 119 | 7 | 20 |
| `trash_snack_wrapper` | 71 | 13 | 20 |
| `trash_net` | 115 | 12 | 20 |
| `trash_rope` | 117 | 0 | 20 |
| `trash_tarp` | 102 | 19 | 20 |
| **Total** | **1,310** | **173** | **120** |

Số liệu được lấy từ `grouped_class_distribution.csv`.

120 là số mẫu mục tiêu, không phải số mẫu đã được kiểm tra.

Đây là **kiểm tra chuyên sâu có mục đích** cho sáu lớp ứng viên, **không phải mẫu đại diện thống kê cho toàn bộ 22 lớp**. Theo Option D đang được đề xuất, CNN baseline vẫn có **22 đầu ra**; sáu lớp trên chỉ là nhóm phân tích chuyên sâu, không phải tập nhãn huấn luyện độc lập. Kết quả kiểm tra không đủ để suy luận tỷ lệ lỗi nhãn/chất lượng ảnh của 16 lớp còn lại.

**Phạm vi mở rộng (chưa phê duyệt):** Nếu nhóm cần kết luận về chất lượng toàn bộ 22 lớp, phải lập một kế hoạch lấy mẫu phân tầng bổ sung cho 16 lớp còn lại, ghi rõ cỡ mẫu, mục đích, nguồn lực và người phê duyệt **trước khi xem ảnh**. Không gộp kết quả hai thiết kế lấy mẫu thành một tỷ lệ chung nếu không có phương pháp tổng hợp phù hợp.

## 3. Nguồn dữ liệu

Sử dụng:

- `instances_train_grouped_clean.json`
- `instances_val_grouped_clean.json`
- `train_grouped_manifest.csv`
- `val_grouped_manifest.csv`
- `preprocessed_crops/`
- `preprocessed_resized/`
- `grouped_class_distribution.csv`
- `dataset_report.md`

**Nguyên tắc nguồn dữ liệu:**

- Annotation JSON là nguồn xác định ID và nhãn.
- Ảnh crop là nguồn kiểm tra trực quan chính.
- Ảnh gốc và bounding box được dùng để đối chiếu khi crop không rõ.
- Ảnh resize chỉ dùng để kiểm tra khả năng quan sát sau tiền xử lý.
- Không thay đổi annotation, split hoặc ảnh gốc trong quá trình kiểm tra.

## 4. Thiết kế lấy mẫu

### 4.1. Đơn vị lấy mẫu

Mỗi mẫu tương ứng với một annotation ID và vùng ảnh được crop theo bounding box.

Không lấy mẫu theo số lượng ảnh nguồn vì một ảnh có thể chứa nhiều đối tượng.

### 4.2. Phương pháp lấy mẫu

Áp dụng lấy mẫu ngẫu nhiên phân tầng theo `class_id` và `split`, với seed cố định `42`. **Đơn vị chọn là annotation**, không phải ảnh nguồn hay nhóm video; nhiều annotation có thể cùng một `image_id` hoặc video, nên 120 mẫu không có nghĩa là 120 ảnh/video độc lập. Các tỷ lệ mô tả mẫu không được diễn giải là ước lượng độc lập cho toàn dataset.

**Quy ước tái lập bắt buộc (đề xuất):** dùng Python 3, `random.Random(42)` của thư viện chuẩn; đọc annotation từ JSON nguồn đã khóa phiên bản; với từng cặp `(class_id, split)` theo thứ tự `class_id` tăng dần, `train` trước `val`, sắp xếp danh sách ứng viên theo `annotation_id` tăng dần (so sánh kiểu số nếu ID là số), rồi gọi `rng.sample(candidate_ids, k)`; ghi kết quả đã chọn theo thứ tự lấy mẫu và xuất manifest. Không phụ thuộc thứ tự annotation trong JSON hoặc thứ tự liệt kê tệp. **Phải ghi phiên bản Python, phiên bản script, hash SHA-256 của JSON và manifest** để tái lập; nếu đổi thuật toán hoặc thứ tự duyệt phải tăng phiên bản protocol.

Mỗi lớp lấy tối đa 20 annotation, theo quy tắc:

- Nếu cả Train và Validation đều có ít nhất 5 annotation: lấy 15 Train và 5 Validation.
- Nếu Validation có từ 1 đến 4 annotation: lấy tối đa số mẫu Validation hiện có và bổ sung Train để đạt 20.
- Nếu Validation không có annotation: lấy 20 Train.
- Nếu một split không đủ số mẫu theo chỉ tiêu, bổ sung từ split còn lại.
- Không lấy trùng annotation ID.
- Không chuyển annotation giữa các split.

Với dữ liệu hiện tại, kế hoạch là:

| Class | Train sample | Val sample | Total |
|---|---:|---:|---:|
| `trash_bag` | 15 | 5 | 20 |
| `trash_bottle` | 15 | 5 | 20 |
| `trash_snack_wrapper` | 15 | 5 | 20 |
| `trash_net` | 15 | 5 | 20 |
| `trash_rope` | 20 | 0 | 20 |
| `trash_tarp` | 15 | 5 | 20 |
| **Total** | **95** | **25** | **120** |

Đây là mẫu kiểm tra chất lượng dữ liệu, không phải tập Test độc lập dùng để đánh giá hiệu năng CNN.

### 4.3. Pilot trước khi đánh giá chính thức

Trước khi mở 120 mẫu chính thức, hai reviewer thực hiện một **pilot 10 annotation** từ danh sách riêng được chọn theo quy tắc ghi nhận trước (ưu tiên bao phủ nhiều lớp và các tình huống khó). Pilot dùng để kiểm tra khả năng hiểu các mức `Yes/No/Uncertain`, `Good/Fair/Poor`, quy trình mở crop/ảnh gốc và cách ghi bằng chứng. Ghi lại điểm chưa rõ, điều chỉnh rubric nếu cần và khóa phiên bản rubric trước lượt đánh giá chính thức.

Pilot **không được tự động cộng vào 120 mẫu** hoặc dùng để thay thế mẫu chính thức. Nếu pilot trùng annotation trong danh sách chính thức, cần đánh dấu trùng và xác định quy tắc xử lý trước khi review chính thức; ưu tiên chọn pilot từ phần còn lại của dữ liệu. Không báo cáo kết quả pilot như kết quả chính thức.

### 4.4. Khả năng tái lập

Khi xuất danh sách mẫu, phải ghi lại:

- Phiên bản hoặc checksum của JSON nguồn.
- Random seed.
- Thuật toán lấy mẫu.
- Annotation ID và Image ID.
- Class ID và Class Name.
- Split.
- Đường dẫn ảnh crop.
- Thời điểm xuất mẫu.
- Python version, script commit/hash và PRNG được dùng.
- Thứ tự duyệt strata, cách sort ID và số mẫu chọn mỗi stratum.
- SHA-256 của sample manifest sau xuất.

Không thay thế mẫu sau khi xem kết quả. Nếu ảnh không truy xuất được, giữ bản ghi và đánh dấu `INACCESSIBLE`; không điền `No`/`Poor` thay cho kết quả chưa quan sát. Chỉ tạo danh sách bổ sung theo một quy tắc đã phê duyệt, có manifest và provenance riêng; báo cáo tách biệt số mẫu chọn, số truy xuất được và số đã đánh giá.

**Kiểm tra manifest trước khi review:** đủ 120 annotation theo chỉ tiêu nếu nguồn đáp ứng; không trùng `annotation_id`; đúng nhãn và split theo JSON; mỗi đường dẫn crop gắn với annotation tương ứng; kiểm tra ảnh nguồn/bbox nếu crop không khớp; ghi số `image_id` và video ID duy nhất thực tế (không giả định trước).

## 5. Tiêu chí kiểm tra trực quan

### C1 — Object Visibility

| Giá trị | Định nghĩa |
|---|---|
| `Yes` | Có thể nhận thấy đối tượng chính trong ảnh crop |
| `No` | Không quan sát được đối tượng chính |
| `Uncertain` | Đối tượng bị che khuất hoặc quá khó nhận biết |

### C2 — Label Consistency

| Giá trị | Định nghĩa |
|---|---|
| `Yes` | Đối tượng quan sát được phù hợp với nhãn annotation |
| `No` | Có bằng chứng rõ ràng cho thấy nhãn không phù hợp |
| `Uncertain` | Không đủ thông tin để xác nhận hoặc bác bỏ nhãn |

Người đánh giá không tự sửa class ID hoặc class name trong dataset.

### C3 — Image Quality

| Giá trị | Định nghĩa |
|---|---|
| `Good` | Đối tượng tương đối rõ, đủ chi tiết để nhận biết |
| `Fair` | Có suy giảm chất lượng nhưng vẫn quan sát được đặc trưng chính |
| `Poor` | Mờ, nhiễu, quá nhỏ, bị che khuất hoặc thiếu chi tiết đáng kể |

Đây là thang đánh giá định tính, không phải kết quả đo chất lượng ảnh bằng thuật toán.

### C4 — Material Evidence

| Giá trị | Định nghĩa |
|---|---|
| `Verified` | Có nguồn bằng chứng đáng tin cậy xác nhận vật liệu của đối tượng |
| `Unverified` | Không có đủ bằng chứng xác nhận vật liệu |

Nếu chọn `Verified`, bắt buộc ghi rõ nguồn bằng chứng và loại vật liệu được xác nhận.

Không được chọn `Verified` chỉ vì vật thể trông giống đồ nhựa.

### C5 — Notes

Ghi nhận nguyên nhân ảnh hưởng đến đánh giá, chẳng hạn:

- Crop quá nhỏ.
- Ảnh mờ.
- Đối tượng bị che khuất.
- Nhãn không rõ ràng.
- Không mở được ảnh.
- Vật liệu chưa thể xác minh.

## 6. Quy trình thực hiện

### Step 1 — Chuẩn bị danh sách mẫu

**Owner:** Hùng

- Đọc annotation JSON đã làm sạch.
- Chọn mẫu theo quy tắc tại Mục 4.
- Xuất danh sách annotation ID.
- Liên kết từng mẫu với ảnh crop tương ứng.
- Xác minh đường dẫn và quyền truy cập ảnh.

**Output:** Sample manifest có thể tái lập.

### Step 2 — Kiểm tra độc lập

**Reviewers:** Yến và Hùng

Mỗi người kiểm tra độc lập cùng danh sách ảnh.

Với từng mẫu:

1. Mở ảnh crop.
2. Kiểm tra khả năng nhìn thấy đối tượng.
3. Đối chiếu nhãn với đối tượng.
4. Đánh giá chất lượng ảnh.
5. Kiểm tra bằng chứng vật liệu nếu có.
6. Ghi kết quả và ghi chú.

Không tham khảo kết quả của người còn lại trước khi hoàn thành lượt đánh giá độc lập. Người đánh giá phải ghi `reviewer`, thời điểm review và trạng thái truy xuất. Nếu người chuẩn bị mẫu cũng là reviewer, danh sách được khóa trước khi mở ảnh để hạn chế chọn mẫu theo chủ quan.

### Step 3 — Đối chiếu bất đồng

So sánh kết quả giữa hai người đánh giá.

Các trường hợp cần xem lại:

- Một người chọn `Yes`, người còn lại chọn `No`.
- Một người chọn `Uncertain`, người còn lại đưa ra kết luận chắc chắn.
- Chênh lệch đánh giá chất lượng ảnh.
- Khác biệt về bằng chứng vật liệu.

Khi có bất đồng:

1. Mở lại ảnh crop.
2. Đối chiếu ảnh nguồn và bounding box nếu cần.
3. Ghi nhận lý do bất đồng.
4. Thống nhất kết luận hoặc giữ `Uncertain`.
5. Không ép kết luận khi thiếu bằng chứng.

Lưu cả đánh giá ban đầu và kết quả sau đối chiếu. Trường hợp không thống nhất sau khi đối chiếu phải giữ `Unresolved`, nêu lý do và không ép thành `Agreed`.

**Độ nhất quán giữa người đánh giá:** báo cáo số cặp có đủ hai đánh giá hợp lệ và tỷ lệ đồng thuận thô (percent agreement) **riêng cho** Visibility, Label Consistency và Quality trước khi hòa giải; có thể bổ sung Cohen’s kappa khi phù hợp, nhưng phải giải thích hạn chế khi phân bố nhãn lệch hoặc số mẫu nhỏ. Không đặt ngưỡng kappa tùy tiện làm điều kiện đạt nếu nhóm chưa phê duyệt.

### Step 4 — Tổng hợp

**Owner:** Yến

Tổng hợp theo từng lớp:

- Số mẫu được chọn và số mẫu thực sự kiểm tra.
- Tỷ lệ đối tượng nhìn thấy.
- Tỷ lệ nhãn khớp ảnh.
- Phân bố chất lượng Good/Fair/Poor.
- Số mẫu có và không có bằng chứng vật liệu.
- Số trường hợp bất đồng và chưa giải quyết.

Mọi tỷ lệ phải ghi rõ **tử số, mẫu số**, cách xử lý `Uncertain` và ảnh không truy xuất được. Báo cáo theo lớp và theo split; không suy rộng tỷ lệ trên mẫu kiểm tra thành tỷ lệ của toàn bộ 22 lớp. Không dùng kết quả này như Accuracy/Precision/Recall/F1 của CNN.

### Step 5 — Cập nhật Class Scope Analysis

Cập nhật `docs/class_scope_analysis.md` với:

- Phương pháp và quy mô lấy mẫu.
- Kết quả Visual Inspection theo từng lớp.
- Các vấn đề ảnh hưởng đến chất lượng dữ liệu.
- Giới hạn của việc xác định vật liệu.
- Đề xuất giữ, loại hoặc cần kiểm tra thêm **ở mức bằng chứng và review**, không tự ý thay đổi cấu hình baseline 22 lớp.
- Quyết định còn mở.

Không chốt tập lớp CNN chỉ dựa trên 120 ảnh mẫu.

## 7. Định dạng dữ liệu đầu ra

File kết quả: `docs/visual_inspection_results.csv`.

Một dòng tương ứng với một annotation được chọn, chứa hai **bộ cột đánh giá độc lập** và một bộ cột kết luận đối chiếu. Có thể lưu riêng hai file raw review bất biến trước khi hợp nhất; kết quả tổng hợp không được ghi đè nhận xét ban đầu.

Các trường bắt buộc:

| Field | Nội dung |
|---|---|
| `annotation_id` | ID annotation |
| `image_id` | ID ảnh nguồn |
| `class_id` | ID lớp |
| `class_name` | Tên lớp |
| `split` | Train hoặc Validation |
| `crop_path` | Đường dẫn ảnh crop |
| `reviewer_1` | Người đánh giá thứ nhất |
| `reviewer_1_visibility` | Visibility |
| `reviewer_1_label_match` | Label Consistency |
| `reviewer_1_quality` | Image Quality |
| `reviewer_2` | Người đánh giá thứ hai |
| `reviewer_2_visibility` | Visibility |
| `reviewer_2_label_match` | Label Consistency |
| `reviewer_2_quality` | Image Quality |
| `reviewer_1_material_evidence` | Verified/Unverified/Not assessed |
| `reviewer_2_material_evidence` | Verified/Unverified/Not assessed |
| `reviewer_1_reviewed_at` | Thời điểm reviewer 1 hoàn thành |
| `reviewer_2_reviewed_at` | Thời điểm reviewer 2 hoàn thành |
| `access_status` | Available / Inaccessible |
| `source_image_path` | Tham chiếu ảnh nguồn nếu có |
| `sampling_manifest_id` | Phiên bản manifest đã khóa |
| `final_visibility` | Kết luận sau đối chiếu |
| `final_label_match` | Kết luận sau đối chiếu |
| `final_quality` | Kết luận sau đối chiếu |
| `material_evidence` | Verified hoặc Unverified |
| `material_type` | Loại vật liệu nếu xác minh được |
| `material_evidence_source` | Nguồn xác minh |
| `review_status` | Pending / Agreed / Unresolved |
| `disagreement_reason` | Lý do bất đồng, nếu có |
| `adjudication_notes` | Căn cứ kết luận sau đối chiếu |
| `notes` | Ghi chú |

Một annotation ID chỉ xuất hiện một lần trong file kết quả tổng hợp.

Các giá trị chưa đánh giá phải để trống và có trạng thái `Pending`; không được điền kết quả giả định. `Verified` bắt buộc có `material_type` và `material_evidence_source` truy xuất được; nếu không có, dùng `Unverified`. Cột `review_status` chỉ là trạng thái đối chiếu, không phải nhãn chất lượng ảnh.

**Đầu ra tối thiểu:** `docs/visual_inspection_sample_manifest.csv` (danh sách đã khóa), `docs/visual_inspection_results.csv` (kết quả), và `docs/visual_inspection_summary.md` (tổng hợp, agreement, hạn chế). Đây là **đường dẫn đề xuất**, chưa khẳng định các file đã được tạo.

## 8. Kiểm soát chất lượng

- Không sử dụng ảnh từ nguồn ngoài để thay thế mẫu đã chọn.
- Không chỉnh sửa ảnh nhằm làm thay đổi nội dung đối tượng.
- Không dùng kết quả Visual Inspection để thay đổi nhãn Train/Validation mà không có quy trình sửa dữ liệu riêng.
- Không xem tập Validation là tập Test độc lập.
- Không dùng đánh giá thủ công để tuyên bố mô hình CNN đạt Accuracy hoặc F1.
- Mọi nhận định về vật liệu phải có nguồn bằng chứng.
- Không công bố đường dẫn cá nhân hoặc thông tin truy cập Google Drive không cần thiết trong repository công khai.

## 9. Điều kiện nghiệm thu

Visual Inspection được xem là hoàn thành khi:

- [ ] Protocol được nhóm review và chấp thuận.
- [ ] Sample manifest có đủ ID, split và nguồn ảnh.
- [ ] Danh sách mẫu tái lập được bằng seed, PRNG, thứ tự sort, phiên bản script và hash nguồn.
- [ ] Pilot hoàn thành, rubric được khóa trước review chính thức.
- [ ] Số annotation/image/video duy nhất được ghi nhận, không đồng nhất chúng.
- [ ] Hai người đánh giá độc lập hoàn thành kiểm tra.
- [ ] Kết quả được ghi đầy đủ vào CSV.
- [ ] Bất đồng được đối chiếu và ghi nhận; agreement trước hòa giải được báo cáo.
- [ ] Có báo cáo tổng hợp theo từng lớp.
- [ ] Các trường hợp không xác minh được vật liệu được ghi rõ.
- [ ] `class_scope_analysis.md` được cập nhật dựa trên bằng chứng.
- [ ] Kết quả được review trong Issue #4.

## 10. Phân công

| Vai trò | Người phụ trách | Trách nhiệm |
|---|---|---|
| Protocol owner | Yến | Thiết kế, duy trì quy trình |
| Dataset owner | Hùng | Lấy mẫu, xác nhận metadata và ảnh |
| Independent reviewers | Yến, Hùng | Kiểm tra và đối chiếu |
| CNN reviewer | Hoàng Bách | Đánh giá tác động đến CNN |
| Evaluation reviewer | Ân | Kiểm tra phương pháp và giới hạn đánh giá |
| Inference reviewer | Ngọc Nguyên | Kiểm tra tính nhất quán nhãn đầu ra |

## 11. Giới hạn

Visual Inspection chỉ đánh giá chất lượng ảnh và tính phù hợp sơ bộ của nhãn.

Kết quả không chứng minh vật liệu của tất cả đối tượng trong dataset, không đại diện cho toàn bộ môi trường đại dương và không thay thế kết quả đánh giá CNN trên dữ liệu độc lập.

Mọi kết luận về phạm vi CNN phải xem xét đồng thời bằng chứng thị giác, phân bố dữ liệu, mục tiêu bài toán và tính khả thi của thực nghiệm.

---

**Approval record**

- Prepared by: Phan Hải Yến
- Dataset verification: Pending — Hùng
- CNN feasibility review: Pending — Hoàng Bách
- Evaluation review: Pending — Ân
- Final approval: Pending — Team

**Current status: PROPOSED — READY FOR TEAM REVIEW — NOT APPROVED**

**Change log v1.1:** Đồng bộ Option D (22-class CNN + six-class deep-dive); bổ sung pilot, PRNG/sort/hash, kiểm tra manifest, trạng thái ảnh không truy xuất được, raw independent reviews, agreement và adjudication, giới hạn suy rộng và bộ output có thể audit. Không thay đổi dataset, không thực hiện lấy mẫu, không có kết quả Visual Inspection thực tế.