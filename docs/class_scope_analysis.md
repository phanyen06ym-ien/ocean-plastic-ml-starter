# Class Scope Analysis — Ocean Plastic Waste Classification and Monitoring

**Project:** Phân loại và giám sát rác thải nhựa đại dương  
**Sprint:** Sprint 1 — Foundation  
**Issue:** #4 — S1-ARCH  
**Owner:** Phan Hải Yến  
**Dataset:** TrashCan 1.0 — Instance  
**Version:** 1.2  
**Status:** PROPOSED — NOT APPROVED (Pending Team Review)  
**Proposed approach:** Option D — General Object Classification Baseline + Plastic-Related Class Analysis

---

## 1. Mục đích

Tài liệu phân tích phạm vi nhãn của bộ dữ liệu TrashCan 1.0 — Instance nhằm đề xuất thiết kế bài toán phân loại ảnh đối tượng dưới nước bằng mạng nơ-ron tích chập (CNN), đồng thời duy trì sự liên quan với mục tiêu dài hạn là hỗ trợ giám sát và thu gom rác thải nhựa đại dương.

Các mục tiêu cụ thể:

1. Phân tích toàn bộ 22 lớp đối tượng trong dataset.
2. Đối chiếu các lớp với mục tiêu nghiên cứu rác thải nhựa.
3. Đánh giá mức độ hỗ trợ của dữ liệu Train và Validation.
4. So sánh các phương án lựa chọn tập nhãn.
5. Đề xuất phương án phù hợp với nguồn lực và yêu cầu Sprint 1.
6. Xây dựng chiến lược đánh giá tổng thể và phân tích chuyên sâu nhóm lớp liên quan đến nhựa.
7. Xác định các rủi ro, quyết định còn mở và điều kiện phê duyệt.

**Nguyên tắc khoa học:** Phân loại loại đối tượng không đồng nghĩa với xác định vật liệu cấu thành. Tài liệu không tuyên bố một đối tượng là nhựa nếu chưa có bằng chứng xác minh vật liệu.

## 2. Cơ sở dữ liệu

### 2.1. Nguồn dữ liệu

Dữ liệu phân tích được ghi nhận từ các tài liệu và tệp do nhóm chuẩn bị:

- `instances_train_grouped_clean.json`
- `instances_val_grouped_clean.json`
- `grouped_class_distribution.csv`
- `dataset_report.md`

### 2.2. Thống kê tổng quan

| Chỉ tiêu | Train | Validation | Tổng |
|---|---:|---:|---:|
| Ảnh nguồn | 5.884 | 1.328 | 7.212 |
| Annotation | 9.866 | 2.262 | 12.128 |
| Lớp có mẫu | 22 | 20 | 22 |
| Nhóm video | 249 | 63 | 312 |

Theo Dataset Audit của nhóm, Train và Validation không có video ID giao nhau.

Các thống kê theo lớp là **số annotation**, không phải số ảnh nguồn độc lập. Một ảnh có thể chứa nhiều annotation.

Kết quả này được sử dụng làm cơ sở phân tích ban đầu và cần được xác nhận với phiên bản dữ liệu chính thức trước khi khóa cấu hình thực nghiệm.

## 3. Nguyên tắc lựa chọn lớp

### 3.1. Phân biệt loại đối tượng, loại rác và vật liệu

Ba khái niệm cần được phân biệt:

- **Object category:** Loại đối tượng được quan sát, ví dụ `rov`, `animal_fish`, `trash_bottle`.
- **Waste category:** Loại đối tượng được dataset gắn nhãn là rác, ví dụ `trash_bag`, `trash_net`.
- **Material category:** Vật liệu cấu thành, ví dụ nhựa, kim loại, thủy tinh hoặc vật liệu khác.

Ví dụ, nhãn `trash_bottle` cho biết đối tượng thuộc nhóm chai bị thải bỏ, nhưng không chứng minh chai đó được làm từ nhựa.

Tương tự, `trash_rope`, `trash_net`, `trash_tarp` có thể liên quan đến vật liệu tổng hợp nhưng không mặc nhiên được xác nhận là nhựa.

### 3.2. Tiêu chí đánh giá

| Mã | Tiêu chí | Nội dung |
|---|---|---|
| C1 | Waste relevance | Mức độ liên quan đến rác thải |
| C2 | Plastic relevance | Mức độ liên quan tiềm năng đến rác nhựa |
| C3 | Material evidence | Bằng chứng xác minh vật liệu |
| C4 | Visual distinguishability | Khả năng phân biệt bằng hình ảnh |
| C5 | Training support | Số lượng và chất lượng dữ liệu Train |
| C6 | Validation support | Khả năng đánh giá bằng dữ liệu Validation |
| C7 | Reproducibility | Khả năng tái lập tập nhãn và thực nghiệm |
| C8 | Scope alignment | Mức độ phù hợp với mục tiêu và phạm vi Sprint 1 |

**Quy tắc sử dụng:**

- C1–C2 hỗ trợ xác định nhóm lớp cần phân tích chuyên sâu.
- C3 là điều kiện để đưa ra kết luận về vật liệu.
- C4–C7 đánh giá tính khả thi thực nghiệm.
- C8 đánh giá sự phù hợp với mục tiêu học phần và nguồn lực nhóm.

Các tiêu chí này chưa được chuyển thành thang điểm định lượng chính thức; nhóm cần phê duyệt quy tắc ra quyết định trước khi sử dụng để loại hoặc gộp lớp.

## 4. Phân tích toàn bộ 22 lớp

| ID | Class name | Train | Validation | Nhóm đối tượng | Hướng phân tích |
|---|---|---:|---:|---|---|
| 1 | `rov` | 2.662 | 655 | Thiết bị | Baseline tổng quát |
| 2 | `plant` | 394 | 113 | Thực vật | Baseline tổng quát |
| 3 | `animal_fish` | 658 | 106 | Động vật | Baseline tổng quát |
| 4 | `animal_starfish` | 282 | 116 | Động vật | Baseline tổng quát |
| 5 | `animal_shells` | 248 | 1 | Động vật/vỏ sinh vật | Baseline; Validation hạn chế |
| 6 | `animal_crab` | 271 | 38 | Động vật | Baseline tổng quát |
| 7 | `animal_eel` | 272 | 71 | Động vật | Baseline tổng quát |
| 8 | `animal_etc` | 159 | 76 | Động vật | Baseline tổng quát |
| 9 | `trash_clothing` | 67 | 15 | Rác | Baseline; vật liệu chưa rõ |
| 10 | `trash_pipe` | 150 | 6 | Rác | Baseline; Validation ít |
| 11 | `trash_bottle` | 119 | 7 | Rác | Ứng viên phân tích sâu |
| 12 | `trash_bag` | 786 | 122 | Rác | Ứng viên phân tích sâu |
| 13 | `trash_snack_wrapper` | 71 | 13 | Rác | Ứng viên phân tích sâu |
| 14 | `trash_can` | 321 | 138 | Rác | Baseline; vật liệu chưa rõ |
| 15 | `trash_cup` | 59 | 0 | Rác | Baseline; thiếu Validation |
| 16 | `trash_container` | 454 | 56 | Rác | Baseline; vật liệu chưa rõ |
| 17 | `trash_unknown_instance` | 2.217 | 539 | Rác chưa xác định | Baseline; lớp không đồng nhất |
| 18 | `trash_branch` | 294 | 42 | Rác/vật liệu hữu cơ | Baseline tổng quát |
| 19 | `trash_wreckage` | 48 | 117 | Rác/mảnh vỡ | Baseline; cần kiểm tra phân bố |
| 20 | `trash_tarp` | 102 | 19 | Rác | Ứng viên phân tích sâu |
| 21 | `trash_rope` | 117 | 0 | Rác | Ứng viên; thiếu Validation |
| 22 | `trash_net` | 115 | 12 | Rác | Ứng viên phân tích sâu |

**Nhận xét:**

- Dataset có 22 lớp, bao gồm 14 lớp có tiền tố `trash_` và 8 lớp đối tượng khác.
- Các lớp có mức hỗ trợ Train/Validation rất khác nhau.
- `trash_cup` và `trash_rope` không có annotation Validation.
- `rov` và `trash_unknown_instance` chiếm tỷ trọng lớn trong dữ liệu.
- `trash_wreckage` có số annotation Validation lớn hơn Train, cần kiểm tra ảnh hưởng của sự khác biệt phân bố.

Bảng trên là cơ sở đề xuất, không phải quyết định cuối cùng về class mapping.

## 5. Phân tích nhóm ứng viên liên quan đến nhựa

Sáu lớp được ưu tiên phân tích chuyên sâu theo mức độ liên quan tiềm năng đến rác nhựa:

| Class | Train | Validation | Tổng |
|---|---:|---:|---:|
| `trash_bag` | 786 | 122 | 908 |
| `trash_bottle` | 119 | 7 | 126 |
| `trash_snack_wrapper` | 71 | 13 | 84 |
| `trash_net` | 115 | 12 | 127 |
| `trash_rope` | 117 | 0 | 117 |
| `trash_tarp` | 102 | 19 | 121 |
| **Tổng** | **1.310** | **173** | **1.483** |

Đây là **nhóm phân tích**, không phải sáu nhãn vật liệu nhựa đã được xác minh.

Việc ưu tiên sáu lớp dựa trên ý nghĩa tên loại đối tượng và mức độ liên quan tiềm năng đến mục tiêu nghiên cứu. Kết quả Visual Inspection và tài liệu định nghĩa nhãn cần được dùng để kiểm tra lại sự phù hợp của nhóm này.

Các lớp khác như `trash_pipe`, `trash_cup`, `trash_container` vẫn có thể liên quan đến nhựa và không bị kết luận là không phải nhựa.

## 6. So sánh các phương án thiết kế

### Option A — Phân loại toàn bộ 22 lớp

Huấn luyện CNN với toàn bộ nhãn đối tượng của dataset.

**Ưu điểm:**
- Giữ nguyên không gian nhãn.
- Sử dụng toàn bộ dữ liệu hiện có.
- Thuận lợi xây dựng baseline tổng quát.

**Hạn chế:**
- Chưa tập trung phân tích mục tiêu rác nhựa.
- Không trực tiếp xác minh vật liệu.
- Có lớp thiếu hoặc rất ít Validation.

### Option B — Phân loại 14 lớp rác

Huấn luyện CNN với các lớp có tiền tố `trash_`.

**Ưu điểm:**
- Tập trung hơn vào các loại rác.
- Số lớp ít hơn Option A.

**Hạn chế:**
- Không có khả năng nhận biết đáng tin cậy các đối tượng ngoài 14 lớp.
- Một số lớp rác không liên quan trực tiếp đến nhựa.
- Vẫn có lớp thiếu Validation.

### Option C — Phân loại tập con liên quan đến nhựa

Huấn luyện CNN với tập con lớp được lựa chọn sau kiểm chứng.

**Ưu điểm:**
- Tập trung vào nhóm đối tượng liên quan tiềm năng đến nhựa.
- Phạm vi phân loại hẹp hơn.

**Hạn chế:**
- Dữ liệu ít hơn.
- Một số lớp thiếu mẫu Validation.
- Không có ground truth vật liệu nhựa.
- Có nguy cơ phân loại sai các đối tượng ngoài tập lớp.

### Option D — Baseline tổng quát và phân tích chuyên sâu

Đề xuất huấn luyện **một CNN baseline với 22 lớp đầu ra**, sau đó thực hiện hai cấp độ đánh giá:

1. **Đánh giá tổng thể:** Phân tích hiệu năng của mô hình trên toàn bộ tập nhãn.
2. **Phân tích chuyên sâu:** Xem xét riêng hiệu năng của sáu lớp ứng viên liên quan đến nhựa.

**Ưu điểm:**
- Giữ nguyên không gian nhãn và sử dụng dữ liệu đầy đủ.
- Có baseline tổng quát để phân tích.
- Vẫn tập trung được vào nhóm rác có liên quan tiềm năng đến nhựa.
- Không cần huấn luyện thêm một CNN riêng chỉ để có báo cáo chuyên sâu.

**Hạn chế:**
- Không chứng minh được khả năng xác minh vật liệu nhựa.
- Phân bố dữ liệu không cân bằng.
- Một số lớp không có đủ dữ liệu Validation.
- Việc phân tích chuyên sâu cần tách biệt rõ với đánh giá tổng thể.

**Đề xuất:** Option D.

**Trạng thái:** PROPOSED — NOT APPROVED.

## 7. Class Selection Decision Framework

### 7.1. Mục tiêu ra quyết định

Lựa chọn phương án đáp ứng đồng thời:

- Phù hợp với bài toán phân loại ảnh bằng CNN.
- Sử dụng dữ liệu hiện có một cách có căn cứ.
- Có thể xây dựng và kiểm thử trong Sprint 1.
- Có phương pháp đánh giá minh bạch.
- Không đưa ra tuyên bố vượt quá bằng chứng của dataset.

### 7.2. Điểm khác biệt giữa Option A và Option D

**Option D sử dụng cùng kiến trúc CNN 22 lớp và cùng tập dữ liệu với Option A.** Khác biệt nằm ở **thiết kế câu hỏi nghiên cứu và kế hoạch phân tích kết quả**, không phải một kiến trúc học máy khác:

- **Option A:** chỉ yêu cầu đánh giá năng lực phân loại tổng quát 22 lớp.
- **Option D:** yêu cầu đánh giá tổng quát 22 lớp **và** một báo cáo chuyên sâu đã xác định trước về sáu lớp ứng viên liên quan đến nhựa. Báo cáo chuyên sâu bao gồm support, Precision/Recall/F1 từng lớp, các cặp nhầm lẫn và ví dụ lỗi có thể truy vết.
- **Điều kiện có giá trị:** xác định nhóm sáu lớp trước khi xem kết quả dự đoán, giữ nguyên không gian dự đoán 22 lớp khi tính chỉ số và không mô tả kết quả là nhận diện vật liệu nhựa.
- **Chi phí bổ sung:** công việc kiểm tra dữ liệu, phân tích lỗi và báo cáo; **không yêu cầu CNN thứ hai**.

Nếu nhóm không thực hiện báo cáo chuyên sâu theo quy tắc trên, Option D về thực chất chỉ còn là Option A và phải cập nhật Decision Log tương ứng.

### 7.3. So sánh theo tiêu chí

| Tiêu chí | A | B | C | D |
|---|---|---|---|---|
| Sử dụng toàn bộ không gian nhãn | Có | Không | Không | Có |
| Có baseline tổng quát | Có | Không | Không | Có |
| Tập trung phân tích nhóm liên quan nhựa | Không bắt buộc | Một phần | Có | Có |
| Cần xác minh vật liệu trước khi tuyên bố nhận diện nhựa | Có | Có | Có | Có |
| Có rủi ro thiếu Validation | Có | Có | Có | Có |
| Cần thêm CNN thứ hai | Không | Không | Không | Không |
| Phù hợp với mục tiêu cân bằng khoa học và khả năng hoàn thành | Có điều kiện | Có điều kiện | Có điều kiện | Được đề xuất |

Bảng so sánh phản ánh đặc điểm thiết kế, chưa phải kết quả benchmark thực nghiệm.

### 7.4. Điều kiện phê duyệt Option D

Option D chỉ được phê duyệt khi:

1. Hùng xác nhận dataset và phân bố 22 lớp.
2. Hoàng Bách xác nhận CNN có thể cấu hình 22 đầu ra.
3. Ân xác nhận Evaluation Pipeline hỗ trợ đánh giá tổng thể và theo nhóm.
4. Nhóm thống nhất cách xử lý các lớp thiếu Validation.
5. Nhóm thống nhất rằng sáu lớp ứng viên chỉ được gọi là *plastic-related candidates*, không phải lớp vật liệu nhựa.
6. Class mapping được ghi nhận và dùng nhất quán giữa các module.
7. Quyết định được lưu tại Issue #4 và có review.

Nếu một điều kiện chưa đạt, Option D tiếp tục ở trạng thái Proposed.

### 7.5. Quy tắc thay đổi phương án

Nếu thực nghiệm cho thấy Option D không khả thi trong giới hạn Sprint 1, nhóm có thể xem xét phương án khác.

Mọi thay đổi phải ghi rõ:

- Lý do.
- Bằng chứng.
- Ảnh hưởng đến dữ liệu.
- Ảnh hưởng đến CNN và Evaluation.
- Người đề xuất và người phê duyệt.

Không thay đổi số lớp âm thầm trong code mà không cập nhật tài liệu và class mapping.

## 8. Thiết kế Class Mapping

### 8.1. Nguyên tắc

Đối với Option D, đề xuất giữ nguyên 22 nhãn đối tượng gốc.

CNN cần sử dụng các chỉ số lớp liên tiếp từ `0` đến `21`, trong khi category ID gốc của dataset nằm trong khoảng `1` đến `22`.

Do đó, cần xây dựng ánh xạ:

`original_category_id → model_class_index → class_name`

Ví dụ đề xuất:

| Original category ID | Model index | Class name |
|---|---:|---|
| 1 | 0 | `rov` |
| 2 | 1 | `plant` |
| 3 | 2 | `animal_fish` |
| ... | ... | ... |
| 20 | 19 | `trash_tarp` |
| 21 | 20 | `trash_rope` |
| 22 | 21 | `trash_net` |

Đây là quy tắc ánh xạ đề xuất dựa trên ID tăng dần, chưa phải mapping đã được phê duyệt.

### 8.2. Yêu cầu kiểm thử

- Mỗi original category ID có đúng một model index.
- Model index không trùng lặp.
- Số model index bằng số lớp đầu ra CNN.
- Train, Validation, Evaluation và Inference dùng cùng phiên bản mapping.
- Không thay đổi thứ tự lớp khi tải checkpoint.

**Bằng chứng cần có:** File mapping chính thức, kiểm thử ánh xạ và thông tin phiên bản mapping.

## 9. Chiến lược Validation và Evaluation

### 9.1. Nguyên tắc chia dữ liệu

- Giữ nguyên Train/Validation theo nhóm video.
- Không chuyển annotation từ Train sang Validation chỉ để tăng support của một lớp.
- Không sử dụng augmentation để tạo bằng chứng đánh giá độc lập từ dữ liệu Train.
- Nếu cần tập đánh giá bổ sung, phải xác định nguồn và phương pháp tạo tập riêng, bảo đảm không rò rỉ dữ liệu.

### 9.2. Đánh giá tổng thể 22 lớp

Các chỉ số dự kiến:

- Accuracy.
- Precision theo lớp.
- Recall theo lớp.
- F1-score theo lớp.
- Macro F1-score.
- Confusion Matrix 22 × 22.
- Support theo lớp.

Phải công bố số lượng mẫu Validation thực tế của từng lớp.

### 9.3. Xử lý lớp thiếu Validation

Hai lớp cần lưu ý:

| Class | Train | Validation | Hướng xử lý |
|---|---:|---:|---|
| `trash_cup` | 59 | 0 | Không kết luận Recall/F1 có ý nghĩa thực nghiệm trên lớp này |
| `trash_rope` | 117 | 0 | Không kết luận Recall/F1 có ý nghĩa thực nghiệm trên lớp này |

Đối với lớp không có mẫu dương trong Validation:

- Báo cáo `support = 0`.
- Không diễn giải Recall như một kết quả đo được.
- Ghi rõ F1 của lớp không có đủ điều kiện đánh giá có thể không xác định về mặt diễn giải.
- Nếu thư viện gán giá trị số mặc định khi gặp trường hợp không xác định, phải ghi rõ cấu hình `zero_division` và không coi số đó là bằng chứng hiệu năng.
- Không tuyên bố mô hình hoạt động tốt trên lớp chưa có mẫu đánh giá dương.

**Quy tắc báo cáo đề xuất (phải được Ân review trước khi khóa):**

1. **Macro F1 trên các lớp có support dương (primary descriptive metric):** tính F1 từng lớp có `support > 0` trong Validation, rồi lấy trung bình không trọng số. Ghi rõ danh sách lớp được tính; với số liệu hiện tại dự kiến là 20 lớp. Đây **không phải** Macro F1 đại diện cho đủ 22 lớp.
2. **Macro F1 trên toàn bộ 22 nhãn (supplementary metric):** gọi `f1_score(y_true, y_pred, labels=list(range(22)), average="macro", zero_division=0)` theo scikit-learn (nếu dùng thư viện này). Các lớp không có cả true positive lẫn predicted positive được quy ước F1 = 0 theo thiết lập trên; số này **không chứng minh** hiệu năng của lớp thiếu Validation.
3. **Báo cáo theo lớp:** luôn kèm `support`, Precision/Recall/F1 và ghi chú `N/A — no positive validation samples` ở Recall của `trash_cup` và `trash_rope`; nếu lớp không có ground truth nhưng có dự đoán dương, vẫn ghi nhận các false positive và có thể tính Precision theo quy tắc đã khai báo.
4. **Báo cáo chính:** công bố cả hai cách tổng hợp, số lớp tham gia, cấu hình `labels`, `zero_division`, và phiên bản thư viện. Không so sánh hai Macro F1 này như thể cùng một định nghĩa.

Các chỉ số tổng hợp phải công bố quy tắc đưa lớp vào phép tính. Không sử dụng một giá trị Macro F1 duy nhất để che giấu vấn đề thiếu support.

### 9.4. Phân tích chuyên sâu sáu lớp ứng viên

Sáu lớp được phân tích:

- `trash_bag`
- `trash_bottle`
- `trash_snack_wrapper`
- `trash_net`
- `trash_rope`
- `trash_tarp`

Các chỉ số được tính từ **cùng dự đoán của CNN baseline 22 lớp**, với `y_true` và `y_pred` trên **toàn bộ Validation**, không lọc bỏ mẫu thuộc 16 lớp còn lại trước khi tính confusion matrix và các thống kê one-vs-rest.

**Quy tắc tính chuyên sâu:** đối với mỗi lớp ứng viên `c`, tính `TP_c`, `FP_c`, `FN_c` trên toàn bộ 22 lớp: mọi ảnh thật thuộc lớp khác nhưng bị dự đoán thành `c` đều tính là `FP_c`; mọi ảnh thật thuộc `c` nhưng bị dự đoán thành bất kỳ lớp khác đều tính là `FN_c`. Sau đó mới trích Precision/Recall/F1 của sáu lớp. Không chuẩn hóa lại softmax trên sáu lớp và không thay đổi mẫu số để làm đẹp chỉ số.

Nếu muốn báo cáo một giá trị tổng hợp sáu lớp, nêu rõ đây là trung bình mô tả trên các lớp có support dương (hiện dự kiến năm lớp), còn `trash_rope` phải ghi `N/A` cho Recall và không đưa ra kết luận thực nghiệm về khả năng nhận đúng lớp này.

Không huấn luyện lại mô hình sáu lớp chỉ để tạo kết quả phân tích chuyên sâu.

Đối với từng lớp:

- Báo cáo support.
- Báo cáo Precision, Recall và F1 nếu có đủ điều kiện diễn giải.
- Phân tích các lớp thường bị nhầm lẫn.
- Ghi nhận ảnh dự đoán sai để phân tích lỗi.

Nếu tạo bảng tổng hợp riêng sáu lớp, phải xác định rõ tập mẫu và quy tắc tính. Không được loại bỏ dự đoán sai sang các lớp ngoài nhóm chỉ để cải thiện chỉ số.

### 9.5. Tránh Evaluation Leakage

- Không sử dụng Validation như dữ liệu huấn luyện.
- Không dùng kết quả Validation để điều chỉnh liên tục mô hình rồi tuyên bố đây là đánh giá độc lập cuối cùng.
- Các quyết định lựa chọn mô hình và điều chỉnh siêu tham số phải được ghi nhận.
- Nếu cần đánh giá cuối cùng độc lập, nhóm phải thiết kế một tập Test phù hợp, có nguồn gốc và quy tắc tách dữ liệu rõ ràng.

**Trạng thái:** Chiến lược đề xuất, chưa có kết quả đánh giá thực nghiệm.

## 10. Out-of-Scope và Closed-Set Classification

### 10.1. Giới hạn của mô hình

CNN baseline 22 lớp là mô hình phân loại **closed-set**.

Mô hình được thiết kế để lựa chọn một trong 22 lớp đã học. Khi gặp đối tượng không thuộc 22 lớp, mô hình vẫn có thể trả về một nhãn đã biết.

Do đó:

- Không mặc nhiên xem nhãn dự đoán là bằng chứng đối tượng chắc chắn thuộc dataset.
- Không mặc nhiên xem điểm softmax cao là bằng chứng mô hình nhận biết đúng.
- Không tuyên bố có khả năng phát hiện vật thể chưa từng học nếu chưa có thực nghiệm phù hợp.

### 10.2. Phạm vi kiểm thử đề xuất

Đối với Option D, ROV, động vật và thực vật **không phải đối tượng ngoài tập nhãn**, vì đã có nhãn trong 22 lớp.

Để kiểm tra giới hạn ngoài tập nhãn, cần sử dụng ảnh hoặc đối tượng không thuộc các nhãn được huấn luyện, có nguồn gốc được ghi nhận và không trùng với dữ liệu Train/Validation.

Các kịch bản đề xuất:

1. Ảnh đối tượng thuộc 22 lớp.
2. Ảnh đối tượng ngoài 22 lớp.
3. Ảnh không có đối tượng rõ ràng.
4. Ảnh có chất lượng thấp.
5. Ảnh đầu vào không hợp lệ.

### 10.3. Tiêu chí báo cáo

- Ghi nhận nhãn dự đoán và điểm softmax.
- Phân tích trường hợp mô hình tự tin nhưng sai.
- Phân biệt lỗi dữ liệu đầu vào với lỗi phân loại.
- Không cam kết cơ chế từ chối đối tượng ngoài tập nhãn nếu chưa được thiết kế và kiểm chứng.

Kiểm thử out-of-scope trong Sprint 1 nhằm **minh họa giới hạn**, không chứng minh hệ thống đã có khả năng Open-Set Recognition.

## 11. Kế hoạch Visual Inspection

### 11.1. Mục tiêu

Kiểm tra chất lượng và tính phù hợp của ảnh crop, hỗ trợ phân tích nhãn và các hạn chế dữ liệu trước khi chốt thiết kế thực nghiệm.

### 11.2. Phạm vi ưu tiên

Giữ kế hoạch kiểm tra chuyên sâu sáu lớp ứng viên liên quan đến nhựa.

Kế hoạch ban đầu dự kiến tối đa 20 annotation cho mỗi lớp, tổng mục tiêu 120 annotation.

Cần thực hiện theo `visual_inspection_protocol.md` sau khi protocol được review và phê duyệt.

### 11.3. Giới hạn

Kết quả kiểm tra sáu lớp **là kiểm tra có mục đích (purposive/stratified within candidate classes)**, không phải mẫu ngẫu nhiên đại diện cho toàn bộ 22 lớp. Không được suy rộng tỷ lệ lỗi nhãn hoặc chất lượng crop của 120 annotation thành tỷ lệ lỗi toàn dataset.

Đơn vị lấy mẫu là **annotation**, không mặc nhiên là ảnh nguồn độc lập. Nếu nhiều annotation thuộc cùng một ảnh/video, phải lưu ID để báo cáo sự phụ thuộc mẫu.

Nếu nhóm muốn đưa ra kết luận về chất lượng dữ liệu trên toàn bộ không gian nhãn, cần phê duyệt **một kế hoạch lấy mẫu bổ sung trên 16 lớp còn lại**, có số lượng theo lớp/split, quy tắc ngẫu nhiên tái lập và giới hạn suy rộng rõ ràng. Chưa ấn định số lượng bổ sung khi chưa xem nguồn lực và mức độ sai lệch cần phát hiện.

Việc quan sát ảnh không đủ để xác minh chắc chắn vật liệu nhựa khi thiếu nguồn bằng chứng độc lập.

### 11.4. Bằng chứng cần lưu

- Annotation ID.
- Image ID.
- Class ID và class name.
- Split.
- Đường dẫn crop.
- Kết quả đánh giá.
- Người đánh giá.
- Trường hợp bất đồng và kết luận xử lý.
- Thông tin phiên bản dữ liệu và quy tắc lấy mẫu.

**Trạng thái:** Chưa có kết quả Visual Inspection được xác nhận trong tài liệu này.

## 12. Phân tích rủi ro

| ID | Rủi ro | Ảnh hưởng | Biện pháp |
|---|---|---|---|
| R1 | Nhầm nhãn đối tượng với vật liệu | Kết luận khoa học sai | Không tuyên bố xác minh nhựa khi thiếu bằng chứng |
| R2 | Lớp thiếu Validation | Không đánh giá được đầy đủ | Công bố support và giới hạn chỉ số |
| R3 | Mất cân bằng lớp | Mô hình thiên lệch lớp phổ biến | Báo cáo chỉ số theo lớp, Macro F1 và Confusion Matrix |
| R4 | Crop quá nhỏ | Mất thông tin hình ảnh | Kiểm tra chất lượng crop |
| R5 | Rò rỉ dữ liệu theo video | Đánh giá quá lạc quan | Duy trì split theo nhóm video |
| R6 | Không nhất quán class mapping | Sai nhãn dự đoán và đánh giá | Dùng mapping chung có phiên bản |
| R7 | Closed-set prediction | Gán nhầm đối tượng ngoài phạm vi | Kiểm thử và công bố giới hạn |
| R8 | Nhóm sáu lớp không đại diện toàn bộ rác nhựa | Kết luận vượt quá phạm vi | Gọi là nhóm ứng viên, ghi rõ tiêu chí lựa chọn |
| R9 | Điều chỉnh thiết kế dựa quá nhiều vào Validation | Evaluation leakage | Ghi nhận quá trình lựa chọn và thiết kế đánh giá độc lập |

## 13. Kế hoạch kiểm chứng và phê duyệt

### Gate 1 — Dataset Verification

- [x] Có thống kê 22 lớp và số annotation từ Dataset Audit.
- [x] Có thống kê Train/Validation theo lớp.
- [x] Ghi nhận lớp thiếu Validation.
- [ ] Xác nhận phiên bản dataset chính thức dùng cho CNN.
- [ ] Xác nhận class mapping.

### Gate 2 — Visual Inspection

- [ ] Review và phê duyệt protocol.
- [ ] Kiểm tra ảnh đại diện sáu lớp ứng viên.
- [ ] Ghi nhận lỗi nhãn, chất lượng crop và mức độ quan sát được.
- [ ] Xác định có cần mở rộng kiểm tra sang các lớp khác hay không.
- [ ] Tổng hợp kết quả có bằng chứng.

### Gate 3 — CNN Feasibility

- [ ] Xác nhận CNN baseline có 22 đầu ra.
- [ ] Kiểm thử input/output shape.
- [ ] Kiểm thử class mapping.
- [ ] Xác nhận khả năng lưu và tải checkpoint nhất quán.

### Gate 4 — Evaluation Design

- [ ] Xác nhận chỉ số đánh giá tổng thể.
- [ ] Xác nhận quy tắc đánh giá sáu lớp ứng viên.
- [ ] Xác nhận xử lý lớp có `support = 0`.
- [ ] Xác nhận phương pháp phân tích lỗi và giới hạn closed-set.

### Gate 5 — Team Approval

- [ ] Yến trình bày Option D và bằng chứng.
- [ ] Hùng review dữ liệu.
- [ ] Hoàng Bách review CNN.
- [ ] Ân review Evaluation.
- [ ] Ngọc Nguyên review hợp đồng inference và class mapping.
- [ ] Nhóm thống nhất phương án.
- [ ] Ghi quyết định chính thức vào Issue #4.

## 14. Decision Log

| ID | Quyết định | Trạng thái |
|---|---|---|
| D01 | Sử dụng TrashCan 1.0 — Instance làm cơ sở thực nghiệm | Working assumption |
| D02 | Giữ 22 lớp đối tượng làm baseline | Proposed |
| D03 | Phân tích chuyên sâu sáu lớp ứng viên liên quan nhựa | Proposed |
| D04 | Lựa chọn Option D | Proposed — Not approved |
| D05 | Ánh xạ category ID gốc sang model index `0–21` | Proposed |
| D06 | Giữ nguyên split theo nhóm video | Proposed |
| D07 | Quy tắc đánh giá lớp thiếu Validation | Proposed |
| D08 | Phương pháp kiểm thử out-of-scope | Proposed |
| D09 | Phạm vi Visual Inspection mở rộng | Open |
| D10 | Phê duyệt cấu hình CNN và Evaluation cuối cùng | Open |

Không chuyển trạng thái Proposed sang Approved nếu chưa có bằng chứng review và quyết định của nhóm.

## 15. Điều kiện nghiệm thu

Tài liệu được xem là hoàn thành khi:

1. Có bảng phân tích đầy đủ 22 lớp.
2. Có số liệu Train/Validation và nguồn đối chiếu.
3. Có so sánh các phương án A/B/C/D.
4. Có lý do đề xuất Option D.
5. Có quy tắc class mapping.
6. Có chiến lược đánh giá tổng thể và nhóm ứng viên.
7. Có phương án xử lý các lớp thiếu Validation.
8. Có mô tả giới hạn closed-set và out-of-scope.
9. Có kết quả Visual Inspection hoặc ghi nhận rõ các hạng mục còn chờ, tùy theo tiêu chí nghiệm thu Issue #4.
10. Có review của các thành viên liên quan.
11. Có quyết định chính thức được ghi nhận trong Issue #4 trước khi áp dụng làm cấu hình khóa.

## 16. Kết luận

Phương án D được đề xuất nhằm cân bằng giữa chất lượng khoa học và khả năng hoàn thành Sprint 1.

Theo phương án này, nhóm dự kiến sử dụng một CNN baseline phân loại 22 loại đối tượng dưới nước, đồng thời phân tích chuyên sâu sáu lớp rác có khả năng liên quan đến nhựa.

Cách tiếp cận cho phép nghiên cứu khả năng phân loại đối tượng của CNN trên không gian nhãn gốc và xem xét riêng những loại rác có ý nghĩa đối với định hướng đề tài.

Tuy nhiên, phương án này **không đồng nghĩa với nhận diện chính xác vật liệu nhựa**, cũng chưa giải quyết hoàn toàn vấn đề mất cân bằng lớp, thiếu Validation hoặc nhận biết đối tượng ngoài tập nhãn.

**Final document status: PROPOSED — NOT APPROVED.**

Tập nhãn và cấu hình thực nghiệm chỉ được khóa sau khi hoàn thành các bước kiểm chứng và được nhóm phê duyệt.

---

## References và Data Provenance

| Nguồn | Vai trò | Truy vết / tình trạng |
|---|---|---|
| `instances_train_grouped_clean.json` | COCO Train | Tệp nhóm cung cấp; checksum và phiên bản phát hành: **chưa xác nhận** |
| `instances_val_grouped_clean.json` | COCO Validation | Tệp nhóm cung cấp; checksum và phiên bản phát hành: **chưa xác nhận** |
| `grouped_class_distribution.csv` | Thống kê lớp | Cần ghi đường dẫn Drive/revision và checksum khi khóa dữ liệu |
| `dataset_report.md` | Dataset Audit Sprint 1 | Cần ghi commit/revision đã review |
| `docs/problem_statement.md` | Ranh giới bài toán | Đối chiếu phiên bản mới nhất được nhóm review |
| `docs/visual_inspection_protocol.md` | Quy tắc kiểm tra ảnh | Chưa phê duyệt protocol; cần ghi phiên bản khi thực thi |
| [GitHub Issue #4](https://github.com/phanyen06ym-ien/ocean-plastic-ml-starter/issues/4) | Quyết định phạm vi | Chưa có phê duyệt Option D |

**Provenance gate trước khi thực nghiệm:** Hùng ghi rõ nguồn tải, phiên bản/phát hành dataset nếu xác định được, SHA-256 của hai COCO JSON, phiên bản manifest/crop, commit hoặc revision của script preprocessing, ngày kiểm tra và người xác nhận. Không điền giả checksum, thời gian hoặc commit. Những thông tin này chưa có trong tài liệu nên được giữ ở trạng thái **PENDING**.
