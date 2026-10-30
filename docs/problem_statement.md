# Problem Statement — Ocean Plastic Waste Classification and Monitoring

**Project:** Phân loại và giám sát rác thải nhựa đại dương  
**Sprint:** Sprint 1 — Foundation  
**Issue:** #4 — S1-ARCH  
**Dataset:** TrashCan 1.0 — Instance  
**Version:** 1.1 — Review Revision  
**Status:** DRAFT — Pending Class Scope Approval

## 1. Bối cảnh và vấn đề (WHY)

Rác thải trong môi trường biển và dưới nước đặt ra thách thức đối với hoạt động quan sát, giám sát và thu gom. Việc nhận biết đối tượng qua hình ảnh có thể gặp khó khăn do điều kiện ánh sáng, độ đục của nước, vật thể bị che khuất và sự đa dạng về hình dạng.

Trong định hướng phát triển hệ thống hỗ trợ giám sát và robot thu gom rác, nhận biết đối tượng từ hình ảnh là một thành phần quan trọng. Tuy nhiên, trước khi xây dựng hệ thống hoàn chỉnh, cần kiểm chứng khả năng phân loại đối tượng bằng mô hình học máy trên dữ liệu phù hợp.

Dự án lựa chọn Convolutional Neural Network (CNN) để nghiên cứu bài toán phân loại ảnh đối tượng dưới nước.

**Ranh giới bài toán:** Sprint 1 chỉ phát triển thành phần phân loại ảnh đối tượng đã được cắt sẵn. Thành phần này có thể được tích hợp vào hệ thống giám sát trong tương lai, nhưng chưa thực hiện phát hiện, định vị, theo dõi hoặc thu gom rác.

## 2. Đối tượng sử dụng và hưởng lợi (WHO)

### 2.1. Đối tượng hướng đến trong tương lai

- Nhóm nghiên cứu môi trường biển và thị giác máy tính.
- Đơn vị phát triển hệ thống giám sát môi trường dưới nước.
- Nhóm phát triển robot hỗ trợ nhận biết và thu gom rác.

### 2.2. Đối tượng sử dụng trong Sprint 1

- Nhóm sinh viên phát triển và kiểm thử mô hình.
- Giảng viên đánh giá phương pháp và kết quả học máy.
- Người thực hiện phân tích dữ liệu và kết quả dự đoán.

Sản phẩm Sprint 1 là một hệ thống thực nghiệm phục vụ học tập và kiểm chứng kỹ thuật, chưa phải giải pháp giám sát môi trường được triển khai thực tế.

## 3. Phát biểu bài toán (WHAT)

Cho một ảnh RGB chứa đối tượng dưới nước đã được xác định và cắt từ ảnh nguồn, xây dựng mô hình CNN dự đoán nhãn đối tượng trong tập lớp được nhóm thống nhất.

### 3.1. Đầu vào

- Ảnh RGB của một đối tượng đã crop.
- Kích thước đầu vào sau tiền xử lý: 224 × 224 pixel.
- Quy trình biến đổi màu, resize, padding và chuẩn hóa phải được xác định trong Data Contract.
- Huấn luyện, đánh giá và suy luận phải sử dụng các phép tiền xử lý tương thích.

### 3.2. Đầu ra

- Nhãn lớp dự đoán.
- Vector điểm softmax tương ứng với các lớp.
- Thông tin phục vụ phân tích kết quả, chẳng hạn class ID và class name.

Điểm softmax biểu thị mức độ ưu tiên tương đối giữa các lớp mà mô hình được huấn luyện để phân biệt. **Không mặc nhiên coi đây là xác suất đúng đã được hiệu chuẩn.**

### 3.3. Loại bài toán

**Supervised Multi-class Image Classification.**

Số lớp đầu ra CNN và ánh xạ nhãn chính thức sẽ được xác định sau bước Class Scope Review.

Mô hình chỉ phân loại trong tập nhãn được cấu hình. Nếu gặp đối tượng ngoài tập nhãn, mô hình vẫn có thể trả về một lớp đã học; khả năng phát hiện hoặc từ chối đối tượng ngoài phạm vi chưa thuộc cam kết của Sprint 1.

## 4. Cơ sở dữ liệu thực nghiệm

Dự án sử dụng TrashCan 1.0 — Instance, bộ dữ liệu ảnh dưới nước với annotation theo định dạng COCO.

### 4.1. Thống kê dữ liệu

| Chỉ tiêu | Train | Validation | Tổng |
|---|---:|---:|---:|
| Ảnh nguồn | 5.884 | 1.328 | 7.212 |
| Annotation | 9.866 | 2.262 | 12.128 |
| Lớp có mẫu | 22 | 20 | 22 |
| Nhóm video | 249 | 63 | 312 |

Theo kết quả Dataset Audit đã cung cấp, không có video ID giao nhau giữa Train và Validation.

Các kiểm tra cấu trúc JSON, ID và bounding box được ghi nhận là hợp lệ trên những file đã kiểm tra.

Số annotation không tương đương số ảnh nguồn độc lập.

### 4.2. Hạn chế dữ liệu

- `trash_cup`: 59 annotation Train, 0 Validation.
- `trash_rope`: 117 annotation Train, 0 Validation.
- `animal_shells`: 1 annotation Validation.
- Phân bố lớp không đồng đều.
- Dữ liệu chứa cả rác, động vật, thực vật và ROV.
- Nhãn đối tượng không đồng nghĩa với nhãn vật liệu.

Do đó, kết quả phân loại các loại đối tượng không được trình bày như bằng chứng nhận diện chính xác vật liệu nhựa.

## 5. Mục tiêu Sprint 1

1. Phân tích dataset và xây dựng cơ sở lựa chọn phạm vi nhãn.
2. Thiết kế CNN baseline với số lớp đầu ra cấu hình được.
3. Chuẩn bị quy trình tiền xử lý và Data Contract.
4. Xây dựng khung huấn luyện có thể tái lập.
5. Chuẩn bị Evaluation Pipeline với Accuracy, Precision, Recall, Macro F1-score và Confusion Matrix.
6. Xây dựng chương trình inference cơ bản.
7. Hoàn thiện tài liệu kiến trúc, kế hoạch thực nghiệm và bằng chứng kiểm thử.

Sprint 1 ưu tiên chứng minh các thành phần có thể kết nối và hoạt động đúng về mặt kỹ thuật. Kết quả hiệu năng của mô hình chỉ được công bố khi có thực nghiệm phù hợp.

## 6. Phạm vi và giới hạn

### 6.1. Trong phạm vi

- Phân loại ảnh đối tượng đã crop.
- Phân tích và kiểm định dữ liệu.
- Thiết kế CNN baseline.
- Kiểm thử forward pass.
- Chuẩn bị huấn luyện và đánh giá.
- Demo inference bằng mô hình mẫu.
- Thiết kế kế hoạch thực nghiệm.

### 6.2. Ngoài phạm vi

- Object Detection và segmentation.
- Phân loại nhiều đối tượng trực tiếp trong một ảnh nguồn.
- Theo dõi đối tượng trong video.
- Xác minh vật liệu nhựa từ nhãn TrashCan.
- Phát hiện đáng tin cậy đối tượng ngoài tập nhãn.
- Điều khiển robot.
- Giám sát đại dương thời gian thực.
- Cam kết chỉ số hiệu năng trước khi thực nghiệm.

## 7. Tiêu chí nghiệm thu Sprint 1

Các tiêu chí dưới đây là **điều kiện kiểm thử đề xuất**, chưa phải kết quả đã PASS.

| ID | Hạng mục | Điều kiện nghiệm thu | Bằng chứng cần có |
|---|---|---|---|
| AC-01 | Class Scope | Có danh sách lớp, class ID, class name và quyết định review | Class Scope Analysis, Decision Log |
| AC-02 | Data Contract | Định nghĩa input shape, màu ảnh, resize, padding, normalization và label mapping | Data Contract |
| AC-03 | CNN Baseline | Với batch đầu vào hợp lệ, forward pass trả tensor kích thước `[B, K]`, trong đó `K` là số lớp cấu hình | Unit/smoke test |
| AC-04 | Label Mapping | Output index ánh xạ nhất quán đến class ID và class name | Mapping test |
| AC-05 | Evaluation | Tính được các chỉ số từ dữ liệu kiểm thử có nhãn và đối chiếu được với kết quả kỳ vọng | Evaluation test |
| AC-06 | Inference | Nhận ảnh hợp lệ và trả kết quả đúng cấu trúc; có xử lý lỗi đầu vào không hợp lệ | Inference test |
| AC-07 | Reproducibility | Ghi nhận cấu hình, seed, phiên bản dữ liệu và dependency cần thiết | Config, README, test log |
| AC-08 | GitHub Evidence | Có Issues, PRs, review và bằng chứng tương ứng với deliverable | GitHub links |

Trong đó:

- `B` là batch size.
- `K` là số lớp được phê duyệt.
- Các phép thử kỹ thuật có thể dùng mô hình hoặc dữ liệu mẫu.
- Việc vượt qua smoke test không chứng minh mô hình có độ chính xác cao.
- Các lớp thiếu mẫu Validation phải được ghi nhận giới hạn khi đánh giá.

## 8. Các quyết định còn mở

| ID | Quyết định | Trạng thái |
|---|---|---|
| D01 | Lựa chọn tập nhãn CNN | Open |
| D02 | Chốt class mapping | Open |
| D03 | Chốt chi tiết Data Contract | Open |
| D04 | Chốt phương án đánh giá lớp thiếu Validation | Open |
| D05 | Chốt cấu hình CNN baseline | Open |

Các quyết định phải được ghi nhận trong tài liệu liên quan và review trước khi áp dụng làm cấu hình chính thức.

## 9. Kết luận

Sprint 1 tập trung xây dựng nền tảng phân loại ảnh đối tượng dưới nước bằng CNN, sử dụng TrashCan 1.0 — Instance làm dữ liệu thực nghiệm.

Kết quả mong đợi là một quy trình kỹ thuật có thể kiểm chứng, bao gồm phân tích dữ liệu, mô hình baseline, đánh giá và inference.

Việc lựa chọn tập nhãn cần dựa trên Class Scope Analysis, kết quả Visual Inspection, tính khả thi của CNN và chiến lược đánh giá. Chưa có quyết định chính thức về số lớp hoặc tuyên bố hệ thống đã nhận diện được vật liệu nhựa.

**Current status: DRAFT — Pending Class Scope Approval and Technical Review.**