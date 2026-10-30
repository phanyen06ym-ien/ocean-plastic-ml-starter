# Problem Statement — Ocean Plastic Waste Classification and Monitoring

## 1. Bối cảnh (Background)

Rác thải nhựa trong môi trường biển là một vấn đề môi trường đáng quan tâm. Các loại rác như chai nhựa, túi nilon và vật dụng nhựa có thể xuất hiện trong nhiều điều kiện môi trường khác nhau, gây khó khăn cho việc quan sát, xác định và thu gom.

Việc ứng dụng học máy và thị giác máy tính có thể hỗ trợ tự động nhận diện rác thải từ hình ảnh, giảm sự phụ thuộc vào quá trình quan sát thủ công.

## 2. Vấn đề cần giải quyết (Problem)

Quá trình xác định rác thải bằng mắt thường có thể gặp khó khăn khi số lượng vật thể lớn, ảnh có nhiều đối tượng hoặc điều kiện ánh sáng không thuận lợi.

Nhóm tập trung nghiên cứu khả năng sử dụng mạng nơ-ron tích chập (CNN) để tự động phân loại các đối tượng rác thải nhựa dựa trên hình ảnh.

## 3. Mục tiêu (Objectives)

- Tìm kiếm và khảo sát bộ dữ liệu ảnh rác thải có nhãn.
- Phân tích, làm sạch và tiền xử lý dữ liệu.
- Xây dựng mô hình CNN để phân loại rác thải.
- Huấn luyện và đánh giá mô hình trên các tập dữ liệu riêng biệt.
- Thử nghiệm khả năng dự đoán trên hình ảnh mới.
- Định hướng mở rộng sang phát hiện vị trí rác và giám sát qua ảnh/video.

## 4. Đối tượng sử dụng (Target Users)

- Nhóm nghiên cứu ứng dụng AI trong môi trường.
- Đơn vị giám sát, thu gom rác thải.
- Nhóm phát triển robot hoặc hệ thống tự động hỗ trợ làm sạch môi trường biển.

## 5. Phạm vi giữa kỳ (Midterm Scope)

**Trong phạm vi:**
- Image Classification sử dụng CNN.
- Khảo sát dataset, EDA và tiền xử lý ảnh.
- Huấn luyện, kiểm định và đánh giá mô hình.
- Dự đoán loại rác từ một ảnh đầu vào.

**Ngoài phạm vi giữa kỳ:**
- Điều khiển robot thực tế.
- Tự động gắp hoặc thu gom rác.
- Xác định tọa độ vật thể bằng Object Detection.
- Giám sát video thời gian thực.

## 6. Đầu vào và đầu ra (Input/Output)

**Input:** Ảnh màu RGB chứa một đối tượng rác cần phân loại, có thể là ảnh được cắt từ dữ liệu gán nhãn.

**Output:** Nhãn loại rác dự đoán và điểm xác suất/độ tin cậy do mô hình trả về.

Danh sách nhãn chính thức sẽ được xác định sau khi kiểm kê dữ liệu.

## 7. Phương pháp (Methodology)

Pipeline nghiên cứu dự kiến:

Dataset → Data Exploration → Preprocessing → CNN → Training → Validation → Evaluation → Prediction

Mô hình được xây dựng bằng Python và thư viện học sâu phù hợp với nội dung môn học.

## 8. Đánh giá (Evaluation)

Các chỉ số dự kiến:
- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion Matrix.
- Training Loss và Validation Loss.

Kết quả thực nghiệm sẽ được bổ sung sau quá trình huấn luyện, không giả định trước độ chính xác.

## 9. Hạn chế và rủi ro (Limitations)

- Dataset có thể không bao phủ đủ loại rác nhựa cần nghiên cứu.
- Một số lớp có thể mất cân bằng nghiêm trọng.
- Hình ảnh trong dataset có thể khác điều kiện thực tế dưới nước.
- Mô hình classification chưa xác định được vị trí của nhiều vật thể trong một ảnh.
- Kết quả trên tập kiểm tra không đồng nghĩa với hiệu quả trên robot thực tế.

## 10. Định hướng cuối kỳ (Future Work)

Từ mô hình phân loại cơ sở, nhóm dự kiến phát triển bài toán Object Detection nhằm phát hiện vị trí, phân loại và thống kê vật thể rác thải trong ảnh/video, phục vụ mô phỏng hệ thống hỗ trợ robot thu gom.