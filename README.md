# Ocean Plastic Waste Classification and Monitoring

## 1. Giới thiệu

Dự án Phân loại và giám sát rác thải nhựa đại dương thuộc môn Học máy và Khai phá dữ liệu.

Mục tiêu là nghiên cứu và xây dựng mô hình mạng nơ-ron tích chập (CNN) để phân loại hình ảnh rác thải nhựa, hướng tới hỗ trợ hệ thống giám sát và robot thu gom rác trong môi trường nước.

## 2. Mục tiêu giữa kỳ

- Khảo sát và lựa chọn bộ dữ liệu ảnh có nhãn.
- Phân tích dữ liệu khám phá (EDA).
- Tiền xử lý dữ liệu ảnh.
- Xây dựng và huấn luyện CNN baseline.
- Đánh giá mô hình bằng các chỉ số phù hợp.
- Thử nghiệm dự đoán trên ảnh mới.

## 3. Quy trình học máy

Dataset → EDA → Preprocessing → CNN → Training → Validation → Evaluation → Prediction

## 4. Công nghệ dự kiến

- Python
- NumPy, Pandas, Matplotlib
- TensorFlow / Keras
- Scikit-learn
- Git và GitHub

## 5. Cấu trúc dự án

- `data/`: dữ liệu gốc và dữ liệu đã xử lý.
- `src/eda.py`: phân tích dữ liệu.
- `src/preprocessing.py`: tiền xử lý.
- `src/train.py`: xây dựng và huấn luyện CNN.
- `src/predict.py`: dự đoán ảnh.
- `models/`: lưu checkpoint mô hình.
- `results/`: lưu kết quả thực nghiệm.
- `docs/`: tài liệu dự án.

## 6. Phân công nhóm

| Thành viên | Vai trò |
|---|---|
| TV1 | Leader, Problem Analysis, Documentation |
| TV2 | Dataset, EDA, Preprocessing |
| TV3 | CNN Architecture, Training |
| TV4 | Evaluation, Optimization |
| TV5 | Inference, Demo, Testing |

## 7. Trạng thái phát triển

**Sprint 1 — Foundation:** Đang triển khai.

Hiện tại project đã được khởi tạo trên GitHub. Các module học máy đang trong quá trình phát triển, chưa có kết quả huấn luyện chính thức.

## 8. Định hướng cuối kỳ

- Phát hiện và định vị rác thải bằng Object Detection.
- Nghiên cứu cải thiện ảnh dưới nước bằng GAN nếu phù hợp.
- Tối ưu mô hình nhẹ để hỗ trợ robot.
- Phát triển demo giám sát rác qua ảnh/video.

## 9. Ghi chú

Danh sách nhãn, số lượng dữ liệu và các kết quả thực nghiệm sẽ được cập nhật sau khi hoàn thành kiểm kê dataset và huấn luyện mô hình.