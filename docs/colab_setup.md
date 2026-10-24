# Hướng Dẫn Thiết Lập & Chạy Trên Google Colab
## Dự Án: Phân Loại Rác Thải Nhựa Môi Trường Biển (TrashCan 1.0 — Instance)

Tài liệu này hướng dẫn cách thiết lập môi trường và chạy các Jupyter Notebooks (đặc biệt là `notebooks/04_evaluation.ipynb`) trên **Google Colab**.

---

## 1. Khởi Động Notebook Trên Google Colab

1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Chọn thẻ **GitHub**, nhập URL repository:
   ```
   https://github.com/phanyen06ym-ien/ocean-plastic-ml-starter
   ```
3. Chọn nhánh phát triển (ví dụ: `feat/sprint1-evaluation-pipeline` hoặc `main`).
4. Mở notebook cần làm việc: `notebooks/04_evaluation.ipynb`.

---

## 2. Thiết Lập Môi Trường Làm Việc

Trong cell đầu tiên của notebook trên Colab, thực hiện clone repository và cài đặt các phụ thuộc:

```python
# 1. Clone repository nếu chưa có
import os
if not os.path.exists("ocean-plastic-ml-starter"):
    !git clone https://github.com/phanyen06ym-ien/ocean-plastic-ml-starter.git
    %cd ocean-plastic-ml-starter
else:
    %cd ocean-plastic-ml-starter

# 2. Cài đặt các thư viện cần thiết
!pip install -r requirements.txt -q

# 3. Đảm bảo sys.path trỏ đến project root
import sys
if "." not in sys.path:
    sys.path.insert(0, ".")
```

---

## 3. Cấu Hình Tăng Tốc Phần Cứng (Hardware Accelerator)

Khi thực hiện huấn luyện mô hình CNN hoặc trích xuất đặc trưng quy mô lớn:
1. Vào menu **Runtime** (Thời gian chạy) -> **Change runtime type** (Thay đổi loại thời gian chạy).
2. Chọn **Hardware accelerator:** **T4 GPU** (hoặc GPU phù hợp).
3. Nhấn **Save**.

Đối với module Đánh giá (`04_evaluation.ipynb`):
- Module đánh giá sử dụng các phép tính đại số và trực quan hóa (NumPy, Scikit-learn, Matplotlib), có thể chạy mượt mà ngay cả trên CPU thông thường của Colab.

---

## 4. Lưu Trữ Kết Quả Vào Google Drive (Tùy Chọn)

Để lưu lại các biểu đồ và báo cáo đánh giá vào Google Drive cá nhân:

```python
from google.colab import drive
drive.mount('/content/drive')

# Sao chép kết quả sang Drive
!cp -r results /content/drive/MyDrive/ocean-plastic-results/
```

---

## 5. Danh Sách Các Notebook Trong Dự Án

| Notebook | Tên Công Việc | Mục Đích |
|---|---|---|
| `notebooks/01_trashcan_eda.ipynb` | S1-DATA | Khám phá dữ liệu, phân tích phân bố lớp, độ phân giải ảnh. |
| `notebooks/02_preprocessing.ipynb` | S1-DATA | Tiền xử lý dữ liệu, crop bounding box, resize 224x224. |
| `notebooks/03_cnn_training.ipynb` | S1-CNN | Huấn luyện kiến trúc CNN Baseline và lưu checkpoints. |
| `notebooks/04_evaluation.ipynb` | S1-EVAL | Đánh giá mô hình, tính metrics, vẽ CM, Learning Curves, phân tích lỗi. |
