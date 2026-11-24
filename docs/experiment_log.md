# Sổ thí nghiệm

Mỗi run một dòng. Tất cả chỉ số trong bảng là trên **Validation** (không dùng Test). Macro F1 tổng thể tính trên 22 lớp; Macro F1 nhóm nhựa tính trên 6 lớp liên quan nhựa theo ADR-001. Các số liệu của các run trước E03G được giữ theo kết quả đã ghi trong kế hoạch thực nghiệm. E03G được bổ sung từ log huấn luyện do người chạy cung cấp; cần đối chiếu với `results/metrics/E03G_val.json` nếu file metric chuẩn được xuất/merge.

## 1. Bảng tổng hợp

| EXP_ID | Ngày | Người chạy | Môi trường | Dữ liệu | Seed | Thay đổi so với mốc | Macro F1 val | Accuracy val | Macro F1 nhóm nhựa val | Ghi chú |
|---|---|---|---|---|---:|---|---:|---:|---:|---|
| E01 | 05/11/2026 | Yến | Colab T4, TF 2.21 | v1_crop128 | 42 | CNN baseline + class weights | 0,0757 | 0,2190 | 0,0000 | best epoch 7 |
| E02 | 05/11/2026 | Yến | Colab T4, TF 2.21 | v1_crop128 | 42 | E01 + augmentation | 0,0648 | 0,1610 | 0,0165 | H2 không được xác nhận |
| E03 | 08/11/2026 | Bách | CPU local, PyTorch | v1_crop128 | 42 | MobileNetV2 ImageNet, huấn luyện 2 pha | **0,2034** | 0,4052 | 0,1228 | Chọn theo `val_loss`; pha 1, epoch 2 |
| E03a | 10/11/2026 | Bách | CPU, PyTorch | v1_crop128 | 42 | Dropout 0,6 (E03: 0,4) | 0,2099 | — | 0,1377 | Chọn theo `val_loss`; pha 1, epoch 2 |
| E03c | 10/11/2026 | Bách | CPU, PyTorch | v1_crop128 | 42 | Learning rate pha 1 = 3e-4 | 0,1955 | — | 0,1083 | Chọn theo `val_loss`; pha 1, epoch 6 |
| E03d | 10/11/2026 | Bách | CPU, PyTorch | v1_crop128 | 42 | Learning rate backbone pha 2 = 3e-5 | 0,2034 | — | 0,1228 | Checkpoint theo `val_loss` vẫn ở pha 1, epoch 2 |
| E03f | 10/11/2026 | Bách | CPU, PyTorch | v1_crop128 | 43 | Seed 43; cấu hình như E03 | 0,2122 | — | 0,1168 | Dùng để ước lượng biến động theo seed |
| E03d-F1 | 10/11/2026 | Bách | CPU, PyTorch | v1_crop128 | 42 | E03d, checkpoint chọn theo Macro F1 | 0,2398 | 0,4994 | 0,1076 | `bestf1_phase2.pt`, pha 2 epoch 6 |
| E03f-F1 | 10/11/2026 | Bách | CPU, PyTorch | v1_crop128 | 43 | E03f, checkpoint chọn theo Macro F1 | 0,2372 | 0,4826 | 0,1357 | `bestf1_phase2.pt`, pha 2 epoch 9 |
| E03G | 24/11/2026 | Bách | CPU local, PyTorch | v1_crop128 | 42 | E03 với thời lượng huấn luyện mở rộng; patience 12 | **0,2034** | **0,4052** | **0,1228** | Checkpoint chọn theo `val_loss`; kết quả cuối trùng E03 ở độ chính xác báo cáo |

`—` nghĩa là chỉ số không có trong tài liệu kết quả đang dùng; không tự suy ra hoặc điền giá trị.

## 2. Ghi chú E03

- Pha 1 (đóng băng backbone) dừng ở epoch 8; `val_loss` tốt nhất tại epoch 2 (2,205).
- Pha 2 (mở từ block 14, learning rate backbone/head tương ứng 1e-5/1e-4) dừng ở epoch 7; `val_loss` tốt nhất tại epoch 1 (2,275). Theo quy tắc chọn checkpoint bằng `val_loss`, `best.pt` là checkpoint pha 1.
- Macro F1 Validation của pha 2 cao hơn: 0,219 tại epoch 1 và tối đa 0,233 tại epoch 7. Tuy nhiên `val_loss` không tốt hơn, vì vậy checkpoint pha 2 không được chọn theo quy tắc hiện tại.
- So với E01, E03 đồng thời thay framework Keras sang PyTorch, dùng chuẩn hóa ImageNet mean/std và thay đầu phân loại. Các yếu tố cố định gồm split, seed 42, class weights `balanced`, không augmentation, batch 64 và ảnh 128×128.
- Thời gian suy luận E03 đo trên CPU 4 luồng, batch 1: 32,4 ms/ảnh. Cần đo E01 trong cùng điều kiện trước khi so sánh tốc độ.
- Checkpoint `*.pt` (khoảng 9 MB) không commit lên GitHub; lưu trên Drive.

## 3. Ghi chú các run tinh chỉnh

- E03a tăng dropout từ 0,4 lên 0,6; Macro F1 Validation tăng nhẹ từ 0,2034 lên 0,2099.
- E03c giảm learning rate pha 1 từ 1e-3 xuống 3e-4; Macro F1 Validation là 0,1955.
- E03d tăng learning rate backbone pha 2 lên 3e-5; kết quả checkpoint theo `val_loss` vẫn lấy từ pha 1, nên thay đổi learning rate pha 2 không ảnh hưởng checkpoint được chọn.
- E03f giữ cấu hình E03 nhưng đổi seed từ 42 sang 43; Macro F1 Validation tăng từ 0,2034 lên 0,2122. Chênh lệch 0,009 cho thấy cần thận trọng khi diễn giải các cải thiện nhỏ giữa các cấu hình.
- E03d-F1 và E03f-F1 là các kết quả đánh giá bổ sung khi chọn checkpoint theo Macro F1 Validation, không phải checkpoint được chọn theo quy tắc `val_loss` ban đầu.

## 4. Ghi chú E03G — Extended Training

**Cấu hình**
- Pha 1: tối đa 30 epoch; Early Stopping patience 12.
- Pha 2: tối đa 40 epoch; Early Stopping patience 12.
- Theo dõi `val_loss`, mode `min`, bật `restore_best_weights`.
- Giữ cấu hình lõi E03: MobileNetV2 pretrained ImageNet, `v1_crop128`, input 128×128, 22 lớp, batch size 64, seed 42.
- `num_workers: 0` để tránh lỗi multiprocessing đã gặp trên Windows.

**Diễn biến**
- Pha 1 dừng tại epoch 14; epoch tốt nhất theo `val_loss` là epoch 2.
- Pha 2 dừng tại epoch 13; epoch tốt nhất theo `val_loss` là epoch 1.
- Trong pha 2, training loss giảm từ 0,6157 (epoch 1) xuống 0,2312 (epoch 13), trong khi validation loss tăng từ 2,3028 lên 2,5321 sau nhiều dao động.
- Macro F1 Validation cao nhất quan sát được trong log pha 2 là 0,236 tại epoch 9; accuracy cao nhất quan sát được là 0,470 tại epoch 10. Đây là các giá trị theo từng epoch trong log, không phải chỉ số của checkpoint cuối được chọn theo `val_loss`.

**Chỉ số checkpoint cuối được báo cáo**
- Validation Macro F1: 0,2034098 (xấp xỉ 0,2034).
- Validation Accuracy: 0,4051627 (xấp xỉ 0,4052).
- Plastic6 Validation Macro F1: 0,1227852 (xấp xỉ 0,1228).
- Tổng tham số: 2.252.054; tham số trainable ở pha 2: 1.709.526.
- Kích thước checkpoint: 9,23 MB.
- CPU inference, batch 1: 31,396 ms/ảnh; `torch_threads=4`.

**Kết luận**
E03G kiểm tra giả thuyết rằng kéo dài thời gian huấn luyện và tăng patience có thể cải thiện E03. Với checkpoint cuối được chọn theo `val_loss`, E03G đạt các chỉ số tương đương E03. Training loss tiếp tục giảm trong khi validation loss nhìn chung tăng ở pha fine-tuning, là dấu hiệu overfitting. Mặc dù log pha 2 ghi nhận Macro F1 tối đa 0,236, epoch này không phải epoch có `val_loss` tốt nhất. Do đó không được ghi 0,236 như Macro F1 của checkpoint E03G cuối cùng. E03G không chứng minh được lợi ích của việc kéo dài lịch huấn luyện.

## 5. Các điểm cần xác nhận / giới hạn

- Quy tắc chọn checkpoint đang có điểm chưa thống nhất: `experiment_plan.md` mô tả chọn theo `val_loss`, trong khi kế hoạch đánh giá và bảng ứng viên có sử dụng Macro F1 Validation. Reviewer cần xác nhận tiêu chí chính thức trước khi chốt checkpoint bàn giao.
- Các kết quả E03d-F1/E03f-F1 cho thấy checkpoint chọn theo Macro F1 có thể khác đáng kể với checkpoint chọn theo `val_loss`; cần ghi rõ tiêu chí mỗi khi báo cáo.
- Kết quả E03G được nhập từ log huấn luyện và JSON tổng kết do người chạy cung cấp. Nếu pipeline xuất file metric chuẩn, cần kiểm tra và đồng bộ `results/metrics/E03G_val.json`, đường dẫn run và checkpoint thực tế.
- Không dùng tập Test để chọn epoch, siêu tham số hoặc checkpoint. Tập Test chỉ đánh giá một lần sau khi nhóm chốt mô hình cuối.
