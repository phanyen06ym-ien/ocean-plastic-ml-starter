# Ảnh demo (tập Kiểm tra, mô hình E04)

Tạo bằng `python scripts/make_demo_images.py` (chạy trên Colab, 25/11/2026). Ba ảnh đầu mô hình đoán đúng, hai ảnh sau đoán sai.

| Ảnh | Lớp thật | Mô hình đoán | Độ tin cậy | Ghi chú |
|---|---|---|---|---|
| `1_dung_sao_bien.jpg` | animal_starfish | animal_starfish | 99,9% | Lớp có F1 cao nhất (0,89): hình dạng đặc trưng. |
| `2_dung_tui_nilon.jpg` | trash_bag | trash_bag | 99,1% | Lớp nhựa mô hình nhận dạng tốt nhất (F1 0,34). |
| `3_dung_lon.jpg` | trash_can | trash_can | 83,2% | Rác kim loại, hình dạng rõ (F1 0,49). |
| `4_sai_vo_bao_bi_thanh_tui_nilon.jpg` | trash_snack_wrapper | trash_bag | 86,9% | Cả 10 ảnh vỏ bao bì trong tập Kiểm tra đều bị đoán thành túi nilon: hai lớp cùng là màng nhựa mỏng, sáng màu. |
| `5_sai_chai_thanh_rac_khac.jpg` | trash_bottle | trash_other | 55,4% | Lỗi phổ biến nhất của nhóm nhựa: lớp gộp trash_other hút nhầm (11/17 ảnh chai). |
