<!--
Tiêu đề PR: <type>(s<N>-<area>): <mô tả ngắn>
Ví dụ:     feat(s2-data): build grouped train/val/test split
Reviewer:  @phanyen06ym-ien (bắt buộc) · Labels + milestone giống issue
-->

## 1. Tóm tắt

<!-- 1–3 câu: PR này làm gì và vì sao. -->

## 2. Issue liên quan

<!-- Closes #n nếu hoàn thành toàn bộ issue; Refs #n nếu chỉ một phần. -->
Closes #

## 3. Thay đổi

| File | Thay đổi |
|---|---|
| | |

## 4. Cách kiểm tra

<!-- Lệnh / notebook / cell để reviewer chạy lại. Ghi rõ DATA_VERSION, EXP_ID, đường dẫn Drive nếu có. -->
```bash
pytest -q
```

## 5. Checklist của tác giả

- [ ] Tên nhánh `<type>/s<N>-<area>-<mo-ta>` và tiêu đề PR đúng quy ước.
- [ ] Đã cập nhật từ `main` mới nhất, không xung đột.
- [ ] Code/test chạy qua; notebook chạy lại được từ đầu trên Colab.
- [ ] Không commit dữ liệu, checkpoint, file > 5 MB, token hay đường dẫn cá nhân.
- [ ] Số liệu báo cáo có nguồn (run trên Drive / file trong `results/metrics/`).
- [ ] Tài liệu liên quan đã cập nhật (README, `docs/`).
- [ ] Đã gắn labels, milestone; reviewer @phanyen06ym-ien.

## 6. Bằng chứng / kết quả

<!-- Ảnh chụp output, bảng số liệu, đường dẫn file kết quả. Ghi "Không áp dụng" nếu PR chỉ là tài liệu. -->

## 7. Ghi chú cho reviewer

<!-- Điểm cần Yến tập trung xem, quyết định còn mở, rủi ro. -->

---
<!-- Phần dành cho Yến khi review -->
**Review (Yến):** ☐ Đúng quy ước ☐ Đủ DoD của issue ☐ Approve
