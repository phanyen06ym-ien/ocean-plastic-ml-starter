# Hướng dẫn chạy trên Google Colab

## 1. Chuẩn bị (một lần)

1. **Drive:** tạo thư mục `MyDrive/ocean-plastic-ml/` (notebook tự tạo `data/raw`, `data/processed`, `runs`). Nếu làm chung, Yến chia sẻ thư mục này cho cả nhóm và mỗi người *Add shortcut to Drive* vào `MyDrive/`.
2. **Dataset:** tải `dataset.zip` của TrashCan 1.0 từ [UMN Data Repository](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7), đặt vào `MyDrive/ocean-plastic-ml/data/raw/`.
3. **Token GitHub (chỉ người đẩy kết quả):** GitHub → Settings → Developer settings → Fine-grained token, quyền *Contents: Read and write* cho repo này. Trong Colab: biểu tượng 🔑 *Secrets* → thêm `GH_TOKEN`, bật *Notebook access*. Không dán token vào cell.

## 2. Chạy pipeline Sprint 2

1. Mở Colab → *File → Open notebook → GitHub* → `phanyen06ym-ien/ocean-plastic-ml-starter` → `notebooks/00_pipeline_colab.ipynb`.
2. *Runtime → Change runtime type → T4 GPU*.
3. Chạy lần lượt từng cell (hoặc *Runtime → Run all*). Notebook tự clone repo, giải nén dữ liệu, tiền xử lý, kiểm thử, huấn luyện E01/E02, đánh giá Validation, demo và đẩy kết quả lên nhánh `exp/s2-run-results-*`.
4. Mở link PR in ra ở cell cuối, điền template, reviewer **@phanyen06ym-ien**.

| Bước | Lưu ở đâu |
|---|---|
| Dữ liệu đã xử lý | Drive `data/processed/v1_crop128.zip` (lần sau dùng lại, không xử lý lại) |
| Checkpoint, history, config | Drive `runs/<EXP_ID>_<YYYYMMDD-HHMM>/` |
| Chỉ số, hình, báo cáo | Repo: `results/metrics/`, `results/figures/`, `docs/reports/sprint2_results.md` |

## 3. Khi Colab ngắt kết nối

- Tiền xử lý đã lưu `.zip` trên Drive → chạy lại cell 3 sẽ giải nén thay vì xử lý lại.
- Huấn luyện dở: `run_training(cfg, PROCESSED, RUNS, run_dir=RUNS/"E01_<thời gian>", resume=True)` chạy tiếp từ `last.keras`.

## 4. Làm việc với notebook khác

- Notebook chỉ **gọi hàm trong `src/`**, không chép code vào cell.
- Lưu notebook bằng *File → Save a copy in GitHub* và **chọn nhánh của mình**, không chọn `main`.

## 5. Chạy trên máy cá nhân (VS Code)

```bash
pip install -r requirements.txt
pytest -q                                   # test TensorFlow tự bỏ qua nếu chưa cài
python -m src.preprocessing --raw-root <thư-mục-trashcan> --out-dir data/processed/v1_crop128 --mapping-out configs/class_mapping.json
python main.py --image <ảnh.jpg> --model <runs/.../best.keras>
```
