"""Giao diện demo: tải ảnh bất kỳ → mô hình cuối E04 dự đoán loại đối tượng dưới nước.

Chạy:
    pip install -r requirements.txt
    python demo_app.py                                   # mở http://127.0.0.1:7860
    python demo_app.py --model runs/E04_20261122-0853/best.pt --port 7860
    python demo_app.py --share                           # tạo link công khai tạm thời (cần Internet)

Mô hình phân loại ảnh crop của MỘT đối tượng. Với ảnh chụp cả cảnh, dùng công cụ cắt (crop) trong khung ảnh
để khoanh vùng quanh vật thể trước khi bấm "Nhận dạng". Logic suy luận dùng chung với main.py (src/predict.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from src.predict import InferenceError, Predictor
    from src.preprocessing import ensure_utf8_stdio
except ImportError:  # pragma: no cover
    from predict import InferenceError, Predictor
    from preprocessing import ensure_utf8_stdio

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "runs" / "E04_20261122-0853" / "best.pt"
EXAMPLES_DIR = ROOT / "data" / "processed" / "v2_crop224" / "test"
LOW_CONFIDENCE = 0.40

VI_NAMES = {
    "rov": "ROV (thiết bị lặn)", "plant": "Thực vật", "animal_fish": "Cá", "animal_starfish": "Sao biển",
    "animal_other": "Động vật khác", "animal_crab": "Cua", "animal_eel": "Lươn biển",
    "animal_shells": "Vỏ sò, ốc", "animal_etc": "Động vật khác",
    "trash_other": "Rác khác", "trash_unknown_instance": "Rác không xác định", "trash_pipe": "Ống",
    "trash_bottle": "Chai", "trash_bag": "Túi nilon", "trash_snack_wrapper": "Vỏ bao bì thực phẩm",
    "trash_can": "Lon", "trash_container": "Thùng, hộp chứa", "trash_branch": "Cành cây",
    "trash_tarp": "Tấm bạt", "trash_rope": "Dây thừng", "trash_net": "Lưới",
    "trash_cup": "Cốc", "trash_clothing": "Quần áo", "trash_wreckage": "Mảnh vỡ, xác tàu",
}


def label(class_name: str) -> str:
    """Tên hiển thị: 'Túi nilon (trash_bag)'."""
    vi = VI_NAMES.get(class_name)
    return f"{vi} ({class_name})" if vi else class_name


def to_pil(value: Any) -> Image.Image | None:
    """Nhận giá trị từ gr.Image / gr.ImageEditor (PIL, numpy hoặc dict có 'composite') → ảnh PIL RGB."""
    if value is None:
        return None
    if isinstance(value, dict):  # gr.ImageEditor
        value = value.get("composite") or value.get("background")
        if value is None:
            return None
    if not isinstance(value, Image.Image):
        value = Image.fromarray(value)
    return value.convert("RGB")


def center_square(img: Image.Image) -> Image.Image:
    """Cắt vùng vuông ở giữa ảnh, giống cửa sổ vuông khi tạo dữ liệu v2 (tránh méo khi resize)."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def prepare(value: Any, square: bool) -> Image.Image | None:
    """Ảnh đưa vào mô hình: RGB, tuỳ chọn cắt vuông ở giữa."""
    img = to_pil(value)
    if img is None:
        return None
    return center_square(img) if square else img


def summarize(preds: list, plastic_names: set[str] | None = None) -> str:
    """Đoạn Markdown kết luận cho top-1: tên lớp, độ tin cậy, có thuộc nhóm nhựa không, cảnh báo khi không chắc."""
    top = preds[0]
    lines = [f"### {label(top.class_name)}", f"Độ tin cậy: **{top.confidence:.1%}**"]
    lines.append("🟠 Thuộc nhóm **rác liên quan nhựa**." if top.plastic_related
                 else "Không thuộc nhóm 6 lớp liên quan nhựa.")
    if top.confidence < LOW_CONFIDENCE:
        lines.append(f"⚠️ Mô hình **không chắc chắn** (dưới {LOW_CONFIDENCE:.0%}). "
                     "Thử cắt sát vật thể hơn hoặc dùng ảnh rõ hơn.")
    return "\n\n".join(lines)


def make_predict_fn(predictor: Predictor):
    """Hàm callback cho Gradio: (ảnh, cắt vuông) → (ảnh đã xử lý, nhãn top-5, kết luận)."""
    n = len(predictor.mapping)

    def predict(value, square=True):
        img = prepare(value, square)
        if img is None:
            return None, {}, "Hãy tải lên một ảnh."
        try:
            preds = predictor.predict_image(img, k=n)
        except InferenceError as e:
            return None, {}, f"❌ {e}"
        size = predictor.input_size
        shown = img.resize((size, size), Image.BILINEAR)
        return shown, {label(p.class_name): float(p.confidence) for p in preds}, summarize(preds)

    return predict


def example_images(folder: Path = EXAMPLES_DIR, per_class: int = 1, limit: int = 12) -> list[list[str]]:
    """Một vài ảnh Test làm ví dụ (ưu tiên các lớp nhựa); rỗng nếu chưa tải dữ liệu."""
    if not folder.exists():
        return []
    order = ["trash_bag", "trash_net", "trash_rope", "trash_tarp", "trash_bottle", "trash_snack_wrapper",
             "animal_fish", "animal_starfish", "rov", "trash_can", "animal_crab", "trash_pipe"]
    out = []
    for cls in order:
        files = sorted((folder / cls).glob("*.jpg"))[:per_class] if (folder / cls).exists() else []
        out += [[str(f)] for f in files]
    return out[:limit]


def build_app(predictor: Predictor, model_name: str):
    """Tạo giao diện Gradio."""
    import gradio as gr

    predict = make_predict_fn(predictor)
    with gr.Blocks(title="Demo phân loại rác thải dưới nước") as app:
        gr.Markdown(
            "# Phân loại rác thải nhựa dưới nước\n"
            f"Mô hình: **{model_name}** · {len(predictor.mapping)} lớp · ảnh đầu vào "
            f"{predictor.input_size} × {predictor.input_size}.\n\n"
            "Tải ảnh lên (hoặc chụp bằng webcam, dán từ clipboard). Mô hình nhận dạng **một** đối tượng: "
            "với ảnh có nhiều vật, dùng công cụ **cắt** trong khung ảnh để khoanh vùng quanh vật thể.")
        with gr.Row():
            with gr.Column():
                inp = gr.ImageEditor(type="pil", label="Ảnh đầu vào", sources=["upload", "webcam", "clipboard"],
                                     transforms=["crop"], layers=False, brush=False, eraser=False, height=420)
                square = gr.Checkbox(value=True, label="Cắt vuông ở giữa (giống dữ liệu huấn luyện)")
                with gr.Row():
                    btn = gr.Button("Nhận dạng", variant="primary")
                    gr.ClearButton([inp])
            with gr.Column():
                summary = gr.Markdown()
                probs = gr.Label(num_top_classes=5, label="Top 5 lớp")
                shown = gr.Image(label="Ảnh mô hình nhìn thấy", height=224, interactive=False)
        btn.click(predict, [inp, square], [shown, probs, summary])
        examples = example_images()
        if examples:
            ex_img = gr.Image(type="pil", visible=False)
            gr.Examples(examples, inputs=[ex_img], label="Ảnh ví dụ từ tập Kiểm tra")
            ex_img.change(lambda v: v, ex_img, inp)
        gr.Markdown("Lưu ý: đồ án môn học, kết quả mang tính minh hoạ. Macro F1 trên tập Kiểm tra: 0,329; "
                    "nhóm lớp nhựa còn yếu (0,132).")
    return app


def main(argv: list[str] | None = None) -> None:
    """Đọc tham số dòng lệnh, nạp mô hình một lần rồi mở giao diện."""
    ensure_utf8_stdio()            # thông báo tiếng Việt / emoji không lỗi trên console Windows
    ap = argparse.ArgumentParser(description="Giao diện demo nhận dạng ảnh rác dưới nước (Gradio)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="checkpoint .pt/.keras (mặc định: E04)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="tạo link công khai tạm thời")
    args = ap.parse_args(argv)
    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"Không tìm thấy checkpoint {model}. Tải thư mục runs/E04_20261122-0853/ từ Drive "
                         "(MyDrive/ocean-plastic-ml/runs/) vào thư mục runs/ của repo.")
    try:
        import gradio  # noqa: F401
    except ImportError:
        raise SystemExit("Chưa cài gradio. Chạy: pip install -r requirements.txt")
    predictor = Predictor(model)
    build_app(predictor, model.parent.name).launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
