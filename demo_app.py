"""Giao diện demo: tải ảnh bất kỳ → mô hình cuối E04 dự đoán loại đối tượng dưới nước.

Chạy:
    pip install -r requirements.txt
    python demo_app.py                                   # mở http://127.0.0.1:7860
    python demo_app.py --model runs/E04_20261122-0853/best.pt --port 7860
    python demo_app.py --share                           # tạo link công khai tạm thời (cần Internet)

Mô hình phân loại ảnh crop của MỘT đối tượng. Với ảnh chụp cả cảnh, nên cắt sát quanh vật thể trước khi tải lên. Logic suy luận dùng chung với main.py (src/predict.py).
"""
from __future__ import annotations

import argparse
import html
import time
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

DESCRIPTIONS = {
    "rov": "Thiết bị lặn điều khiển từ xa hoặc một phần của nó (càng gắp, khung, đèn).",
    "plant": "Rong, tảo hoặc thực vật dưới nước.",
    "animal_fish": "Cá.", "animal_starfish": "Sao biển.", "animal_crab": "Cua.", "animal_eel": "Lươn biển.",
    "animal_other": "Sinh vật biển khác (vỏ sò, ốc, mực, sinh vật đáy…).",
    "trash_other": "Rác không xếp được vào lớp cụ thể (lớp gộp: rác không rõ loại, cốc, quần áo, mảnh vỡ).",
    "trash_pipe": "Đoạn ống, thường bằng nhựa hoặc kim loại.",
    "trash_bottle": "Chai, thường là chai nhựa hoặc thuỷ tinh.",
    "trash_bag": "Túi nilon, túi nhựa.",
    "trash_snack_wrapper": "Vỏ bao bì thực phẩm, thường là màng nhựa mỏng.",
    "trash_can": "Lon kim loại.", "trash_container": "Thùng, hộp, can chứa.",
    "trash_branch": "Cành cây, gỗ trôi dạt.",
    "trash_tarp": "Tấm bạt, tấm nhựa lớn.", "trash_rope": "Dây thừng, dây buộc.", "trash_net": "Lưới đánh cá, lưới.",
}


def group_of(class_name: str, plastic_related: bool) -> tuple[str, str]:
    """Nhóm đối tượng và màu hiển thị: rác liên quan nhựa / rác khác / sinh vật / thực vật / thiết bị."""
    if plastic_related:
        return "Rác liên quan nhựa", "#e8590c"
    if class_name.startswith("trash"):
        return "Rác khác", "#a61e4d"
    if class_name.startswith("animal"):
        return "Sinh vật biển", "#1971c2"
    if class_name == "plant":
        return "Thực vật", "#2f9e44"
    return "Thiết bị (ROV)", "#495057"


def label(class_name: str) -> str:
    """Tên hiển thị: 'Túi nilon (trash_bag)'."""
    vi = VI_NAMES.get(class_name)
    return f"{vi} ({class_name})" if vi else class_name


def to_pil(value: Any) -> Image.Image | None:
    """Nhận giá trị từ gr.Image / gr.ImageEditor (PIL, numpy hoặc dict có 'composite') → ảnh PIL RGB."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):  # đường dẫn file (nút chọn ảnh, ảnh ví dụ)
        with Image.open(value) as im:
            im.load()
            return im.convert("RGB")
    if isinstance(value, dict):  # gr.ImageEditor
        composite = value.get("composite")              # không dùng `or`: mảng numpy không có giá trị đúng/sai
        value = composite if composite is not None else value.get("background")
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


PLACEHOLDER = ("<div style='padding:28px;text-align:center;opacity:.7'>Kết quả nhận dạng sẽ hiện ở đây.<br>"
               "Chọn hoặc kéo thả một ảnh vào khung bên trái.</div>")


def _pct(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",") + "%"


def summarize(preds: list, original_size: tuple[int, int] | None = None, elapsed_ms: float | None = None) -> str:
    """Thẻ HTML thông tin nhận dạng: tên lớp, nhóm, độ tin cậy, mô tả, lớp đứng thứ hai, cảnh báo khi không chắc."""
    top = preds[0]
    group, color = group_of(top.class_name, top.plastic_related)
    vi = html.escape(VI_NAMES.get(top.class_name, top.class_name))
    rows = [("Tên lớp gốc", f"<code>{html.escape(top.class_name)}</code>"),
            ("Nhóm", f"<b style='color:{color}'>{group}</b>"),
            ("Mô tả", html.escape(DESCRIPTIONS.get(top.class_name, "")))]
    if len(preds) > 1:
        rows.append(("Khả năng thứ hai", f"{html.escape(label(preds[1].class_name))} – {_pct(preds[1].confidence)}"))
    if original_size:
        rows.append(("Kích thước ảnh gốc", f"{original_size[0]} × {original_size[1]} px"))
    if elapsed_ms is not None:
        rows.append(("Thời gian xử lý", f"{elapsed_ms:.0f} ms"))
    table = "".join(f"<tr><td style='padding:4px 12px 4px 0;opacity:.7;white-space:nowrap'>{k}</td>"
                    f"<td style='padding:4px 0'>{v}</td></tr>" for k, v in rows)
    bar = (f"<div style='background:rgba(127,127,127,.25);border-radius:6px;height:12px;margin:6px 0 2px'>"
           f"<div style='width:{top.confidence * 100:.1f}%;background:{color};height:12px;border-radius:6px'></div></div>")
    plastic = ("<div style='margin-top:10px;padding:8px 12px;border-radius:8px;background:rgba(232,89,12,.12)'>"
               "🟠 Thuộc nhóm <b>rác liên quan nhựa</b>.</div>" if top.plastic_related else
               "<div style='margin-top:10px;opacity:.75'>Không thuộc nhóm 6 lớp liên quan nhựa.</div>")
    warn = ""
    if top.confidence < LOW_CONFIDENCE:
        warn = ("<div style='margin-top:8px;padding:8px 12px;border-radius:8px;background:rgba(240,140,0,.15)'>"
                f"⚠️ Mô hình <b>không chắc chắn</b> (độ tin cậy dưới {LOW_CONFIDENCE:.0%}). "
                "Thử cắt sát vật thể hơn hoặc dùng ảnh rõ hơn.</div>")
    return ("<div style='padding:16px 18px;border:1px solid rgba(127,127,127,.3);border-radius:12px'>"
            "<div style='font-size:13px;opacity:.7'>KẾT QUẢ NHẬN DẠNG</div>"
            f"<div style='font-size:30px;font-weight:700;color:{color};line-height:1.25'>{vi}</div>"
            f"{bar}<div style='font-size:15px'>Độ tin cậy: <b>{_pct(top.confidence)}</b></div>"
            f"<table style='margin-top:10px;font-size:14px'>{table}</table>{plastic}{warn}</div>")


def make_predict_fn(predictor: Predictor):
    """Hàm callback cho Gradio: (ảnh, cắt vuông) → (ảnh mô hình nhìn thấy, xác suất các lớp, thẻ thông tin HTML)."""
    n = len(predictor.mapping)

    def predict(value, square=True):
        img = prepare(value, square)
        if img is None:
            return None, {}, "Hãy tải lên một ảnh."
        original = to_pil(value).size
        t0 = time.perf_counter()
        try:
            preds = predictor.predict_image(img, k=n)
        except InferenceError as e:
            return None, {}, f"❌ {e}"
        elapsed = (time.perf_counter() - t0) * 1000
        size = predictor.input_size
        shown = img.resize((size, size), Image.BILINEAR)
        return (shown, {label(p.class_name): float(p.confidence) for p in preds},
                summarize(preds, original, elapsed))

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
    """Tạo giao diện Gradio: cột trái nhập ảnh và nút bấm, cột phải thẻ kết quả, top-5 và ảnh mô hình nhìn thấy."""
    import gradio as gr

    predict = make_predict_fn(predictor)
    size = predictor.input_size
    with gr.Blocks(title="Demo phân loại rác thải dưới nước") as app:
        gr.Markdown(
            "# 🌊 Phân loại rác thải nhựa dưới nước\n"
            f"Mô hình **{model_name}** (EfficientNet-B0) · {len(predictor.mapping)} lớp · ảnh {size} × {size}. "
            "Chọn một ảnh **chứa một đối tượng** (ảnh dưới nước cho kết quả tốt nhất); "
            "ảnh có nhiều vật nên được cắt sát quanh vật thể trước khi tải lên.")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                inp = gr.Image(type="pil", label="Ảnh đầu vào", sources=["upload", "webcam", "clipboard"],
                               height=340)
                with gr.Row():
                    upload = gr.UploadButton("📁 Chọn ảnh từ máy", file_types=["image"], variant="secondary")
                    btn = gr.Button("🔍 Nhận dạng", variant="primary")
                    clear = gr.Button("🗑️ Xoá")
                square = gr.Checkbox(value=True, label="Cắt vuông ở giữa ảnh (giống dữ liệu huấn luyện)")
                examples = example_images()
                if examples:
                    gr.Examples(examples, inputs=[inp], label="Ảnh ví dụ từ tập Kiểm tra (bấm để thử)",
                                examples_per_page=12)
            with gr.Column(scale=6):
                card = gr.HTML(PLACEHOLDER)
                probs = gr.Label(num_top_classes=5, label="Top 5 lớp có xác suất cao nhất")
                shown = gr.Image(label="Ảnh mô hình nhìn thấy", height=200, interactive=False)

        outputs = [shown, probs, card]

        def auto(value, sq):
            """Có ảnh mới (tải lên, ảnh ví dụ, dán, webcam) hoặc đổi tuỳ chọn → nhận dạng luôn; không có ảnh → để trống."""
            return predict(value, sq) if value is not None else (None, {}, PLACEHOLDER)

        btn.click(predict, [inp, square], outputs)
        inp.change(auto, [inp, square], outputs)
        square.change(auto, [inp, square], outputs)
        upload.upload(lambda f: to_pil(getattr(f, "name", f)), upload, inp)
        clear.click(lambda: (None, None, {}, PLACEHOLDER), None, [inp] + outputs)
        gr.Markdown("<small>Đồ án môn Học máy và Khai phá dữ liệu – Nhóm 4. Kết quả mang tính minh hoạ: "
                    "Macro F1 trên tập Kiểm tra 0,329; nhóm lớp nhựa còn yếu (0,132).</small>")
    return app


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True nếu đã có chương trình khác lắng nghe ở host:port (ví dụ một phiên demo cũ chưa tắt)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def main(argv: list[str] | None = None) -> None:
    """Đọc tham số dòng lệnh, nạp mô hình một lần rồi mở giao diện."""
    ensure_utf8_stdio()            # thông báo tiếng Việt / emoji không lỗi trên console Windows
    ap = argparse.ArgumentParser(description="Giao diện demo nhận dạng ảnh rác dưới nước (Gradio)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="checkpoint .pt/.keras (mặc định: E04)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="tạo link công khai tạm thời")
    args = ap.parse_args(argv)
    if port_in_use(args.port):     # báo ngay, trước khi mất vài chục giây nạp mô hình
        raise SystemExit(f"Cổng {args.port} đang được dùng (có thể một phiên demo cũ vẫn chạy: mở "
                         f"http://127.0.0.1:{args.port} để xem, hoặc tắt cửa sổ đó). "
                         f"Hoặc chạy cổng khác: python demo_app.py --port {args.port + 1}")
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
