"""Kiểm thử phần chọn ảnh của scripts/make_demo_images.py (không cần dữ liệu ảnh hay checkpoint)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mdi", ROOT / "scripts" / "make_demo_images.py")
mdi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mdi)


def test_plan_has_three_correct_two_wrong_and_all_exist_in_test_predictions():
    df = mdi.candidates()
    assert sum(t == p for _, t, p, _ in mdi.PLAN) == 3 and sum(t != p for _, t, p, _ in mdi.PLAN) == 2
    for _, true, pred, _ in mdi.PLAN:
        path = mdi.choose(df, true, pred)
        assert path.startswith(f"test/{true}/")


def test_choose_prefers_highest_score():
    df = mdi.candidates()
    pool = sorted(df[(df["true"] == "trash_bag") & (df["pred"] == "trash_bag")]["path"])
    assert mdi.choose(df, "trash_bag", "trash_bag", {pool[-1]: 0.9, pool[0]: 0.1}) == pool[-1]
