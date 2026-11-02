"""Kiểm thử phần tính chỉ số hai cấp (tổng thể + nhóm nhựa) với nhãn có kết quả biết trước."""
import json

import numpy as np

from src.evaluation_runner import comparison_table, group_metrics, summarize

MAPPING = [
    {"class_index": 0, "category_id": 1, "class_name": "rov", "plastic_related": False},
    {"class_index": 1, "category_id": 2, "class_name": "trash_bag", "plastic_related": True},
    {"class_index": 2, "category_id": 3, "class_name": "trash_net", "plastic_related": True},
    {"class_index": 3, "category_id": 4, "class_name": "trash_rope", "plastic_related": True},
]


def test_group_metrics_known_values():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    g = group_metrics(y_true, y_pred, [1, 2, 3])
    assert g["classes_evaluated"] == [1, 2] and g["classes_without_samples"] == [3]
    # bag: P=2/3 R=1 F1=0.8 ; net: P=1 R=0.5 F1=2/3
    assert np.isclose(g["macro_f1"], (0.8 + 2 / 3) / 2)


def test_summarize_and_comparison(tmp_path):
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    probs = np.eye(4)[y_pred]
    r = summarize(y_true, y_pred, probs, MAPPING, "E01", "val")
    assert r["num_classes"] == 4 and r["num_samples"] == 6
    assert np.isclose(r["overall"]["accuracy"], 4 / 6)
    assert r["macro_f1_present_classes"] > r["overall"]["macro_f1"]   # lớp rope 0 mẫu kéo macro tổng xuống
    p = tmp_path / "E01_val.json"
    p.write_text(json.dumps(r, default=float))
    t = comparison_table([p])
    assert list(t["exp_id"]) == ["E01"] and "plastic_macro_f1" in t.columns
