"""Phân tích lỗi theo video và theo kích thước bbox (src.evaluate.error_analysis)."""
import json

import pandas as pd
import pytest

from src.evaluate import error_analysis, join_predictions_manifest, scores_by_size, scores_by_video


def _write(tmp_path):
    man = pd.DataFrame({
        "uid": ["a", "b", "c", "d", "e"],
        "crop_status": ["ok", "ok", "ok", "ok", "too_small"],
        "crop_path": ["test/x/a.jpg", "test/x/b.jpg", "test/y/c.jpg", "test/y/d.jpg", ""],
        "video_id": [1, 1, 2, 2, 2],
        "bbox_w": [20, 200, 50, 300, 5],
        "bbox_h": [40, 150, 100, 90, 5],
    })
    pred = pd.DataFrame({"path": ["test/x/a.jpg", "test/x/b.jpg", "test/y/c.jpg", "test/y/d.jpg"],
                         "y_true": [0, 0, 1, 1], "y_pred": [1, 0, 1, 1]})
    man.to_csv(tmp_path / "manifest.csv", index=False)
    pred.to_csv(tmp_path / "pred.csv", index=False)
    return tmp_path / "pred.csv", tmp_path / "manifest.csv"


def test_join_and_scores(tmp_path):
    p, m = _write(tmp_path)
    df = join_predictions_manifest(p, m)
    assert list(df["short_side"]) == [20, 150, 50, 90]
    v = scores_by_video(df, plastic=[1])
    assert v.loc["1", "accuracy"] == 0.5 and v.loc["2", "plastic_recall"] == 1.0
    assert pd.isna(v.loc["1", "plastic_recall"])            # video 1 không có mẫu nhựa
    s = scores_by_size(df, plastic=[1]).set_index(["group", "size_bin"])
    assert s.loc[("all", "0-32px"), "recall"] == 0.0 and s.loc[("all", ">=128px"), "num_samples"] == 1
    assert s.loc[("plastic6", "32-64px"), "recall"] == 1.0


def test_error_analysis_writes_files(tmp_path):
    p, m = _write(tmp_path)
    summary = error_analysis(p, m, [1], tmp_path / "out", "T00_test")
    assert summary["num_samples"] == 4
    for f in ("T00_test_by_video.csv", "T00_test_by_size.csv", "T00_test_error_analysis.json"):
        assert (tmp_path / "out" / f).exists()
    json.loads((tmp_path / "out" / "T00_test_error_analysis.json").read_text(encoding="utf-8"))


def test_join_reports_missing_rows(tmp_path):
    p, m = _write(tmp_path)
    pd.DataFrame({"path": ["test/z/zz.jpg"], "y_true": [0], "y_pred": [0]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        join_predictions_manifest(p, m)
