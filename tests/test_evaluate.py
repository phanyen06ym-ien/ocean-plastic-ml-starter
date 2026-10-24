"""
Unit Tests for Evaluation Module (src/evaluate.py)
==================================================
Dự án: Ocean Plastic Waste Classification (TrashCan 1.0 — Instance)
Mã công việc: S1-EVAL
"""

import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluate import (
    compute_classification_metrics,
    generate_classification_report,
    compute_confusion_matrix,
    find_top_confused_pairs,
    identify_misclassified_samples,
    format_comparison_table,
    save_evaluation_results,
)


class TestEvaluateModule(unittest.TestCase):

    def setUp(self):
        self.classes = ["trash_bag", "trash_bottle", "trash_can", "trash_net"]
        self.num_classes = len(self.classes)
        # Toy dataset: 10 samples
        # Class 0: 3 samples (indices 0, 1, 2)
        # Class 1: 3 samples (indices 3, 4, 5)
        # Class 2: 2 samples (indices 6, 7)
        # Class 3: 2 samples (indices 8, 9)
        self.y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
        # Predictions: 7 correct, 3 wrong
        # Index 2: True=0, Pred=1 (missed trash_bag as bottle)
        # Index 5: True=1, Pred=2 (missed bottle as can)
        # Index 9: True=3, Pred=0 (missed net as bag)
        self.y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 0])

    def test_compute_metrics_accuracy(self):
        """Kiểm tra độ chính xác tổng thể (7/10 = 0.70)."""
        metrics = compute_classification_metrics(self.y_true, self.y_pred, target_names=self.classes)
        self.assertAlmostEqual(metrics["accuracy"], 0.70, places=4)
        self.assertEqual(metrics["total_samples"], 10)

    def test_compute_metrics_keys_present(self):
        """Xác minh sự có mặt của tất cả các chỉ số yêu cầu."""
        metrics = compute_classification_metrics(self.y_true, self.y_pred, target_names=self.classes)
        required_keys = [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "per_class",
            "total_samples",
        ]
        for key in required_keys:
            self.assertIn(key, metrics)

    def test_per_class_support(self):
        """Kiểm tra số lượng mẫu support của từng lớp rác."""
        metrics = compute_classification_metrics(self.y_true, self.y_pred, target_names=self.classes)
        self.assertEqual(metrics["per_class"]["trash_bag"]["support"], 3)
        self.assertEqual(metrics["per_class"]["trash_bottle"]["support"], 3)
        self.assertEqual(metrics["per_class"]["trash_can"]["support"], 2)
        self.assertEqual(metrics["per_class"]["trash_net"]["support"], 2)

    def test_confusion_matrix_shape_and_trace(self):
        """Kiểm tra kích thước và đường chéo chính của confusion matrix."""
        cm = compute_confusion_matrix(self.y_true, self.y_pred)
        self.assertEqual(cm.shape, (4, 4))
        self.assertEqual(np.trace(cm), 7)  # 7 mẫu phân loại đúng

    def test_confusion_matrix_normalized_row_sums(self):
        """Kiểm tra ma trận chuẩn hóa theo hàng (tổng mỗi hàng = 1.0)."""
        cm_norm = compute_confusion_matrix(self.y_true, self.y_pred, normalize="true")
        for i in range(4):
            self.assertAlmostEqual(float(np.sum(cm_norm[i, :])), 1.0, places=4)

    def test_find_top_confused_pairs(self):
        """Kiểm tra việc trích xuất các cặp nhầm lẫn phổ biến nhất."""
        cm = compute_confusion_matrix(self.y_true, self.y_pred)
        top_pairs = find_top_confused_pairs(cm, target_names=self.classes, top_k=3)
        self.assertLessEqual(len(top_pairs), 3)
        self.assertTrue(all(p["count"] > 0 for p in top_pairs))
        self.assertTrue(all(p["true_idx"] != p["pred_idx"] for p in top_pairs))

    def test_identify_misclassified_samples(self):
        """Kiểm tra việc nhận diện chính xác các mẫu dự đoán sai."""
        probs = np.zeros((10, 4))
        for i in range(10):
            probs[i, self.y_pred[i]] = 0.85
            probs[i, (self.y_pred[i] + 1) % 4] = 0.15

        errors = identify_misclassified_samples(
            self.y_true,
            self.y_pred,
            probabilities=probs,
            target_names=self.classes,
        )
        self.assertEqual(len(errors), 3)
        err_indices = [e["sample_index"] for e in errors]
        self.assertEqual(err_indices, [2, 5, 9])
        self.assertEqual(errors[0]["severity"], "high_confidence_error")

    def test_classification_report_output(self):
        """Kiểm tra chuỗi báo cáo Classification Report."""
        report = generate_classification_report(self.y_true, self.y_pred, target_names=self.classes)
        self.assertIn("trash_bag", report)
        self.assertIn("accuracy", report)
        self.assertIn("macro avg", report)

    def test_format_comparison_table(self):
        """Kiểm tra tạo bảng so sánh thực nghiệm dạng Markdown."""
        experiments = [
            {
                "exp_id": "EXP-001",
                "model_name": "CNN Baseline",
                "augmentation": "None",
                "val_loss": 0.40,
                "test_accuracy": 0.85,
                "test_macro_f1": 0.84,
            },
            {
                "exp_id": "EXP-002",
                "model_name": "CNN + Aug",
                "augmentation": "Underwater Blur",
                "val_loss": 0.32,
                "test_accuracy": 0.89,
                "test_macro_f1": 0.88,
            },
        ]
        table_md = format_comparison_table(experiments, output_format="markdown")
        self.assertIn("EXP-001", table_md)
        self.assertIn("EXP-002", table_md)
        self.assertIn("Macro F1", table_md)

    def test_save_evaluation_results(self):
        """Kiểm tra lưu kết quả đánh giá ra file JSON và Markdown tạm thời."""
        metrics = compute_classification_metrics(self.y_true, self.y_pred, target_names=self.classes)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test.json")
            md_path = os.path.join(tmpdir, "test.md")

            save_evaluation_results(metrics, json_path, format="json")
            save_evaluation_results(metrics, md_path, format="markdown")

            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(md_path))
            self.assertGreater(os.path.getsize(json_path), 0)
            self.assertGreater(os.path.getsize(md_path), 0)


if __name__ == "__main__":
    unittest.main()
