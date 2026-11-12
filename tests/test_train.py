"""Kiểm thử E01/E02 qua src/train_keras.py; src/train.py dành cho E03 PyTorch."""
from pathlib import Path

import numpy as np
import pytest

from src.train_keras import DEFAULT_CONFIG, compute_class_weights, load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def test_class_weights_balanced_and_zero_for_missing():
    w = compute_class_weights([100, 10, 0, 50])
    assert w[2] == 0.0
    assert w[1] > w[3] > w[0]
    # trung bình có trọng số theo số mẫu = 1 (định nghĩa 'balanced')
    counts = np.array([100, 10, 50])
    assert np.isclose(sum(w[i] * c for i, c in zip([0, 1, 3], counts)) / counts.sum(), 1.0)


def test_class_weights_disabled():
    assert compute_class_weights([1, 2], None) is None


def test_experiment_configs_differ_only_in_augmentation():
    e1 = load_experiment_config(ROOT / "configs/experiments/E01_baseline.yaml")
    e2 = load_experiment_config(ROOT / "configs/experiments/E02_augment.yaml")
    diff = {k for k in set(e1) | set(e2) if e1.get(k) != e2.get(k)}
    assert diff == {"exp_id", "description", "augmentation"}
    assert e1["num_classes"] == 22 and e1["input_shape"] == [128, 128, 3]
