"""E03 and its tuning configs: each tuning run changes exactly one setting (fair comparison)."""
from pathlib import Path

import pytest

from src.train import load_experiment_config

EXP = Path(__file__).resolve().parents[1] / "configs" / "experiments"
TUNING = sorted(EXP.glob("E03?_tune.yaml"))


def _leaves(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _leaves(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def test_at_most_six_tuning_runs_and_no_test_set():
    assert 1 <= len(TUNING) <= 6
    assert load_experiment_config(EXP / "E03_transfer_mobilenetv2.yaml")["data"]["use_test_set"] is False


@pytest.mark.parametrize("path", TUNING, ids=lambda p: p.stem)
def test_tuning_changes_only_lr_or_dropout(path):
    base = dict(_leaves(load_experiment_config(EXP / "E03_transfer_mobilenetv2.yaml")))
    run = dict(_leaves(load_experiment_config(path)))
    diff = {k for k in set(base) | set(run) if base.get(k) != run.get(k)}
    meta = {"exp_id", "description", "changed_factor", "baseline_exp_id"}
    changed = diff - meta
    assert changed, "no setting changed"
    assert all(k == "model.head.dropout" or k.startswith(("training.phase1.learning_rate", "training.phase2.learning_rate"))
               for k in changed), changed
    assert "base_config" not in run
