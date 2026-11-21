"""Experiment configs: E01–E04 only, and the MobileNetV2 config never touches the Test set."""
from pathlib import Path

from src.train import load_experiment_config

EXP = Path(__file__).resolve().parents[1] / "configs" / "experiments"


def test_only_main_experiment_configs():
    ids = sorted(p.name.split("_")[0] for p in EXP.glob("*.yaml"))
    assert ids and all(i in {"E01", "E02", "E03", "E04"} for i in ids), ids
    assert len(ids) == len(set(ids)), "one config file per experiment"


def test_e03_does_not_use_test_set():
    assert load_experiment_config(EXP / "E03_transfer_mobilenetv2.yaml")["data"]["use_test_set"] is False
