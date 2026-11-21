"""Experiment configs: E01–E04 only, and the MobileNetV2 config never touches the Test set."""
from pathlib import Path

import pytest

from src.train import load_experiment_config

EXP = Path(__file__).resolve().parents[1] / "configs" / "experiments"


def test_only_main_experiment_configs():
    ids = sorted(p.name.split("_")[0] for p in EXP.glob("*.yaml"))
    assert ids and all(i in {"E01", "E02", "E03", "E04"} for i in ids), ids
    assert len(ids) == len(set(ids)), "one config file per experiment"


def test_e03_does_not_use_test_set():
    assert load_experiment_config(EXP / "E03_transfer_mobilenetv2.yaml")["data"]["use_test_set"] is False


def _leaves(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _leaves(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def test_e04_keeps_e03_model_and_training_changes_only_data():
    e03 = load_experiment_config(EXP / "E03_transfer_mobilenetv2.yaml")
    e04 = load_experiment_config(EXP / "E04_mobilenetv2_data_v2.yaml")
    assert e04["exp_id"] == "E04" and e04["baseline_exp_id"] == "E03"
    assert e04["model"] == e03["model"] and e04["training"] == e03["training"]
    assert e04["input"]["normalize"] == e03["input"]["normalize"]
    assert e04["data"]["use_test_set"] is False
    assert e04["data"]["class_names"] == e03["data"]["class_names"] and e04["data"]["num_classes"] == 22
    assert e04["input"]["image_size"] == [224, 224] and e04["augmentation"]["enabled"] is True
    a, b = dict(_leaves(e03)), dict(_leaves(e04))
    changed = {k.split(".")[0] for k in set(a) | set(b) if a.get(k) != b.get(k)}
    allowed = {"exp_id", "name", "description", "baseline_exp_id", "changed_factor", "known_extra_differences",
               "baseline_reference", "data", "input", "augmentation", "paths"}
    assert changed <= allowed, changed - allowed


def test_e04_matches_data_contract_v2():
    from src.preprocessing import load_contract
    root = EXP.parents[1]
    e04 = load_experiment_config(EXP / "E04_mobilenetv2_data_v2.yaml")
    contract = load_contract(root / e04["data"]["contract"])
    assert contract["dataset"]["data_version"] == e04["data"]["data_version"] == "v2_crop224"
    assert contract["image"]["input_size"] == e04["input"]["image_size"][0]
    assert contract["split"]["method"] == e04["data"]["split_info"]["method"]
    v1 = load_contract(root / "configs" / "data_contract.yaml")
    assert contract["classes"] == v1["classes"]                       # cùng 22 lớp, cùng thứ tự


def test_e04_train_transform_augments_and_val_transform_does_not():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image
    from src.train import build_transform
    e04 = load_experiment_config(EXP / "E04_mobilenetv2_data_v2.yaml")
    e03 = load_experiment_config(EXP / "E03_transfer_mobilenetv2.yaml")
    img = Image.new("RGB", (224, 224), (120, 80, 40))
    assert tuple(build_transform(e04, train=True)(img).shape) == (3, 224, 224)
    assert len(build_transform(e04, train=True).transforms) == 6
    assert len(build_transform(e04).transforms) == 2
    assert len(build_transform(e03, train=True).transforms) == 2       # E03 không augmentation
