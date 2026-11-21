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


def test_e04_is_the_documented_improved_model():
    e03 = load_experiment_config(EXP / "E03_transfer_mobilenetv2.yaml")
    e04 = load_experiment_config(EXP / "E04_efficientnet_b0_data_v2.yaml")
    assert e04["exp_id"] == "E04" and e04["baseline_exp_id"] == "E03"
    assert e04["data"]["use_test_set"] is False
    assert e04["model"]["backbone"] == "efficientnet_b0" and e04["model"]["head"]["in_features"] == 1280
    assert e04["input"]["image_size"] == [224, 224] and e04["augmentation"]["enabled"] is True
    assert e04["input"]["normalize"] == e03["input"]["normalize"]               # ImageNet mean/std
    t = e04["training"]
    assert t["loss"] == "focal" and t["sampler"]["mode"] == "sqrt_inverse" and t["class_weights"]["mode"] is None
    assert t["checkpoint_monitor"] == "val_macro_f1"
    for ph in ("phase1", "phase2"):
        assert t[ph]["early_stopping"]["monitor"] == "val_macro_f1" and t[ph]["early_stopping"]["mode"] == "max"
    assert t["optimizer"] == e03["training"]["optimizer"] and t["batch_size"] == e03["training"]["batch_size"]


def test_e04_classes_match_mapping_v2():
    import json
    root = EXP.parents[1]
    e04 = load_experiment_config(EXP / "E04_efficientnet_b0_data_v2.yaml")
    mapping = json.loads((root / "configs/class_mapping_v2.json").read_text(encoding="utf-8"))["classes"]
    names = [m["class_name"] for m in sorted(mapping, key=lambda m: m["class_index"])]
    assert names == e04["data"]["class_names"] and len(names) == e04["data"]["num_classes"] == 18
    assert [m["class_index"] for m in mapping if m["plastic_related"]] == e04["data"]["plastic6_indices"]
    assert sorted(e04["evaluation"]["plastic6_classes"]) == sorted(names[i] for i in e04["data"]["plastic6_indices"])


def test_e04_matches_data_contract_v2():
    from src.preprocessing import load_contract
    root = EXP.parents[1]
    e04 = load_experiment_config(EXP / "E04_efficientnet_b0_data_v2.yaml")
    contract = load_contract(root / e04["data"]["contract"])
    assert contract["dataset"]["data_version"] == e04["data"]["data_version"] == "v2_crop224"
    assert contract["image"]["input_size"] == e04["input"]["image_size"][0]
    assert contract["split"]["method"] == e04["data"]["split_info"]["method"]
    v1 = load_contract(root / "configs" / "data_contract.yaml")
    assert contract["classes"]["num_classes"] == e04["data"]["num_classes"] == 18
    assert contract["classes"]["plastic_related_candidates"] == v1["classes"]["plastic_related_candidates"]


def test_e04_train_transform_augments_and_val_transform_does_not():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image
    from src.train import build_transform
    e04 = load_experiment_config(EXP / "E04_efficientnet_b0_data_v2.yaml")
    e03 = load_experiment_config(EXP / "E03_transfer_mobilenetv2.yaml")
    img = Image.new("RGB", (224, 224), (120, 80, 40))
    assert tuple(build_transform(e04, train=True)(img).shape) == (3, 224, 224)
    assert len(build_transform(e04, train=True).transforms) == 6
    assert len(build_transform(e04).transforms) == 2
    assert len(build_transform(e03, train=True).transforms) == 2       # E03 không augmentation


def test_e04_model_sampler_and_focal_loss_run():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.train import build_model, configure_phase, make_sampler, weighted_ce
    e04 = load_experiment_config(EXP / "E04_efficientnet_b0_data_v2.yaml")
    model = build_model(e04, pretrained=False).eval()
    with torch.no_grad():
        assert tuple(model(torch.zeros(2, 3, 224, 224)).shape) == (2, 18)
    configure_phase(model, {**e04["training"]["phase2"], "freeze_backbone": False})
    assert any(p.requires_grad for p in model.features[4].parameters())
    assert not any(p.requires_grad for p in model.features[3].parameters())
    logits, y, w = torch.tensor([[4.0, 0.0], [0.0, 0.1]]), torch.tensor([0, 1]), torch.ones(2)
    assert weighted_ce(logits, y, w, gamma=2.0) < weighted_ce(logits, y, w)  # focal giảm trọng số mẫu dễ
    s = make_sampler([0, 0, 0, 1], e04["training"]["sampler"], 2, torch.Generator().manual_seed(0))
    assert len(list(iter(s))) == 4
