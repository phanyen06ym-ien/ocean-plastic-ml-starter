"""Frozen class order and ImageNet preprocessing, no downloads or real data."""
import copy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.common import class_names, load_class_mapping, validate_instance_mapping
from src.train import load_experiment_config, validate_mobilenet_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_experiment_config(ROOT / "configs/experiments/E03_transfer_mobilenetv2.yaml")


def test_e03_and_all_tuning_configs_keep_frozen_contract():
    for path in (ROOT / "configs/experiments").glob("E03*.yaml"):
        validate_mobilenet_config(load_experiment_config(path))


@pytest.mark.parametrize("field", ["class_index", "category_id", "class_name", "plastic_related"])
def test_mapping_identity_includes_category_ids_and_plastic_flags(field):
    mapping = copy.deepcopy(load_class_mapping())
    mapping[0][field] = mapping[10][field]
    with pytest.raises(ValueError, match="22 Instance"):
        validate_instance_mapping(mapping)


@pytest.mark.parametrize("section,key,value", [
    ("data", "num_classes", 21), ("data", "class_names", sorted(class_names(load_class_mapping()))),
    ("data", "data_version", "v2"), ("data", "use_test_set", True),
    ("input", "image_size", [224, 224]), ("input", "channels", 1),
])
def test_config_drift_fails_before_training(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_mobilenet_config(config)


def test_wrong_normalization_is_rejected(config):
    config["input"]["normalize"] = {"mean": [0, 0, 0], "std": [1, 1, 1]}
    with pytest.raises(ValueError, match="ImageNet"):
        validate_mobilenet_config(config)


def test_training_rejects_changed_dataset_mapping_before_creating_run(config, tmp_path):
    import json
    from src.train import run_training

    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    mapping = load_class_mapping()
    mapping[0]["category_id"] = 999
    (tmp_path / "class_mapping.json").write_text(json.dumps({"classes": mapping}))
    with pytest.raises(ValueError, match="22 Instance"):
        run_training(config, tmp_path, tmp_path / "runs", pretrained=False)
    assert not (tmp_path / "runs").exists()


def test_actual_transform_and_22_output_head(config):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.train import build_model, build_transform

    pixels = np.full((128, 128, 3), 128, dtype=np.uint8)
    tensor = build_transform(config)(Image.fromarray(pixels))
    expected = (torch.tensor([128 / 255] * 3) - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
    assert tensor.shape == (3, 128, 128)
    assert torch.allclose(tensor[:, 0, 0], expected)
    assert build_model(config, pretrained=False).classifier[-1].out_features == 22


def test_loaders_assign_mapping_ids_and_never_open_test(config, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src import train

    names = class_names(load_class_mapping())
    for split in ("train", "val"):
        for name in names:
            path = tmp_path / split / name / "sample.png"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (128, 128), (20, 40, 80)).save(path)
    config["data"]["num_workers"] = 0
    # No Test directory exists; also guard against even attempting to open it.
    original = train.ordered_image_folder
    opened = []

    def guarded(root, *args, **kwargs):
        assert Path(root).name in {"train", "val"}
        opened.append(Path(root).name)
        return original(root, *args, **kwargs)

    monkeypatch.setattr(train, "ordered_image_folder", guarded)
    train_ds, val_ds, _, _ = train.make_loaders(config, tmp_path, names)
    assert opened == ["train", "val"]
    assert train_ds.class_to_idx == val_ds.class_to_idx == dict(zip(names, range(22)))
    assert train_ds.class_to_idx["rov"] == 0
    assert train_ds.class_to_idx["trash_cup"] == 14
    assert set(train_ds.targets) == set(range(22))
    (tmp_path / "train/unknown").mkdir()
    with pytest.raises(ValueError, match="unexpected"):
        original(tmp_path / "train", names)
