import pytest

from src.download import pick_bitstream


def test_pick_bitstream_by_name():
    bs = [{"name": "README.txt"}, {"name": "dataset.zip", "sizeBytes": 10}, {"name": "other.zip", "sizeBytes": 99}]
    assert pick_bitstream(bs)["name"] == "dataset.zip"


def test_pick_bitstream_fallback_largest_zip():
    bs = [{"name": "a.zip", "sizeBytes": 5}, {"name": "b.zip", "sizeBytes": 50}]
    assert pick_bitstream(bs)["name"] == "b.zip"


def test_pick_bitstream_no_zip():
    with pytest.raises(FileNotFoundError):
        pick_bitstream([{"name": "README.txt"}])
