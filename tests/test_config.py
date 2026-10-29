"""Tests of the configuration loader."""

from pathlib import Path

import pytest
import yaml

from churn_project.config import load_config

VALID = {
    "data": {"url": "https://example.com/data.csv", "path": "data/raw.csv"},
    "split": {"test_size": 0.25, "seed": 0},
    "model": {"params": {"C": 1.0}},
    "output": {"dir": "artifacts"},
}


def write(tmp_path: Path, content: dict) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(content))
    return path


def test_valid_config_is_loaded(tmp_path):
    cfg = load_config(write(tmp_path, VALID))
    assert cfg.test_size == 0.25
    assert cfg.data_path == Path("data/raw.csv")
    assert cfg.model_params == {"C": 1.0}


@pytest.mark.parametrize("test_size", [0, 1, 1.5, -0.1])
def test_invalid_test_size_is_rejected(tmp_path, test_size):
    bad = {**VALID, "split": {"test_size": test_size, "seed": 0}}
    with pytest.raises(ValueError, match="test_size"):
        load_config(write(tmp_path, bad))
