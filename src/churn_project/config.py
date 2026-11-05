"""Training configuration, read from a YAML file and checked once."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    data_url: str
    data_path: Path
    test_size: float
    seed: int
    cv_folds: int
    model_type: str
    model_params: dict[str, Any]
    experiment: str
    registered_model: str


def load_config(path: str | Path) -> Config:
    """Read the YAML file at `path` and return an immutable, validated Config."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> Config:
    test_size = float(raw["split"]["test_size"])
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"split.test_size must be in (0, 1), got {test_size}")
    cv_folds = int(raw["split"].get("cv_folds", 5))
    if cv_folds < 2:
        raise ValueError(f"split.cv_folds must be at least 2, got {cv_folds}")

    return Config(
        data_url=raw["data"]["url"],
        data_path=Path(raw["data"]["path"]),
        test_size=test_size,
        seed=int(raw["split"]["seed"]),
        cv_folds=cv_folds,
        model_type=raw["model"].get("type", "logistic_regression"),
        model_params=dict(raw["model"].get("params", {})),
        experiment=raw["mlflow"]["experiment"],
        registered_model=raw["mlflow"]["registered_model"],
    )
