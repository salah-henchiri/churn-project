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
    model_params: dict[str, Any]
    output_dir: Path


def load_config(path: str | Path) -> Config:
    """Read the YAML file at `path` and return an immutable, validated Config."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    test_size = float(raw["split"]["test_size"])
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"split.test_size must be in (0, 1), got {test_size}")

    return Config(
        data_url=raw["data"]["url"],
        data_path=Path(raw["data"]["path"]),
        test_size=test_size,
        seed=int(raw["split"]["seed"]),
        model_params=dict(raw["model"]["params"]),
        output_dir=Path(raw["output"]["dir"]),
    )
