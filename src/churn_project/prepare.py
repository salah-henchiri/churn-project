"""Stage 1 of the pipeline: clean the raw data and split it into train and test.

Usage:  uv run churn-prepare        (normally run by `dvc repro`)
"""

from __future__ import annotations

import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

from churn_project import data
from churn_project.config import Config, load_config
from churn_project.features import TARGET

logger = logging.getLogger(__name__)

OUT_DIR = Path("data/prepared")


def prepare(cfg: Config, out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    df = data.load(cfg.raw_data)
    train, test = train_test_split(
        df, test_size=cfg.test_size, stratify=df[TARGET], random_state=cfg.seed
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = out_dir / "train.csv", out_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    logger.info("train: %d rows, test: %d rows", len(train), len(test))
    return train_path, test_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    prepare(load_config())


if __name__ == "__main__":
    main()
