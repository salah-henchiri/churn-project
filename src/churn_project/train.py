"""Command-line entry point: train the churn model and save it.

Usage:  uv run churn-train --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from churn_project import data
from churn_project.config import load_config
from churn_project.features import CATEGORICAL, NUMERIC, TARGET
from churn_project.model import build_model

logger = logging.getLogger(__name__)


def run(config_path: str) -> dict[str, float]:
    """Train, evaluate and save the model described by the configuration file."""
    cfg = load_config(config_path)
    df = data.load(data.download(cfg.data_url, cfg.data_path))

    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )

    model = build_model(cfg.model_params).fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, p > 0.5)), 4),
        "auc": round(float(roc_auc_score(y_test, p)), 4),
    }
    logger.info("test metrics: %s", metrics)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.output_dir / "model.joblib")
    (cfg.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("model and metrics saved in %s", cfg.output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model.")
    parser.add_argument("--config", default="configs/train.yaml", help="YAML configuration file")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(args.config)


if __name__ == "__main__":
    main()
