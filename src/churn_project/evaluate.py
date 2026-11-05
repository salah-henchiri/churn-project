"""Stage 3 of the pipeline: evaluate the saved model once on the test set.

Usage:  uv run churn-evaluate       (normally run by `dvc repro`)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import skops.io as sio
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from churn_project.features import CATEGORICAL, NUMERIC, TARGET
from churn_project.model import SKOPS_TRUSTED_TYPES

logger = logging.getLogger(__name__)

TEST_PATH = Path("data/prepared/test.csv")
MODEL_PATH = Path("models/model.skops")
METRICS_PATH = Path("reports/metrics.json")


def load_model(path: Path = MODEL_PATH):
    """Load a model saved by the train stage, trusting only the reviewed types."""
    trusted = sorted({t for types in SKOPS_TRUSTED_TYPES.values() for t in types})
    return sio.load(path, trusted=trusted)


def evaluate(
    test_path: Path = TEST_PATH, model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH
) -> dict[str, float]:
    df = pd.read_csv(test_path)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    p = load_model(model_path).predict_proba(X)[:, 1]
    metrics = {
        "test_auc": roc_auc_score(y, p),
        "test_accuracy": accuracy_score(y, p > 0.5),
        "test_log_loss": log_loss(y, p),
        "test_brier": brier_score_loss(y, p),
        "n_test": len(df),
    }
    metrics = {k: round(float(v), 4) if k != "n_test" else int(v) for k, v in metrics.items()}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("test metrics: %s", metrics)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    evaluate()


if __name__ == "__main__":
    main()
