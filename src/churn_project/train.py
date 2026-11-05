"""Train the churn model and record everything in MLflow.

Usage:  uv run churn-train --config configs/train.yaml

The MLflow server is chosen with the environment variable MLFLOW_TRACKING_URI
(for instance http://127.0.0.1:5000). Without it, MLflow writes to ./mlflow.db.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402

from churn_project import data  # noqa: E402
from churn_project.config import Config, load_config  # noqa: E402
from churn_project.features import CATEGORICAL, NUMERIC, TARGET  # noqa: E402
from churn_project.model import SKOPS_TRUSTED_TYPES, build_model  # noqa: E402

logger = logging.getLogger(__name__)


def git_commit() -> str:
    """Hash of the current Git commit, or 'unknown' outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_sha256(path: Path) -> str:
    """Fingerprint of a data file: any change of its content changes the hash."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation_figure(y_true: pd.Series, p: pd.Series) -> plt.Figure:
    """ROC curve and calibration curve of the test predictions."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    fpr, tpr, _ = roc_curve(y_true, p)
    ax[0].plot(fpr, tpr, color="#2b7bd6")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve")
    frac, mean_p = calibration_curve(y_true, p, n_bins=10)
    ax[1].plot(mean_p, frac, "o-", color="#ee6b35")
    ax[1].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[1].set(
        xlabel="mean predicted probability", ylabel="observed churn rate", title="calibration"
    )
    fig.tight_layout()
    return fig


def train_and_log(cfg: Config, run_name: str | None = None) -> tuple[str, dict[str, float]]:
    """Train one model, evaluate it and log the run in MLflow. Returns (run_id, metrics)."""
    path = data.download(cfg.data_url, cfg.data_path)
    df = data.load(path)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )

    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=run_name) as run:
        # 1. lineage: code version, data version, configuration
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "data_sha256": file_sha256(path),
                "model_type": cfg.model_type,
            }
        )
        mlflow.log_params(
            {
                "model_type": cfg.model_type,
                "test_size": cfg.test_size,
                "seed": cfg.seed,
                "cv_folds": cfg.cv_folds,
                **{f"model.{k}": v for k, v in cfg.model_params.items()},
            }
        )
        mlflow.log_input(
            mlflow.data.from_pandas(df, source=cfg.data_url, name="telco_churn", targets=TARGET),
            context="training",
        )

        # 2. model selection metric: cross-validated AUC on the training part only
        model = build_model(cfg.model_type, cfg.model_params)
        cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        # 3. final fit and test metrics
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        p = model.predict_proba(X_test)[:, 1]
        metrics = {
            "cv_auc_mean": float(cv_auc.mean()),
            "cv_auc_std": float(cv_auc.std(ddof=1)),
            "test_auc": float(roc_auc_score(y_test, p)),
            "test_accuracy": float(accuracy_score(y_test, p > 0.5)),
            "test_log_loss": float(log_loss(y_test, p)),
            "test_brier": float(brier_score_loss(y_test, p)),
            "fit_seconds": fit_seconds,
        }
        mlflow.log_metrics(metrics)
        for fold, value in enumerate(cv_auc):
            mlflow.log_metric("cv_auc_fold", float(value), step=fold)

        # 4. artefacts: evaluation plots and the model itself
        fig = evaluation_figure(y_test, p)
        mlflow.log_figure(fig, "plots/evaluation.png")
        plt.close(fig)
        signature = infer_signature(X_train, model.predict_proba(X_train))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=SKOPS_TRUSTED_TYPES[cfg.model_type] or None,
        )
        logger.info(
            "run %s (%s): %s",
            run.info.run_id,
            run_name,
            {k: round(v, 4) for k, v in metrics.items()},
        )
        return run.info.run_id, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model and log it in MLflow.")
    parser.add_argument("--config", default="configs/train.yaml", help="YAML configuration file")
    parser.add_argument("--run-name", default=None, help="name of the MLflow run")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    train_and_log(load_config(args.config), args.run_name)


if __name__ == "__main__":
    main()
