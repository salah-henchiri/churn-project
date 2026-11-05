"""Stage 2 of the pipeline: train the model, save it, and log the run in MLflow.

Usage:  uv run churn-train          (normally run by `dvc repro`)

The MLflow server is chosen with MLFLOW_TRACKING_URI; without it, MLflow
writes to ./mlflow.db.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
import pandas as pd
import skops.io as sio
from mlflow.models import infer_signature
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churn_project.config import Config, load_config
from churn_project.features import CATEGORICAL, NUMERIC, TARGET
from churn_project.lineage import dvc_md5, git_commit
from churn_project.model import SKOPS_TRUSTED_TYPES, build_model

logger = logging.getLogger(__name__)

TRAIN_PATH = Path("data/prepared/train.csv")
MODEL_PATH = Path("models/model.skops")
REPORT_PATH = Path("reports/train.json")


def train(
    cfg: Config,
    train_path: Path = TRAIN_PATH,
    model_path: Path = MODEL_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, float]:
    df = pd.read_csv(train_path)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]

    model = build_model(cfg.model_type, cfg.model_params)
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    model.fit(X, y)
    metrics = {"cv_auc_mean": float(cv_auc.mean()), "cv_auc_std": float(cv_auc.std(ddof=1))}

    # outputs tracked by DVC: the model file and the training report
    model_path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, model_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))

    # the same run in MLflow, with the DVC data version as a tag
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "data_dvc_md5": dvc_md5(cfg.raw_data),
                "model_type": cfg.model_type,
            }
        )
        mlflow.log_params(
            {
                "model_type": cfg.model_type,
                "seed": cfg.seed,
                "cv_folds": cfg.cv_folds,
                **{f"model.{k}": v for k, v in cfg.model_params.items()},
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X, model.predict_proba(X)),
            input_example=X.head(3),
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=SKOPS_TRUSTED_TYPES[cfg.model_type] or None,
        )
        logger.info("MLflow run %s: %s", run.info.run_id, json.dumps(metrics))
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    train(load_config())


if __name__ == "__main__":
    main()
