"""End-to-end tests of training, MLflow logging and the model registry.

Each test uses a temporary MLflow store (a SQLite file in a temporary folder),
so the tests never write into the real tracking server.
"""

import mlflow
import numpy as np
import pytest

from churn_project.config import config_from_dict
from churn_project.features import CATEGORICAL, NUMERIC
from churn_project.registry import best_run_id, load_champion, register_and_promote
from churn_project.train import train_and_log


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # artifacts are written under the temporary folder
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield tmp_path
    mlflow.set_tracking_uri(None)


def make_config(csv, model_type="logistic_regression", params=None):
    return config_from_dict(
        {
            "data": {"url": "https://not-used.example.com", "path": str(csv)},
            "split": {"test_size": 0.25, "seed": 0, "cv_folds": 3},
            "model": {"type": model_type, "params": params or {"max_iter": 1000}},
            "mlflow": {"experiment": "test-churn", "registered_model": "test-model"},
        }
    )


@pytest.fixture
def csv(tmp_path, raw_df):
    path = tmp_path / "raw.csv"
    raw_df.to_csv(path, index=False)
    return path


def test_run_records_params_metrics_tags_and_model(mlflow_store, csv):
    run_id, metrics = train_and_log(make_config(csv), run_name="unit-test")

    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["model_type"] == "logistic_regression"
    assert {"cv_auc_mean", "test_auc", "test_log_loss"} <= set(run.data.metrics)
    assert len(run.data.tags["data_sha256"]) == 64
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    assert hasattr(model, "predict_proba")


def test_gradient_boosting_model_can_be_logged_and_loaded(mlflow_store, csv, raw_df):
    params = {"max_iter": 20, "max_leaf_nodes": 8, "random_state": 0}
    run_id, _ = train_and_log(make_config(csv, "hist_gradient_boosting", params))
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    p = model.predict_proba(
        raw_df[NUMERIC + CATEGORICAL]
        .head(5)
        .assign(TotalCharges=lambda d: d["TotalCharges"].str.strip().replace("", "0").astype(float))
    )
    assert p.shape == (5, 2)


def test_best_run_is_registered_as_champion(mlflow_store, csv, raw_df):
    for c in (0.01, 1.0):
        train_and_log(make_config(csv, params={"C": c, "max_iter": 1000}))
    run_id = best_run_id("test-churn")
    version = register_and_promote(run_id, "test-model")
    assert version == 1

    champion = load_champion("test-model")
    direct = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    X = raw_df[NUMERIC + CATEGORICAL].head(10).copy()
    X["TotalCharges"] = X["TotalCharges"].str.strip().replace("", "0").astype(float)
    np.testing.assert_allclose(champion.predict_proba(X), direct.predict_proba(X))
