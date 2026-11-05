"""Tests of the three pipeline stages, run on fake data in a temporary folder.

Each test uses a temporary MLflow store, so it never writes into the real one.
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from churn_project.config import config_from_dict
from churn_project.evaluate import evaluate, load_model
from churn_project.features import CATEGORICAL, NUMERIC
from churn_project.lineage import dvc_md5
from churn_project.prepare import prepare
from churn_project.registry import best_run_id, load_champion, register_and_promote
from churn_project.train import train


@pytest.fixture
def workdir(tmp_path, monkeypatch, raw_df):
    """A temporary project folder with the fake raw data and its own MLflow store."""
    monkeypatch.chdir(tmp_path)
    Path("data/raw").mkdir(parents=True)
    raw_df.to_csv("data/raw/telco_churn.csv", index=False)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield tmp_path
    mlflow.set_tracking_uri(None)


def make_config(model_type="logistic_regression", params=None):
    return config_from_dict(
        {
            "data": {"raw": "data/raw/telco_churn.csv"},
            "split": {"test_size": 0.25, "seed": 0, "cv_folds": 3},
            "model": {"type": model_type, "params": params or {"max_iter": 1000}},
            "mlflow": {"experiment": "test-churn", "registered_model": "test-model"},
        }
    )


def run_pipeline(cfg):
    prepare(cfg)
    train(cfg)
    return evaluate()


def test_prepare_writes_a_stratified_split(workdir):
    train_path, test_path = prepare(make_config())
    tr, te = pd.read_csv(train_path), pd.read_csv(test_path)
    assert len(tr) + len(te) == 200
    assert len(te) == 50
    assert abs(tr["Churn"].mean() - te["Churn"].mean()) < 0.05


def test_pipeline_writes_model_and_metrics(workdir):
    metrics = run_pipeline(make_config())
    assert Path("models/model.skops").exists()
    assert set(json.loads(Path("reports/train.json").read_text())) == {
        "cv_auc_mean",
        "cv_auc_std",
    }
    assert metrics["n_test"] == 50
    assert 0.0 <= metrics["test_auc"] <= 1.0


def test_same_inputs_give_the_same_results(workdir):
    """Functional reproducibility: identical split, metrics and predictions."""
    cfg = make_config()
    first = run_pipeline(cfg)
    split_1 = Path("data/prepared/train.csv").read_bytes()
    X = pd.read_csv("data/prepared/test.csv")[NUMERIC + CATEGORICAL]
    p_1 = load_model().predict_proba(X)
    second = run_pipeline(cfg)
    assert Path("data/prepared/train.csv").read_bytes() == split_1
    assert first == second
    np.testing.assert_array_equal(load_model().predict_proba(X), p_1)


def test_mlflow_run_records_the_dvc_data_version(workdir):
    Path("data/raw/telco_churn.csv.dvc").write_text(
        yaml.safe_dump({"outs": [{"md5": "0123456789abcdef", "path": "telco_churn.csv"}]})
    )
    prepare(make_config())
    train(make_config())
    run = mlflow.search_runs(experiment_names=["test-churn"]).iloc[0]
    assert run["tags.data_dvc_md5"] == "0123456789abcdef"


def test_untracked_data_is_reported(tmp_path):
    assert dvc_md5(tmp_path / "telco_churn.csv") == "untracked"


def test_gradient_boosting_model_is_saved_and_loaded(workdir):
    params = {"max_iter": 20, "max_leaf_nodes": 8, "random_state": 0}
    metrics = run_pipeline(make_config("hist_gradient_boosting", params))
    assert metrics["n_test"] == 50


def test_best_run_is_registered_as_champion(workdir):
    prepare(make_config())
    for c in (0.01, 1.0):
        train(make_config(params={"C": c, "max_iter": 1000}))
    run_id = best_run_id("test-churn")
    assert register_and_promote(run_id, "test-model") == 1
    X = pd.read_csv("data/prepared/test.csv")[NUMERIC + CATEGORICAL]
    direct = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    np.testing.assert_allclose(
        load_champion("test-model").predict_proba(X), direct.predict_proba(X)
    )
