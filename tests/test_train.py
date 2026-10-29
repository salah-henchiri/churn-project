"""End-to-end test of the training command, on the fake data set (no download)."""

import json

import joblib
import yaml

from churn_project.train import run


def test_training_writes_model_and_metrics(tmp_path, raw_df):
    csv = tmp_path / "raw.csv"
    raw_df.to_csv(csv, index=False)
    config = {
        "data": {"url": "https://not-used.example.com", "path": str(csv)},
        "split": {"test_size": 0.25, "seed": 0},
        "model": {"params": {"max_iter": 1000}},
        "output": {"dir": str(tmp_path / "artifacts")},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))

    metrics = run(str(config_path))

    assert set(metrics) == {"accuracy", "auc"}
    assert 0.0 <= metrics["auc"] <= 1.0
    saved = json.loads((tmp_path / "artifacts" / "metrics.json").read_text())
    assert saved == metrics
    model = joblib.load(tmp_path / "artifacts" / "model.joblib")
    assert hasattr(model, "predict_proba")
