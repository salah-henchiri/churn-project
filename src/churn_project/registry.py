"""Model registry: register the best run and give it the alias 'champion'.

Usage:  uv run churn-register --experiment churn --model-name churn-model
"""

from __future__ import annotations

import argparse

import mlflow
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

SELECTION_METRIC = "cv_auc_mean"  # chosen on cross-validation, never on the test set


def best_run_id(experiment: str, metric: str = SELECTION_METRIC) -> str:
    """Id of the finished run of `experiment` with the highest `metric`."""
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if runs.empty:
        raise LookupError(f"no finished run in experiment {experiment!r}")
    return str(runs.loc[0, "run_id"])


def register_and_promote(run_id: str, model_name: str, alias: str = "champion") -> int:
    """Register the model of `run_id` as a new version and point `alias` to it."""
    version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, version.version)
    client.set_model_version_tag(model_name, version.version, "selected_by", SELECTION_METRIC)
    return int(version.version)


def load_champion(model_name: str) -> Pipeline:
    """Load the scikit-learn pipeline that currently has the alias 'champion'."""
    return mlflow.sklearn.load_model(f"models:/{model_name}@champion")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the best run as the champion model.")
    parser.add_argument("--experiment", default="churn")
    parser.add_argument("--model-name", default="churn-model")
    args = parser.parse_args()
    run_id = best_run_id(args.experiment)
    version = register_and_promote(run_id, args.model_name)
    print(f"run {run_id} registered as {args.model_name} version {version} (alias: champion)")


if __name__ == "__main__":
    main()
