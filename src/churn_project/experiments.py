"""Run several model configurations as MLflow runs of one experiment.

Usage:  uv run churn-experiments --config configs/train.yaml --runs configs/experiments.yaml
"""

from __future__ import annotations

import argparse
import copy
import logging

import yaml

from churn_project.config import config_from_dict
from churn_project.train import train_and_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model configurations in MLflow.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--runs", default="configs/experiments.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    with open(args.config, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(args.runs, encoding="utf-8") as f:
        runs = yaml.safe_load(f)["runs"]

    print(f"{'run':24s} {'cv_auc':>14s} {'test_auc':>9s} {'fit (s)':>8s}")
    for entry in runs:
        raw = copy.deepcopy(base)
        raw["model"] = {"type": entry["type"], "params": entry["params"]}
        _, m = train_and_log(config_from_dict(raw), run_name=entry["name"])
        print(
            f"{entry['name']:24s} {m['cv_auc_mean']:.4f} ± {m['cv_auc_std']:.4f} "
            f"{m['test_auc']:9.4f} {m['fit_seconds']:8.2f}"
        )


if __name__ == "__main__":
    main()
