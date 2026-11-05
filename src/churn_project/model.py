"""The churn model: preprocessing and classifier in one scikit-learn Pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn_project.features import build_preprocessor

CLASSIFIERS = {
    "logistic_regression": LogisticRegression,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}

# Types that the safe "skops" format refuses unless they are explicitly trusted
# (Section 3.4). We trust them because we build and save these models ourselves.
SKOPS_TRUSTED_TYPES = {
    "logistic_regression": [],
    "hist_gradient_boosting": ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
}


def build_model(
    model_type: str = "logistic_regression", params: dict[str, Any] | None = None
) -> Pipeline:
    """Return an unfitted pipeline: preprocessing followed by the chosen classifier."""
    if model_type not in CLASSIFIERS:
        raise ValueError(f"unknown model type {model_type!r}; choose from {sorted(CLASSIFIERS)}")
    return Pipeline(
        [
            ("pre", build_preprocessor()),
            ("clf", CLASSIFIERS[model_type](**(params or {}))),
        ]
    )
