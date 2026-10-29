"""The churn model: preprocessing and classifier in one scikit-learn Pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn_project.features import build_preprocessor


def build_model(params: dict[str, Any] | None = None) -> Pipeline:
    """Return an unfitted pipeline. `params` are passed to LogisticRegression."""
    return Pipeline(
        [
            ("pre", build_preprocessor()),
            ("clf", LogisticRegression(**(params or {}))),
        ]
    )
