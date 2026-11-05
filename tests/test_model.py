"""Tests of the model pipeline."""

import numpy as np
import pytest

from churn_project.data import clean
from churn_project.features import CATEGORICAL, NUMERIC, TARGET
from churn_project.model import build_model


@pytest.fixture
def fitted(raw_df):
    df = clean(raw_df)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    return build_model("logistic_regression", {"max_iter": 1000}).fit(X, y), X


def test_probabilities_are_valid(fitted):
    model, X = fitted
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (len(X),)
    assert np.all((p >= 0) & (p <= 1))


def test_training_is_deterministic(raw_df):
    df = clean(raw_df)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    p1 = build_model().fit(X, y).predict_proba(X)
    p2 = build_model().fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p1, p2)


def test_unknown_category_does_not_crash(fitted):
    """The encoder ignores unknown categories (see Section 1.5.1): documented behaviour."""
    model, X = fitted
    X_new = X.head(3).copy()
    X_new["Contract"] = "Month to month"
    assert model.predict_proba(X_new).shape == (3, 2)


def test_column_order_does_not_matter(fitted):
    model, X = fitted
    shuffled = X[list(reversed(X.columns))]
    np.testing.assert_allclose(model.predict_proba(X), model.predict_proba(shuffled))


def test_gradient_boosting_pipeline(raw_df):
    df = clean(raw_df)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    model = build_model("hist_gradient_boosting", {"max_iter": 20, "random_state": 0}).fit(X, y)
    assert model.predict_proba(X).shape == (len(X), 2)


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValueError, match="unknown model type"):
        build_model("random_forest")
