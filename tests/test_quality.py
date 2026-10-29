"""Quality test on the real data: the model must not become worse than expected.

The threshold 0.82 is below the lowest test AUC observed over 100 random splits
(0.827, Section 2.8): the test fails on a real regression, not on random noise.
It is skipped when the data file has not been downloaded.
"""

from pathlib import Path

import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from churn_project.data import load
from churn_project.features import CATEGORICAL, NUMERIC, TARGET
from churn_project.model import build_model

DATA = Path("data/raw/telco_churn.csv")
MIN_AUC = 0.82


@pytest.mark.skipif(not DATA.exists(), reason="real data not downloaded")
def test_model_auc_is_above_threshold():
    df = load(DATA)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    model = build_model({"max_iter": 1000}).fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    assert auc >= MIN_AUC, f"AUC {auc:.3f} is below the threshold {MIN_AUC}"
