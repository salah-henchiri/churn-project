"""Feature lists and the preprocessing step of the model."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "TechSupport",
    "OnlineSecurity",
    "SeniorCitizen",
]
TARGET = "Churn"


def build_preprocessor() -> ColumnTransformer:
    """Standardise the numeric columns and one-hot encode the categorical ones.

    The encoder returns a dense matrix so that every model type accepts it.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
