"""Shared test fixtures: a small, fake but realistic raw data set."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """200 fake customers with the same columns and value types as the real file."""
    rng = np.random.default_rng(0)
    n = 200
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(18, 119, n).round(2)
    total = (tenure * monthly).round(2).astype(str)
    total[tenure == 0] = " "  # blank, as in the real file
    return pd.DataFrame(
        {
            "customerID": [f"{i:04d}-TEST" for i in range(n)],
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Churn": rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
        }
    )
