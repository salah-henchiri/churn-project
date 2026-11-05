"""Loading and cleaning of the Telco Customer Churn data.

The raw file is not downloaded by the code any more: it is versioned with DVC
(Section 4) and obtained with `dvc pull`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from churn_project.features import CATEGORICAL, NUMERIC, TARGET

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [*NUMERIC, *CATEGORICAL, TARGET]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw data.

    - checks that every required column is present;
    - converts TotalCharges from text to a number (blank for new customers -> 0);
    - converts the target Churn from "Yes"/"No" to 1/0.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")

    out = df[REQUIRED_COLUMNS].copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    if not out[TARGET].isin(["Yes", "No"]).all():
        raise ValueError("Churn must only contain 'Yes' or 'No'")
    out[TARGET] = (out[TARGET] == "Yes").astype(int)
    return out


def load(path: Path) -> pd.DataFrame:
    """Read the raw CSV file and clean it."""
    df = clean(pd.read_csv(path))
    logger.info("loaded %d rows, churn rate %.3f", len(df), df[TARGET].mean())
    return df
