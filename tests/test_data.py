"""Tests of the data cleaning step."""

import pandas as pd
import pytest

from churn_project.data import clean


def test_total_charges_becomes_numeric(raw_df):
    out = clean(raw_df)
    assert pd.api.types.is_float_dtype(out["TotalCharges"])


def test_blank_total_charges_become_zero(raw_df):
    out = clean(raw_df)
    new_customers = raw_df["tenure"] == 0
    assert new_customers.any()
    assert (out.loc[new_customers, "TotalCharges"] == 0).all()


def test_target_is_binary(raw_df):
    out = clean(raw_df)
    assert set(out["Churn"].unique()) <= {0, 1}


def test_missing_column_is_rejected(raw_df):
    with pytest.raises(ValueError, match="missing columns"):
        clean(raw_df.drop(columns=["Contract"]))


@pytest.mark.parametrize("bad_value", ["yes", "1", "maybe"])
def test_unexpected_target_value_is_rejected(raw_df, bad_value):
    raw_df.loc[0, "Churn"] = bad_value
    with pytest.raises(ValueError, match="Churn"):
        clean(raw_df)


def test_raw_data_is_not_modified(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)
