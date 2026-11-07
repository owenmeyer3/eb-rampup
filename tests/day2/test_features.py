from cmath import sqrt
import pandas as pd

from eb_rampup.data import generate_groups
from eb_rampup.features import (
    FEATURE_COLUMNS,
    build_features,
    experience_loss_ratio,
    sample_weights,
    credibility
)

def test_columns_match_contract(tiny_groups):
    X = build_features(tiny_groups)
    assert list(X.columns) == FEATURE_COLUMNS


def test_loss_ratio(tiny_groups):
    """Check that the experience_loss_ratio function works as expected."""
    lr = experience_loss_ratio(tiny_groups)
    pd.testing.assert_series_equal(lr, pd.Series([0.7, 0.6, 5.0], index=tiny_groups.index), check_names=False)

def test_credibility(tiny_groups):
    """Check that the credibility function works as expected."""
    cred = credibility(tiny_groups)
    pd.testing.assert_series_equal(cred, pd.Series([sqrt(0.1).real, 1.0, 0.1], index=tiny_groups.index), check_names=False)

def test_ohe(tiny_groups):
    """Check that the one-hot encoding works as expected."""
    X = build_features(tiny_groups)
    assert X["industry_construction"].iloc[0] == 1
    assert X["industry_technology"].iloc[1] == 1
    assert X["industry_retail"].iloc[2] == 1
    assert X["region_midwest"].iloc[0] == 1
    assert X["region_west"].iloc[1] == 1
    assert X["region_midwest"].iloc[2] == 1

def test_all_industries_exist(tiny_groups):
    X = build_features(tiny_groups)
    industries_cols_in_X = [col for col in X.columns if col.startswith("industry_")]
    expected_industry_cols = [col for col in FEATURE_COLUMNS if col.startswith("industry_")]
    assert set(industries_cols_in_X) == set(expected_industry_cols) 

def test_for_missing_values():
    generated_df = generate_groups(500)
    X = build_features(generated_df)
    assert not X.isnull().values.any(), "There are missing values in the feature DataFrame."

def test_build_features_does_not_modify_input(tiny_groups):
    original = tiny_groups.copy(deep=True)

    build_features(tiny_groups)

    pd.testing.assert_frame_equal(tiny_groups, original)

def test_all_groups_exist_in_features():
    generated_df = generate_groups(500)
    X = build_features(generated_df)
    assert set(generated_df.index) == set(X.index), "Not all groups from the original DataFrame are present in the features DataFrame."

def test_weights_are_equal_to_premium(tiny_groups):
    weights = sample_weights(tiny_groups)
    expected = pd.Series([100.0, 1_000.0, 10.0], index=tiny_groups.index)

    pd.testing.assert_series_equal(weights, expected)