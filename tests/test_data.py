import numpy as np
import pandas as pd
import pytest
import re

from eb_rampup.data import generate_groups


@pytest.fixture(scope="module")
def groups():
    """One small dataset, built once and shared by every test in this file."""
    return generate_groups(n=2_000, seed=7)


def test_returns_requested_number_of_rows(groups):
    assert len(groups) == 2_000


def test_has_expected_columns(groups):
    expected = {
        "group_id", "industry", "region", "group_size", "avg_age", "pct_female",
        "avg_salary", "elimination_days", "benefit_pct", "loss_ratio_next",
    }
    assert expected.issubset(groups.columns)

def test_default_inputs(groups):
    default_groups = generate_groups(
        n=2_000, #n: int,
        seed=7, #seed: int = 42,
        age_shift = 0.0, #age_shift: float = 0.0,
        industry_weights = None, #industry_weights: dict[str, float] | None = None,
        claims_trend = 1.0, #claims_trend: float = 1.0,
        id_prefix = "G", #id_prefix: str = "G",
    ) 
    pd.testing.assert_frame_equal(groups, default_groups)

def test_age_shift(groups):
    age_shift = 5.0
    shifted_groups = generate_groups(n=2_000, seed=7, age_shift=age_shift)
    groups_mean_age = np.mean(groups['avg_age'])
    shifted_groups_mean_age = np.mean(shifted_groups['avg_age'])
    assert groups_mean_age+age_shift == pytest.approx(shifted_groups_mean_age, abs=0.5)

def test_claims_trend(groups):
    ct_1 = 1.2
    shifted_groups = generate_groups(n=2_000, seed=7, claims_trend=ct_1)
    groups_lr = np.mean(groups['loss_ratio_next'])
    shifted_groups_lr = np.mean(shifted_groups['loss_ratio_next'])
    assert groups_lr/shifted_groups_lr== pytest.approx(1/ct_1, abs=0.5)

def test_indt_weights_leave_out_other_indts():
    groups_1 = generate_groups(n=2_000, seed=7, industry_weights={"Healthcare": 1.0})
    all_groups = groups_1['industry'].unique()
    assert len(all_groups) == 1
    assert "Healthcare" in all_groups

def test_unk_indt():
    with pytest.raises(ValueError, match=re.escape("Unknown industries: ['unk']")):
        generate_groups(n=2_000, seed=7, industry_weights={"unk": 1.0})

def test_group_id_prefix():
    groups_1=generate_groups(n=2_000, seed=7, id_prefix='T')
    assert (groups_1['group_id'].str.startswith('T')).all()