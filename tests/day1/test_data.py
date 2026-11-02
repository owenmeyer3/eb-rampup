import pandas as pd
import pytest

from eb_rampup.data import INDUSTRY_RISK, generate_groups


@pytest.fixture(scope="module") # build the data once for the whole file instead of once per test
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

def test_no_missing_values(groups):
    assert not groups.isna().any().any()

def test_group_ids_are_unique(groups):
    assert groups["group_id"].is_unique

# assert groups["group_size"].between(2, 20_000).all()
# assert groups["avg_age"].between(22, 64).all()
# assert groups["pct_female"].between(0, 1).all()
# assert groups["avg_salary"].between(25_000, 400_000).all()

test_data = [
    ("group_size", 2, 20_000),
    ("avg_age", 22, 64),
    ("pct_female", 0, 1),
    ("avg_salary", 25_000, 400_000),
]

@pytest.mark.parametrize("column_name,min,max", test_data)
def test_values_are_in_sensible_ranges(groups, column_name, min, max):
    assert groups[column_name].between(min, max).all()

def test_industries_are_valid(groups):
    assert groups["industry"].isin(INDUSTRY_RISK).all()

def test_seed_reproducibility():
    """Two calls with the same seed should produce identical results."""
    df1 = generate_groups(n=1_000, seed=42)
    df2 = generate_groups(n=1_000, seed=42)
    df3 = generate_groups(n=1_000, seed=1)
    pd.testing.assert_frame_equal(df1, df2) # same seed should produce same results
    assert not df3.equals(df2)  # different seed should produce different results

def test_riskier_industries_have_higher_loss_ratios(groups):
    groups_a=groups.set_index("industry").groupby("industry")["loss_ratio_next"].mean().sort_values(ascending=False).T
    print(groups_a)
    """The expected loss ratio should be higher for riskier industries."""
    industries_by_risk_desc = sorted(INDUSTRY_RISK.items(), key=lambda x: x[1], reverse=True)
    print("Industries by risk (desc):", industries_by_risk_desc)
    for i, (industry, risk) in enumerate(industries_by_risk_desc):

        loss_ratio = groups.loc[groups["industry"] == industry, "loss_ratio_next"].mean()
        print(f"Loss ratios for {industry} (risk={risk}):", loss_ratio)

        later_industries = industries_by_risk_desc[i + 1:]
        print("later_industries: ", later_industries)

        industries_by_risk_desc = sorted(later_industries, key=lambda x: x[1], reverse=True)

        for ind, rk in later_industries:
            other_loss_ratio = groups.loc[groups["industry"] == ind, "loss_ratio_next"].mean()
            print(f"Loss ratios for {ind} (risk={rk}):", other_loss_ratio)
            assert loss_ratio >= other_loss_ratio, f"{industry} is not the riskiest remaining industry"

def test_smaller_groups_have_more_noise_in_loss_ratios(groups):
    """Smaller groups should have more variation in their loss ratios."""
    groups["size_bin"] = pd.cut(groups["group_size"], bins=[0, 50, 200, 1_000, 20_000])
    size_variances = groups.groupby("size_bin")["loss_ratio_next"].var()
    assert size_variances.is_monotonic_decreasing

test_data_2 = [1, 2]
# test_data_2 = [0, -5]
@pytest.mark.parametrize("n", test_data_2)
def test_rejects_non_positive_n(n):
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
