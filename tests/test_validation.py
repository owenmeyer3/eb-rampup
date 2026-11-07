import numpy as np
import pytest

from eb_rampup.data import generate_groups
from eb_rampup.validation import validate_groups


def test_generated_data_is_valid():
    assert validate_groups(generate_groups(2_000, seed=5)) == []

def test_tinygroup_enough_rows(tiny_groups):
    assert validate_groups(tiny_groups,1) == []

def test_tinygroup_too_few_rows(tiny_groups):
    assert any("expected at least" in problem for problem in validate_groups(tiny_groups))

def test_when_missing_col(tiny_groups):
    df = tiny_groups.drop(columns=["avg_age"])
    assert any("missing columns" in problem for problem in validate_groups(df, min_rows=1))

def test_when_nulls(tiny_groups):
    tiny_groups.loc[0, "avg_salary"] = np.nan
    assert any("null values" in problem for problem in validate_groups(tiny_groups, min_rows=1))

def test_when_duplicate_group_id(tiny_groups):
    tiny_groups.loc[1, "group_id"] = tiny_groups.loc[0, "group_id"]
    assert any("duplicate group_ids" in problem for problem in validate_groups(tiny_groups, min_rows=1))

@pytest.mark.parametrize(
    "col,test_val", [
    ("group_size", -1), ("avg_age", -1), ("pct_female", 10), ("avg_salary", 2_000_000), ("loss_ratio_next", 51)
    ]
)
def test_when_out_of_range(col,test_val,tiny_groups):
    tiny_groups.loc[0, col] = test_val
    assert any("values outside" in problem and col in problem for problem in validate_groups(tiny_groups, min_rows=1))

def test_multiple_problems(tiny_groups):
    tiny_groups.loc[0, "avg_salary"] = np.nan
    tiny_groups.loc[1, "group_id"] = tiny_groups.loc[0, "group_id"]
    problems = validate_groups(tiny_groups, min_rows=1)
    assert any("null values" in problem for problem in problems)
    assert any("duplicate group_ids" in problem for problem in problems)