"""Checks that a batch of employer-group data is fit to train on.

Returns every problem found (not just the first), so one failed run tells you
everything that needs fixing.
"""

import pandas as pd

from eb_rampup.data import INDUSTRY_RISK, REGIONS, YEARS

REQUIRED_COLUMNS = [
    "group_id", "industry", "region", "group_size", "avg_age", "pct_female",
    "avg_salary", "elimination_days", "benefit_pct", "loss_ratio_next",
    *[f"premium_y{y}" for y in YEARS],
    *[f"claims_y{y}" for y in YEARS],
]
RANGES = {
    "group_size": (2, 20_000),
    "avg_age": (18, 70),
    "pct_female": (0, 1),
    "avg_salary": (10_000, 1_000_000),
    "loss_ratio_next": (0, 50),
}


def validate_groups(df: pd.DataFrame, min_rows: int = 1_000) -> list[str]:
    problems = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        # Later checks would just fail on the missing columns, so stop here
        return [f"missing columns: {missing}"]

    if len(df) < min_rows:
        problems.append(f"only {len(df)} rows, expected at least {min_rows}")

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    for col, n in null_counts[null_counts > 0].items():
        problems.append(f"{col}: {n} null values")

    if df["group_id"].duplicated().any():
        problems.append(f"{df['group_id'].duplicated().sum()} duplicate group_ids")

    for col, (low, high) in RANGES.items():
        bad = (~df[col].between(low, high)) & df[col].notna()
        if bad.any():
            problems.append(f"{col}: {bad.sum()} values outside [{low}, {high}]")

    for col in [f"premium_y{y}" for y in YEARS]:
        if (df[col] <= 0).any():
            problems.append(f"{col}: non-positive premiums")

    unknown = set(df["industry"].dropna()) - set(INDUSTRY_RISK)
    if unknown:
        problems.append(f"unknown industries: {sorted(unknown)}")
    unknown = set(df["region"].dropna()) - set(REGIONS)
    if unknown:
        problems.append(f"unknown regions: {sorted(unknown)}")

    return problems
