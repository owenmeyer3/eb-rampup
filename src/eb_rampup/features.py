"""Turn raw employer-group rows into the model's numeric feature matrix.

The serving container only ever sees a list of numbers per group, so the
column ORDER in FEATURE_COLUMNS is the contract between training and serving.
"""

import numpy as np
import pandas as pd

from eb_rampup.data import INDUSTRY_RISK, REGIONS, YEARS

TARGET = "loss_ratio_next"
EXPERIENCE_CAP = 5.0  # cap extreme small-group loss ratios


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_")

INDUSTRIES = sorted(INDUSTRY_RISK)
REGION_LIST = sorted(REGIONS)

NUMERIC_COLUMNS = [
    "log_group_size",
    "avg_age",
    "pct_female",
    "log_avg_salary",
    "elimination_days",
    "benefit_pct",
    "experience_lr",
]
FEATURE_COLUMNS = (
    NUMERIC_COLUMNS
    + [f"industry_{_slug(i)}" for i in INDUSTRIES]
    + [f"region_{_slug(r)}" for r in REGION_LIST]
)

def experience_loss_ratio(df: pd.DataFrame) -> pd.Series:
    """Premium-weighted loss ratio over the history years, capped."""
    claims = sum(df[f"claims_y{y}"] for y in YEARS)
    premium = sum(df[f"premium_y{y}"] for y in YEARS)
    return (claims / premium).clip(upper=EXPERIENCE_CAP)

def credibility(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.sqrt(df["group_size"] / 1000), index=df.index).clip(upper=1)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a float DataFrame with exactly FEATURE_COLUMNS, in order."""
    unknown = set(df["industry"]) - set(INDUSTRIES)
    if unknown:
        raise ValueError(f"Unknown industries: {sorted(unknown)}")
    unknown = set(df["region"]) - set(REGION_LIST)
    if unknown:
        raise ValueError(f"Unknown regions: {sorted(unknown)}")

    out = pd.DataFrame(index=df.index)
    out["log_group_size"] = np.log(df["group_size"])
    out["avg_age"] = df["avg_age"]
    out["pct_female"] = df["pct_female"]
    out["log_avg_salary"] = np.log(df["avg_salary"])
    out["elimination_days"] = df["elimination_days"]
    out["benefit_pct"] = df["benefit_pct"]
    out["experience_lr"] = experience_loss_ratio(df)
    out["credibility"] = credibility(df)
    for industry in INDUSTRIES:
        out[f"industry_{_slug(industry)}"] = (df["industry"] == industry).astype(float)
    for region in REGION_LIST:
        out[f"region_{_slug(region)}"] = (df["region"] == region).astype(float)

    return out[FEATURE_COLUMNS].astype(float)


def sample_weights(df: pd.DataFrame) -> pd.Series:
    """Weight each group by its average annual premium (bigger groups count more)."""
    return sum(df[f"premium_y{y}"] for y in YEARS) / len(YEARS)
