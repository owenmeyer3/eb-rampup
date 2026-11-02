"""Synthetic employer-group data for a group long-term disability (LTD) book."""

import numpy as np
import pandas as pd

INDUSTRY_RISK = {
    "Professional Services": 0.80,
    "Finance": 0.85,
    "Technology": 0.75,
    "Education": 0.95,
    "Healthcare": 1.15,
    "Retail": 1.10,
    "Manufacturing": 1.25,
    "Construction": 1.45,
    "Transportation": 1.35,
}
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
ELIMINATION_DAYS = [90, 180]
BENEFIT_PCTS = [0.50, 0.60, 0.667]
YEARS = [1, 2, 3]

BASE_LOSS_RATIO = 0.70
PREMIUM_RATE = 0.006  # annual premium per $1 of covered payroll


def expected_loss_ratio(df: pd.DataFrame) -> pd.Series:
    """The 'true' loss ratio each group would have with no randomness."""
    industry = df["industry"].map(INDUSTRY_RISK)
    age = np.exp(0.035 * (df["avg_age"] - 42))  # older groups claim more
    female = 1 + 0.4 * (df["pct_female"] - 0.5)
    elimination = np.where(df["elimination_days"] == 180, 0.85, 1.0)
    benefit = df["benefit_pct"] / 0.60
    return BASE_LOSS_RATIO * industry * age * female * elimination * benefit


def generate_groups(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate n synthetic employer groups with 3 years of history and a target.

    Small groups get much noisier yearly results than large ones, the same
    reason real underwriters trust a 5,000-life group's history more than a
    20-life group's.
    """
    rng = np.random.default_rng(seed)
    industries = list(INDUSTRY_RISK)

    df = pd.DataFrame(
        {
            "group_id": [f"G{i:06d}" for i in range(n)],
            "industry": rng.choice(industries, size=n),
            "region": rng.choice(REGIONS, size=n),
            "group_size": np.clip(rng.lognormal(4.0, 1.2, size=n), 2, 20_000).astype(int),
            "avg_age": np.clip(rng.normal(42, 6, size=n), 22, 64).round(1),
            "pct_female": rng.beta(5, 5, size=n).round(3),
            "avg_salary": np.clip(rng.lognormal(np.log(65_000), 0.35, size=n), 25_000, 400_000).round(-2),
            "elimination_days": rng.choice(ELIMINATION_DAYS, size=n, p=[0.8, 0.2]),
            "benefit_pct": rng.choice(BENEFIT_PCTS, size=n, p=[0.3, 0.5, 0.2]),
        }
    )

    true_lr = expected_loss_ratio(df)
    premium = (df["group_size"] * df["avg_salary"] * PREMIUM_RATE).round(2)
    noise_sd = 0.10 + 1.5 / np.sqrt(df["group_size"])

    for year in YEARS:
        yearly_lr = true_lr * rng.lognormal(-(noise_sd**2) / 2, noise_sd)
        df[f"premium_y{year}"] = premium
        df[f"claims_y{year}"] = (premium * yearly_lr).round(2)

    df["loss_ratio_next"] = (true_lr * rng.lognormal(-(noise_sd**2) / 2, noise_sd)).round(4)
    return df


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate synthetic employer groups.")
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/groups.parquet")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_groups(args.n, args.seed).to_parquet(out, index=False)
    print(f"Wrote {args.n:,} groups to {out}")
