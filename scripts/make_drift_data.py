"""Create two "next year" tables in BigQuery, each with a different kind of change.

    python scripts/make_drift_data.py --project $PROJECT_ID
"""

import argparse

from google.cloud import bigquery

from eb_rampup.data import generate_groups

# Scenario B's industry mix: more healthcare and construction, less tech and finance
SHIFTED_MIX = {
    "Healthcare": 2.0, "Construction": 2.0, "Transportation": 1.5, "Manufacturing": 1.5,
    "Retail": 1.0, "Education": 1.0, "Finance": 0.6, "Technology": 0.5,
    "Professional Services": 0.5,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--n", type=int, default=20_000)
    args = p.parse_args()

    tables = {
        # A: same kinds of groups, but claims costs up 15% (medical/wage inflation)
        "eb_rampup.groups_trend": generate_groups(
            args.n, seed=101, claims_trend=1.15, id_prefix="T"),
        # B: an older, riskier book of business, and costs up further
        "eb_rampup.groups_shift": generate_groups(
            args.n, seed=202, claims_trend=1.40, age_shift=5,
            industry_weights=SHIFTED_MIX, id_prefix="S"),
    }
    client = bigquery.Client(project=args.project)
    config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    for table, df in tables.items():
        client.load_table_from_dataframe(df, table, job_config=config).result()
        print(f"Loaded {len(df):,} rows into {table}")


if __name__ == "__main__":
    main()