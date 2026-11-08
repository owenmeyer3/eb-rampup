"""Batch scoring: read groups from BigQuery, score with the champion, append results.

    python -m eb_rampup.score --project P --table eb_rampup.groups --output eb_rampup.scores
"""

import argparse
from datetime import datetime, timezone

import pandas as pd

from eb_rampup.serving import LoadedModel, apply_business_rules, load_model_dir


def score_frame(model: LoadedModel, df: pd.DataFrame) -> pd.DataFrame:
    scores = apply_business_rules(df, model.predict(df))
    scores["model_version"] = model.version
    scores["scored_at"] = datetime.now(timezone.utc)
    return scores


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Score groups with the champion model.")
    p.add_argument("--project", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--model-name", default="loss-ratio-ltd")
    p.add_argument("--model-dir", help="score with a local model instead of the champion")
    p.add_argument("--table", default="eb_rampup.groups")
    p.add_argument("--output", default="eb_rampup.scores")
    args = p.parse_args(argv)

    from google.cloud import bigquery

    from eb_rampup.bq import load_table

    if args.model_dir:
        model = load_model_dir(args.model_dir)
    else:
        from eb_rampup.registry import load_champion

        model = load_champion(args.project, args.region, args.model_name)

    client = bigquery.Client(project=args.project)
    scores = score_frame(model, load_table(client, args.table))

    # Append, so every run is kept: day 5's monitoring compares runs over time
    job = client.load_table_from_dataframe(
        scores, args.output,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    )
    job.result()
    print(f"Wrote {len(scores):,} scores from model {model.version} to {args.output}")


if __name__ == "__main__":
    main()
