"""Train the loss-ratio model. Runs the same way on your laptop and on Vertex AI.

    python -m eb_rampup.train                              # local, generated data
    python -m eb_rampup.train --project P --table eb_rampup.groups   # from BigQuery
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from eb_rampup.features import FEATURE_COLUMNS, TARGET, build_features, sample_weights

PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
}
MODEL_FILENAME = "model.bst"  # the name Vertex's prebuilt XGBoost container looks for


def split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Random train/test split, reproducible via the seed."""
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(df)) < test_frac
    return df[~is_test], df[is_test]


def train_model(X: pd.DataFrame, y: pd.Series, w: pd.Series, params: dict | None = None):
    model = xgb.XGBRegressor(**(params or PARAMS))
    # Train on a plain array: the serving container sends arrays without
    # column names, so the model must not expect names.
    model.fit(X.to_numpy(), y.to_numpy(), sample_weight=w.to_numpy())
    return model


def weighted_mae(y_true, y_pred, w) -> float:
    y_true, y_pred, w = map(np.asarray, (y_true, y_pred, w))
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))


def evaluate(model, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Score the model and two baselines on the test set (lower is better)."""
    X_test = build_features(test_df)
    y, w = test_df[TARGET], sample_weights(test_df)
    pred = model.predict(X_test.to_numpy())

    # Baseline 1: everyone gets the book's overall loss ratio
    w_train = sample_weights(train_df)
    book_lr = float(np.sum(train_df[TARGET] * w_train) / np.sum(w_train))
    # Baseline 2: each group gets its own past loss ratio
    own_history = X_test["experience_lr"]

    return {
        "mae": weighted_mae(y, pred, w),
        "baseline_book_mae": weighted_mae(y, np.full(len(y), book_lr), w),
        "baseline_history_mae": weighted_mae(y, own_history, w),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }


def to_local_path(uri: str) -> Path:
    """Vertex mounts buckets at /gcs/, so gs://bucket/x is readable at /gcs/bucket/x."""
    if uri.startswith("gs://"):
        return Path("/gcs") / uri.removeprefix("gs://")
    return Path(uri)


def save_artifacts(model, model_dir: str, metrics: dict) -> Path:
    out = to_local_path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(out / MODEL_FILENAME))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "features.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    return out


def load_data(args) -> pd.DataFrame:
    if args.table:
        from google.cloud import bigquery

        from eb_rampup.bq import load_table

        return load_table(bigquery.Client(project=args.project), args.table)
    from eb_rampup.data import generate_groups

    return generate_groups(args.n_rows, seed=args.seed)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Train the loss-ratio model.")
    parser.add_argument("--project")
    parser.add_argument("--table", help="dataset.table in BigQuery; omit to use generated data")
    parser.add_argument("--n-rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    # Vertex AI sets AIP_MODEL_DIR when the job has a base output directory
    parser.add_argument("--model-dir", default=os.environ.get("AIP_MODEL_DIR", "artifacts/model"))
    args = parser.parse_args(argv)

    df = load_data(args)
    train_df, test_df = split(df, seed=args.seed)
    model = train_model(build_features(train_df), train_df[TARGET], sample_weights(train_df))
    metrics = evaluate(model, train_df, test_df)

    out = save_artifacts(model, args.model_dir, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {out}")
    return metrics


if __name__ == "__main__":
    # main(["--n-rows", "300", ...])
    main()
