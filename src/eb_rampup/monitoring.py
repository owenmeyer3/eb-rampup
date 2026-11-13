"""Measure how much new data differs from the data a model was trained on."""

import numpy as np
import pandas as pd

from eb_rampup.features import TARGET, experience_loss_ratio, sample_weights
from eb_rampup.train import weighted_mae

NUMERIC_FEATURES = [
    "group_size", "avg_age", "pct_female", "avg_salary",
    "elimination_days", "benefit_pct", "experience_lr",
]
CATEGORICAL_FEATURES = ["industry", "region"]
EPSILON = 1e-4  # stands in for an empty bin, so log() never sees zero


def _psi_from_shares(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, EPSILON, None)
    actual = np.clip(actual, EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index for a numeric column.

    Bins come from the expected (reference) data's quantiles, so each bin holds
    about the same share of reference rows. PSI then measures how far the
    actual data's shares have moved.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    inner = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.concatenate([[-np.inf], inner, [np.inf]])
    e_counts = np.histogram(expected, edges)[0]
    a_counts = np.histogram(actual, edges)[0]
    return _psi_from_shares(e_counts / e_counts.sum(), a_counts / a_counts.sum())


def psi_categorical(expected, actual) -> float:
    """PSI for a categorical column, using each category as a bin."""
    e_shares = pd.Series(expected).value_counts(normalize=True)
    a_shares = pd.Series(actual).value_counts(normalize=True)
    categories = e_shares.index.union(a_shares.index)
    e = e_shares.reindex(categories, fill_value=0).to_numpy()
    a = a_shares.reindex(categories, fill_value=0).to_numpy()
    return _psi_from_shares(e, a)


def _with_experience(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(experience_lr=experience_loss_ratio(df))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    """PSI for every monitored input, keyed by column name."""
    reference, current = _with_experience(reference), _with_experience(current)
    drift = {c: psi(reference[c], current[c]) for c in NUMERIC_FEATURES}
    drift.update({c: psi_categorical(reference[c], current[c]) for c in CATEGORICAL_FEATURES})
    return drift


def build_report(model, reference: pd.DataFrame, current: pd.DataFrame,
                 training_mae: float) -> dict:
    """Everything the retraining policy needs to make a decision."""
    report = {
        "model_version": model.version,
        "feature_psi": feature_drift(reference, current),
        "prediction_psi": psi(model.predict(reference), model.predict(current)),
        "mae_at_training": training_mae,
        "mae_now": None,
    }
    if TARGET in current.columns:  # outcomes have arrived for this batch
        report["mae_now"] = weighted_mae(
            current[TARGET], model.predict(current), sample_weights(current)
        )
    return report