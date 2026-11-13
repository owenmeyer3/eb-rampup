import numpy as np
import pytest

from eb_rampup.data import generate_groups
from eb_rampup.monitoring import build_report, feature_drift, psi, psi_categorical


def test_identical_data_has_no_drift():
    values = np.random.default_rng(0).normal(42, 6, 5_000)
    assert psi(values, values) == pytest.approx(0.0, abs=1e-9)

def test_large_mean_shift():
    values = np.random.default_rng(0).normal(42, 6, 5_000)
    values_1 = np.random.default_rng(0).normal(47, 6, 5_000)
    assert psi(values, values_1) > 0.25

def test_small_mean_shift():
    values = np.random.default_rng(0).normal(42, 6, 5_000)
    values_1 = np.random.default_rng(0).normal(42.5, 6, 5_000)
    assert psi(values, values_1) < 0.1

def test_stable_pop():
    values = np.random.default_rng(0).normal(42, 6, 10_000)
    values_1 = np.random.default_rng(0).normal(42, 6, 10_000)
    assert psi(values, values_1) < 0.2

def test_binary_column_dist_shift():
    elimination_days_vals = [90, 180]
    np.random.seed(42)
    split_20_80 = np.random.choice(elimination_days_vals, size=10_000, p=[0.2, 0.8])
    split_50_50 = np.random.choice(elimination_days_vals, size=10_000, p=[0.5, 0.5])
    assert psi(split_20_80, split_50_50) > 0.25

def test_categorical_new_category_is_flagged():
    before = ["Retail"] * 500 + ["Finance"] * 500
    after = ["Retail"] * 400 + ["Finance"] * 400 + ["Healthcare"] * 200
    assert psi_categorical(before, after) > 0.25


def test_feature_drift_finds_what_changed():
    before = generate_groups(10_000, seed=1)
    after = generate_groups(10_000, seed=2, age_shift=5, id_prefix="S")
    drift = feature_drift(before, after)
    assert drift["avg_age"] > 0.25    # changed on purpose
    assert drift["region"] < 0.02     # untouched

class AgeModel:
    """Fake model: predicts from age, so we know exactly how predictions should move."""

    version = "fake-1"

    def predict(self, df):
        return df["avg_age"].to_numpy() / 60

def test_report_includes_prediction_drift_and_current_error():
    before = generate_groups(5_000, seed=1)
    after = generate_groups(5_000, seed=2, age_shift=5, id_prefix="S")
    report = build_report(AgeModel(), before, after, training_mae=0.12)
    assert report["model_version"] == "fake-1"
    assert report["prediction_psi"] > 0.25
    assert report["mae_now"] is not None


def test_report_without_outcomes_has_no_current_error():
    before = generate_groups(1_000, seed=1)
    after = generate_groups(1_000, seed=2).drop(columns=["loss_ratio_next"])
    report = build_report(AgeModel(), before, after, training_mae=0.12)
    assert report["mae_now"] is None