"""Fixtures here are available to every test file without importing."""

import pandas as pd
import pytest

RANGES = {
    "group_size": (2, 20_000),
    "avg_age": (18, 70),
    "pct_female": (0, 1),
    "avg_salary": (10_000, 1_000_000),
    "loss_ratio_next": (0, 50),
}

@pytest.fixture(scope="function")
def tiny_groups() -> pd.DataFrame:
    """Three hand-made groups with numbers easy to check by hand."""
    return pd.DataFrame(
        {
            "group_id": ["A", "B", "C"],
            "industry": ["Construction", "Technology", "Retail"],
            "region": ["Midwest", "West", "Midwest"],
            "group_size": [100, 1_000, 10],
            "avg_age": [45.0, 35.0, 50.0],
            "pct_female": [0.2, 0.5, 0.8],
            "avg_salary": [50_000.0, 120_000.0, 30_000.0],
            "elimination_days": [90, 180, 90],
            "benefit_pct": [0.6, 0.667, 0.5],
            "premium_y1": [100.0, 1_000.0, 10.0],
            "premium_y2": [100.0, 1_000.0, 10.0],
            "premium_y3": [100.0, 1_000.0, 10.0],
            "claims_y1": [50.0, 500.0, 0.0],
            "claims_y2": [70.0, 600.0, 0.0],
            "claims_y3": [90.0, 700.0, 300.0],
            "loss_ratio_next": [0.7, 0.6, 1.2],
        }
    )

@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """A small real model, trained and saved once for the whole test run."""
    from eb_rampup.data import generate_groups
    from eb_rampup.features import TARGET, build_features, sample_weights
    from eb_rampup.train import save_artifacts, train_model

    df = generate_groups(300, seed=3)
    fast = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.3, "random_state": 0}
    model = train_model(build_features(df), df[TARGET], sample_weights(df), fast)
    out = tmp_path_factory.mktemp("model")
    save_artifacts(model, str(out), {"mae": 0.0})
    return out


@pytest.fixture
def valid_group() -> dict:
    """One group as an API caller would send it."""
    return {
        "group_id": "API-1",
        "industry": "Manufacturing",
        "region": "Midwest",
        "group_size": 250,
        "avg_age": 44.0,
        "pct_female": 0.35,
        "avg_salary": 68_000.0,
        "elimination_days": 90,
        "benefit_pct": 0.6,
        "premium_y1": 100_000.0,
        "premium_y2": 102_000.0,
        "premium_y3": 104_000.0,
        "claims_y1": 70_000.0,
        "claims_y2": 81_000.0,
        "claims_y3": 66_000.0,
    }
