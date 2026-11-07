import json
import time
from pathlib import Path

import numpy as np
import pytest

from eb_rampup.data import generate_groups
from eb_rampup.features import TARGET, build_features, sample_weights
from eb_rampup.train import (
    MODEL_FILENAME,
    evaluate,
    save_artifacts,
    split,
    to_local_path,
    train_model,
    weighted_mae,
)

FAST = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3, "random_state": 0}


@pytest.fixture(scope="module")
def small_df():
    return generate_groups(200, seed=11)


@pytest.fixture(scope="module")
def fast_model(small_df):
    return train_model(build_features(small_df), small_df[TARGET], sample_weights(small_df), FAST)


def test_training_smoke(small_df):
    start = time.perf_counter()
    model = train_model(build_features(small_df), small_df[TARGET], sample_weights(small_df), FAST)
    preds = model.predict(build_features(small_df).to_numpy())
    assert preds.shape == (len(small_df),)
    assert np.isfinite(preds).all()
    assert time.perf_counter() - start < 5

def test_weighted_mae():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])
    w = np.array([1, 1, 2])
    # mae = (0*1 + 0*1 + 1*2) / (1+1+2) = 2/4 = 0.5
    assert weighted_mae(y_true, y_pred, w) == 0.5

@pytest.mark.parametrize(
    "uri, expected",
    [
        ("/path/to/model.bst", "/path/to/model.bst"),
        ("gs://path/to/model.bst", "/gcs/path/to/model.bst"),
        ("/gcs/path/to/model.bst", "/gcs/path/to/model.bst"),
    ],
)
def test_to_local_path(uri, expected):
    assert to_local_path(uri) == Path(expected)

@pytest.mark.parametrize(
    "model_dir", ["artifacts/model/"]
)
def test_save_artifacts(model_dir, fast_model):
    out = save_artifacts(fast_model, model_dir, FAST)
    # Check that the files exist
    assert (out / MODEL_FILENAME).exists()
    assert (out / "features.json").exists()
    assert (out / "metrics.json").exists()
    # Check that the metrics file is valid JSON
    with open(out / "metrics.json") as f:
        loaded_params = json.load(f)
    assert loaded_params == FAST

def test_model_evaluation(fast_model, small_df):
    train_df, test_df = split(small_df, test_frac=0.2, seed=42)
    results = evaluate(fast_model, train_df, test_df)
    # Check that the results dictionary has the expected keys
    expected_keys = {"mae", "baseline_book_mae", "baseline_history_mae", "n_train", "n_test"}
    assert expected_keys.issubset(results.keys())   
