# import json
import numpy as np
import pandas as pd
import pytest, json

from eb_rampup.data import generate_groups
from eb_rampup.features import FEATURE_COLUMNS
from eb_rampup.score import score_frame
from eb_rampup.serving import (
    FeatureMismatchError,
    LoadedModel,
    apply_business_rules,
    load_model_dir,
)

class FakeBooster():
    def predict(self, groups):
        return [0.5, 1.0, 2.0]

def test_loaded_model_scores_every_row(model_dir):
    model = load_model_dir(model_dir)
    df = generate_groups(50, seed=8)
    preds = model.predict(df)
    assert preds.shape == (50,)
    assert np.isfinite(preds).all()

def test_load_model_reads_requested_model(tiny_groups):
    booster =FakeBooster()
    model = LoadedModel(booster, FEATURE_COLUMNS, target_transform="log")
    assert np.log(model.predict(tiny_groups)) == pytest.approx([0.5, 1.0, 2.0])

def test_no_transoform(tiny_groups):
    booster =FakeBooster()
    log_model = LoadedModel(booster, FEATURE_COLUMNS, target_transform="log")
    raw_model = LoadedModel(booster, FEATURE_COLUMNS, target_transform=None)
    assert np.log(log_model.predict(tiny_groups)) == pytest.approx(raw_model.predict(tiny_groups))

def test_missing_feature(tiny_groups):
    booster =FakeBooster()
    model=LoadedModel(booster, FEATURE_COLUMNS[:-1], target_transform=None)

    with pytest.raises(FeatureMismatchError, match="expects"):
        model.predict(tiny_groups)

def test_reads_target_transform_when_present(model_dir, tmp_path):
    for name in ["model.bst", "features.json"]:
        (tmp_path / name).write_bytes((model_dir / name).read_bytes())
    assert load_model_dir(tmp_path).target_transform is None
    (tmp_path / "model_info.json").write_text(json.dumps({"target_transform": "log"}))
    assert load_model_dir(tmp_path).target_transform == "log"

@pytest.mark.parametrize(
    "size, prediction, refer, reason",
    [
        (250, 0.80, False, ""),
        (5, 0.80, True, "fewer than 10 lives"),
        (250, 1.90, True, "predicted loss ratio above 1.5"),
        (5, 1.90, True, "fewer than 10 lives; predicted loss ratio above 1.5"),
        (10, 1.50, False, ""),   # boundaries: exactly 10 lives and exactly 1.5 are fine
    ],
)
def test_business_rules(size, prediction, refer, reason):
    df = pd.DataFrame({"group_id": ["G"], "group_size": [size]})
    row = apply_business_rules(df, [prediction]).iloc[0]
    assert row["refer_to_underwriter"] == refer
    assert row["refer_reason"] == reason

def test_score_frame_columns(model_dir):

    model = load_model_dir(model_dir, version="3")
    df = generate_groups(20, seed=8)
    scores = score_frame(model, df)
    assert list(scores.columns) == [
        "group_id", "predicted_loss_ratio", "refer_to_underwriter",
        "refer_reason", "model_version", "scored_at",
    ]
    assert (scores["model_version"] == "3").all()