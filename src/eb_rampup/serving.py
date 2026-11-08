"""Load a trained model and score groups with it.

Batch scoring and the API both use this module, so they can't drift apart.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from eb_rampup.features import build_features

REFER_MIN_LIVES = 10
REFER_MAX_LOSS_RATIO = 1.5


class FeatureMismatchError(RuntimeError):
    """The model was trained on different features than the code now builds."""


@dataclass
class LoadedModel:
    booster: xgb.Booster
    features: list[str]
    target_transform: str | None = None
    version: str = "local"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = build_features(df)
        if list(X.columns) != self.features:
            differences = sorted(set(X.columns) ^ set(self.features))
            raise FeatureMismatchError(
                f"Model {self.version} expects {len(self.features)} features, "
                f"code builds {len(X.columns)}; columns that differ: {differences or 'order only'}"
            )
        raw = self.booster.predict(xgb.DMatrix(X.to_numpy()))
        return np.exp(raw) if self.target_transform == "log" else raw


def load_model_dir(path, version: str = "local") -> LoadedModel:
    """Load the files save_artifacts wrote: model.bst, features.json, model_info.json."""
    path = Path(path)
    booster = xgb.Booster()
    booster.load_model(str(path / "model.bst"))
    features = json.loads((path / "features.json").read_text())
    info_file = path / "model_info.json"  # only exists if you added the log-target change
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    return LoadedModel(booster, features, info.get("target_transform"), version)


def apply_business_rules(df: pd.DataFrame, predictions) -> pd.DataFrame:
    """Turn raw predictions into what underwriters see, including referral flags."""
    out = pd.DataFrame(
        {
            "group_id": df["group_id"].to_numpy(),
            "predicted_loss_ratio": np.round(np.asarray(predictions, dtype=float), 4),
        }
    )
    small = df["group_size"].to_numpy() < REFER_MIN_LIVES
    high = out["predicted_loss_ratio"].to_numpy() > REFER_MAX_LOSS_RATIO
    reasons = []
    for is_small, is_high in zip(small, high):
        r = []
        if is_small:
            r.append(f"fewer than {REFER_MIN_LIVES} lives")
        if is_high:
            r.append(f"predicted loss ratio above {REFER_MAX_LOSS_RATIO}")
        reasons.append("; ".join(r))
    out["refer_to_underwriter"] = small | high
    out["refer_reason"] = reasons
    return out
