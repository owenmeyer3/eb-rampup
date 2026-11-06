# Day 2: Feature Code, Training, and the Model Registry

**Tue Nov 3 · about 2.5–3 hours**

By the end of today you'll have tested feature and training code in your package, a model trained on Vertex AI as your `pipeline-sa` service account, and that model registered in the Vertex AI Model Registry as version 1, alias `champion`.

| Part | What you do | Time |
| --- | --- | --- |
| 1 | Concepts: SageMaker → Vertex, and the training–serving contract | 10 min |
| 2 | Update the project | 10 min |
| 3 | Feature code and tests | 45 min |
| 4 | Training code and tests | 35 min |
| 5 | Train locally from BigQuery | 10 min |
| 6 | Test-first: add a credibility feature | 15 min |
| 7 | Train on Vertex AI and register the model | 35 min |
| 8 | Bonus: a BigQuery ML baseline | 15 min |
| 9 | Commit, self-check, and report back | 5 min |

---

## Part 1: Concepts (10 min)

### SageMaker → Vertex AI training

| SageMaker | Vertex AI |
| --- | --- |
| Estimator / `CreateTrainingJob` | `CustomJob` |
| `entry_point` script | `python_module` inside your package |
| Framework container (e.g., XGBoost image) | Prebuilt container (`executor_image_uri`) |
| `SM_MODEL_DIR` → `/opt/ml/model` → uploaded to S3 | `AIP_MODEL_DIR` → a `gs://` folder, mounted at `/gcs/...` |
| Execution role | Service account (`pipeline-sa`), passed with `service_account=` |
| Model package group + versions | Model with versions |
| Approval status (`Approved`) | Version **aliases** (`champion`, `challenger`) |

The biggest difference: on SageMaker, your script writes to a local folder and SageMaker uploads it. On Vertex, your script writes to a **Cloud Storage path** given in `AIP_MODEL_DIR`. Because Vertex mounts buckets at `/gcs/<bucket>/` (Cloud Storage FUSE, which we discussed), plain file writes work once you convert `gs://bucket/...` to `/gcs/bucket/...`.

### The training–serving contract

A trained model file is useless unless whatever serves it later feeds it **exactly** the inputs it learned from. Two things make up that contract today:

1. **Feature columns, in the same order.** Vertex's prebuilt XGBoost serving container receives each group as a plain list of numbers, with no column names. If training used `[age, size, ...]` and serving sends `[size, age, ...]`, you get confident, wrong predictions and no error. That's why today's feature code uses a fixed `FEATURE_COLUMNS` list, saves it next to the model, and has tests guarding it.
2. **Library versions.** A model saved by one XGBoost version should be loaded by the same version. Google's dedicated XGBoost *training* images are old (XGBoost 1.x), while the XGBoost *serving* image is 2.1. So today trains on a maintained Python 3.10 image that ships XGBoost 2.1.3, and the package pins `xgboost==2.1.3` everywhere. On day 4, you'll build your own container, which is how most teams avoid this problem entirely.

When these two get out of sync, it's called **training–serving skew**, and preventing it is a core MLOps job.

---

## Part 2: Update the project (10 min)

### 2a. `pyproject.toml`

Make three changes: lower the Python requirement (the training image uses Python 3.10), add XGBoost pinned, and add two dev tools.

```toml
[project]
name = "eb-rampup"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "numpy",
  "pandas",
  "pyarrow",
  "google-cloud-bigquery",
  "db-dtypes",
  "xgboost==2.1.3",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "build", "google-cloud-aiplatform"]
```

Keep the rest as it was. While you're there, add `--strict-markers` to `addopts`:

```toml
addopts = "-m 'not integration' --strict-markers"
```

Why `google-cloud-aiplatform` goes in `dev`, not `dependencies`: it's only used on your laptop to *launch* jobs. The training code itself never imports it, so the training container doesn't need it.

### 2b. `.gitignore`

Add the two new output folders:

```
artifacts/
dist/
```

### 2c. Reinstall

```bash
cd ~/path/to/eb-rampup
source .venv/bin/activate
pip install -e ".[dev]"
python -c "import xgboost; print(xgboost.__version__)"     # expect 2.1.3
```

> **Mac troubleshooting:** if importing XGBoost fails with `Library not loaded: ... libomp.dylib`, run `brew install libomp` and try again.

---

## Part 3: Feature code and tests (45 min)

### 3a. The feature code

Create `src/eb_rampup/features.py`. Read it before moving on; the design choices are explained below.

```python
"""Turn raw employer-group rows into the model's numeric feature matrix.

The serving container only ever sees a list of numbers per group, so the
column ORDER in FEATURE_COLUMNS is the contract between training and serving.
"""

import numpy as np
import pandas as pd

from eb_rampup.data import INDUSTRY_RISK, REGIONS, YEARS

TARGET = "loss_ratio_next"
EXPERIENCE_CAP = 5.0  # cap extreme small-group loss ratios


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_")


INDUSTRIES = sorted(INDUSTRY_RISK)
REGION_LIST = sorted(REGIONS)

NUMERIC_COLUMNS = [
    "log_group_size",
    "avg_age",
    "pct_female",
    "log_avg_salary",
    "elimination_days",
    "benefit_pct",
    "experience_lr",
]
FEATURE_COLUMNS = (
    NUMERIC_COLUMNS
    + [f"industry_{_slug(i)}" for i in INDUSTRIES]
    + [f"region_{_slug(r)}" for r in REGION_LIST]
)


def experience_loss_ratio(df: pd.DataFrame) -> pd.Series:
    """Premium-weighted loss ratio over the history years, capped."""
    claims = sum(df[f"claims_y{y}"] for y in YEARS)
    premium = sum(df[f"premium_y{y}"] for y in YEARS)
    return (claims / premium).clip(upper=EXPERIENCE_CAP)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a float DataFrame with exactly FEATURE_COLUMNS, in order."""
    unknown = set(df["industry"]) - set(INDUSTRIES)
    if unknown:
        raise ValueError(f"Unknown industries: {sorted(unknown)}")
    unknown = set(df["region"]) - set(REGION_LIST)
    if unknown:
        raise ValueError(f"Unknown regions: {sorted(unknown)}")

    out = pd.DataFrame(index=df.index)
    out["log_group_size"] = np.log(df["group_size"])
    out["avg_age"] = df["avg_age"]
    out["pct_female"] = df["pct_female"]
    out["log_avg_salary"] = np.log(df["avg_salary"])
    out["elimination_days"] = df["elimination_days"]
    out["benefit_pct"] = df["benefit_pct"]
    out["experience_lr"] = experience_loss_ratio(df)
    for industry in INDUSTRIES:
        out[f"industry_{_slug(industry)}"] = (df["industry"] == industry).astype(float)
    for region in REGION_LIST:
        out[f"region_{_slug(region)}"] = (df["region"] == region).astype(float)

    return out[FEATURE_COLUMNS].astype(float)


def sample_weights(df: pd.DataFrame) -> pd.Series:
    """Weight each group by its average annual premium (bigger groups count more)."""
    return sum(df[f"premium_y{y}"] for y in YEARS) / len(YEARS)
```

Design choices worth noticing:

- **Hard-coded category lists, not `pd.get_dummies`.** `get_dummies` only creates columns for categories present in *that* DataFrame. Score a batch that happens to contain only tech companies, and you get different columns than training had. The fixed lists guarantee the same columns every time.
- **Unknown categories raise an error.** Scoring a "Space Mining" company as if it were no industry at all would be silently wrong. Rejecting it loudly is safer, especially once this runs behind an API.
- **`experience_lr`** is the group's own 3-year loss ratio, premium-weighted like your Day 1 SQL. The model learns how much to trust it, and since `log_group_size` is also a feature, it can learn to trust it more for big groups. That's credibility, learned from the data.
- **Sample weights** make large groups count more in training and evaluation, for the same reason premium-weighted loss ratios were right on Day 1.

### 3b. Shared fixtures with `conftest.py`

Fixtures defined in a file named `tests/conftest.py` are available to **every** test file automatically, with no import needed. Use it for test data that several files share. Create `tests/conftest.py`:

```python
"""Fixtures here are available to every test file without importing."""

import pandas as pd
import pytest


@pytest.fixture
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
```

**Why hand-made data instead of `generate_groups`?** With three rows you can work out the right answer on paper, so a test can check exact values, not just shapes. Group C is deliberately extreme: a 10-life group with one big year of claims.

This fixture has no `scope=`, so each test gets a **fresh copy**. That matters when a test modifies the data, as one challenge below does.

### 3c. The first test

Create `tests/test_features.py`:

```python
import numpy as np
import pandas as pd
import pytest

from eb_rampup.data import generate_groups
from eb_rampup.features import (
    FEATURE_COLUMNS,
    build_features,
    experience_loss_ratio,
    sample_weights,
)


def test_columns_match_contract(tiny_groups):
    X = build_features(tiny_groups)
    assert list(X.columns) == FEATURE_COLUMNS
```

Run `pytest tests/test_features.py -v` and confirm it passes.

### 3d. Challenges: write these yourself (30 min)

Solutions are at the end; try first.

1. **Experience loss ratio by hand.** Work out group A, B, and C's 3-year loss ratio on paper, then assert `experience_loss_ratio(tiny_groups)` matches. Watch group C. *Hint:* floats rarely compare exactly, so use `pytest.approx`, e.g. `assert values == pytest.approx([0.1, 0.2])`.
2. **One-hot encoding is correct.** Group A should have `industry_construction == 1.0`, and every row should have exactly one industry column set.
3. **A one-industry batch still has every column.** Pass only group B (`tiny_groups.iloc[[1]]`, with double brackets to keep it a DataFrame) and check the columns still equal `FEATURE_COLUMNS`. This is the test that guards against the `get_dummies` bug.
4. **No missing values** in features built from 500 generated groups.
5. **The input DataFrame isn't modified.** Functions that quietly change their input are a classic pandas bug. *Hint:* copy before, compare after with `pd.testing.assert_frame_equal`.
6. **Unknown industries are rejected** with a message naming the bad value. *Hint:* `pytest.raises(..., match=...)`, which we discussed.
7. **Weights equal average annual premium.**

### 3e. Mocking: testing code that talks to BigQuery (worked example)

On Day 1, you queried BigQuery in a notebook. Training needs that too, so it moves into the package now. Create `src/eb_rampup/bq.py`:

```python
"""Small BigQuery helpers. The client is passed in so tests can use a fake."""

import pandas as pd


def load_table(client, table: str, max_rows: int | None = None) -> pd.DataFrame:
    """Read a whole table into a DataFrame.

    Uses list_rows (the table-read API) instead of a SELECT query, so it isn't
    billed for bytes scanned.
    """
    return client.list_rows(table, max_results=max_rows).to_dataframe()
```

Notice that `client` is a **parameter**, not created inside the function. This design choice is called *dependency injection*, and it makes the function testable: a test can pass in a fake client instead of a real one.

The fake comes from Python's built-in `unittest.mock.MagicMock`, an object that accepts any method call, records it, and returns whatever you tell it to. Create `tests/test_bq.py`:

```python
from unittest.mock import MagicMock

import pandas as pd

from eb_rampup.bq import load_table


def test_load_table_reads_requested_table():
    fake_client = MagicMock()
    expected = pd.DataFrame({"x": [1, 2]})
    # "when list_rows(...).to_dataframe() is called, return expected"
    fake_client.list_rows.return_value.to_dataframe.return_value = expected

    result = load_table(fake_client, "eb_rampup.groups")

    # Did our code call the client the way we intended?
    fake_client.list_rows.assert_called_once_with("eb_rampup.groups", max_results=None)
    pd.testing.assert_frame_equal(result, expected)
```

This test runs in milliseconds, needs no credentials, and costs nothing. It checks *your code's use of the client*, not BigQuery itself; that's what the Day 1 integration test is for. Together they cover both sides.

**🧩 Mini-challenge:** write a second test checking that `max_rows=100` is passed through to `list_rows`.

✅ **Check:** `pytest -v` passes, about 25 tests.

---

## Part 4: Training code and tests (35 min)

### 4a. The training code

Create `src/eb_rampup/train.py`. It runs the same way on your laptop and on Vertex AI.

```python
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
    "max_depth": 4,
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
    main()
```

Things to notice:

- **Two baselines.** A model is only worth deploying if it beats the simple alternatives: charging everyone the book's average rate, or charging each group its own past loss ratio. The metrics record both, so anyone reviewing the model can see the comparison.
- **`main(argv=None)`** accepts a list of arguments, so a test can call `main(["--n-rows", "300", ...])` without a command line.
- **Artifacts are saved together:** the model, its metrics, and its feature list. Anyone loading this model later knows exactly what inputs it expects.

### 4b. The first test: a smoke test

A **smoke test** answers "does it run at all?" quickly. It uses tiny data and tiny settings, so it doesn't check model quality. Create `tests/test_train.py`:

```python
import json
import time
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from eb_rampup.data import generate_groups
from eb_rampup.features import FEATURE_COLUMNS, TARGET, build_features, sample_weights
from eb_rampup.train import (
    MODEL_FILENAME,
    evaluate,
    main,
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
```

Notice `fast_model` is a fixture that **uses another fixture** (`small_df`). Fixtures can build on each other.

### 4c. Challenges (25 min)

8. **`weighted_mae` by hand.** Pick two predictions with errors 1 and 3 and weights 1 and 3, work out the answer on paper, and assert it.
9. **`split` is reproducible and complete.** The same seed gives the same split, train and test sizes add up to the whole, and no row is in both. *Hint:* compare `.index` values; Python sets have `isdisjoint`.
10. **`to_local_path`** turns `gs://bucket/runs/1/model` into `/gcs/bucket/runs/1/model` and leaves a local path alone. Use `parametrize` with both cases.
11. **Save and reload gives identical predictions.** This tests the training–serving handoff: save with `save_artifacts`, check `model.bst` and `features.json` exist, then load the file the way a serving container would and compare predictions. *Hint:* pytest has a built-in fixture, `tmp_path`, that gives each test its own empty temporary folder. To load: `booster = xgb.Booster(); booster.load_model(path)`, then `booster.predict(xgb.DMatrix(X))`. Compare with `np.testing.assert_allclose`.
12. **The model is actually good.** Two tests on 5,000 generated groups: the model's MAE is at least 20% better than the book-rate baseline, and on **small groups only** (under 100 lives) it beats the own-history baseline. Why small groups? Think about Day 1's volatility query; the answer is in the self-check.
13. **`main` runs end to end** with 300 generated rows, writing to `tmp_path`.

✅ **Check:** `pytest -v` passes, about 35 tests, in a few seconds.

---

## Part 5: Train locally from BigQuery (10 min)

First on generated data, then from your real table:

```bash
python -m eb_rampup.train
python -m eb_rampup.train --project $PROJECT_ID --table eb_rampup.groups
ls artifacts/model     # model.bst  metrics.json  features.json
```

Look at the printed metrics. You should see the model's `mae` well below `baseline_book_mae` and somewhat below `baseline_history_mae`. **Write down the three numbers**; you'll compare against them in Part 6 and on Vertex.

Why is the model only modestly better than history overall? Because the metric is premium-weighted, it's dominated by large groups, and a large group's own history is already a good predictor. Your Part 4 challenge 12 test shows where the model really earns its keep.

---

## Part 6: Test-first, add a credibility feature (15 min)

Actuaries use a **credibility factor** to say how much to trust a group's own history: Z = √(lives ÷ 1,000), capped at 1. A 250-life group gets Z = 0.5; anything over 1,000 lives gets full credibility, 1.0.

Add it as a feature, test first:

1. **Write the test** in `tests/test_features.py`. Work out the expected values for 4,000, 1,000, 250, and 10 lives, then write a parametrized test that imports `credibility` from `eb_rampup.features` and checks each case. *Hint:* `tiny_groups.iloc[[0]].assign(group_size=size)` gives a one-row DataFrame with the size you want.
2. **Run it and watch it fail** (`ImportError`: the function doesn't exist yet).
3. **Implement `credibility`** in `features.py`, add `"credibility"` to `NUMERIC_COLUMNS`, and set it in `build_features`.
4. **Run all tests.** Everything should pass, including `test_columns_match_contract`, since `FEATURE_COLUMNS` updated with it.
5. **Retrain** (`python -m eb_rampup.train --project $PROJECT_ID --table eb_rampup.groups`) and compare `mae` with your Part 5 numbers.

**The real question:** did it help? Gradient-boosted trees can often learn this relationship on their own from `log_group_size`, so the improvement may be tiny or nothing. That's a useful lesson: a feature that makes perfect domain sense can still add nothing to a particular model. **Measure, don't assume.** Decide whether to keep it, and note one more thing: adding a feature changed the contract. A model trained before this change expects one fewer input, which is exactly why each model's `features.json` is saved alongside it.

---

## Part 7: Train on Vertex AI and register the model (35 min)

### 7a. The launcher script

Training code lives in the package. The code that *launches* training on the cloud is separate. On day 3, a pipeline replaces this script. Create `scripts/train_on_vertex.py`:

```python
"""Build the package, train on Vertex AI, and register the model.

    python scripts/train_on_vertex.py --project $PROJECT_ID --bucket $BUCKET \
        --service-account pipeline-sa@$PROJECT_ID.iam.gserviceaccount.com
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from google.cloud import aiplatform, storage

# Google's dedicated XGBoost *training* images are old (XGBoost 1.x), so we use a
# maintained Python 3.10 image that ships XGBoost 2.1.3, matching the 2.1 serving image.
TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest"
SERVE_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"
MODEL_NAME = "loss-ratio-ltd"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_package() -> Path:
    for old in Path("dist").glob("*.tar.gz"):
        old.unlink()
    subprocess.run([sys.executable, "-m", "build", "--sdist"], check=True)
    return next(Path("dist").glob("*.tar.gz"))


def upload(bucket: str, local: Path, dest: str) -> str:
    storage.Client().bucket(bucket).blob(dest).upload_from_filename(str(local))
    return f"gs://{bucket}/{dest}"


def read_json(uri: str) -> dict:
    bucket, _, path = uri.removeprefix("gs://").partition("/")
    return json.loads(storage.Client().bucket(bucket).blob(path).download_as_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--bucket", required=True)
    p.add_argument("--table", default="eb_rampup.groups")
    p.add_argument("--service-account", required=True)
    p.add_argument("--alias", help="alias for a new version, e.g. challenger")
    args = p.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    base_dir = f"gs://{args.bucket}/training/{run_id}"
    aiplatform.init(project=args.project, location=args.region,
                    staging_bucket=f"gs://{args.bucket}/staging")

    # 1. Package the code and put it where Vertex can install it
    package = build_package()
    package_uri = upload(args.bucket, package, f"training/{run_id}/{package.name}")
    print(f"Uploaded {package_uri}")

    # 2. Run training (like a SageMaker training job)
    job = aiplatform.CustomJob(
        display_name=f"{MODEL_NAME}-train-{run_id}",
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "python_package_spec": {
                "executor_image_uri": TRAIN_IMAGE,
                "package_uris": [package_uri],
                "python_module": "eb_rampup.train",
                "args": ["--project", args.project, "--table", args.table],
            },
        }],
        base_output_dir=base_dir,  # Vertex sets AIP_MODEL_DIR = base_dir/model/
    )
    job.run(service_account=args.service_account)

    # 3. Register the trained model as a new version
    model_dir = f"{base_dir}/model"
    metrics = read_json(f"{model_dir}/metrics.json")
    existing = aiplatform.Model.list(filter=f'display_name="{MODEL_NAME}"')
    if existing:
        aliases = [args.alias] if args.alias else []
    else:
        aliases = ["champion"]  # the very first version is the champion by default

    model = aiplatform.Model.upload(
        display_name=MODEL_NAME,
        artifact_uri=model_dir,
        serving_container_image_uri=SERVE_IMAGE,
        parent_model=existing[0].resource_name if existing else None,
        is_default_version=not existing,
        version_aliases=aliases or None,
        version_description=(
            f"MAE {metrics['mae']:.4f} | book baseline {metrics['baseline_book_mae']:.4f}"
            f" | history baseline {metrics['baseline_history_mae']:.4f}"
        ),
        labels={"git_sha": git_sha(), "run_id": run_id},
    )
    print(json.dumps(metrics, indent=2))
    print(f"Registered {model.resource_name}, version {model.version_id}, aliases {aliases}")


if __name__ == "__main__":
    main()
```

What it does, in order:

1. **Builds your package** into a `.tar.gz` (`python -m build --sdist`), the same thing `pip install` builds behind the scenes, and uploads it to your bucket.
2. **Runs a `CustomJob`.** Vertex starts the training image, runs `pip install` on your package (which also installs your `dependencies`), then runs `python -m eb_rampup.train` with your arguments, as `pipeline-sa`. Submitting a job that runs as a service account is the PassRole-style permission we discussed (Service Account User); as project owner, you already have it.
3. **Registers the model.** It points the registry at the saved `model.bst`, tells it which serving image can load it (used on day 4), records the metrics in the version description, and labels it with the Git commit, so you can always trace a model version back to the exact code that trained it.

### 7b. Run it

Commit first, so the `git_sha` label points at real code:

```bash
git add . && git commit -m "Day 2: features, training, tests"

export BUCKET="${PROJECT_ID}-data"
cd /Users/owenmeyer/Desktop/ht-prep/eb-rampup
python ../scripts/train_on_vertex.py \
  --project "$PROJECT_ID" \
  --bucket "$BUCKET" \
  --service-account "pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com"
```

The job takes roughly 5–10 minutes, mostly starting the machine and pulling the image. The script prints a console link; open it and watch the logs. Each run costs a few cents.

### 7c. Look at what you made

In the console:

- **Vertex AI → Training → Custom jobs:** your job, its logs, and its machine type.
- **Cloud Storage → your bucket → `training/<run_id>/model/`:** `model.bst`, `metrics.json`, `features.json`.
- **Vertex AI → Model Registry → `loss-ratio-ltd`:** version 1 with alias `champion`, your metrics in the description, and the `git_sha` label.

✅ **Check:** the Vertex metrics should match your local BigQuery run from Part 5 (or Part 6, if you kept the credibility feature) almost exactly. Same code, same data, same seed. If they differ noticeably, something in the environment differs, and that's worth investigating.

**🧩 Challenge:** change one thing in `PARAMS` (try `max_depth: 6`), commit, and run again with `--alias challenger`. You'll now have version 2 in the registry next to the champion. Which is better? On day 3, a pipeline makes that comparison and the promotion decision automatically.

### 7d. Troubleshooting

- **Permission errors in the job logs** (BigQuery or Storage access denied): check that `pipeline-sa` has the four roles from Day 1, using the `get-iam-policy` command we used.
- **A permission error about the service account itself** when submitting: paste me the message; it's usually a missing Service Account User grant.
- **`AIP_MODEL_DIR` problems** or files ending up in `artifacts/model` inside the container: `base_output_dir` wasn't set on the job.
- Anything else: paste the error and the last 20 lines of the job log.

---

## Part 8: Bonus, a BigQuery ML baseline (15 min)

BigQuery ML trains models with SQL, inside the warehouse. For simple problems it's a fast way to get a baseline, and some teams use it in production. Run this in the BigQuery console:

```sql
CREATE OR REPLACE MODEL eb_rampup.lr_linear
OPTIONS (
  model_type = 'LINEAR_REG',
  input_label_cols = ['loss_ratio_next'],
  data_split_method = 'RANDOM',
  data_split_eval_fraction = 0.2
) AS
SELECT
  industry,
  region,
  LOG(group_size) AS log_group_size,
  avg_age,
  pct_female,
  LOG(avg_salary) AS log_avg_salary,
  elimination_days,
  benefit_pct,
  LEAST((claims_y1 + claims_y2 + claims_y3)
      / (premium_y1 + premium_y2 + premium_y3), 5) AS experience_lr,
  loss_ratio_next
FROM eb_rampup.groups;

SELECT * FROM ML.EVALUATE(MODEL eb_rampup.lr_linear);
```

BigQuery ML one-hot encodes `industry` and `region` for you. The training data is a few MB, so this costs pennies at most.

**🧩 Challenge:** `ML.EVALUATE` reports an **unweighted** `mean_absolute_error`, so it isn't comparable to your premium-weighted MAE. Add an unweighted MAE to your Python `evaluate` (a test first!), and compare. A linear model can't capture interactions like "trust history more for bigger groups," so how close does it get?

---

## Part 9: Commit, self-check, and report back (5 min)

```bash
ruff check .
pytest
git add . && git commit -m "Day 2: Vertex training and model registry" && git push
```

**Self-check** (answers at the end):

1. Why does `build_features` use fixed category lists instead of `pd.get_dummies`?
2. What's `AIP_MODEL_DIR`, and what's its SageMaker equivalent?
3. Why does the model beat the own-history baseline by much more on small groups than on large ones?
4. The `load_table` test uses a fake client. What does it prove, and what does it *not* prove?
5. Why must the XGBoost version match between the training and serving containers?
6. Why does the script label each model version with a Git commit?

**Report back** for day 3:

- Anything that broke or confused you, especially in Part 7
- Your three metrics locally and on Vertex, and whether the credibility feature helped
- How long each part took

---

## Answers and solutions

### Part 3d: feature test solutions

```python
def test_experience_loss_ratio_by_hand(tiny_groups):
    lr = experience_loss_ratio(tiny_groups)
    # A: (50+70+90)/300 = 0.7   B: 1800/3000 = 0.6   C: 300/30 = 10, capped at 5
    assert lr.tolist() == pytest.approx([0.7, 0.6, 5.0])


def test_one_hot_encoding(tiny_groups):
    X = build_features(tiny_groups)
    industry_cols = [c for c in FEATURE_COLUMNS if c.startswith("industry_")]
    assert X.loc[0, "industry_construction"] == 1.0
    assert (X[industry_cols].sum(axis=1) == 1.0).all()   # exactly one industry per row


def test_single_industry_batch_still_has_all_columns(tiny_groups):
    one_row = tiny_groups.iloc[[1]]                       # only Technology
    X = build_features(one_row)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X["industry_construction"].iloc[0] == 0.0


def test_no_missing_values():
    X = build_features(generate_groups(500, seed=1))
    assert not X.isna().any().any()


def test_does_not_modify_input(tiny_groups):
    before = tiny_groups.copy()
    build_features(tiny_groups)
    pd.testing.assert_frame_equal(tiny_groups, before)


def test_unknown_industry_is_rejected(tiny_groups):
    tiny_groups.loc[0, "industry"] = "Space Mining"
    with pytest.raises(ValueError, match="Space Mining"):
        build_features(tiny_groups)


def test_weights_are_average_annual_premium(tiny_groups):
    assert sample_weights(tiny_groups).tolist() == pytest.approx([100, 1_000, 10])
```

`test_unknown_industry_is_rejected` modifies `tiny_groups`. That's safe only because the fixture has no `scope`, so each test gets a fresh copy. With `scope="module"`, this test would corrupt the data for every test after it, a confusing kind of failure that depends on test order.

### Part 3e: mini-challenge

```python
def test_load_table_passes_max_rows():
    fake_client = MagicMock()
    load_table(fake_client, "eb_rampup.groups", max_rows=100)
    fake_client.list_rows.assert_called_once_with("eb_rampup.groups", max_results=100)
```

### Part 4c: training test solutions

```python
def test_weighted_mae_by_hand():
    # errors 1 and 3, weights 1 and 3 -> (1*1 + 3*3) / 4 = 2.5
    assert weighted_mae([0, 0], [1, 3], [1, 3]) == pytest.approx(2.5)


def test_split_is_reproducible_and_complete(small_df):
    train_a, test_a = split(small_df, seed=1)
    train_b, _ = split(small_df, seed=1)
    assert train_a.index.equals(train_b.index)
    assert len(train_a) + len(test_a) == len(small_df)
    assert set(train_a.index).isdisjoint(test_a.index)


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("gs://bucket/runs/1/model", Path("/gcs/bucket/runs/1/model")),
        ("artifacts/model", Path("artifacts/model")),
    ],
)
def test_to_local_path(uri, expected):
    assert to_local_path(uri) == expected


def test_save_and_reload_gives_same_predictions(fast_model, small_df, tmp_path):
    out = save_artifacts(fast_model, str(tmp_path), {"mae": 0.1})

    assert (out / MODEL_FILENAME).exists()
    assert json.loads((out / "features.json").read_text()) == FEATURE_COLUMNS

    booster = xgb.Booster()
    booster.load_model(str(out / MODEL_FILENAME))
    X = build_features(small_df).to_numpy()
    reloaded = booster.predict(xgb.DMatrix(X))
    np.testing.assert_allclose(reloaded, fast_model.predict(X), rtol=1e-5)


@pytest.fixture(scope="module")
def trained():
    df = generate_groups(5_000, seed=21)
    train_df, test_df = split(df, seed=21)
    model = train_model(build_features(train_df), train_df[TARGET], sample_weights(train_df))
    return model, train_df, test_df


def test_model_clearly_beats_book_rate(trained):
    m = evaluate(*trained)
    assert m["mae"] < 0.8 * m["baseline_book_mae"]


def test_model_beats_own_history_for_small_groups(trained):
    model, _, test_df = trained
    small = test_df[test_df["group_size"] < 100]
    X, w = build_features(small), sample_weights(small)
    model_mae = weighted_mae(small[TARGET], model.predict(X.to_numpy()), w)
    history_mae = weighted_mae(small[TARGET], X["experience_lr"], w)
    assert model_mae < history_mae


def test_main_runs_end_to_end_locally(tmp_path):
    metrics = main(["--n-rows", "300", "--model-dir", str(tmp_path)])
    assert (tmp_path / MODEL_FILENAME).exists()
    assert metrics["n_train"] + metrics["n_test"] == 300
```

The `trained` fixture is module-scoped because training on 5,000 rows takes a moment, and both quality tests can share one model. Sharing is safe here because neither test modifies it.

### Part 6: credibility

```python
# tests/test_features.py
@pytest.mark.parametrize("size, expected", [(4_000, 1.0), (1_000, 1.0), (250, 0.5), (10, 0.1)])
def test_credibility(tiny_groups, size, expected):
    from eb_rampup.features import credibility

    df = tiny_groups.iloc[[0]].assign(group_size=size)
    assert credibility(df).iloc[0] == pytest.approx(expected)
```

```python
# src/eb_rampup/features.py
FULL_CREDIBILITY_LIVES = 1_000


def credibility(df: pd.DataFrame) -> pd.Series:
    """How much to trust a group's own history: sqrt(lives / full), capped at 1."""
    return np.minimum(1.0, np.sqrt(df["group_size"] / FULL_CREDIBILITY_LIVES))

# ...add "credibility" to NUMERIC_COLUMNS, and in build_features:
#     out["credibility"] = credibility(df)
```

Once the feature exists, move the import to the top of the test file with the others. It's inside the test function here only so you could watch the test fail cleanly before writing the code.

### Self-check answers

1. `get_dummies` only makes columns for categories present in the batch, so a batch missing an industry gets different columns than training had. Fixed lists keep the training–serving contract stable.
2. The Cloud Storage folder where Vertex expects the training code to save the model, set automatically when the job has a base output directory. SageMaker's equivalent is `SM_MODEL_DIR` (`/opt/ml/model`).
3. A small group's history is mostly noise (Day 1's volatility query), so its own past loss ratio is a poor predictor, and the model does much better by leaning on industry, age, and the rest. A large group's history is already reliable, so there's less to gain. That's credibility theory, rediscovered by the model.
4. It proves your code calls the client correctly (right method, right table, right arguments) and returns what the client gives back. It doesn't prove the table exists, that permissions work, or that the real API behaves as the fake does. The integration test covers those.
5. A model file is written in a format tied to the library version. Loading it with a different version can fail outright or, worse, load and behave differently, which is training–serving skew.
6. Traceability. Months later, someone asking "what code produced the model that's in production?" can find the exact commit. Auditors and model risk teams ask exactly this, and the job post's "governance artifacts" include it.