# Day 4: Serving, Batch Scoring, a Cloud Run API, and Continuous Deployment

**About 3–3.5 hours.** Parts 8 and 9 can spill into a second sitting.

By the end of today, the champion model will score every group into a BigQuery table, answer single requests through a private API on Cloud Run, and redeploy itself whenever you merge to `main`, after the tests pass and with a smoke test against the live service.

| Part | What you do | Time |
| --- | --- | --- |
| 1 | Concepts: batch vs. online, and one scoring path | 10 min |
| 2 | Update the project | 10 min |
| 3 | The serving module, with tests | 35 min |
| 4 | Batch scoring: locally, then on Vertex | 20 min |
| 5 | The API, with tests | 35 min |
| 6 | Run the container locally, and test it live | 15 min |
| 7 | Deploy to Cloud Run by hand | 20 min |
| 8 | Continuous deployment from GitHub | 35 min |
| 9 | Commit, self-check, and report back | 5 min |

---

## Part 1: Concepts (10 min)

### Batch vs. online

| | Batch scoring | Online scoring (an API) |
| --- | --- | --- |
| **When** | On a schedule: score every group each month | On demand: one group, right now |
| **Who uses it** | Renewal lists, reports, dashboards | A quoting tool, an underwriter's screen |
| **Where results go** | A table (BigQuery) | The HTTP response |
| **Today's version** | A Vertex job writing to `eb_rampup.scores` | A FastAPI app on Cloud Run |
| **SageMaker equivalent** | Batch Transform, or a Processing job | A real-time endpoint |

The job posting asks for both: "scoring pipelines" and models "delivered into business workflows."

### One scoring path, not two

The biggest risk in serving is **training–serving skew**. It's easy to end up with feature logic written once in Python for training, again in SQL for batch scoring, and a third time in the API. Each copy drifts a little, and nobody notices because nothing errors.

So today, **both** batch and online scoring call the same `LoadedModel.predict`, which calls the same `build_features` that training uses. There's one path, tested once.

That's also why we're not using Vertex's built-in Batch Prediction or the prebuilt XGBoost serving container today. Both expect inputs that are *already* features, so the feature logic would have to live somewhere else. They're great when a model is self-contained (an image classifier, say, or a model whose features are already in a table). Ours does its own feature engineering, so we serve it with our own code, from the image you built on Day 3.

### Guarding the contract

The model files you saved on Day 2 include `features.json`, the exact features the model was trained on. Today's loader **checks** that the code's features still match it before predicting. If you add a feature (like credibility) and forget to retrain, scoring fails with a clear error instead of quietly feeding the model columns in the wrong positions.

### Cloud Run vs. a Vertex endpoint

| | Cloud Run | Vertex AI endpoint |
| --- | --- | --- |
| What it runs | Any container that answers HTTP | A registered model, in a serving container |
| Scales to zero | Yes: free when idle | No: bills per hour while deployed |
| Good for | Custom logic around the model, like ours | GPU models, built-in monitoring, traffic splits between model versions |

AWS equivalents: Cloud Run is like ECS on Fargate behind a load balancer, but with no cluster or load balancer to manage. A Vertex endpoint is a SageMaker endpoint.

---

## Part 2: Update the project (10 min)

### 2a. `pyproject.toml`

Update the optional dependencies:

```toml
[project.optional-dependencies]
pipeline = ["google-cloud-aiplatform", "google-cloud-storage"]
api = ["fastapi", "uvicorn"]
dev = [
  "pytest", "ruff", "build", "google-cloud-aiplatform", "kfp>=2.7,<3", "pyyaml",
  "fastapi", "uvicorn", "httpx",
]
```

- `google-cloud-storage` is now listed explicitly because today's code imports it directly. It was already being installed as part of `google-cloud-aiplatform`, but the rule from Day 3 applies: if your code imports it, declare it.
- `api` is a new extra for the image. `dev` gets the same packages, plus `httpx`, which FastAPI's test client needs.

Then reinstall: `pip install -e ".[dev]"`.

### 2b. Dockerfile

One image serves every purpose. Pipeline and batch jobs say exactly which command to run, while Cloud Run runs the image's default command, so add a default that starts the API:

```dockerfile
FROM python:3.11-slim

# XGBoost uses OpenMP for multithreading
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[pipeline,api]"

# Cloud Run sends requests to $PORT. Pipeline and batch jobs override this command.
CMD ["sh", "-c", "exec uvicorn eb_rampup.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

`${PORT:-8080}` means "use the `PORT` environment variable, or 8080 if it isn't set." Cloud Run sets `PORT` and expects the app to listen on it. `exec` makes the server the container's main process, so it receives Cloud Run's shutdown signal and stops cleanly.

### 2c. APIs

```bash
gcloud services enable run.googleapis.com iamcredentials.googleapis.com
```

`iamcredentials` is the API that issues short-lived service account tokens. It's what impersonation (and Part 8's GitHub login) uses.

---

## Part 3: The serving module, with tests (35 min)

### 3a. The code

Create `src/eb_rampup/serving.py`:

```python
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
```

Things to notice:

- **`FeatureMismatchError` is a `RuntimeError`, not a `ValueError`.** That distinction matters in Part 5. A `ValueError` from `build_features` means the *caller* sent bad data (an unknown industry), which is the caller's problem. A feature mismatch means *our* deployment is broken, which is our problem. The API turns the first into a "fix your request" response and lets the second surface as a server error.
- **`model_info.json` is optional.** If you made the log-target change on Day 3, it says the model predicts log loss ratios, and `predict` converts them back. Models trained before that change don't have the file and pass predictions through unchanged. Either way, the decision is recorded *with the model*, not assumed by the code.
- **`apply_business_rules`** turns a number into a decision an underwriter can act on: groups under 10 lives, or with a predicted loss ratio above 1.5, are referred for human review, with the reason spelled out. This kind of rule usually comes from the business, and it changes more often than the model, so it lives in its own small, tested function.

Next, `src/eb_rampup/registry.py`, which finds the champion and downloads its files:

```python
"""Find and download the champion model from the Vertex AI Model Registry."""

from pathlib import Path

from eb_rampup.serving import LoadedModel, load_model_dir


def find_champion(project: str, region: str, model_name: str):
    """Return the registry version aliased 'champion', or None."""
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    models = aiplatform.Model.list(filter=f'display_name="{model_name}"')
    if not models:
        return None
    registry = aiplatform.models.ModelRegistry(models[0])
    for version in registry.list_versions():
        if "champion" in version.version_aliases:
            return registry.get_model(version=version.version_id)
    return None


def download_dir(gcs_uri: str, dest) -> Path:
    """Copy every file under a gs:// folder to a local folder."""
    from google.cloud import storage

    bucket, _, prefix = gcs_uri.removeprefix("gs://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    dest = Path(dest)
    for blob in storage.Client().list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        target = dest / blob.name[len(prefix):]
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
    return dest


def load_champion(project: str, region: str, model_name: str, dest="/tmp/champion") -> LoadedModel:
    champion = find_champion(project, region, model_name)
    if champion is None:
        raise RuntimeError(f"No version of {model_name} has the champion alias")
    local = download_dir(champion.uri, dest)
    return load_model_dir(local, version=str(champion.version_id))
```

`find_champion` searches the versions explicitly, the same fix you made to the pipeline on Day 3. The `aiplatform` and `storage` imports are inside the functions, so importing this module doesn't require those packages; only calling it does.

### 3b. Shared fixtures

Add two fixtures to the bottom of `tests/conftest.py`. The first trains a small *real* model once per test run, so tests can load actual model files:

```python
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
```

`model_dir` is session-scoped (built once, shared everywhere), which is safe because no test modifies the files. `valid_group` is function-scoped, since several tests change a field to make it invalid.

### 3c. The first test

Create `tests/test_serving.py`:

```python
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from eb_rampup.data import generate_groups
from eb_rampup.features import FEATURE_COLUMNS
from eb_rampup.score import score_frame
from eb_rampup.serving import (
    FeatureMismatchError,
    LoadedModel,
    apply_business_rules,
    load_model_dir,
)


def test_loaded_model_scores_every_row(model_dir):
    model = load_model_dir(model_dir)
    df = generate_groups(50, seed=8)
    preds = model.predict(df)
    assert preds.shape == (50,)
    assert np.isfinite(preds).all()
```

(It imports `score_frame`, which you'll create in Part 4. Until then, comment out that import, or create Part 4's file first.)

### 3d. Challenges (25 min)

1. **The log transform is reversed.** Build a `LoadedModel` whose booster is a `MagicMock` returning `np.log([0.5, 1.0, 2.0])`, with `target_transform="log"`, and check that `predict(tiny_groups)` returns `[0.5, 1.0, 2.0]`. *Hint:* `booster.predict.return_value = ...`, the same pattern as Day 2's fake BigQuery client. The features list must be `FEATURE_COLUMNS`, or you'll trip challenge 3.
2. **No transform means predictions pass through unchanged.**
3. **A feature mismatch is caught.** Give the model `FEATURE_COLUMNS[:-1]` (one feature short) and check that `FeatureMismatchError` is raised with a helpful message.
4. **`load_model_dir` reads `model_info.json` only when it exists.** Copy `model.bst` and `features.json` from `model_dir` into `tmp_path`, load it (no transform), then add a `model_info.json` saying `"log"` and load it again.
5. **The business rules,** as one parametrized test: a normal group, a small group, a high prediction, both at once, and the **boundaries** (exactly 10 lives and exactly 1.5 should *not* be referred). Boundaries are where off-by-one bugs live, so they deserve their own cases.

✅ **Check:** `pytest tests/test_serving.py -v` passes, once Part 4's file exists.

---

## Part 4: Batch scoring: locally, then on Vertex (20 min)

### 4a. The code

Create `src/eb_rampup/score.py`:

```python
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
```

It **appends** rather than overwriting. Each run adds rows stamped with `scored_at` and `model_version`, so the table becomes a history of every score you've produced. Day 5's monitoring compares runs over time using exactly this, and if anyone asks "what did we predict for this group last March, and with which model?", you have the answer.

**🧩 Challenge 6:** add `test_score_frame_columns` to `tests/test_serving.py`, checking that `score_frame` returns the six expected columns in order and stamps the model version you loaded with (`load_model_dir(model_dir, version="3")`).

### 4b. Run it locally

```bash
python -m eb_rampup.score --project $PROJECT_ID
```

This downloads the champion, scores every group, and appends to `eb_rampup.scores`. Then explore in the BigQuery console:

```sql
SELECT refer_to_underwriter, COUNT(*) AS groups, ROUND(AVG(predicted_loss_ratio), 3) AS avg_pred
FROM eb_rampup.scores
GROUP BY refer_to_underwriter;
```

**If you get a `FeatureMismatchError`,** the guard is working. Your champion was trained on different features than your code builds now (for example, before or after the credibility feature). Retrain through the pipeline so the champion matches the code, then score again.

### 4c. Run it on Vertex

In production, scoring runs on a schedule in the cloud, not on your laptop. Create `scripts/score_on_vertex.py`:

```python
"""Run batch scoring on Vertex AI, from your image, as pipeline-sa.

    python scripts/score_on_vertex.py --project $PROJECT_ID --image $IMAGE \
        --service-account pipeline-sa@$PROJECT_ID.iam.gserviceaccount.com
"""

import argparse
import time

from google.cloud import aiplatform


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--image", required=True)
    p.add_argument("--service-account", required=True)
    p.add_argument("--table", default="eb_rampup.groups")
    p.add_argument("--output", default="eb_rampup.scores")
    args = p.parse_args()

    aiplatform.init(project=args.project, location=args.region)
    job = aiplatform.CustomJob(
        display_name=f"loss-ratio-score-{time.strftime('%Y%m%d-%H%M%S')}",
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "container_spec": {
                "image_uri": args.image,
                "command": ["python", "-m", "eb_rampup.score"],
                "args": [
                    "--project", args.project, "--region", args.region,
                    "--table", args.table, "--output", args.output,
                ],
            },
        }],
    )
    job.run(service_account=args.service_account)


if __name__ == "__main__":
    main()
```

The job uses `container_spec`, meaning "run this command in my image," rather than Day 2's `python_package_spec`, which installed a package into Google's image. Your Day 3 image already has everything.

**Rebuild and push the image first**, since it now contains the scoring and API code:

```bash
git add . && git commit -m "Day 4: serving, scoring, API"
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/eb-rampup/trainer:$(git rev-parse --short HEAD)"
docker buildx build --platform linux/amd64 -t $IMAGE --push .

python scripts/score_on_vertex.py --project $PROJECT_ID --image $IMAGE \
  --service-account pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

Rerun the SQL query afterward: the row counts double, and the table now holds two runs. (On day 9, this job gets scheduled.)

---

## Part 5: The API, with tests (35 min)

### 5a. The code

Create `src/eb_rampup/api.py`:

```python
"""HTTP API that scores one employer group with the champion model.

    MODEL_DIR=artifacts/model uvicorn eb_rampup.api:app --reload
"""

import os
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from eb_rampup.serving import LoadedModel, apply_business_rules, load_model_dir


def load_from_env() -> LoadedModel:
    """MODEL_DIR for a local model; otherwise download the champion."""
    if model_dir := os.environ.get("MODEL_DIR"):
        return load_model_dir(model_dir)
    from eb_rampup.registry import load_champion

    return load_champion(
        os.environ["GCP_PROJECT"],
        os.environ.get("GCP_REGION", "us-central1"),
        os.environ.get("MODEL_NAME", "loss-ratio-ltd"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_from_env()  # once per container start, not per request
    yield


app = FastAPI(title="Loss ratio API", lifespan=lifespan)


class Group(BaseModel):
    group_id: str = "request"
    industry: str
    region: str
    group_size: int = Field(ge=2, le=20_000)
    avg_age: float = Field(ge=18, le=70)
    pct_female: float = Field(ge=0, le=1)
    avg_salary: float = Field(gt=0)
    elimination_days: int = Field(gt=0)
    benefit_pct: float = Field(gt=0, le=1)
    premium_y1: float = Field(gt=0)
    premium_y2: float = Field(gt=0)
    premium_y3: float = Field(gt=0)
    claims_y1: float = Field(ge=0)
    claims_y2: float = Field(ge=0)
    claims_y3: float = Field(ge=0)


class Score(BaseModel):
    predicted_loss_ratio: float
    refer_to_underwriter: bool
    refer_reason: str
    model_version: str


def get_model(request: Request) -> LoadedModel:
    return request.app.state.model


ModelDep = Annotated[LoadedModel, Depends(get_model)]


@app.get("/health")
def health(model: ModelDep) -> dict:
    return {"status": "ok", "model_version": model.version}


@app.post("/score")
def score(group: Group, model: ModelDep) -> Score:
    df = pd.DataFrame([group.model_dump()])
    try:
        predictions = model.predict(df)
    except ValueError as e:  # bad input the schema can't catch, e.g. an unknown industry
        raise HTTPException(status_code=422, detail=str(e)) from e
    row = apply_business_rules(df, predictions).iloc[0]
    return Score(
        predicted_loss_ratio=float(row["predicted_loss_ratio"]),
        refer_to_underwriter=bool(row["refer_to_underwriter"]),
        refer_reason=row["refer_reason"],
        model_version=model.version,
    )
```

How it fits together:

- **`lifespan`** runs once when the server starts. It loads the model into `app.state`, so requests don't each pay for a download. The trade-off: a new champion isn't picked up until the next deploy or restart. That's deliberate. A service's behavior changes only when you deploy it, and every response includes `model_version`, so callers always know which model answered.
- **`Group`** is a Pydantic model. FastAPI validates every request against it automatically, so a missing field, a string where a number belongs, or an age of 120 is rejected with a **422** response before your code runs. That's the schema doing input validation for free.
- **The `ValueError` → 422 conversion** covers what the schema can't check, like an unknown industry. It's a caller error, so they get a 422 with the message. A `FeatureMismatchError` isn't caught, so it becomes a **500** server error, which is correct, because the fault is ours.
- **`get_model` and `Depends`** are **dependency injection**, the same idea as passing the BigQuery client into `load_table` on Day 2. The endpoint asks for "a model" without knowing where it comes from, which is exactly what makes it testable.

### 5b. Testing with dependency overrides

FastAPI lets a test **replace a dependency**. Your tests can swap `get_model` for a fake, so they don't need a trained model, Google Cloud, or a running server. Create `tests/test_api.py`:

```python
import numpy as np
import pytest
from fastapi.testclient import TestClient

from eb_rampup.api import app, get_model, load_from_env
from eb_rampup.serving import load_model_dir


class FakeModel:
    """Stands in for the real model: returns a fixed prediction."""

    version = "test-1"

    def __init__(self, prediction: float = 0.8):
        self.prediction = prediction

    def predict(self, df):
        return np.full(len(df), self.prediction)


@pytest.fixture
def client():
    app.dependency_overrides[get_model] = lambda: FakeModel()
    yield TestClient(app)          # no `with`, so startup (model loading) is skipped
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test-1"}
```

Three things to notice:

- **`FakeModel` is a small handwritten class, not a `MagicMock`.** When a fake needs real behavior (here, returning one prediction per row), a tiny class is clearer than configuring a mock.
- **The fixture uses `yield`.** Code after `yield` runs after the test finishes, even if the test fails. Here it removes the override, so it can't leak into other tests. That's how pytest does setup and teardown.
- **`TestClient(app)` without `with`** skips the `lifespan` startup, so no model is downloaded. The override supplies the model instead.

### 5c. Challenges (20 min)

7. **A valid request returns the prediction,** `refer_to_underwriter` false, and the model version.
8. **A high prediction is referred.** Override with `FakeModel(prediction=2.0)`. Since you're not using the `client` fixture, clear the override in a `try`/`finally`.
9. **Out-of-range inputs return 422:** one parametrized test covering `avg_age` 120, `group_size` 1, `pct_female` 1.5, and `premium_y1` 0.
10. **A missing field returns 422.**
11. **An unknown industry returns 422 with the industry in the message.** This one needs the *real* model, so it calls `build_features`: override `get_model` with `lambda: load_model_dir(model_dir)`.
12. **`load_from_env` uses `MODEL_DIR` when it's set.** pytest's built-in `monkeypatch` fixture sets environment variables just for one test: `monkeypatch.setenv("MODEL_DIR", str(model_dir))`. It undoes the change automatically afterward, so nothing leaks into other tests.

### 5d. Try it locally

```bash
python -m eb_rampup.train                     # if artifacts/model doesn't exist yet
MODEL_DIR=artifacts/model uvicorn eb_rampup.api:app --reload
```

Open `http://localhost:8000/docs`. FastAPI generates interactive documentation from your `Group` model. Click **POST /score → Try it out**, paste the `valid_group` values, and run it. Then try an age of 120 to see the 422. Stop the server with Ctrl+C.

---

## Part 6: Run the container locally, and test it live (15 min)

The container runs the same code, but in production's environment. Run it against your local model:

```bash
docker run --rm -p 8080:8080 --platform linux/amd64 \
  -v "$PWD/artifacts/model:/model" -e MODEL_DIR=/model $IMAGE
```

`-v` mounts your local model folder into the container at `/model`. In a second terminal, run the **live test**:

```python
"""Calls a running API. The same tests check a local container and Cloud Run.

    API_URL=http://localhost:8080 pytest -m integration tests/integration/test_api_live.py
"""

import os

import pytest

httpx = pytest.importorskip("httpx")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def api():
    url = os.environ.get("API_URL")
    if not url:
        pytest.skip("API_URL not set")
    token = os.environ.get("API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(base_url=url, headers=headers, timeout=60) as client:
        yield client


def test_health(api):
    response = api.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score(api, valid_group):
    response = api.post("/score", json=valid_group)
    assert response.status_code == 200
    assert 0 < response.json()["predicted_loss_ratio"] < 5
```

```bash
API_URL=http://localhost:8080 pytest -m integration tests/integration/test_api_live.py -v
```

This test talks to a *running* service over HTTP. It's the same file that tomorrow's CD pipeline runs against Cloud Run, so the check you just ran locally is exactly what gates a production deploy.

**Optional:** run the container against the real champion, using your own Google credentials:

```bash
docker run --rm -p 8080:8080 --platform linux/amd64 \
  -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
  -e GCP_PROJECT=$PROJECT_ID $IMAGE
```

This mounts your Application Default Credentials read-only, so the container can reach the registry. Check `/health` to see the champion's version number.

---

## Part 7: Deploy to Cloud Run by hand (20 min)

Do it once by hand to understand each piece. Part 8 automates it.

### 7a. A service account for the API

The API only needs to *read* the registry and the model files, so it gets its own minimal service account instead of reusing `pipeline-sa`:

```bash
gcloud iam service-accounts create api-sa --display-name="Loss ratio API"
API_SA="api-sa@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${API_SA}" --role="roles/aiplatform.viewer" --condition=None
gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
  --member="serviceAccount:${API_SA}" --role="roles/storage.objectViewer"
```

The storage grant is on **one bucket**, not the whole project. If the API were compromised, it could read model files and nothing else. One service account per workload, with the least access each needs, is how you'd set things up at work.

### 7b. Deploy

```bash
gcloud run deploy loss-ratio-api \
  --image $IMAGE --region $REGION \
  --service-account $API_SA \
  --set-env-vars GCP_PROJECT=$PROJECT_ID,GCP_REGION=$REGION,MODEL_NAME=loss-ratio-ltd \
  --memory 1Gi --max-instances 2 \
  --no-allow-unauthenticated
```

- **`--no-allow-unauthenticated`** makes the service private. Only callers with permission (the Cloud Run Invoker role) can reach it. As project owner, you have that.
- **`--max-instances 2`** is a cost guard, so a flood of requests can't scale it up indefinitely.
- It scales to **zero** when idle, so it costs nothing between requests. The trade-off is a **cold start** of several seconds on the first request after idle time, while it starts a container and downloads the model.

### 7c. Call it

```bash
URL=$(gcloud run services describe loss-ratio-api --region $REGION --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)

curl -H "Authorization: Bearer $TOKEN" $URL/health
```

Try it without the header to see the private service refuse you (403). Then run the live test against Cloud Run:

```bash
API_URL=$URL API_TOKEN=$TOKEN pytest -m integration tests/integration/test_api_live.py -v
```

The same test, now against production.

**If startup fails,** check **Cloud Run → loss-ratio-api → Logs**. The most likely causes are a missing permission for `api-sa` (look for 403s mentioning `aiplatform` or `storage`) or a `FeatureMismatchError` (see Part 4b).

---

## Part 8: Continuous deployment from GitHub (35 min)

Now make a merge to `main` do everything from Part 7, automatically, but only if the tests pass.

### 8a. Let GitHub log in to Google without a key

This is **Workload Identity Federation**, which we discussed when you asked about impersonation. GitHub gives each workflow run a signed identity token. You'll tell Google to trust tokens from GitHub, but only from *your* repo, and to swap them for short-lived credentials for a deployer service account.

```bash
export GITHUB_REPO="YOUR_GITHUB_USER/eb-rampup"   # change this
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')

# 1. A pool for outside identities, and a provider that trusts GitHub
gcloud iam workload-identity-pools create github --location=global \
  --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc eb-rampup \
  --location=global --workload-identity-pool=github \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == '${GITHUB_REPO}'"

# 2. A deployer service account with only what deploying needs
gcloud iam service-accounts create github-deployer --display-name="GitHub deployer"
DEPLOYER_SA="github-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${DEPLOYER_SA}" --role="roles/run.admin" --condition=None
gcloud artifacts repositories add-iam-policy-binding eb-rampup --location=$REGION \
  --member="serviceAccount:${DEPLOYER_SA}" --role="roles/artifactregistry.writer"
gcloud iam service-accounts add-iam-policy-binding $API_SA \
  --member="serviceAccount:${DEPLOYER_SA}" --role="roles/iam.serviceAccountUser"
gcloud run services add-iam-policy-binding loss-ratio-api --region=$REGION \
  --member="serviceAccount:${DEPLOYER_SA}" --role="roles/run.invoker"

# 3. Let workflows from your repo act as the deployer
gcloud iam service-accounts add-iam-policy-binding $DEPLOYER_SA \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.repository/${GITHUB_REPO}"
```

Each grant maps to something you've seen:

| Grant | Why |
| --- | --- |
| `run.admin` | Deploy new revisions of Cloud Run services |
| `artifactregistry.writer` on the repo | Push images, but only to this repository |
| `serviceAccountUser` on `api-sa` | Deploy a service that **runs as** `api-sa`: the PassRole case |
| `run.invoker` on the service | Call the private service for the smoke test |
| `workloadIdentityUser` on the deployer | Let GitHub workflows from your repo, and only your repo, act as the deployer. This is the trust policy from our AssumeRole discussion. |

The **attribute condition** is the security-critical line. Without it, *any* GitHub repository could ask to use your pool. Google's guidance is to always set one ([google-github-actions/auth](https://github.com/google-github-actions/auth)).

### 8b. Tell the workflow where things are

None of these values are secrets (they're names, not credentials), so they go in **repository variables**, not secrets. On GitHub: **Settings → Secrets and variables → Actions → Variables → New repository variable**. Add:

| Name | Value |
| --- | --- |
| `GCP_PROJECT` | your project ID |
| `WIF_PROVIDER` | `projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/github/providers/eb-rampup` |
| `DEPLOYER_SA` | `github-deployer@<PROJECT_ID>.iam.gserviceaccount.com` |
| `API_SA` | `api-sa@<PROJECT_ID>.iam.gserviceaccount.com` |

Print the exact `WIF_PROVIDER` value with `echo projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/providers/eb-rampup`.

### 8c. The workflow

Replace `.github/workflows/ci.yml` with:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"   # match the Docker image
          cache: pip

      - name: Install
        run: pip install -e ".[dev]"

      - name: Lint
        run: ruff check .

      - name: Unit tests
        run: pytest -v

  deploy:
    needs: test                     # only runs if every test passed
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write               # lets this job get a GitHub identity token for Google
    env:
      PROJECT_ID: ${{ vars.GCP_PROJECT }}
      REGION: us-central1
      SERVICE: loss-ratio-api
      IMAGE: us-central1-docker.pkg.dev/${{ vars.GCP_PROJECT }}/eb-rampup/trainer:${{ github.sha }}
    steps:
      - uses: actions/checkout@v5

      - name: Authenticate to Google Cloud (no key file)
        uses: google-github-actions/auth@v3
        with:
          workload_identity_provider: ${{ vars.WIF_PROVIDER }}
          service_account: ${{ vars.DEPLOYER_SA }}

      - uses: google-github-actions/setup-gcloud@v3

      - name: Build and push the image
        run: |
          gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
          docker build -t $IMAGE .
          docker push $IMAGE

      - name: Deploy to Cloud Run
        id: deploy
        run: |
          gcloud run deploy $SERVICE --image $IMAGE --region $REGION \
            --service-account ${{ vars.API_SA }} \
            --set-env-vars GCP_PROJECT=$PROJECT_ID,GCP_REGION=$REGION,MODEL_NAME=loss-ratio-ltd \
            --memory 1Gi --max-instances 2 --no-allow-unauthenticated --quiet
          echo "url=$(gcloud run services describe $SERVICE --region $REGION --format='value(status.url)')" >> "$GITHUB_OUTPUT"

      - name: Get a token to call the private service
        id: token
        uses: google-github-actions/auth@v3
        with:
          workload_identity_provider: ${{ vars.WIF_PROVIDER }}
          service_account: ${{ vars.DEPLOYER_SA }}
          token_format: id_token
          id_token_audience: ${{ steps.deploy.outputs.url }}
          id_token_include_email: true

      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"
          cache: pip

      - name: Smoke test the live service
        env:
          API_URL: ${{ steps.deploy.outputs.url }}
          API_TOKEN: ${{ steps.token.outputs.id_token }}
        run: |
          pip install -e ".[dev]"
          pytest -m integration tests/integration/test_api_live.py -v
```

Reading the new `deploy` job:

- **`needs: test`** means deploy runs only if every test passed. This one line is what makes CI a gate rather than a report.
- **`if: ...`** means deploy runs only for pushes to `main`. Pull requests get tested, not deployed.
- **`id-token: write`** lets the job request its GitHub identity token. Without it, the Google login step fails.
- **Image tag = `github.sha`:** every deployed image traces back to its exact commit, like your model labels.
- **The second `auth` step** gets an **ID token** for the service's URL, which is what Cloud Run checks for private services, so the smoke test can call it.
- **The smoke test** is the same `test_api_live.py` you ran against the local container and by hand against Cloud Run.

### 8d. Watch it work

```bash
git add . && git commit -m "Day 4: continuous deployment" && git push
```

Open the **Actions** tab. You'll see `test` run, then `deploy`, with each step's logs. When it's green, check **Cloud Run → loss-ratio-api → Revisions**: there's a new revision, deployed by the deployer service account, with the commit SHA in its image tag.

Then prove the gate works: make a test fail on a branch, open a pull request, and see that `deploy` never runs. Revert it before merging.

### 8e. A known gap, to fix later

If the smoke test fails, the bad revision is **already serving traffic**: the test runs after the switch. The safer pattern is to deploy with `--no-traffic --tag candidate`, smoke-test the candidate's own URL, and only then move traffic to it (`gcloud run services update-traffic --to-latest`). That's a good stretch goal, and worth knowing the name of: it's a form of **blue/green deployment**.

### Troubleshooting

- **`Permission 'iam.serviceAccounts.getAccessToken' denied`** in the auth step: the `workloadIdentityUser` binding's `principalSet` doesn't match your repo exactly (check the owner name and capitalization), or `id-token: write` is missing.
- **`The attribute condition must reference...`** or the provider rejects tokens: check `GITHUB_REPO` matches `owner/repo` exactly.
- **`Permission denied on push`:** the Artifact Registry grant is missing or was made on a different repository name.
- **`iam.serviceaccounts.actAs` denied during deploy:** the `serviceAccountUser` grant on `api-sa` is missing.
- **Smoke test 403:** the deployer lacks `run.invoker` on the service.

### Cost and cleanup

Cloud Run, the Docker builds on GitHub's runners, and Workload Identity Federation are all free at this scale. Images add up, though: each push stores another image in Artifact Registry, a few hundred MB each. Every few days, delete old ones in the console (**Artifact Registry → eb-rampup → trainer**), keeping the one Cloud Run is using.

---

## Part 9: Commit, self-check, and report back (5 min)

**Self-check** (answers at the end):

1. Why do batch scoring and the API share `LoadedModel.predict` instead of each having their own code?
2. What does `features.json` protect against, and what happens if it's violated?
3. Why does an unknown industry return a 422, while a feature mismatch returns a 500?
4. How does the API test avoid needing a real model or Google Cloud?
5. What makes the GitHub workflow a *gate* on deployment, and what keeps pull requests from deploying?
6. GitHub never has a key to your project. How does it deploy?

**Report back** for day 5:

- What broke or confused you, especially in Parts 7 and 8
- Whether the first CD run went green, and what the smoke test showed
- How long each part took

---

## Answers and solutions

### Part 3d: serving test solutions

```python
def test_log_transform_is_reversed(tiny_groups):
    booster = MagicMock()
    booster.predict.return_value = np.log([0.5, 1.0, 2.0])
    model = LoadedModel(booster, FEATURE_COLUMNS, target_transform="log")
    assert model.predict(tiny_groups) == pytest.approx([0.5, 1.0, 2.0])


def test_no_transform_passes_predictions_through(tiny_groups):
    booster = MagicMock()
    booster.predict.return_value = np.array([0.5, 1.0, 2.0])
    model = LoadedModel(booster, FEATURE_COLUMNS)
    assert model.predict(tiny_groups) == pytest.approx([0.5, 1.0, 2.0])


def test_feature_mismatch_is_caught(tiny_groups):
    model = LoadedModel(MagicMock(), FEATURE_COLUMNS[:-1], version="7")
    with pytest.raises(FeatureMismatchError, match="Model 7 expects"):
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
```

`test_reads_target_transform_when_present` copies the files instead of writing into `model_dir` directly. `model_dir` is session-scoped and shared, so adding `model_info.json` to it would change what every later test loads.

### Part 4a: `score_frame` solution

```python
def test_score_frame_columns(model_dir):
    model = load_model_dir(model_dir, version="3")
    scores = score_frame(model, generate_groups(20, seed=2))
    assert list(scores.columns) == [
        "group_id", "predicted_loss_ratio", "refer_to_underwriter",
        "refer_reason", "model_version", "scored_at",
    ]
    assert (scores["model_version"] == "3").all()
```

Column order matters here because it's the schema of a BigQuery table that other people will query.

### Part 5c: API test solutions

```python
def test_score_returns_prediction(client, valid_group):
    response = client.post("/score", json=valid_group)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_loss_ratio"] == pytest.approx(0.8)
    assert body["refer_to_underwriter"] is False
    assert body["model_version"] == "test-1"


def test_high_prediction_is_referred(valid_group):
    app.dependency_overrides[get_model] = lambda: FakeModel(prediction=2.0)
    try:
        body = TestClient(app).post("/score", json=valid_group).json()
    finally:
        app.dependency_overrides.clear()
    assert body["refer_to_underwriter"] is True


@pytest.mark.parametrize(
    "field, bad_value",
    [("avg_age", 120), ("group_size", 1), ("pct_female", 1.5), ("premium_y1", 0)],
)
def test_out_of_range_input_is_rejected(client, valid_group, field, bad_value):
    valid_group[field] = bad_value
    assert client.post("/score", json=valid_group).status_code == 422


def test_missing_field_is_rejected(client, valid_group):
    del valid_group["industry"]
    assert client.post("/score", json=valid_group).status_code == 422


def test_unknown_industry_with_real_model(model_dir, valid_group):
    app.dependency_overrides[get_model] = lambda: load_model_dir(model_dir)
    try:
        response = TestClient(app).post("/score", json={**valid_group, "industry": "Space Mining"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 422
    assert "Space Mining" in response.json()["detail"]


def test_load_from_env_uses_model_dir(monkeypatch, model_dir):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    assert load_from_env().version == "local"
```

The `try`/`finally` blocks matter. `app` is a single shared object, so an override left behind by a failing test would silently change the behavior of every test after it, the same order-dependence trap as a shared fixture someone modifies.

### Self-check answers

1. To avoid training–serving skew. With one code path, batch and online scoring use exactly the features that training used, and the logic is tested once. Separate copies drift without anything erroring.
2. It protects against the model receiving different features, or the same features in a different order, than it was trained on. If the code's features don't match, `predict` raises `FeatureMismatchError` instead of returning confident, wrong numbers.
3. An unknown industry is the caller's mistake, so they get a 422 telling them what to fix. A feature mismatch is our deployment's mistake, which the caller can't fix, so it's a 500 that shows up in our error monitoring.
4. Dependency injection. The endpoint gets its model from `get_model`, and the test replaces that with a fake through `app.dependency_overrides`. `TestClient` without `with` also skips the startup that would load the real model.
5. `needs: test` means deploy only runs after every test passes. The `if:` condition limits deploys to pushes on `main`, so pull requests are tested but never deployed.
6. Workload Identity Federation. GitHub gives the workflow a signed identity token, Google checks it came from your repo (the attribute condition and the `principalSet` binding), and issues short-lived credentials for the deployer service account. No long-lived secret exists anywhere.
