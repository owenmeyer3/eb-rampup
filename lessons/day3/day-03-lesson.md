# Day 3: Vertex AI Pipelines, a Container Image, and CI

**About 3 hours**

By the end of today, one pipeline run will validate your data, train a model, compare it with the current champion, and register it only if it's better. Your package will run from a container image you built, and GitHub will run your tests on every push.

| Part | What you do | Time |
| --- | --- | --- |
| 1 | Concepts: pipelines, components, and artifacts | 10 min |
| 2 | Update the project | 10 min |
| 3 | Data validation and promotion rules, with tests | 40 min |
| 4 | Build a container image | 30 min |
| 5 | Write the pipeline, and test that it compiles | 30 min |
| 6 | Run it on Vertex AI: promote, skip, and fail | 35 min |
| 7 | CI with GitHub Actions | 25 min |
| 8 | Commit, self-check, and report back | 5 min |

---

## Part 1: Concepts (10 min)

### How the pieces map

| Vertex AI Pipelines (KFP) | SageMaker Pipelines | Step Functions |
| --- | --- | --- |
| **Pipeline:** a Python function decorated with `@dsl.pipeline` | Pipeline definition | State machine |
| **Component:** one step, a function decorated with `@dsl.component` | Step (ProcessingStep, TrainingStep) | Task state |
| **Artifact:** a file or folder passed between steps (`Dataset`, `Model`, `Metrics`) | Step properties / S3 outputs | Payload passed between states (by S3 reference) |
| **Parameter:** a small value (`str`, `int`, `float`, `bool`) | Pipeline parameter | Input JSON |
| `dsl.If` | ConditionStep | Choice state |
| **Pipeline root:** a `gs://` folder where artifacts are stored | Default bucket | Your S3 bucket |
| Compiled YAML (the pipeline spec) | Pipeline JSON definition | ASL JSON |

Vertex AI Pipelines runs pipelines written with **Kubeflow Pipelines (KFP)**, an open-source SDK. You write Python, KFP **compiles** it into a YAML spec, and Vertex runs that spec. Each step runs in its own container, on its own machine, as your service account.

### Artifacts vs. parameters

This is the most important KFP idea:

- **Parameters** are small values, passed directly: a table name, a threshold, a yes/no decision.
- **Artifacts** are files: a dataset, a model folder. A step doesn't receive the data itself. It gets a **path** (`data.path`) where the previous step wrote its output. On Vertex, that path is under `/gcs/`, the Cloud Storage FUSE mount from Day 2, so ordinary file reads and writes work.

Vertex records every artifact, and which step produced or consumed it, in a metadata store. That's what powers the **lineage** view: from any registered model, you can trace back to the exact dataset and run that produced it. It's governance for free.

### Thin components

Component code runs inside a container, so it's awkward to test directly. So today's components follow the rule from our Step Functions discussion: **keep each step thin.** It moves data in and out and calls a tested function from your package. The real logic, validation and the promotion rule, lives in `eb_rampup`, where pytest can reach it.

### Why a container image today

Every step needs your package installed. On Day 2, Vertex installed it into Google's image at job start. For a pipeline, the standard approach is to build **one image with your package already inside** and run every step from it. That also ends the version juggling from Day 2: your image has exactly the XGBoost version your package pins. If you've built images for ECS, this is the same workflow, with Artifact Registry in place of ECR.

---

## Part 2: Update the project (10 min)

### 2a. `pyproject.toml`

Add a `pipeline` extra (what the pipeline steps need inside the image) and add KFP to `dev`:

```toml
[project.optional-dependencies]
pipeline = ["google-cloud-aiplatform"]
dev = ["pytest", "ruff", "build", "google-cloud-aiplatform", "kfp>=2.7,<3"]
```

Then add `pythonpath` to the pytest settings, so tests can import the `pipelines/` folder you'll create:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = ["integration: talks to real GCP resources (slow, costs money)"]
addopts = "-m 'not integration' --strict-markers"
```

Here's how the three dependency lists now line up with where code runs:

| List | Installed where | Why |
| --- | --- | --- |
| `dependencies` | Everywhere | What the package's own code needs |
| `[pipeline]` | The container image | Pipeline steps also talk to the Model Registry |
| `[dev]` | Your laptop and CI | Testing, linting, building, and defining pipelines |

`kfp` is only in `dev`, because it's needed to *define and compile* the pipeline, not inside the image. (KFP installs its own small runtime into each step automatically.)

### 2b. Reinstall

```bash
pip install -e ".[dev]"
```

You changed the dependency lists, so this is the reinstall case from our rule of thumb.

### 2c. Enable Artifact Registry and allow the pipeline to run as `pipeline-sa`

```bash
gcloud services enable artifactregistry.googleapis.com

SA="pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud iam service-accounts add-iam-policy-binding $SA \
  --member="serviceAccount:${SA}" --role="roles/iam.serviceAccountUser"
```

The second command lets `pipeline-sa` act as itself. A pipeline launches each step as a separate job running as `pipeline-sa`, so it needs the PassRole-style permission we discussed, Service Account User, on that account. It's granted on the service account itself, so it's scoped to just that one.

---

## Part 3: Data validation and promotion rules, with tests (40 min)

### 3a. Data validation

On Day 1, your tests checked that the *generator* produces good data. In production, data comes from upstream systems you don't control, so every pipeline run checks the *actual batch* before training on it. These are the same kinds of checks, running in a different place. Create `src/eb_rampup/validation.py`:

```python
"""Checks that a batch of employer-group data is fit to train on.

Returns every problem found (not just the first), so one failed run tells you
everything that needs fixing.
"""

import pandas as pd

from eb_rampup.data import INDUSTRY_RISK, REGIONS, YEARS

REQUIRED_COLUMNS = [
    "group_id", "industry", "region", "group_size", "avg_age", "pct_female",
    "avg_salary", "elimination_days", "benefit_pct", "loss_ratio_next",
    *[f"premium_y{y}" for y in YEARS],
    *[f"claims_y{y}" for y in YEARS],
]
RANGES = {
    "group_size": (2, 20_000),
    "avg_age": (18, 70),
    "pct_female": (0, 1),
    "avg_salary": (10_000, 1_000_000),
    "loss_ratio_next": (0, 50),
}


def validate_groups(df: pd.DataFrame, min_rows: int = 1_000) -> list[str]:
    problems = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        # Later checks would just fail on the missing columns, so stop here
        return [f"missing columns: {missing}"]

    if len(df) < min_rows:
        problems.append(f"only {len(df)} rows, expected at least {min_rows}")

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    for col, n in null_counts[null_counts > 0].items():
        problems.append(f"{col}: {n} null values")

    if df["group_id"].duplicated().any():
        problems.append(f"{df['group_id'].duplicated().sum()} duplicate group_ids")

    for col, (low, high) in RANGES.items():
        bad = (~df[col].between(low, high)) & df[col].notna()
        if bad.any():
            problems.append(f"{col}: {bad.sum()} values outside [{low}, {high}]")

    for col in [f"premium_y{y}" for y in YEARS]:
        if (df[col] <= 0).any():
            problems.append(f"{col}: non-positive premiums")

    unknown = set(df["industry"].dropna()) - set(INDUSTRY_RISK)
    if unknown:
        problems.append(f"unknown industries: {sorted(unknown)}")
    unknown = set(df["region"].dropna()) - set(REGIONS)
    if unknown:
        problems.append(f"unknown regions: {sorted(unknown)}")

    return problems
```

It returns a **list** of problems instead of raising on the first one. When a 2 a.m. run fails, you want to see every problem in one log, not fix one, rerun, and find the next.

### 3b. The promotion rule

Create `src/eb_rampup/promotion.py`:

```python
"""The rule for whether a newly trained model replaces the champion."""


def should_promote(
    challenger: dict, champion: dict | None, min_improvement: float = 0.0
) -> tuple[bool, str]:
    """Decide whether the challenger becomes champion.

    challenger / champion are the metrics dicts saved with each model.
    min_improvement is a fraction: 0.02 means "must be at least 2% better".
    """
    if challenger["mae"] >= challenger["baseline_book_mae"]:
        return False, "challenger is no better than charging everyone the book rate"
    if champion is None:
        return True, "no champion yet"

    target = champion["mae"] * (1 - min_improvement)
    if challenger["mae"] < target:
        return True, f"MAE {challenger['mae']:.4f} beats champion {champion['mae']:.4f}"
    return False, f"MAE {challenger['mae']:.4f} does not beat target {target:.4f}"
```

Three rules, in order:

1. Never promote a model that's no better than charging everyone the book rate. This is a sanity floor that catches a broken training run even when there's no champion to compare against.
2. If there's no champion, the first sane model wins.
3. Otherwise, the challenger must beat the champion by at least `min_improvement`. Requiring a margin keeps the champion from changing over noise-level differences. Every change of champion is a production change someone may have to explain.

This decision lives in a plain Python function, not in pipeline condition logic, so you can test every case in milliseconds. It's the same advice as for Step Functions: keep decisions out of the orchestration language.

### 3c. The first tests

Create `tests/test_validation.py`:

```python
import numpy as np
import pytest

from eb_rampup.data import generate_groups
from eb_rampup.validation import validate_groups


def test_generated_data_is_valid():
    assert validate_groups(generate_groups(2_000, seed=5)) == []
```

And `tests/test_promotion.py`:

```python
import pytest

from eb_rampup.promotion import should_promote


def metrics(mae, baseline=0.30):
    return {"mae": mae, "baseline_book_mae": baseline}


def test_first_model_is_promoted():
    promote, reason = should_promote(metrics(0.12), champion=None)
    assert promote
    assert reason == "no champion yet"
```

`metrics()` is a small **test helper**: a plain function, not a fixture, that builds the dictionaries the rule expects. Helpers like this keep each test focused on the numbers that matter.

### 3d. Challenges (25 min)

Use the `tiny_groups` fixture from Day 2 for the validation tests. Remember that it's function-scoped, so each test can safely modify its own copy.

**Validation:**

1. `tiny_groups` is valid with `min_rows=1`.
2. With the default `min_rows`, a "too few rows" problem is reported.
3. A missing column returns **only** the missing-column problem. (Why does the function stop early there? Look at the comment.)
4. A null `avg_salary` is reported. *Hint:* `np.nan`.
5. Duplicate `group_id`s are reported.
6. Out-of-range values are reported for `group_size`, `avg_age`, `pct_female`, and `loss_ratio_next`. Use one parametrized test with a bad value for each.
7. Two different problems in the same batch produce two messages.

**Promotion:**

8. One parametrized test covering: better (promote), worse (don't), a tie (don't), better but under a 2% margin (don't), better by more than 2% (promote). *Hint:* with a champion MAE of 0.12 and a 2% margin, the target is 0.1176.
9. A model worse than the book-rate baseline is never promoted, even when there's no champion.

✅ **Check:** `pytest -v` passes.

---

## Part 4: Build a container image (30 min)

### 4a. The Dockerfile

Create `Dockerfile` in the repo root:

```dockerfile
FROM python:3.11-slim

# XGBoost uses OpenMP for multithreading
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[pipeline]"
```

And `.dockerignore`, so local junk doesn't get copied into the image (the same idea as `.gitignore`):

```
.venv/
.git/
data/
artifacts/
dist/
notebooks/
tests/
**/__pycache__/
*.yaml
```

Notice what's installed: `pip install ".[pipeline]"`, without `-e` and without `dev`. The image gets a real copy of your package plus what the pipeline steps need, and nothing for testing. That's the `pip install .` vs. `pip install -e` difference we discussed, put to use.

### 4b. Create a repository and build

```bash
gcloud artifacts repositories create eb-rampup \
  --repository-format=docker --location=$REGION

# Let Docker push to Artifact Registry (like `aws ecr get-login-password | docker login`)
gcloud auth configure-docker ${REGION}-docker.pkg.dev

git add . && git commit -m "Day 3: validation, promotion, Dockerfile"
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/eb-rampup/trainer:$(git rev-parse --short HEAD)"

docker buildx build --platform linux/amd64 -t $IMAGE --push .
```

Three things to notice:

- **The image tag is the Git commit.** Like the model labels on Day 2, it ties every image to the exact code inside it. Commit *before* building, or the tag will point at code that doesn't match the image.
- **`--platform linux/amd64` matters on a Mac.** Apple Silicon Macs build ARM images by default, and Vertex machines are x86. Without this flag, the image builds fine and then fails on Vertex with an `exec format error`.
- **`--push`** builds and uploads in one step.

The first build takes a few minutes. The image is several hundred MB; Artifact Registry's free tier covers 0.5 GB, so storage costs at most a few cents a month.

### 4c. Smoke-test the image locally

Before Vertex runs it, check the container works:

```bash
docker run --rm --platform linux/amd64 $IMAGE \
  python -m eb_rampup.train --n-rows 500 --model-dir /tmp/model
```

You should see metrics printed. This is a **container smoke test**: it catches a broken Dockerfile in a minute instead of in the middle of a cloud run. It's slower than usual on a Mac because Docker emulates x86, but it only needs to finish.

**Every code change now needs a new image.** The pipeline runs the code baked into the image, not the code on your laptop. Forgetting to rebuild is one of the most common "my fix didn't work" moments in MLOps. On day 4, CI will build images for you.

---

## Part 5: Write the pipeline, and test that it compiles (30 min)

### 5a. The pipeline

Create a `pipelines/` folder at the repo root with an empty `pipelines/__init__.py`, so it can be imported. Then create `pipelines/training_pipeline.py`:

```python
"""Vertex AI pipeline: validate -> train -> compare with champion -> register if better.

Each component is a thin wrapper: it moves data in and out, and calls tested
functions from the eb_rampup package for the real work.
"""

import os
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

# The image is fixed when components are defined, so it comes from an
# environment variable. The placeholder lets tests compile without a real image.
TRAINER_IMAGE = os.environ.get("TRAINER_IMAGE", "trainer:placeholder")
SERVE_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"


@dsl.component(base_image=TRAINER_IMAGE)
def validate_data(project: str, table: str, min_rows: int, data: Output[Dataset]):
    from google.cloud import bigquery

    from eb_rampup.bq import load_table
    from eb_rampup.validation import validate_groups

    df = load_table(bigquery.Client(project=project), table)
    problems = validate_groups(df, min_rows=min_rows)
    if problems:
        raise ValueError("Data validation failed:\n- " + "\n- ".join(problems))

    df.to_parquet(data.path, index=False)
    data.metadata["row_count"] = len(df)


@dsl.component(base_image=TRAINER_IMAGE)
def train(data: Input[Dataset], model: Output[Model], metrics: Output[Metrics]):
    import pandas as pd

    from eb_rampup.features import TARGET, build_features, sample_weights
    from eb_rampup.train import evaluate, save_artifacts, split, train_model

    df = pd.read_parquet(data.path)
    train_df, test_df = split(df)
    fitted = train_model(build_features(train_df), train_df[TARGET], sample_weights(train_df))
    results = evaluate(fitted, train_df, test_df)

    save_artifacts(fitted, model.path, results)  # model.path is a folder under /gcs/
    for name, value in results.items():
        metrics.log_metric(name, value)  # shows up in the pipeline console


@dsl.component(base_image=TRAINER_IMAGE)
def compare_with_champion(
    model: Input[Model], project: str, region: str, model_name: str, min_improvement: float
) -> NamedTuple("Outputs", [("promote", bool), ("reason", str)]):
    import json
    from collections import namedtuple
    from pathlib import Path

    from google.api_core.exceptions import NotFound
    from google.cloud import aiplatform

    from eb_rampup.promotion import should_promote
    from eb_rampup.train import to_local_path

    aiplatform.init(project=project, location=region)
    challenger = json.loads((Path(model.path) / "metrics.json").read_text())

    champion = None
    existing = aiplatform.Model.list(filter=f'display_name="{model_name}"')
    if existing:
        try:
            champ = aiplatform.Model(model_name=f"{existing[0].resource_name}@champion")
            champion = json.loads((to_local_path(champ.uri) / "metrics.json").read_text())
        except NotFound:
            pass  # versions exist, but none is the champion

    promote, reason = should_promote(challenger, champion, min_improvement)
    print(reason)
    return namedtuple("Outputs", ["promote", "reason"])(promote, reason)


@dsl.component(base_image=TRAINER_IMAGE)
def register_model(
    model: Input[Model], project: str, region: str, model_name: str,
    serving_image: str, git_sha: str,
):
    import json
    from pathlib import Path

    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    metrics = json.loads((Path(model.path) / "metrics.json").read_text())
    existing = aiplatform.Model.list(filter=f'display_name="{model_name}"')

    registered = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model.uri,
        serving_container_image_uri=serving_image,
        parent_model=existing[0].resource_name if existing else None,
        is_default_version=True,
        version_aliases=["champion"],  # moves the alias off the old champion
        version_description=f"MAE {metrics['mae']:.4f} | promoted by pipeline",
        labels={"git_sha": git_sha},
    )
    print(f"Registered version {registered.version_id} as champion")


@dsl.pipeline(name="loss-ratio-training")
def training_pipeline(
    project: str,
    table: str = "eb_rampup.groups",
    region: str = "us-central1",
    model_name: str = "loss-ratio-ltd",
    serving_image: str = SERVE_IMAGE,
    min_rows: int = 1_000,
    min_improvement: float = 0.0,
    git_sha: str = "unknown",
):
    validated = validate_data(project=project, table=table, min_rows=min_rows)
    trained = train(data=validated.outputs["data"])
    compared = compare_with_champion(
        model=trained.outputs["model"], project=project, region=region,
        model_name=model_name, min_improvement=min_improvement,
    )
    # `== True` is required: this builds a pipeline condition, not a Python if
    with dsl.If(compared.outputs["promote"] == True, name="beats-champion"):  # noqa: E712
        register_model(
            model=trained.outputs["model"], project=project, region=region,
            model_name=model_name, serving_image=serving_image, git_sha=git_sha,
        )
```

Read it with these questions in mind:

- **Why are the imports inside each function?** Each component runs in its own container. KFP ships only the function's source code to that container, not the rest of the file, so anything the function needs must be imported inside it.
- **How do steps connect?** Passing `validated.outputs["data"]` into `train` creates both the dependency (train runs after validation) and the data handoff. There's no separate "run this after that" wiring.
- **What does `data.metadata["row_count"]` do?** It attaches a value to the artifact, visible in the console. It's a cheap way to make a run self-describing.
- **What's that `# noqa: E712`?** Ruff normally flags `== True` as bad style, since in plain Python you'd just write `if x:`. Here, `compared.outputs["promote"]` isn't a real boolean yet. It's a placeholder for a value that only exists when the pipeline runs, and `== True` builds a pipeline condition from it. The comment tells Ruff this one is intentional.
- **How does the champion get replaced?** `register_model` uploads the new version with alias `champion`. Aliases are unique per model, and assigning one that's already in use **moves** it off the old version ([Google's docs](https://docs.cloud.google.com/vertex-ai/docs/model-registry/model-alias?hl=en)). The old champion stays in the registry, so rolling back is just moving the alias back.

### 5b. The runner

Create `pipelines/run_pipeline.py`:

```python
"""Compile the pipeline and submit it to Vertex AI.

    TRAINER_IMAGE=$IMAGE python -m pipelines.run_pipeline \
        --project $PROJECT_ID --bucket $BUCKET \
        --service-account pipeline-sa@$PROJECT_ID.iam.gserviceaccount.com
"""

import argparse
import os
import subprocess

from google.cloud import aiplatform
from kfp import compiler

from pipelines.training_pipeline import training_pipeline

PIPELINE_FILE = "training_pipeline.yaml"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--bucket", required=True)
    p.add_argument("--service-account", required=True)
    p.add_argument("--table", default="eb_rampup.groups")
    p.add_argument("--min-improvement", type=float, default=0.0)
    args = p.parse_args()

    if "TRAINER_IMAGE" not in os.environ:
        raise SystemExit("Set TRAINER_IMAGE to the image you pushed to Artifact Registry")

    compiler.Compiler().compile(training_pipeline, PIPELINE_FILE)

    aiplatform.init(project=args.project, location=args.region)
    job = aiplatform.PipelineJob(
        display_name="loss-ratio-training",
        template_path=PIPELINE_FILE,
        pipeline_root=f"gs://{args.bucket}/pipeline-root",
        parameter_values={
            "project": args.project,
            "region": args.region,
            "table": args.table,
            "min_improvement": args.min_improvement,
            "git_sha": git_sha(),
        },
        # Steps read live BigQuery data, so a cached result could be stale
        enable_caching=False,
    )
    job.submit(service_account=args.service_account)  # prints a console link


if __name__ == "__main__":
    main()
```

`enable_caching=False` deserves a note. Vertex can skip a step if it already ran with identical inputs, which saves time and money. But `validate_data`'s inputs are just a project and a table *name*. If the table's contents changed, caching would reuse yesterday's data without noticing. Caching is great for steps whose inputs fully describe their work, and dangerous for steps that read from the outside world.

### 5c. Test that the pipeline compiles

Compiling needs no cloud and takes a second, which makes it a perfect CI check: a broken pipeline definition fails on your laptop instead of after you've submitted it. Create `tests/test_pipeline.py`:

```python
import pytest

pytest.importorskip("kfp")
yaml = pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    from kfp import compiler

    from pipelines.training_pipeline import training_pipeline

    out = tmp_path_factory.mktemp("pipeline") / "pipeline.yaml"
    compiler.Compiler().compile(training_pipeline, str(out))
    return yaml.safe_load(out.read_text())


def test_pipeline_compiles(spec):
    assert spec["pipelineInfo"]["name"] == "loss-ratio-training"
```

`tmp_path_factory` appears here for exactly the reason we discussed: the fixture is module-scoped, so it can't use the function-scoped `tmp_path`.

**🧩 Challenges (10 min):** the compiled spec is just nested dictionaries. Its top-level steps are the keys of `spec["root"]["dag"]["tasks"]`, and KFP names steps after their functions with dashes (`validate_data` → `validate-data`).

10. Assert that `validate-data`, `train`, and `compare-with-champion` are top-level steps.
11. Assert that `register-model` is **not** a top-level step, because it lives inside the condition, but that it does appear somewhere in `spec["components"]`.

Test 11 is worth a moment. It checks the most important property of this pipeline, that registration is gated, without running anything. That's the "test the state machine definition" layer from our Step Functions discussion.

✅ **Check:** `pytest -v` passes, and `python -c "from pipelines.training_pipeline import training_pipeline"` works.

---

## Part 6: Run it on Vertex AI: promote, skip, and fail (35 min)

You'll run the pipeline three times, each showing a different path through it.

### Run 1: promotion

```bash
export BUCKET="${PROJECT_ID}-data"
export TRAINER_IMAGE=$IMAGE

python -m pipelines.run_pipeline \
  --project $PROJECT_ID --bucket $BUCKET \
  --service-account pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

The command prints a console link. Open it and watch the graph fill in. Each step takes a few minutes, mostly machine startup.

What to expect: the Day 2 champion was trained from the same code, data, and seed, so the new model should land within a hair of it. (Not always exactly: BigQuery doesn't guarantee row order, and a different order gives a slightly different train/test split.) So `register-model` may run or be skipped, decided by a noise-level difference. Open `compare-with-champion`'s logs to see the printed reason.

That's exactly the problem `min_improvement` solves. Rerun with `--min-improvement 0.01` added, and the champion only changes for a real improvement of 1% or more.

Then explore the run:

- **Click the `train` step's `metrics` artifact:** your MAE and baselines, logged with `log_metric`.
- **Click the `data` artifact:** its `row_count` metadata and its `gs://` location.
- **Click the `model` artifact → View lineage:** the chain from data → train → model. This is what you'd show an auditor.

### Run 2: force a promotion

To see the other branch, make a real improvement. Raise `n_estimators` in `PARAMS` in `train.py` to 600, and run the unit tests. Then commit, **rebuild the image**, and run again:

```bash
git commit -am "Train with more trees"
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/eb-rampup/trainer:$(git rev-parse --short HEAD)"
docker buildx build --platform linux/amd64 -t $IMAGE --push .
export TRAINER_IMAGE=$IMAGE

python -m pipelines.run_pipeline \
  --project $PROJECT_ID --bucket $BUCKET \
  --service-account pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

If the MAE improves, `register-model` runs, and in the **Model Registry** the `champion` alias moves to the new version. If it doesn't improve, the gate did its job: more trees isn't automatically better. Try a different change, like `max_depth: 5`.

### Run 3: a data validation failure

Make a deliberately broken copy of your table in the BigQuery console:

```sql
CREATE OR REPLACE TABLE eb_rampup.groups_bad AS
SELECT * REPLACE (
  IF(MOD(ABS(FARM_FINGERPRINT(group_id)), 50) = 0, NULL, avg_salary) AS avg_salary,
  IF(MOD(ABS(FARM_FINGERPRINT(group_id)), 100) = 0, 150.0, avg_age) AS avg_age
)
FROM eb_rampup.groups;
```

This nulls out about 2% of salaries and sets about 1% of ages to 150. Run the pipeline against it:

```bash
python -m pipelines.run_pipeline --table eb_rampup.groups_bad \
  --project $PROJECT_ID --bucket $BUCKET \
  --service-account pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

`validate-data` should fail, and the rest of the pipeline should never run. Open its logs: you should see both problems listed. That's a pipeline protecting production from bad data.

Each run costs roughly 5–10 cents.

### Troubleshooting

- **`exec format error`:** the image was built for ARM. Rebuild with `--platform linux/amd64`.
- **Image pull permission denied:** grant the Vertex AI service agent read access to your images: `gcloud artifacts repositories add-iam-policy-binding eb-rampup --location=$REGION --member="serviceAccount:service-$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')@gcp-sa-aiplatform-cc.iam.gserviceaccount.com" --role=roles/artifactregistry.reader`
- **`does not have permission iam.serviceAccounts.actAs`:** the self-grant in Part 2c didn't take, or hasn't propagated yet. Rerun it and wait a minute.
- **`ModuleNotFoundError: eb_rampup` in a step:** the step is using the placeholder image. Check that `TRAINER_IMAGE` was exported *before* running the runner.
- **A fix "didn't work":** check that you rebuilt and pushed the image, and that `TRAINER_IMAGE` has the new tag.
- Anything else: paste the error and the failing step's log.

---

## Part 7: CI with GitHub Actions (25 min)

### 7a. The workflow

GitHub Actions runs whatever is in `.github/workflows/` when its trigger happens. Create `.github/workflows/ci.yml`:

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
```

Line by line:

- **`on`:** run on every push to `main` and on every pull request.
- **`runs-on: ubuntu-latest`:** GitHub provides a fresh Linux machine each run. Nothing carries over from your laptop, which is the point: if tests pass here, they don't depend on anything only you have.
- **`python-version: "3.11"`:** the same version as your image, so CI tests what production runs. `cache: pip` reuses downloaded packages between runs, which makes them faster.
- **Install, lint, test:** the same commands you run locally. Integration tests stay skipped, thanks to your `addopts`. CI has no GCP credentials, and doesn't need any for unit tests.

GitHub Actions is free for public repos and includes 2,000 minutes a month for private repos on the free plan. This workflow uses about a minute per run.

### 7b. Add a status badge

Create or edit `README.md` and add, replacing `YOUR_USER`:

```markdown
![CI](https://github.com/YOUR_USER/eb-rampup/actions/workflows/ci.yml/badge.svg)
```

Push, then open the **Actions** tab on GitHub and watch the run. The badge on your README turns green when it passes.

### 7c. Break it on purpose, through a pull request

This is the workflow you'll use at work: changes go through pull requests, and CI checks them before they merge.

```bash
git checkout -b break-something
```

Change the expected value in `test_weighted_mae_by_hand` from `2.5` to `2.6`, then:

```bash
git commit -am "Break a test on purpose"
git push -u origin break-something
```

Open a pull request on GitHub. Within a minute, the PR shows a red ✗ with a link to the failing test's output. Fix the value, commit, and push to the same branch: the PR updates and turns green. Merge it, then clean up:

```bash
git checkout main && git pull && git branch -d break-something
```

Many teams also turn on a rule that **blocks merging until CI passes**, under Settings → Branches (or Rulesets). On GitHub's free plan, that's only available for public repos, so you may not be able to enable it on a private one; the Hartford's GitHub almost certainly has it on.

---

## Part 8: Commit, self-check, and report back (5 min)

```bash
ruff check . && pytest
git add . && git commit -m "Day 3: pipeline and CI" && git push
```

**Self-check** (answers at the end):

1. What's the difference between a KFP artifact and a parameter? Give an example of each from today's pipeline.
2. Why are imports inside each component function?
3. You fixed a bug in `features.py`, ran the pipeline, and the bug is still there. What's the most likely reason?
4. Why is caching turned off for this pipeline?
5. Why does the promotion rule require a margin, and why does it check the book-rate baseline even when there's no champion?
6. Which of today's tests run in CI, and which checks only run inside the pipeline?

**Report back** for day 4:

- What broke or confused you, especially in Parts 4 and 6
- What happened in each of the three runs (promoted, skipped, or failed, and why)
- How long each part took

---

## Answers and solutions

### Part 3d: validation test solutions

```python
def test_tiny_groups_valid_with_low_min_rows(tiny_groups):
    assert validate_groups(tiny_groups, min_rows=1) == []


def test_too_few_rows(tiny_groups):
    problems = validate_groups(tiny_groups, min_rows=1_000)
    assert any("only 3 rows" in p for p in problems)


def test_missing_column_stops_early(tiny_groups):
    problems = validate_groups(tiny_groups.drop(columns=["avg_age"]), min_rows=1)
    assert problems == ["missing columns: ['avg_age']"]


def test_nulls_are_reported(tiny_groups):
    tiny_groups.loc[0, "avg_salary"] = np.nan
    assert "avg_salary: 1 null values" in validate_groups(tiny_groups, min_rows=1)


def test_duplicate_ids_are_reported(tiny_groups):
    tiny_groups.loc[1, "group_id"] = "A"
    assert "1 duplicate group_ids" in validate_groups(tiny_groups, min_rows=1)


@pytest.mark.parametrize(
    "column, bad_value",
    [("group_size", 0), ("avg_age", 120), ("pct_female", 1.5), ("loss_ratio_next", -1)],
)
def test_out_of_range_values(tiny_groups, column, bad_value):
    tiny_groups.loc[0, column] = bad_value
    problems = validate_groups(tiny_groups, min_rows=1)
    assert any(p.startswith(f"{column}:") for p in problems)


def test_reports_all_problems_at_once(tiny_groups):
    tiny_groups.loc[0, "avg_age"] = 120
    tiny_groups.loc[1, "industry"] = "Space Mining"
    assert len(validate_groups(tiny_groups, min_rows=1)) == 2
```

The function stops early on missing columns because every later check reads those columns and would crash with a `KeyError`, burying the real message.

### Part 3d: promotion test solutions

```python
@pytest.mark.parametrize(
    "challenger_mae, champion_mae, min_improvement, expected",
    [
        (0.10, 0.12, 0.0, True),     # better
        (0.13, 0.12, 0.0, False),    # worse
        (0.12, 0.12, 0.0, False),    # a tie is not an improvement
        (0.119, 0.12, 0.02, False),  # better, but not by 2%
        (0.117, 0.12, 0.02, True),   # better by more than 2%
    ],
)
def test_promotion_rule(challenger_mae, champion_mae, min_improvement, expected):
    promote, _ = should_promote(
        metrics(challenger_mae), metrics(champion_mae), min_improvement
    )
    assert promote is expected


def test_never_promote_a_model_worse_than_the_book_rate():
    promote, reason = should_promote(metrics(0.35, baseline=0.30), champion=None)
    assert not promote
    assert "book rate" in reason
```

The comments next to each case document the *intent*, so when one fails months from now, the reader knows what behavior broke.

### Part 5c: pipeline test solutions

```python
def test_main_steps_run_unconditionally(spec):
    top_level = spec["root"]["dag"]["tasks"]
    for step in ["validate-data", "train", "compare-with-champion"]:
        assert step in top_level


def test_registration_is_gated(spec):
    # register-model must NOT be a top-level step: it lives inside the condition
    assert "register-model" not in spec["root"]["dag"]["tasks"]
    assert "register-model" in str(spec["components"])
```

### Self-check answers

1. A **parameter** is a small value passed directly, like `table`, `min_improvement`, or the `promote` output. An **artifact** is a file or folder the step reads or writes by path, like the validated `data` (a Parquet file) or the trained `model` (a folder with `model.bst`, `metrics.json`, and `features.json`).
2. KFP sends only the function's source code into the step's container, so the function has to bring its own imports.
3. The pipeline runs the code baked into the container image. You probably didn't rebuild and push the image, or `TRAINER_IMAGE` still points at the old tag.
4. `validate_data` reads a BigQuery table by name. If the table's contents changed but its name didn't, a cached step would silently reuse stale data.
5. The margin stops the champion from flipping on noise-level differences, since every change is a production change. The baseline check is a sanity floor: it catches a broken training run even when there's no champion to compare against.
6. CI runs the unit tests: validation and promotion rules, features, training, and the pipeline compiling with registration gated. The data validation of the *real* batch and the champion comparison run inside the pipeline, on every real run. Both kinds are tests: one checks code, the other checks data and models.
