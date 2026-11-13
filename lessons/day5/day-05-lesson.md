# Day 5: Monitoring, Drift, and Retraining

**About 3 hours**

By the end of today you'll be able to tell, from data alone, when your champion is going stale. A written retraining policy will decide what to do about it, and you'll watch two scenarios play out end to end: one where a retrained model goes live automatically, and one where a human has to approve it first.

| Part | What you do | Time |
| --- | --- | --- |
| 1 | Concepts: kinds of drift, PSI, and why insurance makes this hard | 15 min |
| 2 | Simulate a changing world | 15 min |
| 3 | Measure drift, with tests | 40 min |
| 4 | The retraining policy, with tests | 25 min |
| 5 | A monitoring run | 15 min |
| 6 | Pipeline changes: a fair comparison and a review path | 25 min |
| 7 | Scenario A: retrain and promote automatically | 15 min |
| 8 | Scenario B: retrain and hold for review | 20 min |
| 9 | Commit, self-check, and report back | 5 min |

This lesson follows the testing convention in your plan: **no `unittest.mock`**. Test doubles are small handwritten classes.

---

## Part 1: Concepts (15 min)

### Three ways a model goes stale

| Kind | What changes | Example in group disability |
| --- | --- | --- |
| **Data drift** (covariate shift) | The *inputs*: who you insure | The book shifts toward older, more manual-labor workforces |
| **Concept drift** | The *relationship* between inputs and outcomes | Medical and wage inflation make the same group cost 15% more |
| **Model aging** | Nothing measurable yet, but the world it learned from is a year further away | A model trained before a new state paid-leave law |

Data drift doesn't always hurt a model: if it learned the age effect correctly, older groups just get higher, still-correct predictions. Concept drift always hurts, because the model's learned relationship is now wrong.

### Why insurance makes this hard: outcomes arrive late

To measure whether a model is getting worse, you need **actual outcomes**: how a group's claims really turned out. In group disability, that takes a year or more. Claims are filed, then run for months.

So monitoring works on two clocks:

- **Inputs and predictions, available immediately.** If the data or the predictions look different from training, something has changed, even before you know whether the model is wrong. These are **leading indicators**.
- **Performance, available once outcomes arrive.** This is the real verdict, but it lags.

Today's monitoring checks both, and the policy treats them differently.

### PSI: measuring how much a distribution moved

The **population stability index** compares two distributions of the same column. It's the industry standard in insurance and credit risk.

1. Split the **reference** data (what the model trained on) into 10 bins, each holding about 10% of its rows.
2. Count what share of the **current** data falls into each bin.
3. Add up, across bins:

```latex
\text{PSI} = \sum_{\text{bins}} (\text{current share} - \text{reference share}) \times \ln\frac{\text{current share}}{\text{reference share}}
```

Every bin contributes a non-negative amount, so PSI is 0 when nothing moved and grows as distributions separate. The usual rules of thumb:

| PSI | Reading |
| --- | --- |
| Under 0.10 | Stable |
| 0.10 to 0.25 | Moderate shift: worth a look |
| Over 0.25 | Significant shift: the population has changed |

For a categorical column like industry, each category is a bin.

### A flaw in Day 3's pipeline

Day 3's `compare_with_champion` compared the challenger's MAE on **its** test data with the champion's MAE as **stored at training time**, on **older** data. When the world hasn't changed, that's roughly fair. Under drift, it isn't: the champion's stored number describes a world that no longer exists. Today's pipeline fixes that by scoring **both** models on **the same test rows**.

---

## Part 2: Simulate a changing world (15 min)

### 2a. Give the generator some dials

Replace `generate_groups` in `src/eb_rampup/data.py` with this version. It adds four keyword-only options, all defaulting to "no change," so existing calls (and your BigQuery table) produce exactly the same data as before:

```python
def generate_groups(
    n: int,
    seed: int = 42,
    *,
    age_shift: float = 0.0,
    industry_weights: dict[str, float] | None = None,
    claims_trend: float = 1.0,
    id_prefix: str = "G",
) -> pd.DataFrame:
    """Generate n synthetic employer groups with 3 years of history and a target.

    Small groups get much noisier yearly results than large ones, the same
    reason real underwriters trust a 5,000-life group's history more than a
    20-life group's.

    The keyword arguments simulate a changing world (all default to "no change"):
    age_shift moves the average age, industry_weights changes the industry mix,
    claims_trend scales every group's true loss ratio, and id_prefix keeps
    group IDs unique across generated batches.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    rng = np.random.default_rng(seed)
    industries = list(INDUSTRY_RISK)

    industry_p = None
    if industry_weights is not None:
        unknown = set(industry_weights) - set(industries)
        if unknown:
            raise ValueError(f"Unknown industries: {sorted(unknown)}")
        weights = np.array([industry_weights.get(i, 0.0) for i in industries])
        industry_p = weights / weights.sum()

    df = pd.DataFrame(
        {
            "group_id": [f"{id_prefix}{i:06d}" for i in range(n)],
            "industry": rng.choice(industries, size=n, p=industry_p),
            "region": rng.choice(REGIONS, size=n),
            "group_size": np.clip(rng.lognormal(4.0, 1.2, size=n), 2, 20_000).astype(int),
            "avg_age": np.clip(rng.normal(42 + age_shift, 6, size=n), 22, 64).round(1),
            "pct_female": rng.beta(5, 5, size=n).round(3),
            "avg_salary": np.clip(rng.lognormal(np.log(65_000), 0.35, size=n), 25_000, 400_000).round(-2),
            "elimination_days": rng.choice(ELIMINATION_DAYS, size=n, p=[0.8, 0.2]),
            "benefit_pct": rng.choice(BENEFIT_PCTS, size=n, p=[0.3, 0.5, 0.2]),
        }
    )

    true_lr = expected_loss_ratio(df) * claims_trend
    premium = (df["group_size"] * df["avg_salary"] * PREMIUM_RATE).round(2)
    noise_sd = 0.10 + 1.5 / np.sqrt(df["group_size"])

    for year in YEARS:
        yearly_lr = true_lr * rng.lognormal(-(noise_sd**2) / 2, noise_sd)
        df[f"premium_y{year}"] = premium
        df[f"claims_y{year}"] = (premium * yearly_lr).round(2)

    df["loss_ratio_next"] = (true_lr * rng.lognormal(-(noise_sd**2) / 2, noise_sd)).round(4)
    return df
```

- **`age_shift`** moves the average age.
- **`industry_weights`** changes the industry mix.
- **`claims_trend`** scales every group's true loss ratio: concept drift, like cost inflation.
- **`id_prefix`** keeps group IDs unique across batches, since your validation rejects duplicates.

The `*` in the signature makes everything after it **keyword-only**. `generate_groups(100, 1, 5.0)` is an error, so a call can't accidentally pass an age shift where a seed was meant.

### 2b. Challenges: test the dials (10 min)

Add these to `tests/test_data.py`:

1. **Defaults are unchanged:** passing every option at its default produces the same frame as not passing them.
2. **`age_shift=5` makes groups about 5 years older** on average. *Hint:* `pytest.approx(5, abs=0.5)`. Why not exactly 5? Look at the `np.clip` on age.
3. **`claims_trend=1.2` raises the average loss ratio by 20%.** Use the same seed for both, so only the trend differs.
4. **`industry_weights={"Healthcare": 1.0}` produces only healthcare groups.**
5. **An unknown industry in the weights is rejected,** naming the bad value.
6. **`id_prefix="T"` gives IDs `T000000`, `T000001`, ...**

✅ **Check:** `pytest tests/test_data.py -v` passes.

---

## Part 3: Measure drift, with tests (40 min)

### 3a. The code

Create `src/eb_rampup/monitoring.py`:

```python
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
```

Things to notice:

- **Bins come from the reference data's quantiles,** so each holds about 10% of reference rows. The outer edges are infinite, so current values outside the reference's range still land in a bin. A shift *beyond* anything seen in training is exactly what you want to catch.
- **`np.unique` on the edges handles columns with few distinct values.** `elimination_days` only takes 90 or 180, so most quantiles coincide. Without deduplication, `np.histogram` would get repeated edges.
- **`EPSILON` replaces empty bins.** A bin with zero rows would make `log(0)`, so an empty bin counts as a tiny share instead. A brand-new category then produces a large PSI rather than a crash.
- **`experience_lr` is monitored too.** It's computed from claims history, so it moves when **costs** move, even if no group characteristic changed. That makes it an early warning for concept drift, before outcomes arrive. You'll see it fire in Scenario A.
- **`build_report` only computes `mae_now` when outcomes are present.** Without the target column, it reports `None` instead of guessing.

### 3b. The first test

Create `tests/test_monitoring.py`:

```python
import numpy as np
import pytest

from eb_rampup.data import generate_groups
from eb_rampup.monitoring import build_report, feature_drift, psi, psi_categorical


def test_identical_data_has_no_drift():
    values = np.random.default_rng(0).normal(42, 6, 5_000)
    assert psi(values, values) == pytest.approx(0.0, abs=1e-9)
```

### 3c. Challenges (30 min)

These are tests with **known answers**: you construct data where you know how much drift there should be, then check that PSI agrees.

7. **A large shift is severe:** samples from a normal distribution with mean 42 vs. 47 (standard deviation 6) give a PSI above 0.25.
8. **A small shift is below moderate:** 42 vs. 42.5 stays under 0.10.
9. **Two samples of the same population are stable:** two independent 10,000-row draws from the same distribution give PSI under 0.02. This one matters: real data always has *some* random difference, and the test confirms PSI doesn't mistake sampling noise for drift.
10. **A column with few distinct values works:** 80% 90-day / 20% 180-day vs. a 50/50 split gives PSI above 0.25, without errors.
11. **A new category is flagged:** a categorical column that gains a "Healthcare" category at 20% gives PSI above 0.25.
12. **`feature_drift` finds what changed:** compare `generate_groups` with a version using `age_shift=5`. `avg_age` should be above 0.25, and `region`, which you didn't touch, below 0.02.

**13. Testing `build_report` with a handwritten fake.** `build_report` needs a model with `version` and `predict`. A real XGBoost model's predictions are hard to reason about, but a fake whose predictions you can *calculate* makes the test's expectations obvious. Write a small class:

```python
class AgeModel:
    """Fake model: predicts from age, so we know exactly how predictions should move."""

    version = "fake-1"

    def predict(self, df):
        return df["avg_age"].to_numpy() / 60
```

Its predictions are age divided by 60, so if ages shift a lot, predictions must shift a lot too. Use it to test:

- With an `age_shift=5` batch, the report has `model_version == "fake-1"`, a `prediction_psi` above 0.25, and an `mae_now` value.
- When the current batch has no `loss_ratio_next` column (`.drop(columns=["loss_ratio_next"])`), `mae_now` is `None`.

This fake has no `__init__`, because nothing about it varies between tests. That's the "only what's needed" rule: add configuration when a test needs it, not before.

✅ **Check:** `pytest tests/test_monitoring.py -v` passes.

---

## Part 4: The retraining policy, with tests (25 min)

### 4a. The code

Create `src/eb_rampup/retraining.py`:

```python
"""The rules for when to retrain, and whether a retrained model may go live unattended."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrainingPolicy:
    psi_moderate: float = 0.10       # worth a warning
    psi_severe: float = 0.25         # the population has really changed
    max_mae_increase: float = 0.10   # 10% worse than at training time
    max_model_age_days: int = 365


@dataclass(frozen=True)
class Decision:
    retrain: bool
    auto_promote: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def decide(policy: RetrainingPolicy, report: dict, model_age_days: int) -> Decision:
    psis = {**report["feature_psi"], "predictions": report["prediction_psi"]}
    severe = sorted(k for k, v in psis.items() if v >= policy.psi_severe)
    moderate = sorted(k for k, v in psis.items() if policy.psi_moderate <= v < policy.psi_severe)

    reasons, warnings = [], []
    if severe:
        reasons.append(f"severe drift in {severe}")
    if moderate:
        warnings.append(f"moderate drift in {moderate}")

    mae_now, mae_then = report["mae_now"], report["mae_at_training"]
    if mae_now is not None and mae_now > mae_then * (1 + policy.max_mae_increase):
        reasons.append(f"MAE rose from {mae_then:.4f} to {mae_now:.4f}")

    if model_age_days > policy.max_model_age_days:
        reasons.append(f"model is {model_age_days} days old")

    retrain = bool(reasons)
    # A big shift in who we insure deserves a human look before a new model goes live
    auto_promote = retrain and not severe
    return Decision(retrain, auto_promote, tuple(reasons), tuple(warnings))
```

The policy, in words:

| Situation | Retrain? | Go live automatically? |
| --- | --- | --- |
| Everything stable | No | n/a |
| Moderate drift only | No, but warn | n/a |
| Error up more than 10% | Yes | Yes, if it beats the champion |
| Model older than a year | Yes | Yes, if it beats the champion |
| **Severe drift**, in any input or in predictions | Yes | **No: hold for human review** |

Why hold on severe drift, even when the new model wins? A model that wins on a population it has barely seen may be winning for reasons nobody has checked. When who you insure changes substantially, an underwriter or actuary should look before pricing shifts. Automatic promotion is reserved for "same kind of business, model needs refreshing."

Three design choices worth noticing:

- **Thresholds are parameters, not constants buried in the code.** Model risk teams set these, and they'll want to change them without a code review of the logic.
- **`Decision` records reasons and warnings as text.** When the policy acts at 2 a.m., the log says why.
- **`frozen=True`** makes both dataclasses immutable. Nobody can quietly change a threshold or a decision after it's made.

### 4b. The first test

Create `tests/test_retraining.py`:

```python
import pytest

from eb_rampup.retraining import RetrainingPolicy, decide

POLICY = RetrainingPolicy()


def report(psi: float = 0.0, prediction_psi: float = 0.0, mae_now=0.12, mae_then: float = 0.12):
    """Test helper: a monitoring report with every feature at the same PSI."""
    return {
        "feature_psi": {"avg_age": psi, "industry": 0.0},
        "prediction_psi": prediction_psi,
        "mae_now": mae_now,
        "mae_at_training": mae_then,
    }


def test_stable_data_and_steady_error_means_no_retraining():
    decision = decide(POLICY, report(), model_age_days=30)
    assert not decision.retrain
    assert decision.reasons == ()
```

`report()` is a **test helper**, like `metrics()` on Day 3. Each test passes only what's different from a quiet, healthy report, which keeps every case one readable line.

### 4c. Challenges (15 min)

14. **One parametrized test covering the policy table:** moderate drift (PSI 0.15), error up 17% (0.12 → 0.14), error up 8% (just under the limit), a 400-day-old model, severe drift in a feature, severe drift in predictions, severe drift *and* higher error, and no outcomes yet (`mae_now=None`). Check both `retrain` and `auto_promote` for each.
15. **Moderate drift produces the exact warning text.**
16. **The reasons explain the decision,** in order: severe drift and higher error together give both messages.
17. **Thresholds are configurable:** a policy with `max_mae_increase=0.05` retrains on the 8% rise that the default ignores.
18. **Model age in days.** Create `src/eb_rampup/monitor.py` first (Part 5a), then test `model_age_days` with two fixed dates 60 days apart. Passing `now` in, rather than reading the clock inside the function, is what makes this testable, the same dependency-injection idea as the BigQuery client and the API's model.

✅ **Check:** `pytest tests/test_retraining.py -v` passes.

---

## Part 5: A monitoring run (15 min)

### 5a. The runner

Create `src/eb_rampup/monitor.py`:

```python
"""A monitoring run: compare current data with the champion's training data, and decide.

    python -m eb_rampup.monitor --project P \
        --reference eb_rampup.groups --current eb_rampup.groups_trend
"""

import argparse
import json
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from eb_rampup.monitoring import build_report
from eb_rampup.retraining import Decision, RetrainingPolicy, decide


def model_age_days(created: datetime, now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    return (now - created).days


def run(project: str, region: str, model_name: str, reference_table: str,
        current_table: str, policy: RetrainingPolicy = RetrainingPolicy(),
        report_dir: str = "artifacts/monitoring") -> tuple[dict, Decision]:
    from google.cloud import bigquery

    from eb_rampup.bq import load_table
    from eb_rampup.registry import download_dir, find_champion
    from eb_rampup.serving import load_model_dir

    champion = find_champion(project, region, model_name)
    if champion is None:
        raise RuntimeError(f"No version of {model_name} has the champion alias")
    local = download_dir(champion.uri, tempfile.mkdtemp(), project)
    model = load_model_dir(local, version=str(champion.version_id))
    training_mae = json.loads((local / "metrics.json").read_text())["mae"]

    client = bigquery.Client(project=project)
    reference = load_table(client, reference_table)
    current = load_table(client, current_table)

    report = build_report(model, reference, current, training_mae)
    created = getattr(champion, "version_create_time", None) or champion.create_time
    decision = decide(policy, report, model_age_days(created))

    run_at = datetime.now(timezone.utc)
    report.update(
        run_at=run_at.isoformat(),
        reference_table=reference_table,
        current_table=current_table,
        decision=asdict(decision),
    )
    out = Path(report_dir) / f"report-{run_at:%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Report saved to {out}")
    return report, decision


def main(argv=None) -> Decision:
    p = argparse.ArgumentParser(description="Check the champion for drift.")
    p.add_argument("--project", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--model-name", default="loss-ratio-ltd")
    p.add_argument("--reference", required=True, help="table the champion was trained on")
    p.add_argument("--current", required=True, help="the newest batch of data")
    args = p.parse_args(argv)

    report, decision = run(args.project, args.region, args.model_name,
                           args.reference, args.current)
    for name, value in sorted(report["feature_psi"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<18} PSI {value:.3f}")
    print(f"  {'predictions':<18} PSI {report['prediction_psi']:.3f}")
    if report["mae_now"] is not None:
        print(f"  MAE at training {report['mae_at_training']:.4f}, now {report['mae_now']:.4f}")
    print(f"Retrain: {decision.retrain}   Auto-promote: {decision.auto_promote}")
    for line in decision.reasons + decision.warnings:
        print(f"  - {line}")
    return decision


if __name__ == "__main__":
    main()
```

It loads the champion (downloading its files, including `metrics.json` for the training-time MAE), reads the two tables, builds the report, applies the policy, and saves everything as JSON in `artifacts/monitoring/`. That file is your record of what the policy saw and decided.

### 5b. Make two "next year" tables

Create `scripts/make_drift_data.py`:

```python
"""Create two "next year" tables in BigQuery, each with a different kind of change.

    python scripts/make_drift_data.py --project $PROJECT_ID
"""

import argparse

from google.cloud import bigquery

from eb_rampup.data import generate_groups

# Scenario B's industry mix: more healthcare and construction, less tech and finance
SHIFTED_MIX = {
    "Healthcare": 2.0, "Construction": 2.0, "Transportation": 1.5, "Manufacturing": 1.5,
    "Retail": 1.0, "Education": 1.0, "Finance": 0.6, "Technology": 0.5,
    "Professional Services": 0.5,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--n", type=int, default=20_000)
    args = p.parse_args()

    tables = {
        # A: same kinds of groups, but claims costs up 15% (medical/wage inflation)
        "eb_rampup.groups_trend": generate_groups(
            args.n, seed=101, claims_trend=1.15, id_prefix="T"),
        # B: an older, riskier book of business, and costs up further
        "eb_rampup.groups_shift": generate_groups(
            args.n, seed=202, claims_trend=1.40, age_shift=5,
            industry_weights=SHIFTED_MIX, id_prefix="S"),
    }
    client = bigquery.Client(project=args.project)
    config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    for table, df in tables.items():
        client.load_table_from_dataframe(df, table, job_config=config).result()
        print(f"Loaded {len(df):,} rows into {table}")


if __name__ == "__main__":
    main()
```

```bash
python scripts/make_drift_data.py --project $PROJECT_ID
```

(If your `groups` table has 50,000 rows, add `--n 50000` to match.)

- **`groups_trend` (Scenario A):** the same kinds of groups, but claims cost 15% more. Pure concept drift.
- **`groups_shift` (Scenario B):** groups 5 years older, a riskier industry mix, and costs up further. Data drift and concept drift together.

### 5c. Look before acting

Run monitoring on both, without retraining anything yet:

```bash
python -m eb_rampup.monitor --project $PROJECT_ID \
  --reference eb_rampup.groups --current eb_rampup.groups_trend

python -m eb_rampup.monitor --project $PROJECT_ID \
  --reference eb_rampup.groups --current eb_rampup.groups_shift
```

What to look for in Scenario A:

- Almost every input has a PSI near zero, because the groups look the same.
- **`experience_lr` shows moderate drift.** The cost trend appears in claims history before any new outcomes arrive. That's the leading indicator from Part 1.
- MAE rose well past 10%, so the decision is retrain and auto-promote.

In Scenario B, expect severe drift in `avg_age`, `industry`, `experience_lr`, and predictions, and a decision to retrain but hold for review.

---

## Part 6: Pipeline changes: a fair comparison and a review path (25 min)

### 6a. The new pipeline

Replace `pipelines/training_pipeline.py`. Changes from Day 3 are marked `NEW`:

```python
"""Vertex AI pipeline: validate -> train -> compare with champion -> register.

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
def train(
    data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    test_data: Output[Dataset],          # NEW: so the comparison can reuse the same rows
):
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
    test_df.to_parquet(test_data.path, index=False)


@dsl.component(base_image=TRAINER_IMAGE)
def compare_with_champion(
    model: Input[Model], test_data: Input[Dataset], project: str, region: str,
    model_name: str, min_improvement: float,
) -> NamedTuple("Outputs", [("promote", bool), ("reason", str)]):
    """Score BOTH models on the challenger's test rows, so they're judged on the same data."""
    import json
    import tempfile
    from collections import namedtuple
    from pathlib import Path

    import pandas as pd

    from eb_rampup.features import TARGET, sample_weights
    from eb_rampup.promotion import should_promote
    from eb_rampup.registry import download_dir, find_champion
    from eb_rampup.serving import FeatureMismatchError, load_model_dir
    from eb_rampup.train import weighted_mae

    test_df = pd.read_parquet(test_data.path)
    y, w = test_df[TARGET], sample_weights(test_df)

    challenger = json.loads((Path(model.path) / "metrics.json").read_text())
    challenger["mae"] = weighted_mae(y, load_model_dir(model.path).predict(test_df), w)

    champion = None
    found = find_champion(project, region, model_name)
    if found is None:
        print("No champion yet")
    else:
        local = download_dir(found.uri, tempfile.mkdtemp(), project)
        try:
            champ_model = load_model_dir(local, version=str(found.version_id))
            champion = {"mae": weighted_mae(y, champ_model.predict(test_df), w)}
            print(f"Champion is version {found.version_id}; "
                  f"on this test data its MAE is {champion['mae']:.4f}")
        except FeatureMismatchError as e:
            print(f"Champion can't score today's features, so it can't be compared: {e}")

    promote, reason = should_promote(challenger, champion, min_improvement)
    print(reason)
    return namedtuple("Outputs", ["promote", "reason"])(promote, reason)


@dsl.component(base_image=TRAINER_IMAGE)
def register_model(
    model: Input[Model], project: str, region: str, model_name: str,
    serving_image: str, git_sha: str, alias: str,      # NEW: "champion" or "pending-review"
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
        is_default_version=(alias == "champion"),
        version_aliases=[alias],  # an alias moves off whichever version had it
        version_description=f"MAE {metrics['mae']:.4f} | registered as {alias}",
        labels={"git_sha": git_sha},
    )

    # Verify instead of assuming: fail loudly if the alias didn't land
    registry = aiplatform.models.ModelRegistry(registered)
    holders = [v.version_id for v in registry.list_versions() if alias in v.version_aliases]
    if holders != [registered.version_id]:
        raise RuntimeError(f"Expected version {registered.version_id} to be {alias}, found {holders}")
    print(f"Registered version {registered.version_id} as {alias}")


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
    auto_promote: bool = True,           # NEW: False sends a winner to human review
):
    validated = validate_data(project=project, table=table, min_rows=min_rows)
    trained = train(data=validated.outputs["data"])
    compared = compare_with_champion(
        model=trained.outputs["model"], test_data=trained.outputs["test_data"],
        project=project, region=region, model_name=model_name,
        min_improvement=min_improvement,
    )
    # `== True` is required: this builds a pipeline condition, not a Python if
    with dsl.If(compared.outputs["promote"] == True, name="beats-champion"):  # noqa: E712
        with dsl.If(auto_promote == True, name="auto-promote"):  # noqa: E712
            register_model(
                model=trained.outputs["model"], project=project, region=region,
                model_name=model_name, serving_image=serving_image, git_sha=git_sha,
                alias="champion",
            )
        with dsl.Else():
            register_model(
                model=trained.outputs["model"], project=project, region=region,
                model_name=model_name, serving_image=serving_image, git_sha=git_sha,
                alias="pending-review",
            )
```

What changed:

- **`train` also outputs its test rows** (`test_data`), so the comparison can use exactly the same rows.
- **`compare_with_champion` scores both models on those rows.** It downloads the champion, loads it with the same `load_model_dir` the API uses (so the log transform, if you added it, is handled), and computes both MAEs on today's data. If the champion was trained on different features, it can't be compared, so the step says so and treats it as no champion.
- **`register_model` takes an `alias`**: `"champion"` for automatic promotion, `"pending-review"` otherwise. Only a champion becomes the default version, and the postcondition check confirms whichever alias was requested actually landed.
- **A nested condition:** if the challenger wins *and* `auto_promote` is true, it becomes the champion. If it wins but `auto_promote` is false, it's registered as `pending-review`, and the current champion stays in place.

`dsl.If` and `dsl.Else` work like Python's `if`/`else`, except both branches appear in the pipeline graph, and the unused one shows as skipped when the run happens.

### 6b. Let the runner pass `auto_promote`

In `pipelines/run_pipeline.py`, add an argument after `--min-improvement`:

```python
    p.add_argument("--auto-promote", action=argparse.BooleanOptionalAction, default=True,
                   help="--no-auto-promote sends a winning model to review instead")
```

and pass it in `parameter_values`:

```python
            "auto_promote": args.auto_promote,
```

`BooleanOptionalAction` creates both `--auto-promote` and `--no-auto-promote` from one line.

### 6c. Pipeline tests

Add to `tests/test_pipeline.py`:

19. **`auto_promote` is a pipeline parameter:** check it's in `spec["root"]["inputDefinitions"]["parameters"]`.
20. **Both registration paths exist:** `"champion"` and `"pending-review"` both appear in the compiled spec.

### 6d. Rebuild the image

The package changed (`data.py`, plus the new `monitoring.py`, `retraining.py`, and `monitor.py`), and the comparison step imports `registry` and `serving`. Commit, then rebuild with a new tag:

```bash
ruff check . && pytest
git add . && git commit -m "Day 5: drift monitoring and retraining policy"
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/eb-rampup/trainer:$(git rev-parse --short HEAD)"
docker buildx build --platform linux/amd64 -t $IMAGE --push .
export TRAINER_IMAGE=$IMAGE
```

Pushing also runs CI and redeploys the API through your candidate flow. The API code didn't change, so that's just a fresh revision with identical behavior.

---

## Part 7: Scenario A: retrain and promote automatically (15 min)

Create `scripts/monitor_and_retrain.py`, which connects monitoring to the pipeline:

```python
"""Run monitoring and, if the policy says so, start the training pipeline.

    TRAINER_IMAGE=$IMAGE python scripts/monitor_and_retrain.py --project $PROJECT_ID \
        --bucket $BUCKET --service-account pipeline-sa@$PROJECT_ID.iam.gserviceaccount.com \
        --reference eb_rampup.groups --current eb_rampup.groups_trend
"""

import argparse
import subprocess
import sys

from eb_rampup.monitor import main as monitor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--service-account", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--current", required=True)
    args = p.parse_args()

    decision = monitor(["--project", args.project,
                        "--reference", args.reference, "--current", args.current])
    if not decision.retrain:
        print("No retraining needed.")
        return

    # Data window: train on the newest batch only (see the lesson for alternatives)
    subprocess.run(
        [sys.executable, "-m", "pipelines.run_pipeline",
         "--project", args.project, "--bucket", args.bucket,
         "--service-account", args.service_account,
         "--table", args.current,
         "--auto-promote" if decision.auto_promote else "--no-auto-promote"],
        check=True,
    )


if __name__ == "__main__":
    main()
```

**The data window:** this script trains on **the newest batch only**. That's the simplest choice, and it's right when the world has clearly changed. The trade-off is less data. Actuaries handle this differently: they keep several years of experience but **trend** older years up to today's cost level, so more data can be used without teaching the model yesterday's prices. Either approach is a policy decision, and it belongs in writing, like the thresholds.

Run Scenario A:

```bash
python scripts/monitor_and_retrain.py --project $PROJECT_ID --bucket $BUCKET \
  --service-account pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --reference eb_rampup.groups --current eb_rampup.groups_trend
```

It prints the monitoring summary, decides to retrain with auto-promotion, and submits the pipeline. In the console, follow the run:

- **`compare-with-champion` logs** show both MAEs on the same test rows. The challenger, trained on today's costs, should beat the champion.
- **The graph** shows `beats-champion` → `auto-promote` taken and the `Else` branch skipped.
- **The Model Registry** shows the `champion` alias on a new version, with the old champion still listed for rollback.

---

## Part 8: Scenario B: retrain and hold for review (20 min)

The champion was just trained on `groups_trend`, so that's now the reference:

```bash
python scripts/monitor_and_retrain.py --project $PROJECT_ID --bucket $BUCKET \
  --service-account pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --reference eb_rampup.groups_trend --current eb_rampup.groups_shift
```

This time the decision is retrain **without** auto-promotion, so the pipeline runs with `auto_promote=false`. The challenger should still beat the champion (costs rose again), but it takes the `Else` branch: registered as **`pending-review`**, while the existing champion keeps serving.

### Review, then approve

Now play the reviewer. Open the saved monitoring report in `artifacts/monitoring/` and the new version's metrics. The questions a real review would ask:

- What changed in the population, and is it expected (a new large client, a sales push into construction)?
- Do the new model's predictions make sense for the groups that changed, like older construction groups?
- Is anything in the new data suspicious, such as a data feed error rather than a real change?

When satisfied, approve with `scripts/approve.py`:

```python
"""Approve a model version that's waiting for review: make it the champion.

    python scripts/approve.py --project $PROJECT_ID --version 4
"""

import argparse

from google.cloud import aiplatform


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--model-name", default="loss-ratio-ltd")
    p.add_argument("--version", required=True)
    args = p.parse_args()

    aiplatform.init(project=args.project, location=args.region)
    model = aiplatform.Model.list(filter=f'display_name="{args.model_name}"')[0]
    registry = aiplatform.models.ModelRegistry(model)

    waiting = [v.version_id for v in registry.list_versions()
               if "pending-review" in v.version_aliases]
    if args.version not in waiting:
        raise SystemExit(f"Version {args.version} isn't waiting for review (waiting: {waiting})")

    registry.add_version_aliases(["champion"], version=args.version)   # moves it off the old one
    registry.remove_version_aliases(["pending-review"], version=args.version)
    print(f"Version {args.version} is now the champion")


if __name__ == "__main__":
    main()
```

```bash
python scripts/approve.py --project $PROJECT_ID --version <the pending version number>
```

The `champion` alias moves to the approved version. The script refuses to promote anything that isn't waiting for review, so it can't be used to bypass the pipeline's comparison.

Who approved it is recorded automatically. Changes to model aliases are admin actions, and Google Cloud's audit logs capture who made them and when. That's part of the governance trail, alongside the monitoring report and the pipeline run.

**The served API doesn't change until it restarts.** It loads the champion at startup, so a promotion takes effect on the next deploy or new instance. In Day 4's design, that's deliberate: a service's behavior changes when you deploy it. To pick up the new champion now, redeploy (a push to `main`), or start a fresh revision.

### About Vertex AI Model Monitoring

Google offers a managed version of what you built: **Vertex AI Model Monitoring** computes feature drift for registered models on a schedule, against data in BigQuery, and can send alerts. It's a good fit when a team runs many models and wants standard checks without maintaining code. You built your own today because it's transparent (you know exactly what PSI is computing), because the retraining policy is yours to define, and because performance with delayed outcomes needs custom logic either way. It's worth asking in your first week whether the team uses the managed service, its own checks, or both.

---

## Part 9: Commit, self-check, and report back (5 min)

```bash
ruff check . && pytest
git add . && git commit -m "Day 5: scenarios run" && git push
```

**Self-check** (answers at the end):

1. Why monitor inputs and predictions at all, when performance is what actually matters?
2. In Scenario A, no group characteristic changed. Why did `experience_lr` drift?
3. Why must the champion and challenger be scored on the same test rows?
4. Why does severe drift hold promotion for review, even when the challenger wins?
5. Why does `psi` use the reference data's quantiles for bins, with infinite outer edges?
6. `AgeModel` has no `__init__`, while Day 4's `FakeModel` had one. What decides whether a fake needs one?

**Report back** for day 6:

- What the monitoring summaries showed for A and B, and whether the pipelines promoted or held as expected
- Anything that broke, especially in the pipeline changes
- How long each part took

---

## Answers and solutions

### Part 2b: generator test solutions

```python
def test_defaults_are_unchanged_by_drift_options():
    pd.testing.assert_frame_equal(
        generate_groups(200, seed=5),
        generate_groups(200, seed=5, age_shift=0.0, claims_trend=1.0, id_prefix="G"),
    )


def test_age_shift_makes_groups_older():
    base = generate_groups(5_000, seed=1)["avg_age"].mean()
    shifted = generate_groups(5_000, seed=1, age_shift=5)["avg_age"].mean()
    assert shifted - base == pytest.approx(5, abs=0.5)   # clipping at 64 trims a little


def test_claims_trend_raises_loss_ratios():
    base = generate_groups(5_000, seed=1)["loss_ratio_next"].mean()
    trended = generate_groups(5_000, seed=1, claims_trend=1.2)["loss_ratio_next"].mean()
    assert trended / base == pytest.approx(1.2, rel=0.01)


def test_industry_weights_change_the_mix():
    df = generate_groups(5_000, seed=1, industry_weights={"Healthcare": 1.0})
    assert set(df["industry"]) == {"Healthcare"}


def test_unknown_industry_weight_is_rejected():
    with pytest.raises(ValueError, match="Space Mining"):
        generate_groups(10, industry_weights={"Space Mining": 1.0})


def test_id_prefix_keeps_batches_distinct():
    assert generate_groups(3, id_prefix="T")["group_id"].tolist() == ["T000000", "T000001", "T000002"]
```

The age shift isn't exactly 5 because ages are clipped at 64: a group drawn at 62 can only move 2 years.

### Part 3c: monitoring test solutions

```python
def test_large_shift_is_severe():
    rng = np.random.default_rng(1)
    before, after = rng.normal(42, 6, 10_000), rng.normal(47, 6, 10_000)
    assert psi(before, after) > 0.25


def test_small_shift_is_below_moderate():
    rng = np.random.default_rng(2)
    before, after = rng.normal(42, 6, 10_000), rng.normal(42.5, 6, 10_000)
    assert psi(before, after) < 0.10


def test_two_samples_of_the_same_population_are_stable():
    rng = np.random.default_rng(3)
    assert psi(rng.normal(42, 6, 10_000), rng.normal(42, 6, 10_000)) < 0.02


def test_works_on_a_column_with_few_distinct_values():
    before = np.array([90] * 800 + [180] * 200)
    after = np.array([90] * 500 + [180] * 500)
    assert psi(before, after) > 0.25


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
```

### Part 4c: retraining test solutions

```python
@pytest.mark.parametrize(
    "rep, age, retrain, auto_promote",
    [
        (report(psi=0.15), 30, False, False),            # moderate drift: warn only
        (report(mae_now=0.14), 30, True, True),          # error up 17%: retrain, auto
        (report(mae_now=0.13), 30, False, False),        # error up 8%: under the 10% limit
        (report(), 400, True, True),                     # too old: retrain, auto
        (report(psi=0.30), 30, True, False),             # severe drift: hold for review
        (report(prediction_psi=0.30), 30, True, False),  # predictions shifted: hold
        (report(psi=0.30, mae_now=0.14), 30, True, False),  # severe + worse: still hold
        (report(mae_now=None), 30, False, False),        # no outcomes yet: can't judge error
    ],
)
def test_policy(rep, age, retrain, auto_promote):
    decision = decide(POLICY, rep, model_age_days=age)
    assert decision.retrain is retrain
    assert decision.auto_promote is auto_promote


def test_moderate_drift_produces_a_warning():
    decision = decide(POLICY, report(psi=0.15), model_age_days=30)
    assert decision.warnings == ("moderate drift in ['avg_age']",)


def test_reasons_explain_the_decision():
    decision = decide(POLICY, report(psi=0.30, mae_now=0.14), model_age_days=30)
    assert decision.reasons == (
        "severe drift in ['avg_age']",
        "MAE rose from 0.1200 to 0.1400",
    )


def test_policy_thresholds_are_configurable():
    strict = RetrainingPolicy(max_mae_increase=0.05)
    assert decide(strict, report(mae_now=0.13), model_age_days=30).retrain


def test_model_age_in_days():
    from datetime import datetime, timezone

    from eb_rampup.monitor import model_age_days

    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = datetime(2026, 3, 2, tzinfo=timezone.utc)
    assert model_age_days(created, now) == 60
```

### Part 6c: pipeline test solutions

```python
def test_auto_promote_is_a_pipeline_parameter(spec):
    assert "auto_promote" in spec["root"]["inputDefinitions"]["parameters"]


def test_both_registration_paths_exist(spec):
    text = str(spec)
    assert "champion" in text and "pending-review" in text
```

### Self-check answers

1. Because outcomes arrive late: in group disability, a year or more after a prediction. Input and prediction drift are available immediately, so they warn you early that something changed, long before performance numbers can confirm it.
2. `experience_lr` is computed from claims history, and the 15% cost trend applies to past claims as well as future ones. Rising costs show up in history immediately, which makes it an early signal of concept drift even when the groups themselves look the same.
3. Because their errors depend on the data. The champion's stored MAE came from older data. Under drift, comparing it with a challenger's MAE on new data compares two different tests, and either model could win for the wrong reason.
4. A model that wins on a population it has barely seen may be winning for reasons nobody has examined, and a large shift in who you insure is a business event someone should understand before pricing changes. Automatic promotion is for refreshing a model on the same kind of business.
5. Quantile bins give each bin about 10% of the reference rows, so every bin carries weight, including the tails. Infinite outer edges mean current values beyond anything in the reference still land in a bin and register as drift, rather than being dropped.
6. Whether anything about it varies between tests. `AgeModel` always behaves the same, so it needs no configuration. `FakeModel` had to return different predictions in different tests, so it took the value as an argument.
