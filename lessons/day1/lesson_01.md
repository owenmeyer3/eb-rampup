# Day 1: GCP Footing, Your First Tests, and the Data

**Mon Nov 2 · about 2.5 hours**

By the end of today you'll have a GCP project with guardrails, a GitHub repo with tested Python code, and 50,000 synthetic employer groups loaded into BigQuery that you can query from both the console and Python.

| Part | What you do | Time |
| --- | --- | --- |
| 1 | Five concepts that differ from AWS | 10 min |
| 2 | Local setup and the GCP project | 30 min |
| 3 | A least-privilege service account | 15 min |
| 4 | Repo and the data generator | 20 min |
| 5 | Testing: your first test suite | 45 min |
| 6 | Cloud Storage → BigQuery, and SQL | 25 min |
| 7 | Query from Python, plus your first integration test | 15 min |
| 8 | Commit, self-check, and report back | 10 min |

---

## Part 1: Five concepts that differ from AWS (10 min)

Only the parts that will trip up someone who knows AWS:

1. **A project is your AWS account.** Billing, IAM, enabled APIs, and resources all live in a project. Today you'll make one, and deleting it on day 10 cleans up everything.
2. **APIs start switched off.** Every service (BigQuery, Vertex AI, and so on) must be enabled per project before use. "API not enabled" errors are the GCP equivalent of a missing permission.
3. **Two separate logins.** `gcloud auth login` authenticates the CLI. `gcloud auth application-default login` creates **Application Default Credentials (ADC)**, which the Python libraries use. AWS uses one credential chain for both; GCP doesn't, which confuses everyone the first time.
4. **Service accounts replace IAM roles for workloads.** In AWS, a service *assumes a role*. In GCP, a workload *runs as a service account*, and you grant roles *to* that account. GCP "roles" are permission bundles, closer to AWS managed policies.
5. **IAM is granted on resources, not written as policy documents.** No JSON with Effect/Action/Resource. You bind a member (user or service account) to a role on a project, dataset, or bucket.

And one testing concept for today, the **test pyramid**:

- **Unit tests** (the wide base): test one function with no network or cloud. Milliseconds each. You'll write many.
- **Integration tests** (the middle): test your code against a real dependency, like BigQuery. Seconds each, costs a little. You'll write a few.
- **End-to-end tests** (the narrow top): run the whole system. Slow and expensive. You'll write one, on day 9.

Today you build the base and one integration test.

---

## Part 2: Local setup and the GCP project (30 min)

### 2a. Install tools

- **gcloud CLI:** follow the installer for your OS at [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install). On a Mac with Homebrew: `brew install --cask google-cloud-sdk`.
- **Python 3.11+** and **Git**, if you don't have them.
- **Docker** isn't needed until day 4, so install it whenever.

### 2b. Sign up and create the project

Sign up at [cloud.google.com/free](https://cloud.google.com/free). The 90-day trial clock starts now.

Then in your terminal (project IDs must be globally unique, so add something of your own):

```bash
gcloud auth login
export PROJECT_ID="eb-rampup-$(whoami)"   # any unique, lowercase ID
export REGION="us-central1"

gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID

# Link billing (find your billing account ID with the first command)
gcloud billing accounts list
gcloud billing projects link $PROJECT_ID --billing-account=XXXXXX-XXXXXX-XXXXXX

# Turn on the APIs you'll use this week
gcloud services enable storage.googleapis.com bigquery.googleapis.com \
  aiplatform.googleapis.com iam.googleapis.com

# Credentials for Python libraries (the "second login")
gcloud auth application-default login
gcloud auth application-default set-quota-project $PROJECT_ID
```

> Put the two `export` lines in your shell profile, or in a `.env` file you `source`, so they survive new terminal windows. Every command today uses them.

### 2c. Set a budget alert (do not skip)

In the console: **Billing → Budgets & alerts → Create budget**. Scope it to your project, set the amount to **$50**, and keep the alert thresholds at 50%, 90%, and 100%. Alerts only email you; they don't stop spending, which is why the cost rules in the plan matter.

✅ **Check:** `gcloud config list` shows your project, and `gcloud services list --enabled` includes `bigquery.googleapis.com`.

---

## Part 3: A least-privilege service account (15 min)

This is the identity your pipelines will run as from day 3 onward.

```bash
gcloud iam service-accounts create pipeline-sa \
  --display-name="Pipeline runner"

SA="pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com"

for ROLE in roles/bigquery.dataEditor roles/bigquery.jobUser \
            roles/storage.objectAdmin roles/aiplatform.user; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA}" --role="$ROLE" --condition=None
done
```

Notice there's no key file. Avoid downloading service account keys; they're long-lived secrets. GCP services run *as* the account directly, and on day 4 GitHub will impersonate it without a key.

**🧩 Challenge (5 min):** Why does BigQuery need two roles? Run `gcloud iam roles describe roles/bigquery.jobUser` and `gcloud iam roles describe roles/bigquery.dataEditor` and compare the permissions. Then write, in a comment or notes file, the AWS IAM policy you'd write for the same access. *(Answer at the end.)*

---

## Part 4: Repo and the data generator (20 min)

### 4a. Create the repo

Create an empty GitHub repo called `eb-rampup`, clone it, and build this layout:

```
eb-rampup/
├── pyproject.toml
├── .gitignore
├── src/
│   └── eb_rampup/
│       ├── __init__.py      # empty file
│       └── data.py
└── tests/
    └── test_data.py
```

The `src/` layout forces you to install your package before tests can import it, which catches packaging bugs early. It's the standard for production Python.

**`pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "eb-rampup"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy",
  "pandas",
  "pyarrow",
  "google-cloud-bigquery",
  "db-dtypes",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: talks to real GCP resources (slow, costs money)"]
addopts = "-m 'not integration'"
```

That last line means a plain `pytest` skips integration tests. You run them on purpose with `pytest -m integration`.

**`.gitignore`**

```
.venv/
__pycache__/
*.egg-info/
data/
.env
```

Install in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `-e` means "editable": code changes take effect without reinstalling.

### 4b. The generator

Copy this into `src/eb_rampup/data.py`. The domain logic isn't today's learning goal, so it's provided, but read the docstrings and `expected_loss_ratio`: the factors there are what your model will learn on day 2.

```python
"""Synthetic employer-group data for a group long-term disability (LTD) book."""

import numpy as np
import pandas as pd

INDUSTRY_RISK = {
    "Professional Services": 0.80,
    "Finance": 0.85,
    "Technology": 0.75,
    "Education": 0.95,
    "Healthcare": 1.15,
    "Retail": 1.10,
    "Manufacturing": 1.25,
    "Construction": 1.45,
    "Transportation": 1.35,
}
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
ELIMINATION_DAYS = [90, 180]
BENEFIT_PCTS = [0.50, 0.60, 0.667]
YEARS = [1, 2, 3]

BASE_LOSS_RATIO = 0.70
PREMIUM_RATE = 0.006  # annual premium per $1 of covered payroll


def expected_loss_ratio(df: pd.DataFrame) -> pd.Series:
    """The 'true' loss ratio each group would have with no randomness."""
    industry = df["industry"].map(INDUSTRY_RISK)
    age = np.exp(0.035 * (df["avg_age"] - 42))  # older groups claim more
    female = 1 + 0.4 * (df["pct_female"] - 0.5)
    elimination = np.where(df["elimination_days"] == 180, 0.85, 1.0)
    benefit = df["benefit_pct"] / 0.60
    return BASE_LOSS_RATIO * industry * age * female * elimination * benefit


def generate_groups(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate n synthetic employer groups with 3 years of history and a target.

    Small groups get much noisier yearly results than large ones, the same
    reason real underwriters trust a 5,000-life group's history more than a
    20-life group's.
    """
    rng = np.random.default_rng(seed)
    industries = list(INDUSTRY_RISK)

    df = pd.DataFrame(
        {
            "group_id": [f"G{i:06d}" for i in range(n)],
            "industry": rng.choice(industries, size=n),
            "region": rng.choice(REGIONS, size=n),
            "group_size": np.clip(rng.lognormal(4.0, 1.2, size=n), 2, 20_000).astype(int),
            "avg_age": np.clip(rng.normal(42, 6, size=n), 22, 64).round(1),
            "pct_female": rng.beta(5, 5, size=n).round(3),
            "avg_salary": np.clip(rng.lognormal(np.log(65_000), 0.35, size=n), 25_000, 400_000).round(-2),
            "elimination_days": rng.choice(ELIMINATION_DAYS, size=n, p=[0.8, 0.2]),
            "benefit_pct": rng.choice(BENEFIT_PCTS, size=n, p=[0.3, 0.5, 0.2]),
        }
    )

    true_lr = expected_loss_ratio(df)
    premium = (df["group_size"] * df["avg_salary"] * PREMIUM_RATE).round(2)
    noise_sd = 0.10 + 1.5 / np.sqrt(df["group_size"])

    for year in YEARS:
        yearly_lr = true_lr * rng.lognormal(-(noise_sd**2) / 2, noise_sd)
        df[f"premium_y{year}"] = premium
        df[f"claims_y{year}"] = (premium * yearly_lr).round(2)

    df["loss_ratio_next"] = (true_lr * rng.lognormal(-(noise_sd**2) / 2, noise_sd)).round(4)
    return df


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate synthetic employer groups.")
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/groups.parquet")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_groups(args.n, args.seed).to_parquet(out, index=False)
    print(f"Wrote {args.n:,} groups to {out}")
```

Try it:

```bash
python -m eb_rampup.data
python -c "import pandas as pd; print(pd.read_parquet('data/groups.parquet').head().T)"
```

✅ **Check:** you see 16 columns, including `claims_y1` through `claims_y3` and the target, `loss_ratio_next`.

---

## Part 5: Testing, your first test suite (45 min)

This is the most important part of today. Take your time.

### 5a. How pytest works (5 min)

- pytest finds files named `test_*.py` and runs every function named `test_*` in them.
- A test passes if it finishes without an exception. `assert` raises one when its condition is false, and pytest shows you exactly what the values were.
- A **fixture** is a function that builds something tests need. A test asks for it by naming it as a parameter.

### 5b. Your first two tests (10 min)

Put this in `tests/test_data.py`:

```python
import numpy as np
import pandas as pd
import pytest

from eb_rampup.data import INDUSTRY_RISK, generate_groups


@pytest.fixture(scope="module")
def groups():
    """One small dataset, built once and shared by every test in this file."""
    return generate_groups(n=2_000, seed=7)


def test_returns_requested_number_of_rows(groups):
    assert len(groups) == 2_000


def test_has_expected_columns(groups):
    expected = {
        "group_id", "industry", "region", "group_size", "avg_age", "pct_female",
        "avg_salary", "elimination_days", "benefit_pct", "loss_ratio_next",
    }
    assert expected.issubset(groups.columns)
```

Run it:

```bash
pytest -v
```

You should see `2 passed`. Now **break it on purpose**: change `2_000` to `2_001` in the first test and run again. Read the failure message: pytest shows both values. Change it back. Seeing a test fail correctly is how you know it actually tests something.

Notes on choices made here:

- `n=2_000`, not 50,000: unit tests should be fast. Test with the smallest data that proves the point.
- `seed=7`: fixed seeds make tests deterministic. A test that fails randomly ("flaky") is worse than no test, because people learn to ignore it.
- `scope="module"`: build the data once for the whole file instead of once per test.

### 5c. Challenges: write these yourself (25 min)

Write one test for each. Run `pytest -v` after each one. Hints are given; full solutions are at the end, but try first.

1. **No missing values.** *Hint:* `groups.isna()` returns True/False for every cell.
2. **Group IDs are unique.** *Hint:* pandas Series have an `is_unique` attribute.
3. **Values are in sensible ranges:** `group_size` 2 to 20,000, `avg_age` 22 to 64, `pct_female` 0 to 1, `avg_salary` 25,000 to 400,000. Write this as **one test function that runs four times**. *Hint:* look up `@pytest.mark.parametrize`, and use `Series.between(low, high)`.
4. **Every industry is a known one.** *Hint:* compare sets with `INDUSTRY_RISK`.
5. **Same seed gives identical data; a different seed gives different data.** Two tests. *Hint:* `pd.testing.assert_frame_equal` gives much better failure messages than `==`.
6. **Riskier industries have higher loss ratios on average** (Construction beats Technology). This tests behavior, not just shape. Use 20,000 rows so the averages are stable.
7. **Small groups are noisier than large ones.** This is the hardest one. A first attempt might compare the spread of loss ratios for groups under 25 lives with groups over 500 lives. Try it, and if it doesn't work, figure out why before checking the solution. *Hint:* groups differ in *true* risk too. How could you measure only the year-to-year randomness of each group?

### 5d. Test-driven development, in miniature (5 min)

So far you've tested code that already worked. Now do it the other way: write a test for behavior that doesn't exist yet, watch it fail, then make it pass.

The generator should reject `n=0` or a negative `n` with a clear `ValueError`. Write this test first:

```python
@pytest.mark.parametrize("bad_n", [0, -5])
def test_rejects_non_positive_n(bad_n):
    with pytest.raises(ValueError):
        generate_groups(bad_n)
```

Run it. One case fails ("DID NOT RAISE") and one passes. Why does `-5` already pass? *(Answer at the end; it's an important lesson about accidental behavior.)* Now add two lines at the top of `generate_groups` to make both pass on purpose.

✅ **Check:** `pytest` shows about 15 passing tests in under a second.

---

## Part 6: Cloud Storage → BigQuery, and SQL (25 min)

### 6a. Load the data

```bash
export BUCKET="${PROJECT_ID}-data"

# Bucket (like an S3 bucket; names are global)
gcloud storage buckets create gs://$BUCKET --location=$REGION --uniform-bucket-level-access

# Upload
gcloud storage cp data/groups.parquet gs://$BUCKET/raw/groups.parquet

# BigQuery dataset (like a schema/database) in the same region
bq --location=$REGION mk --dataset ${PROJECT_ID}:eb_rampup

# Load: Parquet carries its own schema, so no schema file needed
bq load --source_format=PARQUET eb_rampup.groups gs://$BUCKET/raw/groups.parquet

bq show --schema --format=prettyjson eb_rampup.groups | head -30
```

Keep the bucket and dataset in the same region. BigQuery won't load across regions, and mismatches are a common first-day error.

### 6b. Query in the console

Open **BigQuery** in the console, find `eb_rampup.groups`, and click **Preview** (free) before querying. In BigQuery you pay for the bytes scanned, so `SELECT *` on a big table is an expensive habit. Your table is tiny, but build the habit of selecting only the columns you need. The console shows "This query will process X MB" before you run it.

**🧩 SQL challenges (solutions at the end):**

1. **Loss ratio by industry** across all 3 years. Compute it as total claims ÷ total premium, not the average of each group's ratio. Why does that matter? Try both and compare.
2. **Loss ratio by age band** (20s, 30s, 40s, 50s, 60s). *Hint:* `FLOOR(avg_age / 10) * 10`.
3. **Year-over-year volatility by group size band** (under 25, 25–99, 100–499, 500+). Measure how much each group's claims swing from year 1 to year 2, then average that by band; it's the SQL version of testing challenge #7. This is credibility in one query: it shows why underwriters can't take a small group's history at face value. *Hint:* `ABS(LN(claims_y1 / claims_y2))`.
4. **The 10 large groups (500+ lives) with the worst 3-year loss ratio.** These are the groups an underwriter would review first at renewal.

---

## Part 7: Query from Python, plus your first integration test (15 min)

### 7a. Query with ADC

In a Python shell or notebook (inside your venv):

```python
import os
from google.cloud import bigquery

client = bigquery.Client(project=os.environ["PROJECT_ID"])
sql = """
SELECT industry, ROUND(SUM(claims_y1 + claims_y2 + claims_y3)
                    / SUM(premium_y1 + premium_y2 + premium_y3), 3) AS loss_ratio
FROM eb_rampup.groups
GROUP BY industry
ORDER BY loss_ratio DESC
"""
print(client.query(sql).to_dataframe())
```

No keys or passwords in the code: the library found your ADC from Part 2 automatically. In production, the same code picks up the service account it runs as.

### 7b. Your first integration test

Create `tests/integration/test_bigquery.py`:

```python
import os

import pytest

bigquery = pytest.importorskip("google.cloud.bigquery")

pytestmark = pytest.mark.integration  # applies to every test in this file


@pytest.fixture(scope="module")
def client():
    project = os.environ.get("PROJECT_ID")
    if not project:
        pytest.skip("PROJECT_ID not set")
    return bigquery.Client(project=project)


def test_groups_table_has_all_rows(client):
    rows = client.query("SELECT COUNT(*) AS n FROM eb_rampup.groups").result()
    assert next(iter(rows)).n == 50_000


def test_groups_table_has_no_null_targets(client):
    sql = "SELECT COUNTIF(loss_ratio_next IS NULL) AS nulls FROM eb_rampup.groups"
    assert next(iter(client.query(sql).result())).nulls == 0
```

Then compare:

```bash
pytest           # unit tests only; integration tests are deselected
pytest -m integration -v   # only the BigQuery tests, against the real table
```

This split is the pyramid in practice: unit tests run constantly and everywhere, while integration tests run on purpose because they need credentials, take longer, and cost money.

---

## Part 8: Commit, self-check, and report back (10 min)

```bash
ruff check .           # a linter: catches unused imports and similar
git add .
git commit -m "Day 1: data generator, unit tests, BigQuery integration tests"
git push
```

Check that `data/` and `.venv/` didn't get committed.

**Self-check.** Answer these without looking. Answers are at the end.

1. Your Python script fails with a credentials error, but `gcloud` commands work fine. What's the most likely cause?
2. What's the GCP equivalent of an EC2 instance profile role?
3. Why do the unit tests use 2,000 rows and a fixed seed?
4. Why is "total claims ÷ total premium" usually the right loss ratio, rather than the average of each group's ratio?
5. Which of today's tests would you run in CI on every push, and which wouldn't you? Why?

**Report back.** When you come back for day 2, tell me:

- Anything that broke or confused you, especially in setup and IAM
- How the testing challenges went, particularly #7
- How long each part actually took

I'll adjust day 2 based on that.

---

## Answers and solutions

### Part 3 challenge: two BigQuery roles

`bigquery.dataEditor` grants access to *data* (read and write tables). `bigquery.jobUser` grants permission to *run jobs*, and every query and load is a job billed to a project. They're separate so one project can pay for queries over data owned by another. A rough AWS equivalent would be one policy statement allowing `athena:StartQueryExecution` and related actions, plus another allowing `glue:GetTable` and `s3:GetObject`/`s3:PutObject` on the data location.

### Part 5c: test solutions

```python
def test_no_missing_values(groups):
    assert groups.isna().sum().sum() == 0


def test_group_ids_are_unique(groups):
    assert groups["group_id"].is_unique


@pytest.mark.parametrize(
    "column, low, high",
    [
        ("group_size", 2, 20_000),
        ("avg_age", 22, 64),
        ("pct_female", 0, 1),
        ("avg_salary", 25_000, 400_000),
    ],
)
def test_values_in_range(groups, column, low, high):
    assert groups[column].between(low, high).all()


def test_industries_are_known(groups):
    assert set(groups["industry"]).issubset(INDUSTRY_RISK)


def test_same_seed_gives_same_data():
    pd.testing.assert_frame_equal(generate_groups(100, seed=1), generate_groups(100, seed=1))


def test_different_seed_gives_different_data():
    assert not generate_groups(100, seed=1).equals(generate_groups(100, seed=2))


def test_riskier_industries_have_higher_loss_ratios():
    df = generate_groups(n=20_000, seed=4)
    by_industry = df.groupby("industry")["loss_ratio_next"].mean()
    assert by_industry["Construction"] > by_industry["Technology"]


def test_small_groups_are_noisier_than_large_groups():
    df = generate_groups(n=20_000, seed=3)
    # How much each group's claims swing from year 1 to year 2
    swing = np.log(df["claims_y1"] / df["claims_y2"]).abs()
    small_swing = swing[df["group_size"] < 25].median()
    large_swing = swing[df["group_size"] > 500].median()
    assert small_swing > 2 * large_swing
```

**Why the obvious version of #7 fails:** comparing the spread of loss ratios across groups mixes two things: real differences in risk between groups (a construction firm vs. a tech firm) and year-to-year randomness. The first is large enough to swamp the second. Comparing each group *with itself* across two years cancels out its true risk, leaving only the randomness. That's the same reasoning behind experience rating, and a good example of a test that teaches you something about your data.

### Part 5d: why `-5` already passed

NumPy happens to raise a `ValueError` for a negative size, so the test passed by accident. But `n=0` silently returned an empty DataFrame. Relying on a library's accidental behavior is fragile: an update could change it. The fix makes the behavior intentional, with a clear message:

```python
def generate_groups(n: int, seed: int = 42) -> pd.DataFrame:
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    ...
```

### Part 6b: SQL solutions

```sql
-- 1. Loss ratio by industry (premium-weighted)
SELECT
  industry,
  ROUND(SUM(claims_y1 + claims_y2 + claims_y3)
      / SUM(premium_y1 + premium_y2 + premium_y3), 3) AS loss_ratio,
  ROUND(AVG((claims_y1 + claims_y2 + claims_y3)
          / (premium_y1 + premium_y2 + premium_y3)), 3) AS avg_of_ratios
FROM eb_rampup.groups
GROUP BY industry
ORDER BY loss_ratio DESC;

-- 2. Loss ratio by age band
SELECT
  CAST(FLOOR(avg_age / 10) * 10 AS INT64) AS age_band,
  COUNT(*) AS n_groups,
  ROUND(SUM(claims_y1 + claims_y2 + claims_y3)
      / SUM(premium_y1 + premium_y2 + premium_y3), 3) AS loss_ratio
FROM eb_rampup.groups
GROUP BY age_band
ORDER BY age_band;

-- 3. Volatility by group size band
SELECT
  CASE
    WHEN group_size < 25 THEN '1: <25'
    WHEN group_size < 100 THEN '2: 25-99'
    WHEN group_size < 500 THEN '3: 100-499'
    ELSE '4: 500+'
  END AS size_band,
  COUNT(*) AS n_groups,
  ROUND(AVG(ABS(LN(claims_y1 / claims_y2))), 3) AS avg_yoy_swing
FROM eb_rampup.groups
GROUP BY size_band
ORDER BY size_band;

-- 4. Worst large groups
SELECT
  group_id, industry, group_size,
  ROUND((claims_y1 + claims_y2 + claims_y3)
      / (premium_y1 + premium_y2 + premium_y3), 3) AS loss_ratio_3yr
FROM eb_rampup.groups
WHERE group_size >= 500
ORDER BY loss_ratio_3yr DESC
LIMIT 10;
```

**Why weighting matters (#1):** the average of ratios counts a 5-life group the same as a 5,000-life group, so small, noisy groups distort it, usually upward. Premium-weighted is what the business actually earns or loses.

### Self-check answers

1. You ran `gcloud auth login` but not `gcloud auth application-default login`. The CLI and the Python libraries use different credentials.
2. A service account attached to the resource (for example, a VM or Cloud Run service running *as* that service account).
3. Speed and determinism. Unit tests should run in milliseconds, and fixed seeds keep them from failing randomly.
4. It weights each group by its size (premium), so it reflects the book's real financial result and isn't distorted by small, volatile groups.
5. Unit tests on every push: fast, free, no credentials. Integration tests run separately (manually, nightly, or before deploys), since they need GCP credentials, are slower, and cost money. You'll set this up in CI on day 3.
