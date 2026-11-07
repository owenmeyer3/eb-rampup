# Day 3: Vertex AI Pipelines and CI

**Wed Nov 4 · about 2.5–3 hours**

Today turns the one-off training job from Day 2 into a repeatable workflow. You'll learn how a Vertex AI Pipeline connects validation, training, evaluation, and model registration, then add GitHub Actions so code checks run on every push and pull request.

| Part | What you do | Time |
| --- | --- | --- |
| 1 | Concepts: pipelines, CI, and promotion gates | 15 min |
| 2 | Put the project files under the Git repository | 10 min |
| 3 | Add continuous integration with GitHub Actions | 30 min |
| 4 | Design the Vertex AI Pipeline and its data contract | 20 min |
| 5 | Implement and compile the pipeline | 50 min |
| 6 | Run it, inspect lineage, and test the promotion gate | 35 min |
| 7 | Commit, self-check, and report back | 10 min |

By the end, a GitHub Actions workflow should run lint and tests on every push and pull request. A Vertex AI Pipeline should validate the BigQuery table, train and evaluate a candidate, and register it as `champion` only when it beats the current champion on the same evaluation metric.

---

## Part 1: Concepts (15 min)

### A pipeline is orchestration, not a new training algorithm

Day 2's `eb_rampup.train` module already knows how to load data, split it, train XGBoost, evaluate baselines, and save artifacts. A pipeline coordinates those operations and records their inputs, outputs, status, and relationships.

| SageMaker Pipelines | Vertex AI Pipelines / KFP |
| --- | --- |
| Pipeline definition | Python function decorated with `@dsl.pipeline` |
| Processing or training step | Component / pipeline task |
| Step input and output | Parameters and artifacts |
| Pipeline execution | `PipelineJob` |
| Step dependency | Data dependency or `.after(...)` |
| Conditional step | `dsl.Condition(...)` |

A component is a task that runs in its own container. Its inputs and outputs form a contract with other components. A pipeline is the graph of those tasks. Vertex stores metadata about the run, including artifacts and lineage, so you can trace which data and code produced a model.

The pipeline should make the promotion decision from an evaluation metric computed on a held-out test set. It must not evaluate on the training rows, and it must compare the candidate and champion using the same metric and evaluation data.

### Three different checks

1. **CI tests** check code behavior on small, controlled inputs. They run on every pull request and should be fast and deterministic.
2. **Data validation** checks each real pipeline input batch. It can stop a run when the incoming table violates the expected schema or ranges.
3. **Model evaluation and promotion** check whether a trained candidate meets a quality bar. A candidate that fails the gate should not replace the current champion.

Passing CI does not mean every future BigQuery batch is valid. Passing data validation does not mean a model is good. Keep the checks at the point where they can catch the failure they are designed for.

### CI and CD

- **Continuous integration (CI):** automatically install, lint, and test code changes.
- **Continuous delivery/deployment (CD):** publish or deploy a change after the appropriate review and gates.
- This lesson adds CI. It does **not** make every code push train a model or change production. Cloud training remains an explicit pipeline run; Day 4 covers application deployment.

---

## Part 2: Put the project files under the Git repository (10 min)

The Git repository is the `eb-rampup` directory. Run these commands from it:

```bash
cd ~/path/to/ht-prep/eb-rampup
git rev-parse --show-toplevel
```

The second command should print the `.../ht-prep/eb-rampup` path. The parent `ht-prep` directory is not the Git repository. Files outside `eb-rampup`—including a launcher at `ht-prep/scripts/`—are not included when you commit this repo.

Keep the pipeline code, its launcher, and the CI workflow inside this repo:

```text
eb-rampup/
├── .github/
│   └── workflows/
│       └── tests.yml
├── pipelines/
│   └── training_pipeline.py
├── scripts/
│   └── run_pipeline.py
├── src/
│   └── eb_rampup/
├── tests/
└── pyproject.toml
```

Move or recreate project-owned scripts inside the repo before committing them. Don't move `.git`, `.venv`, or generated artifacts. Confirm the Git status before and after any move so you don't accidentally lose work.

---

## Part 3: Add continuous integration with GitHub Actions (30 min)

### 3a. Check the existing commands locally

From the repository root, activate the environment and run the same checks CI will run:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
```

Fix lint or test failures rather than disabling checks just to get a green badge. The current pytest configuration excludes tests marked `integration` by default, so CI won't contact BigQuery or spend money. Keep the CI suite deterministic and independent of GCP credentials.

### 3b. Create the workflow

Create `.github/workflows/tests.yml`:

```yaml
name: tests

on:
  push:
  pull_request:

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install package and development tools
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev]"
      - name: Lint
        run: ruff check .
      - name: Test
        run: pytest
```

Commit and push the workflow to the GitHub repo. Open the **Actions** tab and watch the run. A local pass is not enough: this verifies the workflow file is in the repo, GitHub can install the project, and the checks work in a clean environment.

### 3c. Prove the failure path

Temporarily make an existing assertion fail, run `pytest` locally, then restore the assertion. If you want to see the GitHub check fail too, push a temporary branch with the failing test and open a pull request; do not merge it. Restore the test and confirm the check turns green.

This demonstrates that CI blocks a broken change. Don't leave an intentionally failing test in the main branch.

### 3d. Add a badge (optional)

After the workflow has run, GitHub's Actions page can provide a status badge. Add the generated Markdown near the top of the repository README if the repo has one. A badge reports the workflow's last status; it is not a substitute for reviewing what the workflow actually tests.

---

## Part 4: Design the Vertex AI Pipeline (20 min)

### 4a. Draw the graph before writing code

Use this graph as the target:

```text
BigQuery table
      │
      ▼
validate_data ── fails the run if the table violates its contract
      │
      ▼
train_and_evaluate ── reuses eb_rampup.train and writes a candidate + metrics
      │
      ▼
compare_with_champion
      │
      ├── candidate is better ──► register candidate and move champion alias
      └── candidate is not better ► record rejection; leave champion unchanged
```

Each box should be a component. The validation output (for example, validated row count) should be an input to training. That data dependency makes the ordering explicit: training cannot start until validation succeeds.

### 4b. Define the data-validation contract

Use the project table `eb_rampup.groups`. At minimum, validate:

- The table has enough rows to produce non-empty train and test splits.
- Required columns exist (the table query itself should fail clearly if they do not).
- `loss_ratio_next` and the feature columns have no nulls.
- Numeric columns stay within documented ranges, such as group size 2–20,000, age 22–64, female percentage 0–1, and salary $25,000–$400,000.
- Premiums are positive, so loss-ratio calculations have a non-zero denominator.

For each invariant, decide whether a violation should **fail the run** or be **reported as a warning**. Make the decision explicit; don't quietly replace invalid values or report success after a failed query.

Write validation against the actual BigQuery table on every pipeline run. Unit-test the validation logic separately with small fixtures or a fake query result; keep GCP integration tests opt-in.

### 4c. Give the gate an unambiguous comparison

Day 2's `evaluate` returns:

- `mae`: candidate's premium-weighted MAE on the held-out split.
- `baseline_book_mae` and `baseline_history_mae`: comparison baselines.
- row counts for train and test.

For this lesson, define "better" as **strictly lower `mae` than the champion's premium-weighted `mae` on the same test data**. Do not compare the candidate's weighted MAE with an unweighted metric or with a metric produced on different data.

The first successful run has no existing champion. Register it as the initial champion and save the evaluation metrics with its artifacts. Later runs must read the current champion's metrics, compare them, and preserve the existing champion if the candidate is not better.

---

## Part 5: Implement and compile the pipeline (50 min)

### 5a. Install the pipeline SDK

Add the authoring SDK to the development dependencies in `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
  "pytest",
  "ruff",
  "build",
  "google-cloud-aiplatform",
  "scikit-learn",
  "kfp>=2,<3",
]
```

Keep `kfp` in the development extra: it is needed to define and compile the pipeline, not by the Day 2 training module. Reinstall:

```bash
pip install -e ".[dev]"
```

### 5b. Build the components

Create `pipelines/training_pipeline.py`. Use KFP v2 (`kfp.dsl` and `kfp.compiler.Compiler`) and implement three components:

1. **`validate_data(project, table) -> row_count`**
   - Use `google-cloud-bigquery` to query counts and invalid-value counts.
   - Raise a clear exception if any blocking invariant fails. A failed component should fail the pipeline.
   - Return the validated count so the next task depends on successful validation.
2. **`train_and_evaluate(project, table, row_count, package_uri, output_uri) -> metrics_uri, model_uri`**
   - `package_uri` is the Cloud Storage URI of the built source distribution; `output_uri` is a unique GCS prefix for this run.
   - In its isolated component container, download and install the package, then invoke the existing `python -m eb_rampup.train --project ... --table ... --model-dir ...` command. Check the subprocess return code.
   - Upload `model.bst`, `metrics.json`, and `features.json` to the run's output prefix. Return the locations needed by the next step.
   - Do not reimplement feature generation or model evaluation in the pipeline file. The package's existing training code remains the source of truth.
3. **`compare_and_register(metrics_uri, model_uri, ...)`**
   - Read the candidate's `metrics.json` and find the current `loss-ratio-ltd` model version with the `champion` alias.
   - If there is a champion, read its saved metrics and compare the two weighted MAEs. If the candidate loses or ties, record the decision and leave the champion unchanged.
   - If no champion exists, register the candidate as the initial champion. If it wins, register a new version and move the `champion` alias from the prior version.
   - Print or persist the reason and metrics for either decision. A gate that silently skips registration is hard to operate.

Use `@dsl.component` for each task. Put imports **inside** a component function (for example, import `google.cloud.bigquery` inside `validate_data`) and list non-standard packages in that component's `packages_to_install`. KFP serializes component functions; it does not automatically package every module imported by the pipeline-definition file.

For multi-output components, use a `NamedTuple` return type with named fields. This gives downstream tasks readable inputs such as `training.outputs["metrics_uri"]` and `training.outputs["model_uri"]`.

### 5c. Define the pipeline

Add a pipeline function that connects the tasks. Its structure should look like this:

```python
@dsl.pipeline(name="loss-ratio-training")
def loss_ratio_pipeline(
    project: str,
    table: str,
    package_uri: str,
    output_uri: str,
):
    validation = validate_data(project=project, table=table)
    training = train_and_evaluate(
        project=project,
        table=table,
        row_count=validation.output,
        package_uri=package_uri,
        output_uri=output_uri,
    )
    compare_and_register(
        metrics_uri=training.outputs["metrics_uri"],
        model_uri=training.outputs["model_uri"],
        project=project,
    )
```

The names and types must match your component definitions. `row_count=validation.output` creates the validation-to-training dependency; the training outputs create the training-to-registration dependency. If you also use a `dsl.Condition`, make the condition operate on a declared metric output and keep the "not promoted" path visible in run metadata.

Compile the graph locally without submitting a cloud job:

```python
from kfp import compiler

compiler.Compiler().compile(
    pipeline_func=loss_ratio_pipeline,
    package_path="dist/loss_ratio_pipeline.json",
)
```

You can keep this compile step in `scripts/compile_pipeline.py` or provide a `--compile-only` option in the launcher. Compiling verifies the graph and component interfaces; it does not run BigQuery queries, train a model, or prove IAM access.

### 5d. Add a launcher inside the repo

Create `scripts/run_pipeline.py` to:

1. Build the package from the repository root (`python -m build --sdist`).
2. Upload the source distribution to a run-specific path in the chosen bucket.
3. Compile the KFP pipeline.
4. Create a `google.cloud.aiplatform.PipelineJob` with the compiled template, a GCS `pipeline_root`, and parameter values for project, table, package URI, and unique output prefix.
5. Submit it using the pipeline service account.

Keep package-building and job-submission code in this launcher. Keep the task logic in components, and keep reusable training and feature logic in `src/eb_rampup/`. Run the launcher from the repo root so `pyproject.toml` and `dist/` resolve correctly.

Do not put a service-account key in the repo or workflow. Locally, use Application Default Credentials. On Vertex, assign the pipeline service account and grant only the needed BigQuery, Storage, and Vertex AI permissions. The identity submitting the pipeline also needs permission to run as that service account.

---

## Part 6: Run it and inspect the results (35 min)

### 6a. Compile before spending cloud time

Run local checks first:

```bash
ruff check .
pytest
python scripts/run_pipeline.py --compile-only
```

Fix component interface, import, or compilation errors before submitting. A compile-only run is fast and does not create Vertex resources.

### 6b. Submit one pipeline run

Use the same project, region, dataset, and bucket as Day 2. Choose a unique run prefix so artifacts from different runs don't overwrite one another:

```bash
python scripts/run_pipeline.py \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --bucket "$BUCKET" \
  --service-account "pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --table "eb_rampup.groups"
```

The launcher should print the PipelineJob resource name or a console URL. Open the run in Vertex AI → Pipelines and inspect:

- The component graph and task statuses.
- The validation output and any surfaced data-quality counts.
- Training logs and the `metrics.json` output.
- The compare/register decision and its reason.
- The model artifact and its link to the pipeline run (lineage).

The initial run should register the first champion. Submit another run with the same data and parameters; it should not promote a model with identical or worse MAE. Then change one training parameter in a controlled way, rebuild the package, and run again. The gate—not a hard-coded alias—decides whether the candidate becomes champion.

If you don't want to pay for a second cloud training run, unit-test the comparison function with explicit cases (no champion, candidate wins, candidate loses, and tie) and inspect the first run's pipeline graph. Do not claim the cloud promotion path is verified unless you actually ran it.

### 6c. Failure handling

- **Validation fails:** inspect which invariant failed; do not bypass the check without understanding the data issue.
- **Permission denied:** confirm which identity is running the component and which resource it needs to access. The submitter and pipeline runtime can be different identities.
- **Package install fails:** verify the GCS package URI is readable by the runtime service account and that dependency versions support the component's Python version.
- **Pipeline compiles but task fails:** compilation only validates the graph. Use that task's logs for its runtime error.
- **Candidate is rejected:** that is a successful gate decision, not a failed pipeline. The run should finish successfully and explain why it preserved the champion.

Delete temporary resources only when you're certain they're no longer needed. Pipeline executions and Cloud Storage artifacts are useful for inspecting lineage, but storage and compute can incur costs.

---

## Part 7: Commit, self-check, and report back (10 min)

```bash
ruff check .
pytest
git status --short
git add .github pipelines scripts pyproject.toml
git commit -m "Day 3: add CI and Vertex training pipeline"
git push
```

Confirm the workflow is green on GitHub. Include the lesson or other project files in the commit if you want to keep them in the repo; review `git status` before committing and do not add credentials, `.venv/`, generated model artifacts, or local data.

**Self-check**

1. What does CI prove, and what does a per-run data validation component prove?
2. Why should a failed data validation stop training rather than return a success-shaped result?
3. What makes validation run before training in the pipeline graph?
4. Why must the candidate and champion metrics use the same test data and weighting?
5. What happens when a candidate ties the champion? Why?
6. What does a green GitHub Actions run not prove about the cloud pipeline?
7. Which identity runs the pipeline components, and what permissions does it need?
8. What does Vertex lineage let you trace from a registered model?

**Report back**

- Whether local CI and GitHub Actions passed
- Whether the pipeline compiled and ran, and where it failed if not
- The validation counts, candidate and champion MAEs, and promotion decision
- Anything confusing about component inputs/outputs, service accounts, or lineage
- How long each part took
