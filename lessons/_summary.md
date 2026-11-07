# 2-Week Ramp-Up Plan: Senior AI/ML Engineer

Oct 28, 2026

## How the plan works

Ten 2–3 hour sessions, Mon Nov 2 to Fri Nov 13, leaving the weekend before your Nov 16 start free. Every session adds to one capstone project, so by day 10 you have a working, demo-able system that mirrors the job.

**The capstone: a group-benefits underwriting assistant.**

- **Week 1 (predictive MLOps on GCP):** a model that predicts the expected loss ratio for an employer group's disability coverage, trained on synthetic data in BigQuery, built and refreshed through a Vertex AI pipeline, monitored for drift, and retrained on a trigger.
- **Week 2 (applied GenAI):** a RAG assistant over synthetic underwriting guidelines, with an evaluation set and guardrails, then an agent that combines the model score and the guidelines into a recommendation an underwriter approves or rejects.

**Emphasis, based on your answers:**

- GCP fluency comes through doing, with AWS used as the reference point (cheat sheet below).
- **Testing and CI/CD are built in from day 1, not saved for the end.** You write tests alongside every piece of code, CI runs them on every push from day 3, and continuous deployment starts on day 4. By day 9 you run one end-to-end test of the whole system.
- Model retraining gets a full session (day 5) since it's your lightest area.
- GenAI gets three sessions. Day 6 builds RAG on Bedrock, which you know, so you learn the concepts without also learning a new cloud; day 7 ports it to Vertex AI.
- Employee Benefits domain knowledge and mentoring each get a focused block on days 9–10.

| Day | Date | Focus | Testing focus | You finish with |
| --- | --- | --- | --- | --- |
| 1 | Mon Nov 2 | GCP setup, IAM, Cloud Storage, BigQuery | pytest basics; data tests | Synthetic data in BigQuery, plus your first passing tests |
| 2 | Tue Nov 3 | Vertex AI training + model registry | Unit tests, fixtures, a training smoke test | A trained, registered loss-ratio model |
| 3 | Wed Nov 4 | Vertex AI Pipelines | CI: GitHub Actions runs tests on every push | A gated pipeline and a CI badge that goes green |
| 4 | Thu Nov 5 | Serving: batch scoring + Cloud Run API | API tests, mocking, integration tests; continuous deployment | A live endpoint that redeploys itself on merge |
| 5 | Fri Nov 6 | Monitoring, drift, retraining | Testing drift math and promotion rules | A drift check that kicks off a gated retrain |
| 6 | Mon Nov 9 | RAG fundamentals (on Bedrock) | Testing code that calls an LLM | A local RAG app over underwriting guidelines |
| 7 | Tue Nov 10 | RAG on Vertex AI + evaluation | Evals as a CI quality gate | The same app on Gemini; a prompt change that hurts quality fails the build |
| 8 | Wed Nov 11 | Agents + guardrails + human-in-the-loop | Testing tools and guardrails | An agent that calls the model and the RAG tool |
| 9 | Thu Nov 12 | End-to-end test, Composer, and the EB domain | End-to-end test of the full system | One command that proves the whole system works, plus an experience-rating notebook |
| 10 | Fri Nov 13 | Runbook, mentoring, demo | Code-review checklist that includes tests | A runbook, a junior-ready task breakdown, and a recorded demo |

If a session runs long, let it spill into the next. Days 9 and 10 are the most compressible.

## AWS → GCP cheat sheet

Most of what you know transfers. The biggest mental shifts are in the bottom rows: projects instead of accounts, service accounts instead of roles, and BigQuery as the center of gravity.

| What you use on AWS | GCP equivalent | What's different |
| --- | --- | --- |
| S3 | Cloud Storage (GCS) | Nearly identical; `gs://` paths, `gcloud storage` CLI |
| SageMaker Training / Processing | Vertex AI custom training jobs | Same container-based idea; prebuilt containers for sklearn/XGBoost |
| SageMaker Model Registry | Vertex AI Model Registry | Versions and aliases instead of approval status |
| SageMaker Pipelines | Vertex AI Pipelines | Written in Kubeflow Pipelines (KFP) Python SDK; serverless, pay per run |
| SageMaker endpoints | Vertex AI endpoints, or Cloud Run | Endpoints bill while deployed; Cloud Run scales to zero |
| SageMaker Batch Transform | Vertex AI batch prediction | Reads from and writes to BigQuery directly |
| SageMaker Model Monitor | Vertex AI Model Monitoring | Similar drift/skew concepts |
| Step Functions | Workflows; Cloud Composer for Airflow | Composer (managed Airflow, like MWAA) is in the job post |
| Lambda | Cloud Run functions | Cloud Run (containers) is the more common default |
| ECS / Fargate | Cloud Run | Simpler; no cluster or task definitions |
| EKS | GKE | Autopilot mode is the low-ops option |
| Glue / Athena / Redshift | BigQuery (+ Dataflow for streaming/ETL) | BigQuery does warehousing, SQL ML, and vector search |
| Bedrock | Vertex AI (Gemini, Model Garden) | Gemini via the `google-genai` SDK on Vertex |
| Bedrock Knowledge Bases | Vertex AI RAG Engine / Vector Search, or BigQuery vector search | BigQuery is cheapest for learning |
| CloudWatch | Cloud Logging + Cloud Monitoring | Logs are queryable across services in one place |
| CodePipeline / CodeBuild | Cloud Build (or GitHub Actions) | Artifact Registry replaces ECR |
| Secrets Manager | Secret Manager | Same idea |
| Account | Project | Billing, IAM, and APIs are all scoped per project |
| IAM role assumed by a service | Service account | Services run *as* a service account; you grant roles *to* it |
| CLI profiles | `gcloud config configurations` + Application Default Credentials | `gcloud auth application-default login` powers local SDK calls |

## Setup and cost guardrails

New Google Cloud accounts get a Free Trial with $300 in credit good for 90 days, and the 90 days start the moment you sign up and can't be paused ([Google Cloud Free Program](https://docs.cloud.google.com/free)). Sign up on day 1, not before. This whole plan should cost well under the credit if you follow these rules:

- **Set a budget alert on day 1** (e.g., $25 and $50 thresholds) in Billing → Budgets & alerts.
- **One project for everything** (e.g., `eb-rampup`), so cleanup is a single project delete at the end.
- **Undeploy Vertex AI endpoints the same session.** They bill per hour while deployed, even idle. Cloud Run scales to zero, so it's the safer default for APIs.
- **Cloud Composer is the expensive one.** Create it, use it, and delete it in the same session on day 9. If you'd rather not, run Airflow locally in Docker; the DAG code is the same.
- **Use BigQuery vector search instead of a Vertex AI Vector Search endpoint** for RAG. A deployed index endpoint bills continuously.
- **Use the smallest model tiers** (a Gemini Flash model; small machine types for training).
- **Keep the code in a GitHub repo** from day 1. Day 3 adds CI to it, and it doubles as your reference once you start.

**Local setup (30 min, first part of day 1):** install the `gcloud` CLI, Python 3.11+, and Docker; `pip install google-cloud-aiplatform google-cloud-bigquery kfp google-genai`; and keep using your existing AWS CLI profile for day 6.

## Week 1: GCP foundations and predictive MLOps

By Friday you have the full lifecycle the job's existing model portfolio runs on: data, training, pipeline, serving, monitoring, and retraining.

### Day 1 (Mon Nov 2): GCP footing and the data

- [ ] Sign up, create the `eb-rampup` project, set budget alerts, install tools
- [ ] Create a GitHub repo with a simple layout (`src/`, `tests/`, `pyproject.toml`) and install pytest
- [ ] Create a service account for your pipelines; grant it only BigQuery, Storage, and Vertex AI roles. Compare against how you'd write the equivalent IAM role in AWS.
- [ ] Write a Python function that generates ~50,000 synthetic employer groups: industry, group size, average age, % female, average salary, benefit design (elimination period, benefit %), region, and 3 years of claim history. Make loss ratio depend on several of these, plus noise.
- [ ] **Testing:** learn the test pyramid (many fast unit tests, fewer integration tests, very few end-to-end tests). Write your first tests for the generator: right columns and types, no nulls, values in sensible ranges, and the same seed gives the same data.
- [ ] Upload to Cloud Storage, load into BigQuery, and write 3–4 SQL queries (loss ratio by industry, by age band) in the console

*Done when:* you can query your data from both the console and a Python notebook using Application Default Credentials, and your tests pass.

### Day 2 (Tue Nov 3): Training and the model registry

- [ ] Split the code into small, testable functions: feature engineering, train, evaluate
- [ ] **Testing:** unit-test the feature functions with tiny hand-made DataFrames (pytest fixtures), and write a smoke test that trains on 100 rows in seconds without crashing
- [ ] Train an XGBoost regressor locally first, on data pulled from BigQuery
- [ ] Package the training code and run it as a Vertex AI custom training job with a prebuilt container (the SageMaker estimator equivalent)
- [ ] Register the model in Vertex AI Model Registry with an alias like `champion`
- [ ] Bonus (15 min): train a quick baseline with BigQuery ML (`CREATE MODEL`) to see how far SQL alone gets you

*Done when:* a model version appears in the registry with its metrics logged.

### Day 3 (Wed Nov 4): Vertex AI Pipelines

- [ ] Learn the KFP basics: components, inputs/outputs, artifacts (map each to SageMaker Pipelines steps)
- [ ] Build a pipeline: feature query → data validation (row counts, nulls, ranges) → train → evaluate → register only if it beats the current `champion`
- [ ] Run it, then open the run graph and lineage view in the console
- [ ] **CI:** add a GitHub Actions workflow that installs your package, runs linting (ruff), and runs all tests on every push and pull request. Break a test on purpose and watch the build fail.
- [ ] **Testing:** note that the pipeline's validation step and "beats the champion" gate are tests too, just ones that run on data and models instead of code

*Done when:* a second pipeline run either promotes or rejects a model based on its metrics, and CI is green on GitHub.

### Day 4 (Thu Nov 5): Serving, batch and real-time

- [ ] Run a Vertex AI batch prediction job from BigQuery to BigQuery (this mirrors the scoring pipelines in the job post)
- [ ] Add a post-processing step: apply a business rule (e.g., cap or flag groups under 10 lives) in SQL
- [ ] Wrap the model in a small FastAPI app, containerize it, push to Artifact Registry, and deploy to Cloud Run
- [ ] **Testing:** write API unit tests with FastAPI's test client and a fake model (mocking), then an integration test that starts the real container locally and calls it
- [ ] **CD:** extend the workflow so a merge to `main` builds the container, pushes it, and deploys to Cloud Run, using Workload Identity Federation instead of a stored key. Add a post-deploy smoke test that calls the live URL.
- [ ] Optional: deploy to a Vertex AI endpoint once to see the difference, then **undeploy it**

*Done when:* scores land in a BigQuery table, and merging a small change to `main` redeploys the API automatically and the smoke test passes.

### Day 5 (Fri Nov 6): Monitoring, drift, and retraining

This is the session to take slowly, since retraining is your lightest area.

- [ ] Generate a "next year" dataset with deliberate drift (older workforce, a new industry mix, higher claim rates)
- [ ] Compute drift yourself first (population stability index per feature, plus prediction error on the new data) so the math isn't a black box
- [ ] **Testing:** test your drift function against cases with known answers (identical data gives PSI ≈ 0; a big shift crosses the threshold)
- [ ] Then set up Vertex AI Model Monitoring on the same data and compare its results to yours
- [ ] Write the retraining policy as code: what triggers it (drift threshold, performance drop, or schedule), what data window it uses, and what must pass before promotion
- [ ] **Testing:** unit-test the promotion rules, since a bug there ships a worse model
- [ ] Wire it up: drift check → trigger the day 3 pipeline → champion/challenger comparison → promote, or hold for human review

*Done when:* drifted data automatically produces a challenger model and a clear promote or hold decision, with the evidence saved. That evidence is what the job post calls governance artifacts.

## Week 2: Applied GenAI, the EB domain, and leading

By Friday you have built the generative and agentic patterns the team is expanding into, plus enough domain grounding and leadership groundwork to contribute in week one.

### Day 6 (Mon Nov 9): RAG fundamentals, on AWS

The point today is the concepts, so stay on Bedrock, which you know.

- [ ] Ask Claude (or write yourself) 15–20 synthetic underwriting guideline docs: eligibility rules, minimum participation, industry exclusions, large-claim referral thresholds, rate guarantee rules
- [ ] Build RAG from scratch in plain Python, without a framework: chunk → embed (Bedrock Titan embeddings) → store in a local vector store or a NumPy array → retrieve top-k → prompt an LLM with citations
- [ ] Experiment: chunk size, top-k, and "answer only from context, otherwise say you don't know"
- [ ] **Testing:** LLM output changes run to run, so split what's testable. Unit-test chunking and retrieval deterministically, and mock the LLM call so tests run fast, free, and offline. Save real LLM checks for day 7's evals.

*Done when:* the app answers 10 guideline questions and cites which doc each answer came from.

### Day 7 (Tue Nov 10): RAG on Vertex AI, and evaluation

- [ ] Port the app: Vertex AI text embeddings, embeddings stored in BigQuery with `VECTOR_SEARCH`, and a Gemini Flash model for generation. Notice how little changes, and that your day 6 unit tests still pass.
- [ ] Build an eval set: 25 questions with expected answers and source docs, including 5 that have no answer in the docs
- [ ] Score it automatically: retrieval hit rate, answer correctness (LLM-as-judge), and refusal on unanswerable questions
- [ ] Change one thing (chunking or prompt), rerun, and compare. This loop is the core of the job post's "prompt evaluation."
- [ ] **CI quality gate:** run the eval set in CI on pull requests that touch prompts or retrieval, and fail the build if scores drop below a threshold you choose

*Done when:* you have a scored eval report you can rerun with one command.

### Day 8 (Wed Nov 11): Agents, guardrails, and human-in-the-loop

- [ ] Build an agent with two tools: `score_group` (calls your Cloud Run model API) and `search_guidelines` (your RAG). Use Google's Agent Development Kit or LangGraph; either teaches the pattern.
- [ ] Task: "Assess this employer group for renewal." The agent scores it, checks the guidelines, and drafts a recommendation with reasons.
- [ ] Add guardrails: validate tool inputs, block made-up numbers (every figure must come from a tool result), and redact PII from inputs
- [ ] Add a human-in-the-loop step: the agent proposes and an underwriter approves or edits. Log both to BigQuery.
- [ ] **Testing:** unit-test each tool and each guardrail on its own (feed the PII redactor sample text; feed the number checker a response with an invented figure). Add 5 agent cases to your eval set that check it called the right tools.

*Done when:* the agent produces a recommendation, a human decision is recorded, and the full trace is logged.

### Day 9 (Thu Nov 12): End-to-end test, Composer, and the EB domain

**Engineering (about 1.5 hours):**

- [ ] **End-to-end test:** write one test that runs the whole system in your GCP project: load a small fresh batch of data → run the pipeline → score through the deployed API → ask the agent about one group → check each step's output. Run it from a manual GitHub Actions workflow, since it's slow and costs a little.
- [ ] Look at your test suite as a pyramid: count unit, integration, and end-to-end tests, and check coverage with pytest-cov to spot untested code
- [ ] Create a small Cloud Composer environment (or local Airflow), write a DAG that runs the monthly batch scoring and the drift check, run it once, then **delete the environment**

**Domain (about 1 hour):** Employee Benefits differs from auto/home in that the risk is an employer group, not a person or a property. Learn:

- Products: group life, short- and long-term disability, absence/leave management, dental, vision, and voluntary/supplemental (accident, critical illness, hospital indemnity)
- How groups are priced: census data, manual rates, experience rating, and credibility (how much a group's own claims count, based on its size)
- The lifecycle in the job post: sales/RFP via brokers → underwriting and quote → policy installation → renewal (often after a multi-year rate guarantee)
- Disability metrics: incidence, duration, and recovery/return-to-work

- [ ] Hands-on: in a notebook, blend manual and experience rates for your synthetic groups using a credibility weight. This connects your model to how underwriters will actually use it.

### Day 10 (Fri Nov 13): Operating it and leading others

- [ ] Write a one-page runbook for your scoring pipeline: what it does, how to tell it's healthy, the 3 most likely failures and how to fix each, and who to escalate to
- [ ] Write a model card: inputs, outputs, training data window, metrics, known limits, and monitoring thresholds
- [ ] Mentoring practice: break a feature (e.g., "add a new input to the model") into 4–6 tickets a junior engineer could pick up, each with acceptance criteria that include the tests to write
- [ ] Write a short code-review checklist, with a testing section (are new paths tested, are mocks hiding real failures, does CI pass)
- [ ] Record a 5-minute walkthrough of the whole system, as if explaining it to a new teammate
- [ ] Delete the GCP project (or at least undeploy everything) once you've saved what you want

*Done when:* someone new could run and fix your system from your docs alone. That's the bar your juniors will need from you.

## Questions to bring to your first week

The plan guesses at the team's stack. These questions will tell you which parts to lean on.

- [ ] Which workloads run on AWS versus GCP, and why the split?
- [ ] What orchestrates the existing model refreshes: Composer, Step Functions, Vertex AI Pipelines, or something else?
- [ ] How are models approved for production today, and who signs off (model risk, actuarial, business owners)?
- [ ] Which models in the portfolio are due for a refresh soonest, and which ones break most often?
- [ ] Which LLMs and agent frameworks has the platform enablement team approved?
- [ ] What does the first generative AI use case in underwriting or service look like, and how will it be evaluated?
- [ ] Who are the junior engineers I'll be guiding, and what are they working on now?
- [ ] How does the team test and deploy today, and what does CI/CD look like for models versus application code?
- [ ] What would a great first 30, 60, and 90 days look like for this role?
