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
    except  (subprocess.CalledProcessError, FileNotFoundError):
        # CalledProcessError is git running and failing; FileNotFoundError is git not being installed at all.
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
