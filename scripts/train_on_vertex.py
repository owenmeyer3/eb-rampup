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
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def build_package() -> Path:
    for old in Path("dist").glob("*.tar.gz"):
        old.unlink()
    subprocess.run([sys.executable, "-m", "build", "--sdist"], check=True)
    return next(Path("dist").glob("*.tar.gz"))


def upload(bucket: str, local: Path, dest: str, project:str) -> str:
    storage.Client(project=project).bucket(bucket).blob(dest).upload_from_filename(str(local))
    return f"gs://{bucket}/{dest}"


def read_json(uri: str, project:str) -> dict:
    bucket, _, path = uri.removeprefix("gs://").partition("/")
    return json.loads(storage.Client(project=project).bucket(bucket).blob(path).download_as_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--bucket", required=True)
    p.add_argument("--table", default="eb_rampup.groups")
    p.add_argument("--service-account", required=True)
    p.add_argument("--alias", help="alias for a new version, e.g. challenger")
    args = p.parse_args()

    commit_sha = git_sha()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    base_dir = f"gs://{args.bucket}/training/{run_id}"
    aiplatform.init(project=args.project, location=args.region,
                    staging_bucket=f"gs://{args.bucket}/staging")

    # 1. Package the code and put it where Vertex can install it
    package = build_package()
    package_uri = upload(args.bucket, package, f"training/{run_id}/{package.name}", args.project)
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
    metrics = read_json(f"{model_dir}/metrics.json", args.project)
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
        labels={"git_sha": commit_sha, "run_id": run_id},
    )
    print(json.dumps(metrics, indent=2))
    print(f"Registered {model.resource_name}, version {model.version_id}, aliases {aliases}")


if __name__ == "__main__":
    main()
