import argparse, time, subprocess, sys
from pathlib import Path
from google.cloud import aiplatform, storage

# Google's dedicated XGBoost *training* images are old (XGBoost 1.x), so we use a
# maintained Python 3.10 image that ships XGBoost 2.1.3, matching the 2.1 serving image.
TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest"
SERVE_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"
MODEL_NAME = "loss-ratio-ltd"

def build_package() -> Path:
    for old in Path("dist").glob("*.tar.gz"):
        old.unlink() # delete existing tar.gz packages in /dist
    subprocess.run([sys.executable, "-m", "build", "--sdist"], check=True) # build package in /dist
    return next(Path("dist").glob("*.tar.gz")) # return first tar.gz filepath in dist

def upload(bucket: str, local: Path, dest: str) -> str:
    storage.Client().bucket(bucket).blob(dest).upload_from_filename(str(local))
    return f"gs://{bucket}/{dest}"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--bucket", required=True)
    p.add_argument("--table", default="eb_rampup.groups")
    p.add_argument("--service-account", required=True)
    args = p.parse_args()

    # 1. Package the code and put it where Vertex can install it
    package = build_package()
    package_uri = upload(args.bucket, package, f"packages/{package.name}")
    print(f"Uploaded {package_uri}")

    pipeline_dir = f"gs://{args.bucket}/pipelines/{MODEL_NAME}-pipeline"

    aiplatform.init(project=args.project, location=args.region, staging_bucket=f"{pipeline_dir}/staging-files")

    job = aiplatform.PipelineJob(
        display_name="loss-ratio-training",
        template_path="dist/training_pipeline.json",
        pipeline_root=pipeline_dir,
        parameter_values={
            "project": args.project,
            "table": args.table,
            "package_uri": package_uri,
            "output_uri": args.output_uri,
        },
        enable_caching=False,
    )

    job.run(service_account=args.service_account)

if __name__ == "__main__":
    main()