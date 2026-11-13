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
    p.add_argument("--bucket", required=True)
    args = p.parse_args()

    aiplatform.init(project=args.project, location=args.region, staging_bucket=f"gs://{args.bucket}/staging")
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