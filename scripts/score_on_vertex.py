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
    p.add_argument("--bucket", required=True)
    args = p.parse_args()

    aiplatform.init(project=args.project, location=args.region,
                    staging_bucket=f"gs://{args.bucket}/staging")
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
