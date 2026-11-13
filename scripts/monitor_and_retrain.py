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