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
        current_table: str, policy: RetrainingPolicy|None = None,
        report_dir: str = "artifacts/monitoring") -> tuple[dict, Decision]:
    from google.cloud import bigquery

    from eb_rampup.bq import load_table
    from eb_rampup.registry import download_dir, find_champion
    from eb_rampup.serving import load_model_dir

    if policy is None: policy = RetrainingPolicy()

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