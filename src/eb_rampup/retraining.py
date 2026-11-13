"""The rules for when to retrain, and whether a retrained model may go live unattended."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrainingPolicy:
    psi_moderate: float = 0.10       # worth a warning
    psi_severe: float = 0.25         # the population has really changed
    max_mae_increase: float = 0.10   # 10% worse than at training time
    max_model_age_days: int = 365


@dataclass(frozen=True)
class Decision:
    retrain: bool
    auto_promote: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def decide(policy: RetrainingPolicy, report: dict, model_age_days: int) -> Decision:
    psis = {**report["feature_psi"], "predictions": report["prediction_psi"]}
    severe = sorted(k for k, v in psis.items() if v >= policy.psi_severe)
    moderate = sorted(k for k, v in psis.items() if policy.psi_moderate <= v < policy.psi_severe)

    reasons, warnings = [], []
    if severe:
        reasons.append(f"severe drift in {severe}")
    if moderate:
        warnings.append(f"moderate drift in {moderate}")

    mae_now, mae_then = report["mae_now"], report["mae_at_training"]
    if mae_now is not None and mae_now > mae_then * (1 + policy.max_mae_increase):
        reasons.append(f"MAE rose from {mae_then:.4f} to {mae_now:.4f}")

    if model_age_days > policy.max_model_age_days:
        reasons.append(f"model is {model_age_days} days old")

    retrain = bool(reasons)
    # A big shift in who we insure deserves a human look before a new model goes live
    auto_promote = retrain and not severe
    return Decision(retrain, auto_promote, tuple(reasons), tuple(warnings))