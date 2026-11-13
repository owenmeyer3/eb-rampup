import pytest

from eb_rampup.retraining import RetrainingPolicy, decide

POLICY = RetrainingPolicy()


def report(psi: float = 0.0, prediction_psi: float = 0.0, mae_now=0.12, mae_then: float = 0.12):
    """Test helper: a monitoring report with every feature at the same PSI."""
    return {
        "feature_psi": {"avg_age": psi, "industry": 0.0},
        "prediction_psi": prediction_psi,
        "mae_now": mae_now,
        "mae_at_training": mae_then,
    }

def test_stable_data_and_steady_error_means_no_retraining():
    decision = decide(POLICY, report(), model_age_days=30)
    assert not decision.retrain
    assert decision.reasons == ()

@pytest.mark.parametrize(
    "rep, age, retrain, auto_promote",
    [
        (report(psi=0.15), 30, False, False),            # moderate drift: warn only
        (report(mae_now=0.14), 30, True, True),          # error up 17%: retrain, auto
        (report(mae_now=0.13), 30, False, False),        # error up 8%: under the 10% limit
        (report(), 400, True, True),                     # too old: retrain, auto
        (report(psi=0.30), 30, True, False),             # severe drift: hold for review
        (report(prediction_psi=0.30), 30, True, False),  # predictions shifted: hold
        (report(psi=0.30, mae_now=0.14), 30, True, False),  # severe + worse: still hold
        (report(mae_now=None), 30, False, False),        # no outcomes yet: can't judge error
    ],
)
def test_policy(rep, age, retrain, auto_promote):
    decision = decide(POLICY, rep, model_age_days=age)
    assert decision.retrain is retrain
    assert decision.auto_promote is auto_promote


def test_moderate_drift_produces_a_warning():
    decision = decide(POLICY, report(psi=0.15), model_age_days=30)
    assert decision.warnings == ("moderate drift in ['avg_age']",)


def test_reasons_explain_the_decision():
    decision = decide(POLICY, report(psi=0.30, mae_now=0.14), model_age_days=30)
    assert decision.reasons == (
        "severe drift in ['avg_age']",
        "MAE rose from 0.1200 to 0.1400",
    )


def test_policy_thresholds_are_configurable():
    strict = RetrainingPolicy(max_mae_increase=0.05)
    assert decide(strict, report(mae_now=0.13), model_age_days=30).retrain


def test_model_age_in_days():
    from datetime import datetime, timezone

    from eb_rampup.monitor import model_age_days

    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = datetime(2026, 3, 2, tzinfo=timezone.utc)
    assert model_age_days(created, now) == 60