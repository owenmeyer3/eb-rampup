import pytest

from eb_rampup.promotion import should_promote

# def should_promote(
#     challenger: dict, champion: dict | None, min_improvement: float = 0.0
# ) -> tuple[bool, str]:
#     """Decide whether the challenger becomes champion.

#     challenger / champion are the metrics dicts saved with each model.
#     min_improvement is a fraction: 0.02 means "must be at least 2% better".
#     """
#     if challenger["mae"] >= challenger["baseline_book_mae"]:
#         return False, "challenger is no better than charging everyone the book rate"
#     if champion is None:
#         return True, "no champion yet"

#     target = champion["mae"] * (1 - min_improvement)
#     if challenger["mae"] < target:
#         return True, f"MAE {challenger['mae']:.4f} beats champion {champion['mae']:.4f}"
#     return False, f"MAE {challenger['mae']:.4f} does not beat target {target:.4f}"


def test_first_model_is_promoted():
    metrics = {"mae": 0.12, "baseline_book_mae": 0.3}
    promote, reason = should_promote(metrics, champion=None)
    assert promote
    assert reason == "no champion yet"

@pytest.mark.parametrize('challenger,champion,exp_prom,exp_reason',[
    ({"mae": 0.12, "baseline_book_mae": 1}, {"mae": 0.2}, True, "beats champion"),
    ({"mae": 0.2, "baseline_book_mae": 1}, {"mae": 0.12}, False, "does not beat target"),
    ({"mae": 0.12, "baseline_book_mae": 1}, {"mae": 0.12}, False, "does not beat target"),
    ({"mae": 0.118, "baseline_book_mae": 1}, {"mae": 0.12}, False, "does not beat target"),
])
def test_champ_compares(challenger,champion,exp_prom,exp_reason):
    promote, reason = should_promote(challenger, champion=champion, min_improvement=0.02)
    assert promote == exp_prom
    assert exp_reason in reason