"""The rule for whether a newly trained model replaces the champion."""


def should_promote(
    challenger: dict, champion: dict | None, min_improvement: float = 0.0
) -> tuple[bool, str]:
    """Decide whether the challenger becomes champion.

    challenger / champion are the metrics dicts saved with each model.
    min_improvement is a fraction: 0.02 means "must be at least 2% better".
    """
    if challenger["mae"] >= challenger["baseline_book_mae"]:
        return False, "challenger is no better than charging everyone the book rate"
    if champion is None:
        return True, "no champion yet"

    target = champion["mae"] * (1 - min_improvement)
    if challenger["mae"] < target:
        return True, f"MAE {challenger['mae']:.4f} beats champion {champion['mae']:.4f}"
    return False, f"MAE {challenger['mae']:.4f} does not beat target {target:.4f}"
