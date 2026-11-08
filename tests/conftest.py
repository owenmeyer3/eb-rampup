"""Fixtures here are available to every test file without importing."""

import pandas as pd
import pytest

RANGES = {
    "group_size": (2, 20_000),
    "avg_age": (18, 70),
    "pct_female": (0, 1),
    "avg_salary": (10_000, 1_000_000),
    "loss_ratio_next": (0, 50),
}

@pytest.fixture(scope="function")
def tiny_groups() -> pd.DataFrame:
    """Three hand-made groups with numbers easy to check by hand."""
    # return pd.DataFrame(
    #     {
    #         "group_id": ["A", "B", "C"],
    #         "industry": ["Construction", "Technology", "Retail"],
    #         "region": ["Midwest", "West", "Midwest"],
    #         "group_size": [100, 1_000, 10],
    #         "avg_age": [45.0, 35.0, 50.0],
    #         "pct_female": [0.2, 0.5, 0.8],
    #         "avg_salary": [50_000.0, 120_000.0, 30_000.0],
    #         "elimination_days": [90, 180, 90],
    #         "benefit_pct": [0.6, 0.667, 0.5],
    #         "premium_y1": [100.0, 1_000.0, 10.0],
    #         "premium_y2": [100.0, 1_000.0, 10.0],
    #         "premium_y3": [100.0, 1_000.0, 10.0],
    #         "claims_y1": [50.0, 500.0, 0.0],
    #         "claims_y2": [70.0, 600.0, 0.0],
    #         "claims_y3": [90.0, 700.0, 300.0],
    #         "loss_ratio_next": [0.7, 0.6, 1.2],
    #     }
    # )
    return pd.DataFrame(
        {
            "group_id": ["A", "B", "C"],
            "industry": ["Construction", "Technology", "Retail"],
            "region": ["Midwest", "West", "Midwest"],
            "group_size": [100, 1_000, 10],
            "avg_age": [45.0, 35.0, 50.0],
            "pct_female": [0.2, 0.5, 0.8],
            "avg_salary": [50_000.0, 120_000.0, 30_000.0],
            "elimination_days": [90, 180, 90],
            "benefit_pct": [0.6, 0.667, 0.5],
            "premium_y1": [100.0, 1_000.0, 10.0],
            "premium_y2": [100.0, 1_000.0, 10.0],
            "premium_y3": [100.0, 1_000.0, 10.0],
            "claims_y1": [50.0, 500.0, 0.0],
            "claims_y2": [70.0, 600.0, 0.0],
            "claims_y3": [90.0, 700.0, 300.0],
            "loss_ratio_next": [0.7, 0.6, 1.2],
        }
    )
# A LR = (50+70+90)/(100+100+100) = 210/300 = 0.7
# B LR = (500+600+700)/(1000+1000+1000) = 1800/3000 = 0.6
# C LR = (0+0+300)/(10+10+10) = 300/30 = 10, but capped at 5.0