import numpy as np
import pytest
from fastapi.testclient import TestClient

from eb_rampup.api import app, get_model, load_from_env
from eb_rampup.serving import load_model_dir, REFER_MAX_LOSS_RATIO


class FakeModel:
    """Stands in for the real model: returns a fixed prediction."""

    version = "test-1"

    def __init__(self, prediction: float = 0.8):
        self.prediction = prediction

    def predict(self, df):
        return np.full(len(df), self.prediction)

# TestClient used for
# Real:   curl  →  network  →  uvicorn  →  ASGI  →  app
# Test:   TestClient  ─────────────────→   ASGI  →  app

@pytest.fixture
def client():
    app.dependency_overrides[get_model] = lambda: FakeModel() # tell fastAPI, when get_model is called, return FakeModel() instead
    yield TestClient(app)          # no `with`, so startup (model loading) is skipped
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test-1"}

test_groups = [
    ({}, 1, {"refer_to_underwriter":False, "model_version": "test-1"}, 200),
    ({}, 2, {"refer_to_underwriter":True, "refer_reason": f"predicted loss ratio above {REFER_MAX_LOSS_RATIO}"}, 200),
    ({"avg_age":120}, 1, {}, 422),
    ({"group_size":1}, 1, {}, 422),
    ({"pct_female":1.5}, 1, {}, 422),
    ({"premium_y1":0}, 1, {}, 422)
]
@pytest.mark.parametrize("group_changes, prediction, expected_score_constraints, expected_status", test_groups)
def test_score_vals(client, valid_group, group_changes, prediction, expected_score_constraints, expected_status):
    client.app.dependency_overrides[get_model] = lambda: FakeModel(prediction=prediction) # swap out model so we get the prediction we want for testing

    response = client.post("/score", json={**valid_group, **group_changes})

    assert response.status_code == expected_status
    score = response.json()
    for c in expected_score_constraints:
        assert expected_score_constraints[c] == score[c]

def test_score_missing_field(client, valid_group):
    valid_group.pop("avg_age")
    response = client.post("/score", json=valid_group)

    assert response.status_code == 422

def test_score_unk_industry(client, valid_group, model_dir):
    client.app.dependency_overrides[get_model] = lambda: load_model_dir(model_dir)# swap out model so we get the prediction we want for testing
    BAD_INDUSTRY = "screwing around"
    response = client.post("/score", json={**valid_group, "industry":BAD_INDUSTRY})

    assert response.status_code == 422
    assert BAD_INDUSTRY in response.json()['detail']

def test_load_from_env_uses_model_dir(monkeypatch, model_dir):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    assert load_from_env().version == "local"