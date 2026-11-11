"""Calls a running API. The same tests check a local container and Cloud Run.

    API_URL=http://localhost:8080 pytest -m integration tests/test_INTEGRATION_client_call.py
"""

import os

import pytest

httpx = pytest.importorskip("httpx")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def api():
    url = os.environ.get("API_URL")
    if not url:
        pytest.skip("API_URL not set")
    token = os.environ.get("API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(base_url=url, headers=headers, timeout=60) as client:
        yield client


def test_health(api):
    response = api.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score(api, valid_group):
    response = api.post("/score", json=valid_group)
    assert response.status_code == 200
    assert 0 < response.json()["predicted_loss_ratio"] < 5