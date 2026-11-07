import os

import pytest

bigquery = pytest.importorskip("google.cloud.bigquery")
pytestmark = pytest.mark.integration # applies to every test in this file

@pytest.fixture(scope="module") # load client once for the whole file instead of once per test
def client():
    project = os.environ.get("PROJECT_ID")
    region = os.environ.get("REGION")
    if not project: raise pytest.skip("PROJECT_ID not set")
    if not region: raise pytest.skip("REGION not set")
    return bigquery.Client(project=project, location=region)

@pytest.fixture(scope="module") # load client once for the whole file instead of once per test
def groups(client):
    return client.query("SELECT COUNT(*) AS CT FROM `eb_rampup.groups`").result()

test_table_data = [
    ("eb_rampup", "groups", 50_000)
]

@pytest.mark.parametrize("schema,table,size", test_table_data)
def test_table_has_all_rows(client, schema, table, size):
    rows = client.query(f"SELECT COUNT(*) AS CT FROM `{schema}.{table}`").result()
    assert next(iter(rows)).CT == size

@pytest.mark.parametrize("schema,table,size", test_table_data)
def test_table_has_no_null_targets(client, schema, table, size):
    rows = client.query(f"SELECT COUNTIF(loss_ratio_next IS NULL) AS N_CT FROM `{schema}.{table}`").result()
    assert next(iter(rows)).N_CT == 0