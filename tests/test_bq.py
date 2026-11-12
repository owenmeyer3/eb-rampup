import pandas as pd

from eb_rampup.bq import load_table

class FakeRowIterator:
    """Stands in for BigQuery's RowIterator: only what load_table uses."""
    def to_dataframe(self):
        return pd.DataFrame({"x": [1, 2]})

class FakeClient:
    def list_rows(self, table, max_results=None):
        return FakeRowIterator()

def test_load_table_reads_requested_table():
    fake_client = FakeClient()

    result = load_table(fake_client, "eb_rampup.groups")

    # Did our code call the client the way we intended?
    pd.testing.assert_frame_equal(result, pd.DataFrame({"x": [1, 2]}))
