from unittest.mock import MagicMock

import pandas as pd

from eb_rampup.bq import load_table


def test_load_table_reads_requested_table():
    fake_client = MagicMock()
    expected = pd.DataFrame({"x": [1, 2]})
    # "when list_rows(...).to_dataframe() is called, return expected"
    fake_client.list_rows.return_value.to_dataframe.return_value = expected

    result = load_table(fake_client, "eb_rampup.groups")

    # Did our code call the client the way we intended?
    fake_client.list_rows.assert_called_once_with("eb_rampup.groups", max_results=None)
    pd.testing.assert_frame_equal(result, expected)
