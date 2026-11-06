"""Small BigQuery helpers. The client is passed in so tests can use a fake."""

import pandas as pd


def load_table(client, table: str, max_rows: int | None = None) -> pd.DataFrame:
    """Read a whole table into a DataFrame.

    Uses list_rows (the table-read API) instead of a SELECT query, so it isn't
    billed for bytes scanned.
    """
    return client.list_rows(table, max_results=max_rows).to_dataframe()
