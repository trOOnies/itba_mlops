import pandas as pd
from dagster_airbyte import build_airbyte_assets
from dagster_fd.constants import AIRBYTE_CONNECTION_ID

airbyte_assets = build_airbyte_assets(
    connection_id=AIRBYTE_CONNECTION_ID,
    destination_tables=["movies_raw", "users_raw", "scores_raw"],
    # asset_key_prefix=["postgres_replica"]
)


def movies_raw() -> pd.DataFrame:
    return pd.DataFrame()


def users_raw() -> pd.DataFrame:
    return pd.DataFrame()


def scores_raw() -> pd.DataFrame:
    return pd.DataFrame()
