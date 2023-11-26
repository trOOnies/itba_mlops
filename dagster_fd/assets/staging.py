import pandas as pd
from dagster import asset, AssetIn
from dagster_fd.assets.transformed import movies, users, scores


@asset(
    ins={
        "movies": AssetIn(),
        "users": AssetIn(),
        "scores": AssetIn()
    }
)
def staged_data(
    movies: pd.DataFrame,
    users: pd.DataFrame,
    scores: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame()
