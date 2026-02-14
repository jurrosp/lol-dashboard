import pandas as pd


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Converteer Unix ms timestamp naar echte datetime
    df["game_datetime"] = pd.to_datetime(df["game_creation"], unit="ms")

    df = df.sort_values("game_datetime")

    # Rolling winrate
    df["winrate_roll10"] = df["win"].rolling(10, min_periods=3).mean()

    return df


def champion_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bouwt champion overzicht met games + winrate.
    """
    if df.empty:
        return df

    g = (
        df.groupby("champion")
        .agg(
            games=("win", "size"),
            winrate=("win", "mean"),
        )
        .sort_values("games", ascending=False)
    )

    return g.reset_index()
