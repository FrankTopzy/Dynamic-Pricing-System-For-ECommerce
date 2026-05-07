import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering shared by training and inference.

    Keep this function in a dedicated module so that serialized sklearn pipelines
    can be unpickled reliably (avoids __main__ lookup issues).
    """
    df = df.copy()
    df["price_gap"] = df["current_price"] - df["competitor_price"]
    df["stock_to_demand_ratio"] = df["stock_level"] / (df["demand_last_week"] + 1)
    df["is_weekend"] = df["day_of_week"].isin(["saturday", "sunday"]).astype(int)
    return df

