import numpy as np, pandas as pd
from config import ITEM_LIST

FEATURES = ["item_code", "dow", "is_weekend", "month", "dayofyear", "is_holiday", "is_event",
            "temperature_c", "rainfall_mm", "price", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
            "lag_1", "lag_7", "lag_14", "roll7", "roll28", "std7", "same_dow_mean4"]
GROUPS = {
    "Day of week": ["dow", "is_weekend", "dow_sin", "dow_cos"],
    "Holiday / event": ["is_holiday", "is_event"],
    "Weather": ["temperature_c", "rainfall_mm"],
    "Recent sales": ["lag_1", "lag_7", "lag_14", "roll7", "roll28", "std7", "same_dow_mean4"],
}

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["item", "date"]).copy()
    d = pd.to_datetime(df["date"])
    df["date"] = d
    df["item_code"] = df["item"].map({k: i for i, k in enumerate(ITEM_LIST)})
    df["dow"], df["month"], df["dayofyear"] = d.dt.dayofweek, d.dt.month, d.dt.dayofyear
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * df.dow / 7), np.cos(2 * np.pi * df.dow / 7)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * df.dayofyear / 365), np.cos(2 * np.pi * df.dayofyear / 365)
    g = df.groupby("item")["sold"]
    for k in (1, 7, 14, 21, 28):
        df[f"lag_{k}"] = g.shift(k)
    df["roll7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    df["std7"] = g.transform(lambda s: s.shift(1).rolling(7).std())
    df["same_dow_mean4"] = df[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1)
    return df
