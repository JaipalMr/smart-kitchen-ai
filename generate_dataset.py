"""Builds a cleaned 10,000-row restaurant dataset (10 items x 1000 days)."""
import numpy as np, pandas as pd
from config import ITEMS, DATA, is_holiday

rng = np.random.default_rng(42)
dates = pd.date_range("2023-10-01", periods=1000, freq="D")
doy = dates.dayofyear.values
temp = 29 + 4.5 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 1.2, len(dates))
monsoon = np.isin(dates.month, [10, 11, 12])
rain = rng.exponential(12, len(dates)) * (rng.random(len(dates)) < np.where(monsoon, 0.45, 0.12))
rain = np.clip(rain, 0, 120)
hol = np.array([is_holiday(d) for d in dates])
dow = dates.dayofweek.values
event = ((rng.random(len(dates)) < 0.03) | ((dow >= 5) & (rng.random(len(dates)) < 0.06))).astype(int)
dow_shape = np.array([0, 0, 0, 0, 0.3, 0.8, 1.3])[dow]

rows = []
for item, p in ITEMS.items():
    trend = 1 + 0.00015 * np.arange(len(dates))
    mult = 1 + p["wk"] * dow_shape
    weather = 1 + p["heat"] * (temp - 30) + p["rain"] * np.minimum(rain, 50) / 10
    demand = p["base"] * mult * weather * trend * (1 + 0.18 * hol) * (1 + 0.12 * event)
    demand = demand * rng.lognormal(0, 0.08, len(dates))
    prepared = np.round(p["base"] * mult * trend * 1.22 * rng.uniform(0.92, 1.18, len(dates)))
    sold = np.minimum(np.round(demand), prepared)
    rows.append(pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "item": item,
        "day_of_week": dates.day_name(), "is_weekend": (dow >= 5).astype(int),
        "month": dates.month, "is_holiday": hol, "is_event": event,
        "temperature_c": temp.round(1), "rainfall_mm": rain.round(1), "price": p["price"],
        "prepared": prepared.astype(int), "sold": sold.astype(int),
        "waste": (prepared - sold).astype(int), "stockout": (np.round(demand) > prepared).astype(int)}))

df = pd.concat(rows).sort_values(["date", "item"]).reset_index(drop=True)
# ---- cleaning checks ----
assert len(df) == 10000 and df.isna().sum().sum() == 0
assert not df.duplicated(["date", "item"]).any()
assert (df.waste >= 0).all() and (df.sold <= df.prepared).all()
df.to_csv(DATA, index=False)
print(df.shape, "saved ->", DATA)
print(df.describe().T[["mean", "min", "max"]].round(1))
