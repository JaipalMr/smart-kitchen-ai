from pathlib import Path
import datetime as dt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "restaurant_sales.csv"
MODEL = ROOT / "models" / "bundle.joblib"
METRICS = ROOT / "models" / "metrics.json"
BACKTEST = ROOT / "models" / "backtest.csv"

# base = average daily units, wk = weekend lift, heat = effect per deg C above 30,
# rain = effect per 10 mm rain, price in INR
ITEMS = {
    "Biryani":      dict(base=100, wk=0.22, heat=-0.005, rain=-0.02, price=220),
    "Meals":        dict(base=70,  wk=0.05, heat=-0.002, rain=0.00,  price=140),
    "Fried Rice":   dict(base=45,  wk=0.15, heat=-0.004, rain=-0.01, price=160),
    "Chicken Curry":dict(base=35,  wk=0.18, heat=-0.003, rain=0.01,  price=190),
    "Parotta":      dict(base=60,  wk=0.12, heat=-0.002, rain=0.02,  price=40),
    "Idli":         dict(base=80,  wk=-0.05,heat=-0.003, rain=0.02,  price=50),
    "Dosa":         dict(base=65,  wk=0.08, heat=-0.002, rain=0.01,  price=70),
    "Lassi":        dict(base=40,  wk=0.10, heat=0.030,  rain=-0.03, price=60),
    "Filter Coffee":dict(base=90,  wk=0.00, heat=-0.015, rain=0.04,  price=30),
    "Veg Noodles":  dict(base=30,  wk=0.12, heat=-0.004, rain=-0.01, price=120),
}
ITEM_LIST = list(ITEMS)

FIXED_HOLIDAYS = {(1, 1), (1, 14), (1, 15), (1, 26), (8, 15), (10, 2), (12, 25)}
DIWALI = {dt.date(2023, 11, 12), dt.date(2024, 10, 31), dt.date(2025, 10, 20), dt.date(2026, 11, 8)}

def is_holiday(d) -> int:
    d = d.date() if hasattr(d, "date") else d
    return int((d.month, d.day) in FIXED_HOLIDAYS or d in DIWALI)
