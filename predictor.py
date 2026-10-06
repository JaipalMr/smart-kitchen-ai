import math, joblib, numpy as np, pandas as pd
from config import DATA, MODEL, ITEM_LIST, is_holiday
from features import add_features, FEATURES, GROUPS

class Predictor:
    def __init__(self):
        self.b = joblib.load(MODEL)
        self.hist = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["item", "date"])
        self.last = self.hist.date.max()
        m = self.hist.assign(m=self.hist.date.dt.month).groupby("m")[["temperature_c", "rainfall_mm"]].mean()
        self.clim = m.to_dict("index")

    @property
    def next_date(self):
        return (self.last + pd.Timedelta(days=1)).date()

    def _ens(self, X):
        return float(sum(w * m.predict(X[FEATURES])[0] for w, m in zip(self.b["weights"], self.b["models"])))

    def _frame(self, item, date, over):
        h = self.hist[self.hist.item == item].tail(60)[["date", "item", "sold", "temperature_c", "rainfall_mm", "is_holiday", "is_event", "price"]]
        f, cur, date = h.copy(), self.last, pd.Timestamp(date)
        if date <= cur:
            raise ValueError(f"date must be after {cur.date()}")
        while cur < date:                       # recursive forecasting for gap days
            cur += pd.Timedelta(days=1)
            final = cur == date
            c = self.clim[cur.month]
            row = dict(date=cur, item=item, sold=np.nan, price=f.price.iloc[-1],
                       temperature_c=(over.get("temperature_c") if final and over.get("temperature_c") is not None else c["temperature_c"]),
                       rainfall_mm=(over.get("rainfall_mm") if final and over.get("rainfall_mm") is not None else c["rainfall_mm"]),
                       is_holiday=(int(over["is_holiday"]) if final and over.get("is_holiday") is not None else is_holiday(cur)),
                       is_event=int(over.get("is_event", 0)) if final else 0)
            f = pd.concat([f, pd.DataFrame([row])], ignore_index=True)
            if not final:
                f.loc[f.index[-1], "sold"] = self._ens(add_features(f).iloc[[-1]])
        return add_features(f).iloc[[-1]]

    def forecast(self, item, date=None, buffer=0.05, **over):
        if item not in ITEM_LIST:
            raise KeyError(item)
        date = date or self.next_date
        X = self._frame(item, date, over)
        p = max(self._ens(X), 0)
        lo = float(self.b["q10"].predict(X[FEATURES])[0]); hi = float(self.b["q90"].predict(X[FEATURES])[0])
        h = self.hist[self.hist.item == item].tail(7)
        return dict(item=item, date=str(pd.Timestamp(date).date()), predicted=round(p), p10=round(min(lo, p)), p90=round(max(hi, p)),
                    recommended=math.ceil(p * (1 + buffer)), typical_prepared=round(float(h.prepared.mean())),
                    typical_waste=round(float(h.waste.mean())))

    def plan(self, date=None, buffer=0.05, **over):
        return [self.forecast(i, date, buffer, **over) for i in ITEM_LIST]

    def explain(self, item, date=None, buffer=0.05, **over):
        date = date or self.next_date
        X = self._frame(item, date, over)
        base = self._ens(X); med = self.b["medians"][item]
        drivers = []
        for g, cols in GROUPS.items():
            Z = X.copy()
            for c in cols: Z[c] = med[c]
            drivers.append(dict(factor=g, impact=round(base - self._ens(Z), 1)))
        drivers.sort(key=lambda d: -abs(d["impact"]))
        typical = round(float(self.hist[self.hist.item == item].sold.mean()))
        top = [d for d in drivers if abs(d["impact"]) >= 1][:3]
        txt = f"For {item} on {pd.Timestamp(date).date()}, demand is forecast at about {round(base)} units (long-run average {typical}). "
        txt += ("Main drivers: " + ", ".join(f"{d['factor']} ({d['impact']:+.0f})" for d in top) + ". ") if top else "No factor moves demand much from normal. "
        r = X.iloc[0]
        txt += f"Same weekday last week sold {r.lag_7:.0f}; the 7-day average is {r.roll7:.0f}."
        return dict(item=item, predicted=round(base), drivers=drivers, explanation=txt)

    def analytics(self):
        h = self.hist
        by_item = h.groupby("item").agg(prepared=("prepared", "sum"), sold=("sold", "sum"), waste=("waste", "sum"), price=("price", "first"))
        by_item["waste_pct"] = (100 * by_item.waste / by_item.prepared).round(1)
        by_item["waste_cost_inr"] = by_item.waste * by_item.price
        mo = h.assign(month=h.date.dt.to_period("M").astype(str)).groupby("month")[["prepared", "waste"]].sum()
        mo["waste_pct"] = (100 * mo.waste / mo.prepared).round(1)
        dw = h.groupby("day_of_week")["waste"].mean().round(1)
        return dict(by_item=by_item.reset_index().to_dict("records"), monthly=mo.reset_index().to_dict("records"),
                    by_weekday=dw.reset_index().to_dict("records"))

    def history(self, item, days=90):
        h = self.hist[self.hist.item == item].tail(days)
        return h[["date", "prepared", "sold", "waste"]].assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d")).to_dict("records")
