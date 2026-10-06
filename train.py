"""Stacked ensemble (RandomForest + XGBoost + HistGradientBoosting) with NNLS-learned weights,
quantile models for P10/P90 prediction intervals, time-based validation and a waste backtest."""
import json, math, joblib, numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from config import DATA, MODEL, METRICS, BACKTEST, ITEM_LIST
from features import add_features, FEATURES

def make_models():
    return {
        "RandomForest": RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0),
        "XGBoost": XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
                                colsample_bytree=0.8, random_state=0, n_jobs=-1),
        "HistGB": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=20, random_state=0),
    }

def score(y, p):
    return dict(MAE=round(mean_absolute_error(y, p), 3), RMSE=round(math.sqrt(mean_squared_error(y, p)), 3),
                MAPE=round(float(np.mean(np.abs(y - p) / y)) * 100, 2), R2=round(r2_score(y, p), 4))

df = pd.read_csv(DATA)
X = add_features(df).dropna(subset=FEATURES)
last = X.date.max(); t0 = last - pd.Timedelta(days=149); v0 = t0 - pd.Timedelta(days=90)
tr, va, te, trva = X[X.date < v0], X[(X.date >= v0) & (X.date < t0)], X[X.date >= t0], X[X.date < t0]
print(f"train {len(tr)} | val {len(va)} | test {len(te)}")

# 1) fit on train, learn blend weights on validation
models = make_models(); vp = []
for n, m in models.items():
    m.fit(tr[FEATURES], tr.sold); vp.append(m.predict(va[FEATURES]))
w, _ = nnls(np.column_stack(vp), va.sold.values); w = w / w.sum()
print("blend weights:", dict(zip(models, w.round(3))))

# 2) refit on train+val, evaluate on untouched future test window
models = make_models()
for m in models.values(): m.fit(trva[FEATURES], trva.sold)
tp = {n: m.predict(te[FEATURES]) for n, m in models.items()}
ens = sum(wi * tp[n] for wi, n in zip(w, models))
q = {}
for a in (0.1, 0.9):
    q[a] = HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=300, learning_rate=0.05,
                                         random_state=0).fit(trva[FEATURES], trva.sold)
lo, hi = q[0.1].predict(te[FEATURES]), q[0.9].predict(te[FEATURES])

y = te.sold.values
metrics = {"test_window": [str(te.date.min().date()), str(te.date.max().date())], "rows": dict(train=len(tr), val=len(va), test=len(te)),
           "weights": dict(zip(models, w.round(4).tolist())),
           "models": {**{n: score(y, p) for n, p in tp.items()}, "Ensemble": score(y, ens),
                      "Baseline: same weekday last week": score(y, te.lag_7.values),
                      "Baseline: 7-day average": score(y, te.roll7.values)},
           "interval_coverage_P10_P90": round(float(np.mean((y >= lo) & (y <= hi))), 3)}

# 3) waste backtest (approximate: observed sales are used as demand)
bt = te[["date", "item", "prepared", "sold", "waste", "stockout"]].copy()
bt["pred"] = ens; bt["p90"] = np.maximum(hi, ens)
policies = {"AI +5% buffer": np.ceil(ens * 1.05), "AI +10% buffer": np.ceil(ens * 1.10), "AI P90 (high service)": np.ceil(bt.p90)}
bt["recommended"] = policies["AI +5% buffer"].astype(int)
metrics["backtest"] = {"historical": dict(waste_units=int(bt.waste.sum()), stockout_pct=round(100 * bt.stockout.mean(), 2))}
for name, rec in policies.items():
    waste = np.maximum(rec - bt.sold, 0)
    metrics["backtest"][name] = dict(waste_units=int(waste.sum()), waste_reduction_pct=round(100 * (1 - waste.sum() / bt.waste.sum()), 1),
                                     shortfall_pct=round(100 * float((bt.sold > rec).mean()), 2))
rf = models["RandomForest"]
metrics["feature_importance"] = dict(sorted(zip(FEATURES, rf.feature_importances_.round(4).tolist()), key=lambda t: -t[1]))
bt.to_csv(BACKTEST, index=False)

# 4) final fit on ALL data and save
models = make_models()
for m in models.values(): m.fit(X[FEATURES], X.sold)
for a in q: q[a].fit(X[FEATURES], X.sold)
joblib.dump(dict(models=list(models.values()), weights=w.tolist(), q10=q[0.1], q90=q[0.9],
                 medians=X.groupby("item")[FEATURES].median().to_dict("index")), MODEL)
METRICS.write_text(json.dumps(metrics, indent=2))
print(json.dumps({k: metrics[k] for k in ("models", "interval_coverage_P10_P90", "backtest")}, indent=2))
