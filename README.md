# Smart Kitchen AI: Food-Waste Prediction & Kitchen Planner

Architecture: `restaurant_sales.csv` -> `features.py` -> `train.py` (RF + XGBoost + HistGB stack, NNLS blend, P10/P90 quantile models) -> `predictor.py` -> `api.py` (FastAPI) -> `dashboard.py` (Streamlit)

## Run
    pip install -r requirements.txt
    python generate_dataset.py      # (optional) regenerates the 10,000-row cleaned dataset
    python train.py                 # trains + writes models/ and metrics
    uvicorn api:app --port 8000     # terminal 1  (Swagger docs: http://localhost:8000/docs)
    streamlit run dashboard.py      # terminal 2

## API
GET /meta, /health, /metrics, /analytics, /history/{item} | POST /plan, /forecast/{item}, /explain/{item}
Body: {"date":"2026-06-28","temperature_c":31,"rainfall_mm":0,"is_holiday":false,"is_event":false,"buffer":0.05}

## Using your own restaurant data
Replace data/restaurant_sales.csv, keeping columns: date,item,is_holiday,is_event,temperature_c,rainfall_mm,price,prepared,sold,waste.
Update ITEMS in config.py with your menu, then re-run train.py.

## Notes
The shipped dataset is synthetic (seeded, reproducible). Say so in your pitch and show the pipeline works on real data too.
`sold` is censored by `prepared` (a sold-out item hides true demand); `stockout` flags those days.
