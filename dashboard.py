"""Streamlit frontend:  streamlit run dashboard.py   (needs the API running on :8000)"""
import os, requests, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config("Smart Kitchen AI", "🍛", layout="wide")

def get(path): return requests.get(API + path, timeout=60).json()
def post(path, body):
    r = requests.post(API + path, json=body, timeout=60)
    if r.status_code != 200: st.error(r.json().get("detail", r.text)); st.stop()
    return r.json()

try: meta = get("/meta")
except Exception: st.error(f"API not reachable at {API}. Start it: uvicorn api:app --port 8000"); st.stop()

st.title("🍛 Smart Kitchen AI")
st.caption("We don't just tell restaurants what they sold yesterday. We tell them how much to prepare tomorrow.")

with st.sidebar:
    st.header("Tomorrow's scenario")
    date = st.date_input("Date", pd.to_datetime(meta["next_date"]), min_value=pd.to_datetime(meta["next_date"]))
    temp = st.slider("Temperature (°C)", 18.0, 42.0, 30.0, 0.5)
    rain = st.slider("Rainfall (mm)", 0.0, 100.0, 0.0, 1.0)
    holiday = st.checkbox("Holiday / festival")
    event = st.checkbox("Local event")
    buf = st.slider("Safety buffer (%)", 0, 25, 5) / 100
scn = dict(date=str(date), temperature_c=temp, rainfall_mm=rain, is_holiday=holiday, is_event=event, buffer=buf)

t1, t2, t3, t4 = st.tabs(["📋 Kitchen plan", "🔍 Item forecast & why", "♻️ Waste analytics", "🧠 Model report"])

with t1:
    df = pd.DataFrame(post("/plan", scn))
    c = st.columns(3)
    c[0].metric("Total recommended", int(df.recommended.sum()), f"{int(df.recommended.sum() - df.typical_prepared.sum())} vs recent habit")
    c[1].metric("Recent avg prepared", int(df.typical_prepared.sum()))
    c[2].metric("Recent avg daily waste", int(df.typical_waste.sum()))
    fig = go.Figure()
    fig.add_bar(name="Recent prepared", x=df.item, y=df.typical_prepared, marker_color="#c9c9c9")
    fig.add_bar(name="AI predicted", x=df.item, y=df.predicted, marker_color="#2a9d8f")
    fig.add_bar(name="AI recommended", x=df.item, y=df.recommended, marker_color="#e76f51")
    fig.update_layout(barmode="group", height=420, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    df["alert"] = (df.typical_prepared > df.recommended * 1.15).map({True: "⚠️ Over-preparing, review", False: "OK"})
    st.dataframe(df[["item", "p10", "predicted", "p90", "recommended", "typical_prepared", "alert"]].rename(columns={
        "p10": "Low (P10)", "p90": "High (P90)", "typical_prepared": "Recent prep"}), hide_index=True, use_container_width=True)

with t2:
    item = st.selectbox("Menu item", meta["items"])
    f = post(f"/forecast/{item}", scn); e = post(f"/explain/{item}", scn)
    c = st.columns(4)
    c[0].metric("Predicted", f["predicted"]); c[1].metric("Range (P10-P90)", f"{f['p10']} - {f['p90']}")
    c[2].metric("Recommended prep", f["recommended"]); c[3].metric("Recent avg prep", f["typical_prepared"])
    st.info(e["explanation"])
    d = pd.DataFrame(e["drivers"])
    st.plotly_chart(px.bar(d, x="impact", y="factor", orientation="h", title="What moves this forecast (units vs a normal day)",
                           color="impact", color_continuous_scale="RdYlGn", color_continuous_midpoint=0), use_container_width=True)
    h = pd.DataFrame(get(f"/history/{item}?days=90"))
    st.plotly_chart(px.line(h, x="date", y=["prepared", "sold", "waste"], title="Last 90 days"), use_container_width=True)

with t3:
    a = get("/analytics"); bi = pd.DataFrame(a["by_item"])
    c = st.columns(3)
    c[0].metric("Total waste (units)", f"{int(bi.waste.sum()):,}")
    c[1].metric("Waste rate", f"{100 * bi.waste.sum() / bi.prepared.sum():.1f}%")
    c[2].metric("Estimated waste cost", f"₹{int(bi.waste_cost_inr.sum()):,}")
    st.plotly_chart(px.bar(bi.sort_values("waste_cost_inr"), x="waste_cost_inr", y="item", orientation="h", title="Waste cost by item (₹)"), use_container_width=True)
    st.plotly_chart(px.line(pd.DataFrame(a["monthly"]), x="month", y="waste_pct", title="Monthly waste %"), use_container_width=True)

with t4:
    m = get("/metrics")
    st.write(f"Tested on unseen window {m['test_window'][0]} to {m['test_window'][1]} ({m['rows']['test']} item-days). Ensemble weights: {m['weights']}")
    st.dataframe(pd.DataFrame(m["models"]).T, use_container_width=True)
    st.caption(f"P10-P90 interval coverage on test: {m['interval_coverage_P10_P90'] * 100:.0f}% (target 80%)")
    st.subheader("Backtest: waste reduction vs historical practice")
    st.dataframe(pd.DataFrame(m["backtest"]).T, use_container_width=True)
    st.caption("Approximate: observed sales stand in for true demand, so shortfall is understated on days when the item sold out.")
    fi = pd.Series(m["feature_importance"]).head(12)
    st.plotly_chart(px.bar(fi[::-1], orientation="h", title="Top feature importances (Random Forest)"), use_container_width=True)
