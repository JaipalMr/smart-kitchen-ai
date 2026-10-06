"""
Smart Kitchen AI - Streamlit Dashboard

Local:
    python -m streamlit run dashboard.py

Cloud:
    Set API_URL in Streamlit Secrets.
"""

import os
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Smart Kitchen AI",
    page_icon="🍛",
    layout="wide"
)


# ==================================================
# API URL
# ==================================================

# Priority:
# 1. Streamlit Secrets
# 2. Environment variable
# 3. Localhost

try:
    API = st.secrets.get(
        "API_URL",
        os.getenv("API_URL", "http://localhost:8000")
    )
except Exception:
    API = os.getenv(
        "API_URL",
        "http://localhost:8000"
    )

API = API.rstrip("/")


# ==================================================
# API FUNCTIONS
# ==================================================

def get(path):
    """GET request to FastAPI."""

    try:
        response = requests.get(
            API + path,
            timeout=60
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        st.error(
            f"❌ API connection failed.\n\n"
            f"API URL: {API}\n\n"
            f"Error: {e}"
        )

        st.stop()


def post(path, body):
    """POST request to FastAPI."""

    try:
        response = requests.post(
            API + path,
            json=body,
            timeout=60
        )

        if response.status_code != 200:

            try:
                error = response.json().get(
                    "detail",
                    response.text
                )
            except Exception:
                error = response.text

            st.error(
                f"API Error ({response.status_code}): {error}"
            )

            st.stop()

        return response.json()

    except requests.exceptions.RequestException as e:

        st.error(
            f"❌ Could not connect to backend.\n\n"
            f"API URL: {API}\n\n"
            f"Error: {e}"
        )

        st.stop()


# ==================================================
# CHECK API
# ==================================================

try:

    health = get("/health")

    if health.get("status") != "ok":
        st.error("Backend API is not healthy.")
        st.stop()

except Exception:

    st.error(
        f"""
        ❌ Smart Kitchen AI backend is not reachable.

        Backend URL:

        {API}

        If running locally, start:

        python -m uvicorn api:app --reload --port 8000
        """
    )

    st.stop()


# ==================================================
# GET MODEL METADATA
# ==================================================

meta = get("/meta")


# ==================================================
# HEADER
# ==================================================

st.title("🍛 Smart Kitchen AI")

st.caption(
    "AI-powered food demand forecasting and inventory "
    "optimization for reducing food waste and preventing stockouts."
)

st.info(
    f"Backend connected successfully: {API}"
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📅 Tomorrow's Scenario")

    date = st.date_input(
        "Date",
        pd.to_datetime(meta["next_date"]),
        min_value=pd.to_datetime(meta["next_date"])
    )

    temp = st.slider(
        "Temperature (°C)",
        18.0,
        42.0,
        30.0,
        0.5
    )

    rain = st.slider(
        "Rainfall (mm)",
        0.0,
        100.0,
        0.0,
        1.0
    )

    holiday = st.checkbox(
        "Holiday / Festival"
    )

    event = st.checkbox(
        "Local Event"
    )

    buf = st.slider(
        "Safety Buffer (%)",
        0,
        25,
        5
    ) / 100


# ==================================================
# SCENARIO
# ==================================================

scenario = {
    "date": str(date),
    "temperature_c": temp,
    "rainfall_mm": rain,
    "is_holiday": holiday,
    "is_event": event,
    "buffer": buf
}


# ==================================================
# TABS
# ==================================================

t1, t2, t3, t4 = st.tabs(
    [
        "📋 Kitchen Plan",
        "🔍 Item Forecast & Why",
        "♻️ Waste Analytics",
        "🧠 Model Report"
    ]
)


# ==================================================
# TAB 1 - KITCHEN PLAN
# ==================================================

with t1:

    st.subheader(
        "AI Recommended Kitchen Preparation"
    )

    plan_data = post(
        "/plan",
        scenario
    )

    df = pd.DataFrame(plan_data)

    # ------------------------------
    # KPI CARDS
    # ------------------------------

    total_recommended = int(
        df["recommended"].sum()
    )

    total_typical = int(
        df["typical_prepared"].sum()
    )

    total_waste = int(
        df["typical_waste"].sum()
    )

    difference = (
        total_recommended -
        total_typical
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Recommended",
        total_recommended,
        difference
    )

    c2.metric(
        "Recent Avg Prepared",
        total_typical
    )

    c3.metric(
        "Recent Avg Daily Waste",
        total_waste
    )


    # ------------------------------
    # BAR CHART
    # ------------------------------

    fig = go.Figure()

    fig.add_bar(
        name="Recent Prepared",
        x=df["item"],
        y=df["typical_prepared"]
    )

    fig.add_bar(
        name="AI Predicted",
        x=df["item"],
        y=df["predicted"]
    )

    fig.add_bar(
        name="AI Recommended",
        x=df["item"],
        y=df["recommended"]
    )

    fig.update_layout(
        barmode="group",
        height=450,
        title="Recent vs AI Recommended Preparation",
        xaxis_title="Menu Item",
        yaxis_title="Units"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ------------------------------
    # ALERT
    # ------------------------------

    df["alert"] = (
        df["typical_prepared"] >
        df["recommended"] * 1.15
    ).map(
        {
            True: "⚠️ Over-preparing",
            False: "✅ OK"
        }
    )


    # ------------------------------
    # TABLE
    # ------------------------------

    display_df = df[
        [
            "item",
            "p10",
            "predicted",
            "p90",
            "recommended",
            "typical_prepared",
            "alert"
        ]
    ].rename(
        columns={
            "item": "Item",
            "p10": "Low (P10)",
            "predicted": "Predicted",
            "p90": "High (P90)",
            "recommended": "Recommended",
            "typical_prepared": "Recent Prep",
            "alert": "Status"
        }
    )

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True
    )


# ==================================================
# TAB 2 - ITEM FORECAST
# ==================================================

with t2:

    st.subheader(
        "Individual Menu Item Forecast"
    )

    item = st.selectbox(
        "Select Menu Item",
        meta["items"]
    )


    forecast = post(
        f"/forecast/{item}",
        scenario
    )

    explanation = post(
        f"/explain/{item}",
        scenario
    )


    # ------------------------------
    # FORECAST KPIs
    # ------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Predicted",
        forecast["predicted"]
    )

    c2.metric(
        "P10 - P90",
        f"{forecast['p10']} - {forecast['p90']}"
    )

    c3.metric(
        "Recommended",
        forecast["recommended"]
    )

    c4.metric(
        "Recent Avg Prep",
        forecast["typical_prepared"]
    )


    # ------------------------------
    # EXPLANATION
    # ------------------------------

    st.info(
        explanation["explanation"]
    )


    # ------------------------------
    # FORECAST DRIVERS
    # ------------------------------

    drivers = pd.DataFrame(
        explanation["drivers"]
    )

    if not drivers.empty:

        fig = px.bar(
            drivers,
            x="impact",
            y="factor",
            orientation="h",
            title="What Moves This Forecast",
            color="impact",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ------------------------------
    # HISTORY
    # ------------------------------

    history = pd.DataFrame(
        get(
            f"/history/{item}?days=90"
        )
    )

    if not history.empty:

        fig = px.line(
            history,
            x="date",
            y=[
                "prepared",
                "sold",
                "waste"
            ],
            title="Last 90 Days"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==================================================
# TAB 3 - WASTE ANALYTICS
# ==================================================

with t3:

    st.subheader(
        "♻️ Food Waste Analytics"
    )

    analytics = get(
        "/analytics"
    )

    by_item = pd.DataFrame(
        analytics["by_item"]
    )

    if not by_item.empty:

        total_waste = int(
            by_item["waste"].sum()
        )

        total_prepared = int(
            by_item["prepared"].sum()
        )

        waste_rate = (
            100 * total_waste /
            total_prepared
            if total_prepared > 0
            else 0
        )

        waste_cost = int(
            by_item["waste_cost_inr"].sum()
        )


        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Waste",
            f"{total_waste:,} units"
        )

        c2.metric(
            "Waste Rate",
            f"{waste_rate:.1f}%"
        )

        c3.metric(
            "Estimated Waste Cost",
            f"₹{waste_cost:,}"
        )


        # ------------------------------
        # WASTE COST BY ITEM
        # ------------------------------

        fig = px.bar(
            by_item.sort_values(
                "waste_cost_inr"
            ),
            x="waste_cost_inr",
            y="item",
            orientation="h",
            title="Waste Cost by Item"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ------------------------------
        # MONTHLY WASTE
        # ------------------------------

        monthly = pd.DataFrame(
            analytics["monthly"]
        )

        if not monthly.empty:

            fig = px.line(
                monthly,
                x="month",
                y="waste_pct",
                title="Monthly Waste Percentage"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ==================================================
# TAB 4 - MODEL REPORT
# ==================================================

with t4:

    st.subheader(
        "🧠 Machine Learning Model Report"
    )

    metrics = get(
        "/metrics"
    )


    st.write(
        f"Test window: "
        f"{metrics['test_window'][0]} "
        f"to "
        f"{metrics['test_window'][1]}"
    )

    st.write(
        f"Test rows: "
        f"{metrics['rows']['test']}"
    )

    st.write(
        f"Ensemble weights: "
        f"{metrics['weights']}"
    )


    # ------------------------------
    # MODEL METRICS
    # ------------------------------

    model_df = pd.DataFrame(
        metrics["models"]
    ).T

    st.dataframe(
        model_df,
        use_container_width=True
    )


    # ------------------------------
    # INTERVAL COVERAGE
    # ------------------------------

    coverage = (
        metrics["interval_coverage_P10_P90"]
        * 100
    )

    st.metric(
        "P10-P90 Interval Coverage",
        f"{coverage:.0f}%"
    )

    st.caption(
        "Target coverage: approximately 80%"
    )


    # ------------------------------
    # BACKTEST
    # ------------------------------

    st.subheader(
        "Backtest: Waste Reduction vs Historical Practice"
    )

    backtest = pd.DataFrame(
        metrics["backtest"]
    ).T

    st.dataframe(
        backtest,
        use_container_width=True
    )


    st.caption(
        "Observed sales are used as a proxy for true demand. "
        "Shortfall may be understated on days when an item sold out."
    )


    # ------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------

    st.subheader(
        "Top Feature Importances"
    )

    feature_importance = pd.Series(
        metrics["feature_importance"]
    ).head(12)

    fig = px.bar(
        feature_importance[::-1],
        orientation="h",
        title="Random Forest Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Smart Kitchen AI • Demand Forecasting • "
    "Inventory Optimization • Food Waste Reduction"
)