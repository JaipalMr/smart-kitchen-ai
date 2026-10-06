"""
Smart Kitchen AI - FastAPI Backend

Local:
    python -m uvicorn api:app --reload --port 8000

Production:
    uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import METRICS, ITEM_LIST
from predictor import Predictor


app = FastAPI(
    title="Smart Kitchen AI",
    description="AI-powered food demand forecasting and inventory optimization system",
    version="1.0.0"
)

# Allow frontend applications such as Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
P = Predictor()


class Scenario(BaseModel):
    date: Optional[str] = Field(
        None,
        description="YYYY-MM-DD. Must be after the last data date."
    )

    temperature_c: Optional[float] = Field(
        None,
        ge=0,
        le=55
    )

    rainfall_mm: Optional[float] = Field(
        None,
        ge=0,
        le=500
    )

    is_holiday: Optional[bool] = None

    is_event: bool = False

    buffer: float = Field(
        0.05,
        ge=0,
        le=0.5,
        description="Safety buffer. 0.05 = 5%"
    )


def scenario_kwargs(scenario: Scenario):
    return {
        "date": scenario.date,
        "buffer": scenario.buffer,
        "temperature_c": scenario.temperature_c,
        "rainfall_mm": scenario.rainfall_mm,
        "is_holiday": scenario.is_holiday,
        "is_event": scenario.is_event,
    }


def guard(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)

    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown item. Choose from: {ITEM_LIST}"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# --------------------------------------------------
# BASIC ROUTES
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "application": "Smart Kitchen AI",
        "status": "running",
        "version": "1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Smart Kitchen AI API"
    }


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

@app.get("/meta")
def meta():
    return {
        "items": ITEM_LIST,
        "last_data_date": str(P.last.date()),
        "next_date": str(P.next_date)
    }


# --------------------------------------------------
# KITCHEN PLAN
# --------------------------------------------------

@app.post("/plan")
def plan(scenario: Scenario):
    return guard(
        P.plan,
        **scenario_kwargs(scenario)
    )


# --------------------------------------------------
# ITEM FORECAST
# --------------------------------------------------

@app.post("/forecast/{item}")
def forecast(item: str, scenario: Scenario):
    return guard(
        P.forecast,
        item,
        **scenario_kwargs(scenario)
    )


# --------------------------------------------------
# FORECAST EXPLANATION
# --------------------------------------------------

@app.post("/explain/{item}")
def explain(item: str, scenario: Scenario):
    return guard(
        P.explain,
        item,
        **scenario_kwargs(scenario)
    )


# --------------------------------------------------
# HISTORICAL DATA
# --------------------------------------------------

@app.get("/history/{item}")
def history(item: str, days: int = 90):
    return guard(
        P.history,
        item,
        days
    )


# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------

@app.get("/analytics")
def analytics():
    return P.analytics()


# --------------------------------------------------
# MODEL METRICS
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    return json.loads(
        METRICS.read_text()
    )