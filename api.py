"""FastAPI server:  uvicorn api:app --reload --port 8000   (docs at /docs)"""
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from config import METRICS, ITEM_LIST
from predictor import Predictor

app = FastAPI(title="Smart Kitchen AI", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
P = Predictor()

class Scenario(BaseModel):
    date: Optional[str] = Field(None, description="YYYY-MM-DD, must be after last data date; default = next day")
    temperature_c: Optional[float] = Field(None, ge=0, le=55)
    rainfall_mm: Optional[float] = Field(None, ge=0, le=500)
    is_holiday: Optional[bool] = None
    is_event: bool = False
    buffer: float = Field(0.05, ge=0, le=0.5, description="safety buffer, 0.05 = 5%")

def kw(s: Scenario):
    return dict(date=s.date, buffer=s.buffer, temperature_c=s.temperature_c, rainfall_mm=s.rainfall_mm,
                is_holiday=s.is_holiday, is_event=s.is_event)

def guard(fn, *a, **k):
    try: return fn(*a, **k)
    except KeyError: raise HTTPException(404, f"Unknown item. Choose from {ITEM_LIST}")
    except ValueError as e: raise HTTPException(400, str(e))

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/meta")
def meta(): return {"items": ITEM_LIST, "last_data_date": str(P.last.date()), "next_date": str(P.next_date)}

@app.post("/plan")
def plan(s: Scenario): return guard(P.plan, **kw(s))

@app.post("/forecast/{item}")
def forecast(item: str, s: Scenario): return guard(P.forecast, item, **kw(s))

@app.post("/explain/{item}")
def explain(item: str, s: Scenario): return guard(P.explain, item, **kw(s))

@app.get("/history/{item}")
def history(item: str, days: int = 90): return guard(P.history, item, days)

@app.get("/analytics")
def analytics(): return P.analytics()

@app.get("/metrics")
def metrics(): return json.loads(METRICS.read_text())
