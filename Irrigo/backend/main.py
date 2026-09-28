"""
Irrigo Backend — FastAPI Application
======================================
Routes:
  GET  /health           — liveness check
  POST /sensor-push      — ESP32 sends a sensor reading → decision returned
  POST /predict          — dashboard manual prediction
  GET  /latest-reading   — most recent reading from DB
  GET  /history          — recent readings + decisions for charting
  GET  /model-info       — training metadata (metrics, features)

Static dashboard files served from ../dashboard/ (when that folder exists).
"""

from __future__ import annotations

import json
import pathlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .database import (
    get_decisions_history,
    get_latest_reading,
    get_readings_history,
    init_db,
    insert_reading,
)
from .decision_engine import decide
from .schemas import (
    HistoryRow,
    IrrigationDecision,
    LatestReading,
    PredictRequest,
    SensorPush,
)

# ── Paths ──────────────────────────────────────────────────────────────────
HERE       = pathlib.Path(__file__).resolve().parent   # …/Irrigo/backend/
IRRIGO_ROOT = HERE.parent                              # …/Irrigo/
ARTIFACTS  = IRRIGO_ROOT / "models" / "artifacts"
DASHBOARD  = IRRIGO_ROOT / "dashboard"


# ── App lifecycle ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB and pre-load model artifacts on startup."""
    print("[Irrigo] Initialising database …")
    init_db()
    print("[Irrigo] Loading ML model …")
    # Import predictor so it loads at startup (raises early if model missing)
    from .predictor import FEATURE_NAMES  # noqa: F401
    print(f"[Irrigo] Model ready. Features: {len(FEATURE_NAMES)}")
    yield
    print("[Irrigo] Shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Irrigo — Explainable Hybrid Irrigation API",
    description=(
        "Hybrid tomato irrigation recommendation system combining "
        "FAO-56 Penman-Monteith ETo, RandomForest 24h soil moisture forecast, "
        "and live sensor / weather data."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — open for now; restrict to dashboard origin in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Mount static dashboard (optional) ──────────────────────────────────────
if DASHBOARD.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD), html=True), name="dashboard")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Liveness check — returns server time and status."""
    return {
        "status":    "ok",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "service":   "Irrigo backend v1.0",
    }


@app.get("/model-info", tags=["System"])
async def model_info():
    """Return training metadata: metrics, feature list, preprocessing info."""
    metrics_path = ARTIFACTS / "physics_metrics.json"
    if not metrics_path.exists():
        metrics_path = ARTIFACTS / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run save_7et_model.py first.",
        )
    with open(metrics_path) as f:
        metrics = json.load(f)
    return {"model_artifacts": str(ARTIFACTS), "metrics": metrics}


@app.post("/sensor-push", response_model=IrrigationDecision, tags=["Sensors"])
async def sensor_push(payload: SensorPush):
    """
    Receive a sensor reading from ESP32 (or any HTTP client).

    1. Persists the raw reading to the database.
    2. Runs the hybrid decision engine.
    3. Returns the full explainable IrrigationDecision.

    The decision is also persisted to the decisions table.
    """
    _now = datetime.now(tz=timezone.utc)

    # Derive temporal columns for ML
    month      = _now.month
    hour       = _now.hour
    day_period = _hour_to_day_period(hour)

    reading_row = {
        "timestamp":               _now.isoformat(),
        "soil_moisture_pct":       payload.soil_moisture_pct,
        "soil_temperature_c":      payload.soil_temperature_c,
        "air_temperature_c":       payload.air_temperature_c,
        "humidity_pct":            payload.humidity_pct,
        "forecast_rainfall_mm":    payload.forecast_rainfall_mm,
        "forecast_temperature_c":  payload.forecast_temperature_c,
        "wind_speed_mps":          payload.wind_speed_mps,
        "solar_radiation":         payload.solar_radiation,
        "water_volume_past_4h_l":  payload.water_volume_past_4h_l,
        "crop_type":               payload.crop_type,
        "growth_stage":            payload.growth_stage,
        "device_id":               payload.device_id,
        "zone_id":                 payload.zone_id,
        "month":                   month,
        "hour":                    hour,
        "day_period":              day_period,
    }
    reading_id = insert_reading(reading_row)
    decision   = decide(payload, reading_id=reading_id)
    return decision


@app.post("/predict", response_model=IrrigationDecision, tags=["Prediction"])
async def predict(payload: PredictRequest):
    """
    Dashboard manual prediction endpoint.
    Accepts a fully-specified PredictRequest (all sensor fields required).
    Does NOT require an existing reading_id.
    """
    decision = decide(payload, reading_id=None)
    return decision


@app.get("/latest-reading", response_model=Optional[LatestReading], tags=["Data"])
async def latest_reading():
    """Return the most recent sensor reading stored in the database."""
    row = get_latest_reading()
    if not row:
        return None
    return LatestReading(**{k: row.get(k) for k in LatestReading.model_fields})


@app.get("/history", tags=["Data"])
async def history(limit: int = 144):
    """
    Return recent readings + decisions joined for dashboard charting.
    Default: last 144 readings (≈ 72 h at 30-min intervals).
    """
    readings  = get_readings_history(limit)
    decisions = get_decisions_history(limit)

    # Build a simple time-keyed merge
    dec_map: dict[int, dict] = {d["reading_id"]: d for d in decisions if d.get("reading_id")}

    rows = []
    for r in readings:
        d = dec_map.get(r["id"], {})
        rows.append({
            "timestamp":              r.get("timestamp"),
            "soil_moisture_pct":      r.get("soil_moisture_pct"),
            "predicted_moisture_24h": d.get("predicted_moisture_24h"),
            "recommend_irrigation":   bool(d["recommend_irrigation"]) if "recommend_irrigation" in d else None,
            "confidence":             d.get("confidence"),
        })
    return {"count": len(rows), "rows": rows}


# ── Helper ────────────────────────────────────────────────────────────────

def _hour_to_day_period(hour: int) -> int:
    if 21 <= hour or hour < 6:
        return 0   # night
    elif 6 <= hour < 12:
        return 1   # morning
    elif 12 <= hour < 17:
        return 2   # midday
    return 3       # evening
