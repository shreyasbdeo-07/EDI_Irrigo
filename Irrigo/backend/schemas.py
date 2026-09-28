"""
Irrigo Backend — Pydantic v2 Schemas
=====================================
Defines all request and response data models.
Input schema is designed to match future ESP32 hardware POST body.
"""

from __future__ import annotations

from enum import IntEnum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# ─── Enumerations ────────────────────────────────────────────────────────────

class GrowthStage(str):
    INITIAL     = "initial"      # Kc = 0.60
    DEVELOPMENT = "development"  # Kc interpolated 0.60 → 1.15
    MID_SEASON  = "mid_season"   # Kc = 1.15
    LATE_SEASON = "late_season"  # Kc = 0.80

class DayPeriod(IntEnum):
    NIGHT   = 0
    MORNING = 1
    MIDDAY  = 2
    EVENING = 3


# ─── ESP32 / hardware POST body ───────────────────────────────────────────────

class SensorPush(BaseModel):
    """
    Payload sent by ESP32 via HTTP POST to /sensor-push.
    All sensor fields are Optional so the backend can flag
    missing/unreliable readings and trigger manual-review mode.
    """
    # Live sensor readings
    soil_moisture_pct:      Optional[float] = Field(None, ge=0, le=100,
                                                    description="Capacitive soil moisture (%)")
    soil_temperature_c:     Optional[float] = Field(None, ge=-10, le=60,
                                                    description="DS18B20 soil temperature (°C)")
    air_temperature_c:      Optional[float] = Field(None, ge=-20, le=60,
                                                    description="DHT22/BME280 air temperature (°C)")
    humidity_pct:           Optional[float] = Field(None, ge=0, le=100,
                                                    description="Relative humidity (%)")
    # Weather forecast (fetched by ESP32 or backend from Open-Meteo)
    forecast_rainfall_mm:   Optional[float] = Field(None, ge=0,
                                                    description="24h rainfall forecast (mm)")
    forecast_temperature_c: Optional[float] = Field(None,
                                                    description="24h mean temperature forecast (°C)")
    wind_speed_mps:         Optional[float] = Field(None, ge=0,
                                                    description="Wind speed (m/s)")
    solar_radiation:        Optional[float] = Field(None, ge=0,
                                                    description="Solar radiation (W/m² or MJ/m²/day)")
    # Context
    crop_type:    str = Field("tomato", description="Crop identifier")
    growth_stage: str = Field(GrowthStage.MID_SEASON,
                              description="FAO-56 growth stage key")
    water_volume_past_4h_l: float = Field(0.0, ge=0,
                                          description="Irrigation water applied in past 4 h (L)")
    # Optional metadata
    device_id: Optional[str] = Field(None, description="ESP32 device identifier")
    zone_id:   Optional[int] = Field(None, description="Field zone number")


# ─── Dashboard manual prediction request ─────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Manual prediction request from the dashboard.
    Maps to the same schema as SensorPush but all fields required
    (the dashboard form validates before sending).
    """
    soil_moisture_pct:      float = Field(..., ge=0,   le=100)
    soil_temperature_c:     float = Field(..., ge=-10, le=60)
    air_temperature_c:      float = Field(..., ge=-20, le=60)
    humidity_pct:           float = Field(..., ge=0,   le=100)
    forecast_rainfall_mm:   float = Field(..., ge=0)
    forecast_temperature_c: float = Field(...)
    wind_speed_mps:         float = Field(..., ge=0)
    solar_radiation:        float = Field(..., ge=0)
    crop_type:              str   = Field("tomato")
    growth_stage:           str   = Field(GrowthStage.MID_SEASON)
    water_volume_past_4h_l: float = Field(0.0, ge=0)
    month:                  int   = Field(..., ge=1, le=12)
    hour:                   int   = Field(..., ge=0, le=23)
    day_period:             int   = Field(..., ge=0, le=3,
                                          description="0=night,1=morning,2=midday,3=evening")


# ─── FAO-56 result ────────────────────────────────────────────────────────────

class FAO56Result(BaseModel):
    eto_mm:           float = Field(description="Reference evapotranspiration (mm/day)")
    etc_mm:           float = Field(description="Crop evapotranspiration ETc (mm/day)")
    kc:               float = Field(description="Crop coefficient Kc used")
    rainfall_mm:      float = Field(description="Expected rainfall next 24 h (mm)")
    net_deficit_mm:   float = Field(description="ETc − rainfall (mm); positive = deficit")
    deficit_flag:     bool  = Field(description="True if net deficit > threshold (1 mm)")


# ─── Irrigation decision response ────────────────────────────────────────────

class IrrigationDecision(BaseModel):
    """
    Full explainable irrigation recommendation returned by /predict.
    """
    recommend_irrigation:   bool  = Field(description="Final irrigation recommendation")
    confidence:             float = Field(ge=0, le=1,
                                          description="Decision confidence score (0–1)")
    requires_manual_review: bool  = Field(description="True when confidence < 0.60 or data missing")
    manual_review_reason:   str   = Field(description="Populated when requires_manual_review is True")

    # ML component
    predicted_moisture_24h: float = Field(description="RF-predicted soil moisture 24h ahead (%)")
    moisture_threshold_pct: float = Field(description="Irrigation trigger threshold (%)")
    ml_flag:                bool  = Field(description="True if predicted_moisture < threshold")

    # FAO-56 component
    fao56: FAO56Result

    # Rainfall component
    forecast_rain_mm:   float = Field(description="24h rainfall forecast (mm)")
    rain_threshold_mm:  float = Field(description="Rainfall amount considered sufficient (mm)")
    rain_sufficient:    bool  = Field(description="True if forecast rain >= rain_threshold")

    # Explainability
    reasons: List[str] = Field(description="Human-readable list of reasons for the decision")

    # Audit
    timestamp: str = Field(description="ISO-8601 UTC timestamp of decision")


# ─── Latest reading response ──────────────────────────────────────────────────

class LatestReading(BaseModel):
    id:                     int
    timestamp:              str
    soil_moisture_pct:      Optional[float]
    soil_temperature_c:     Optional[float]
    air_temperature_c:      Optional[float]
    humidity_pct:           Optional[float]
    forecast_rainfall_mm:   Optional[float]
    wind_speed_mps:         Optional[float]
    solar_radiation:        Optional[float]
    crop_type:              str
    growth_stage:           str
    water_volume_past_4h_l: float
    device_id:              Optional[str]
    zone_id:                Optional[int]


# ─── History row ─────────────────────────────────────────────────────────────

class HistoryRow(BaseModel):
    timestamp:              str
    soil_moisture_pct:      Optional[float]
    predicted_moisture_24h: Optional[float]
    recommend_irrigation:   Optional[bool]
    confidence:             Optional[float]
