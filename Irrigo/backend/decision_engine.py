"""
Irrigo Backend — Hybrid Decision Engine
=========================================
Combines three signals to produce an explainable irrigation recommendation:

  1. ML signal   — RandomForest predicted soil moisture 24 h ahead
  2. FAO-56      — Penman-Monteith water deficit calculation
  3. Rainfall    — Forecast rainfall vs. minimum sufficient threshold

Decision rule (AND logic):
  Recommend irrigation IF:
    (predicted_moisture_24h < MOISTURE_THRESHOLD)
    AND (FAO-56 net deficit > DEFICIT_THRESHOLD)
    AND (forecast rainfall < RAIN_SUFFICIENT_THRESHOLD)

Confidence-aware fallback:
  If any required sensor reading is missing, or confidence < CONFIDENCE_MIN,
  set requires_manual_review = True and do NOT recommend automation.

All decisions are fully logged to SQLite via database.py.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Optional

from .database import insert_reading, insert_decision, get_past_solar_radiation
from .fao56 import run_fao56
from .predictor import predict_both, predict_moisture_24h
from .schemas import (
    FAO56Result,
    IrrigationDecision,
    PredictRequest,
    SensorPush,
)

# ── Thresholds (Tomato, open field) ────────────────────────────────────────
MOISTURE_THRESHOLD_PCT: float  = 30.0   # Below this → ML flag raised
RAIN_SUFFICIENT_MM:     float  = 4.0    # ≥ 4 mm forecast → rain sufficient
DEFICIT_THRESHOLD_MM:   float  = 1.0    # FAO net deficit > 1 mm → flag
CONFIDENCE_MIN:         float  = 0.60   # Below this → manual review required

# Deployment location for FAO-56 (Italy field trial default; override for Junnar)
DEFAULT_LAT_DEG:  float = 40.349306
DEFAULT_ELEV_M:   float = 0.0

# Solar radiation conversion: if units are W/m², divide by 11.574 → MJ/m²/day
WATT_TO_MJ_FACTOR: float = 1.0 / 11.574


def _solar_to_mj(solar_radiation: float) -> float:
    """
    Heuristic unit conversion.
    If the value looks like W/m² (> 50), convert to MJ/m²/day.
    If already MJ/m²/day (< 50), pass through.
    """
    if solar_radiation > 50.0:
        return solar_radiation * WATT_TO_MJ_FACTOR
    return solar_radiation


def _day_of_year(ts: Optional[str] = None) -> int:
    if ts:
        try:
            return datetime.fromisoformat(ts).timetuple().tm_yday
        except Exception:
            pass
    return datetime.now(tz=timezone.utc).timetuple().tm_yday


def _hour_to_day_period(hour: int) -> int:
    """0=night, 1=morning, 2=midday, 3=evening"""
    if 21 <= hour or hour < 6:
        return 0
    elif 6 <= hour < 12:
        return 1
    elif 12 <= hour < 17:
        return 2
    return 3


def _confidence_score(
    soil_moisture_pct:    Optional[float],
    soil_temperature_c:   Optional[float],
    air_temperature_c:    Optional[float],
    humidity_pct:         Optional[float],
    forecast_rainfall_mm: Optional[float],
    wind_speed_mps:       Optional[float],
    solar_radiation:      Optional[float],
) -> tuple[float, list[str]]:
    """
    Compute a 0–1 confidence score based on data completeness.
    Each missing field reduces confidence proportionally.
    Returns (score, list_of_missing_field_names).
    """
    fields = {
        "soil_moisture_pct":    soil_moisture_pct,
        "soil_temperature_c":   soil_temperature_c,
        "air_temperature_c":    air_temperature_c,
        "humidity_pct":         humidity_pct,
        "forecast_rainfall_mm": forecast_rainfall_mm,
        "wind_speed_mps":       wind_speed_mps,
        "solar_radiation":      solar_radiation,
    }
    # Weights: critical fields reduce confidence more if missing
    weights = {
        "soil_moisture_pct":    0.35,  # highest — core ML input
        "air_temperature_c":    0.15,
        "humidity_pct":         0.15,
        "forecast_rainfall_mm": 0.15,
        "wind_speed_mps":       0.05,
        "solar_radiation":      0.10,
        "soil_temperature_c":   0.05,
    }
    missing = [k for k, v in fields.items() if v is None]
    penalty = sum(weights.get(k, 0.05) for k in missing)
    score   = max(0.0, 1.0 - penalty)
    return round(score, 4), missing


def decide(
    payload: SensorPush | PredictRequest,
    reading_id: Optional[int] = None,
    lat_deg: float = DEFAULT_LAT_DEG,
    elev_m: float  = DEFAULT_ELEV_M,
) -> IrrigationDecision:
    """
    Core hybrid decision function.

    Parameters
    ----------
    payload    : SensorPush (ESP32) or PredictRequest (dashboard manual)
    reading_id : FK to readings table row (if already inserted)
    lat_deg    : Deployment latitude for FAO-56
    elev_m     : Elevation for FAO-56 atmospheric pressure

    Returns
    -------
    IrrigationDecision (Pydantic model)
    """
    now_iso = datetime.now(tz=timezone.utc).isoformat()

    # ── Determine temporal context ─────────────────────────────────────────
    if hasattr(payload, "month") and hasattr(payload, "hour"):
        month      = payload.month
        hour       = payload.hour
        day_period = payload.day_period if hasattr(payload, "day_period") else _hour_to_day_period(hour)
    else:
        _now   = datetime.now(tz=timezone.utc)
        month  = _now.month
        hour   = _now.hour
        day_period = _hour_to_day_period(hour)

    doy = _day_of_year()

    # ── Confidence & missing-data detection ────────────────────────────────
    confidence, missing_fields = _confidence_score(
        soil_moisture_pct    = payload.soil_moisture_pct,
        soil_temperature_c   = getattr(payload, "soil_temperature_c", None),
        air_temperature_c    = getattr(payload, "air_temperature_c", None),
        humidity_pct         = getattr(payload, "humidity_pct", None),
        forecast_rainfall_mm = getattr(payload, "forecast_rainfall_mm", None),
        wind_speed_mps       = getattr(payload, "wind_speed_mps", None),
        solar_radiation      = getattr(payload, "solar_radiation", None),
    )

    requires_manual_review = (confidence < CONFIDENCE_MIN)
    manual_review_reason   = ""
    if requires_manual_review:
        manual_review_reason = (
            f"Confidence {confidence:.0%} below threshold {CONFIDENCE_MIN:.0%}. "
            f"Missing or null fields: {', '.join(missing_fields)}. "
            "Please verify sensor data before acting on any recommendation."
        )

    # ── Resolve defaults for missing fields (use safe conservative values) ─
    T_c    = getattr(payload, "air_temperature_c",    None) or 25.0
    rh_pct = getattr(payload, "humidity_pct",         None) or 50.0
    u2     = getattr(payload, "wind_speed_mps",       None) or 1.0
    Rs_raw = getattr(payload, "solar_radiation",      None) or 15.0
    rain   = getattr(payload, "forecast_rainfall_mm", None) or 0.0
    soil_m = payload.soil_moisture_pct or 30.0  # fallback to threshold
    soil_temp_surface = getattr(payload, "soil_temperature_c", None) or T_c - 3.0

    Rs_MJ = _solar_to_mj(Rs_raw)

    # ── FAO-56 calculation ─────────────────────────────────────────────────
    fao_result = run_fao56(
        T_c=T_c, rh_pct=rh_pct, u2_mps=u2,
        Rs_MJ=Rs_MJ, doy=doy, lat_deg=lat_deg,
        rain_mm=rain, growth_stage=payload.growth_stage,
        elev_m=elev_m,
    )
    fao_obj = FAO56Result(**fao_result)

    # ── ML feature vector (7 Physics-Informed Features) ────────────────────
    # Compute 6-hour rolling mean of past solar radiation strictly prior to current reading
    zone_id = getattr(payload, "zone_id", None) or 1
    past_solar = get_past_solar_radiation(
        hours=6.0,
        zone_id=zone_id,
        before_reading_id=reading_id,
    )
    if past_solar:
        solar_roll_6h = float(sum(past_solar) / len(past_solar))
    else:
        # Documented fallback for cold start / initial boot: use instantaneous reading
        solar_roll_6h = float(Rs_raw)

    feature_dict = {
        "weather_temperature_c":                       float(T_c),
        "weather_humidity_pct":                        float(rh_pct),
        "soil_temperature_surface_c":                  float(soil_temp_surface),
        "soil_moisture_pct":                           float(soil_m),
        "weather_radiation_source_value":              float(Rs_raw),
        "weather_radiation_source_value_roll_mean_6h": float(solar_roll_6h),
        "weather_wind_speed":                          float(u2),
    }
    predicted_moisture = predict_moisture_24h(feature_dict)

    # ── Decision flags ─────────────────────────────────────────────────────
    ml_flag      = predicted_moisture < MOISTURE_THRESHOLD_PCT
    rain_sufficient = rain >= RAIN_SUFFICIENT_MM
    fao_flag     = fao_obj.deficit_flag

    # Final recommendation — only if not requiring manual review
    recommend = (ml_flag and fao_flag and not rain_sufficient) \
                and not requires_manual_review

    # ── Build reasons (explainability) ─────────────────────────────────────
    reasons: list[str] = []

    # ML reason
    if ml_flag:
        reasons.append(
            f"ML model predicts soil moisture will drop to {predicted_moisture:.1f}% in 24h "
            f"(threshold: {MOISTURE_THRESHOLD_PCT:.0f}%)."
        )
    else:
        reasons.append(
            f"ML model predicts adequate soil moisture of {predicted_moisture:.1f}% in 24h "
            f"(threshold: {MOISTURE_THRESHOLD_PCT:.0f}%)."
        )

    # FAO-56 reason
    if fao_flag:
        reasons.append(
            f"FAO-56 indicates water deficit of {fao_obj.net_deficit_mm:.2f} mm/day "
            f"(ETo={fao_obj.eto_mm:.2f} mm, ETc={fao_obj.etc_mm:.2f} mm, Kc={fao_obj.kc})."
        )
    else:
        reasons.append(
            f"FAO-56 shows no significant deficit ({fao_obj.net_deficit_mm:.2f} mm/day)."
        )

    # Rainfall reason
    if rain_sufficient:
        reasons.append(
            f"Forecast rainfall ({rain:.1f} mm) is sufficient "
            f"(≥ {RAIN_SUFFICIENT_MM:.0f} mm threshold). Irrigation not needed."
        )
    else:
        reasons.append(
            f"Forecast rainfall ({rain:.1f} mm) is insufficient "
            f"(threshold: {RAIN_SUFFICIENT_MM:.0f} mm)."
        )

    # Confidence reason
    if requires_manual_review:
        reasons.append(
            f"⚠ Manual review required — confidence too low ({confidence:.0%}). "
            f"Missing: {', '.join(missing_fields)}."
        )
    else:
        reasons.append(f"Data confidence: {confidence:.0%} — automatic decision safe.")

    # Summary
    if recommend:
        reasons.insert(0,
            "IRRIGATE: All three indicators (soil moisture forecast, FAO-56 deficit, "
            "insufficient rainfall) point to irrigation need."
        )
    elif requires_manual_review:
        reasons.insert(0,
            "MANUAL REVIEW REQUIRED: Decision withheld due to incomplete sensor data."
        )
    else:
        reasons.insert(0,
            "DO NOT IRRIGATE: At least one condition for irrigation is not met."
        )

    # ── Persist decision ───────────────────────────────────────────────────
    db_payload = {
        "timestamp":               now_iso,
        "recommend_irrigation":    int(recommend),
        "confidence":              confidence,
        "requires_manual_review":  int(requires_manual_review),
        "manual_review_reason":    manual_review_reason,
        "predicted_moisture_24h":  predicted_moisture,
        "moisture_threshold_pct":  MOISTURE_THRESHOLD_PCT,
        "ml_flag":                 int(ml_flag),
        "fao_eto_mm":              fao_obj.eto_mm,
        "fao_etc_mm":              fao_obj.etc_mm,
        "fao_kc":                  fao_obj.kc,
        "fao_deficit_mm":          fao_obj.net_deficit_mm,
        "fao_deficit_flag":        int(fao_obj.deficit_flag),
        "forecast_rain_mm":        rain,
        "rain_threshold_mm":       RAIN_SUFFICIENT_MM,
        "rain_sufficient":         int(rain_sufficient),
        "reasons_json":            json.dumps(reasons),
    }
    insert_decision(reading_id, db_payload)

    return IrrigationDecision(
        recommend_irrigation   = recommend,
        confidence             = confidence,
        requires_manual_review = requires_manual_review,
        manual_review_reason   = manual_review_reason,
        predicted_moisture_24h = predicted_moisture,
        moisture_threshold_pct = MOISTURE_THRESHOLD_PCT,
        ml_flag                = ml_flag,
        fao56                  = fao_obj,
        forecast_rain_mm       = rain,
        rain_threshold_mm      = RAIN_SUFFICIENT_MM,
        rain_sufficient        = rain_sufficient,
        reasons                = reasons,
        timestamp              = now_iso,
    )
