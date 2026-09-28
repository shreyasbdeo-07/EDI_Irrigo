"""
Irrigo Backend — ML Predictor
==============================
Loads the 7-Feature Physics-Informed Ridge model (alpha=500.0) at module import time.
Provides predict_moisture_delta() and predict_moisture_24h() functions used by the decision engine.

Features (7 physics-informed features):
  1. weather_temperature_c
  2. weather_humidity_pct
  3. soil_temperature_surface_c
  4. soil_moisture_pct
  5. weather_radiation_source_value
  6. weather_radiation_source_value_roll_mean_6h
  7. weather_wind_speed
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, Tuple

import joblib
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ARTIFACTS = (
    pathlib.Path(__file__).resolve().parent.parent  # Irrigo/
    / "models"
    / "artifacts"
)

MODEL_PATH   = ARTIFACTS / "physics_ridge_model.joblib"
FEATURE_PATH = ARTIFACTS / "physics_feature_list.json"


# ── Load at import ─────────────────────────────────────────────────────────
def _load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Physics model not found at {MODEL_PATH}. "
            "Run Irrigo/models/save_7et_model.py first."
        )
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH) as f:
        features = json.load(f)
    return model, features


_model, _features = _load_artifacts()

FEATURE_NAMES: list[str] = _features


def predict_moisture_delta(feature_dict: Dict[str, float]) -> float:
    """
    Predict 24-hour soil moisture change (delta, in percentage points).

    Parameters
    ----------
    feature_dict : dict mapping each feature name to its value.
                   Must include all entries in FEATURE_NAMES.

    Returns
    -------
    float — predicted delta (percentage points)
    """
    for f in FEATURE_NAMES:
        if f not in feature_dict:
            raise KeyError(f"Missing required feature for physics model: '{f}'")

    vec = np.array(
        [feature_dict[f] for f in FEATURE_NAMES], dtype=np.float64
    ).reshape(1, -1)

    delta = float(_model.predict(vec)[0])
    return float(round(delta, 4))


def predict_moisture_24h(feature_dict: Dict[str, float]) -> float:
    """
    Predict mean soil moisture 24 hours ahead (%):
      predicted_moisture_24h = current_soil_moisture_pct + predicted_delta
    Clamped to physical moisture range [0.0%, 100.0%].
    """
    current_moisture = float(feature_dict.get("soil_moisture_pct", 30.0))
    delta = predict_moisture_delta(feature_dict)
    predicted_moisture = max(0.0, min(100.0, current_moisture + delta))
    return float(round(predicted_moisture, 4))


def predict_both(feature_dict: Dict[str, float]) -> Tuple[float, float]:
    """
    Convenience function returning (predicted_moisture_24h, predicted_delta).
    """
    delta = predict_moisture_delta(feature_dict)
    current_moisture = float(feature_dict.get("soil_moisture_pct", 30.0))
    predicted_moisture = max(0.0, min(100.0, current_moisture + delta))
    return float(round(predicted_moisture, 4)), float(round(delta, 4))


def feature_names() -> list[str]:
    """Return the ordered list of feature names the model expects."""
    return list(_features)
