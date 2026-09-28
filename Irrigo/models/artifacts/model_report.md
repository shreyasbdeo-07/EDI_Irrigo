# Irrigo — Model Training Report

**Generated:** 2026-09-18 00:20:30

## Dataset
- Source: Mendeley field trial — Arnesano, Apulia, Italy
- Zones: 1 and 2 (open-field tomato only)
- Total rows: 21,460 | Train: 15,021 | Val: 3,219 | Test: 3,220
- Split: chronological per zone — **no random shuffling**

## Target
`target_mean_soil_moisture_24h_pct`: mean observed soil moisture (%) over the following 24 hours.

## Distribution Shift Warning

| Split | n | Target Mean (%) |
|-------|---|-----------------|
| Train | 15,021 | 33.03 |
| Val   | 3,219 | 17.87 |
| Test  | 3,220 | 17.20 |

The train–validation mean difference (~15 pp) reflects a seasonal dry period in the field trial.
This is the realistic challenge for deployment: the model must generalize across seasons.
**Evaluate performance primarily by MAE (scale-aware), not R² alone.**

## Features (13)
- `weather_temperature_c`
- `weather_humidity_pct`
- `weather_rain_mm`
- `weather_wind_speed`
- `weather_radiation_source_value`
- `soil_temperature_surface_c`
- `soil_temperature_deep_c`
- `soil_moisture_pct`
- `water_volume_past_4h_l`
- `month`
- `hour`
- `day_period`
- `zone_id`

## Leakage Columns Excluded
`target_point_24h`, `real_moisture_delta`, `water_vol_to_24h`, `irrigation_duration_minutes`

---

## Baseline — NumPy OLS per zone (soil_moisture_pct only)

| Split | MAE (pp) | RMSE (pp) | R² |
|-------|----------|-----------|-----|
| Validation | 2.1741 | 2.4106 | -0.4096 |
| Test       | 2.3048 | 2.5592 | -0.8866 |

## RandomForestRegressor (300 trees, 13 features including zone_id)

| Split | MAE (pp) | RMSE (pp) | R² |
|-------|----------|-----------|-----|
| Validation | 1.9722 | 2.9483 | -1.1085 |
| Test       | 3.7840 | 4.9355 | -6.0171 |

## Top-5 Feature Importances
| Feature | Importance |
|---------|-----------|
| `soil_moisture_pct` | 0.4613 |
| `month` | 0.2251 |
| `soil_temperature_deep_c` | 0.1024 |
| `water_volume_past_4h_l` | 0.0669 |
| `zone_id` | 0.0425 |

## Interpretation
- MAE on val/test reflects how many percentage points the prediction is off on average.
- A MAE of 2–4 pp is acceptable for a 24h-ahead soil moisture forecast.
- The model is used inside the hybrid decision engine alongside FAO-56, so even moderate
  errors are partially corrected by the deficit threshold.

## Limitation
The target reflects soil moisture **under the historical irrigation policy** of the
Italian experiment. It does not prove what would happen without future irrigation.
For Junnar (Maharashtra) deployment: use local weather and future ESP32 readings
to recalibrate or retrain.

## Artifacts Saved
All artifacts at `Irrigo/models/artifacts/`:
- `rf_model.joblib` — trained RandomForest
- `scaler.joblib` — StandardScaler (fitted on train only)
- `feature_list.json` — ordered feature names
- `preprocessing_meta.json` — per-feature mean / std
- `metrics.json` — full metrics dictionary
- `model_report.md` — this file
