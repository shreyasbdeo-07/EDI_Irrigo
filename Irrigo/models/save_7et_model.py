"""
Irrigo — Train & Export 7-Feature Physics-Informed Regularized Ridge Model
========================================================================
Preserves legacy rf_model.joblib and scaler.joblib.
Exports:
  - artifacts/physics_ridge_model.joblib
  - artifacts/physics_feature_list.json
  - artifacts/physics_metrics.json
"""

import json
import math
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATASET_PATH = ROOT / "tomato_open_field_ml_dataset.csv"
ARTIFACTS_DIR = HERE / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 7 physics features in exact order
FEATURES_7 = [
    "weather_temperature_c",
    "weather_humidity_pct",
    "soil_temperature_surface_c",
    "soil_moisture_pct",
    "weather_radiation_source_value",
    "weather_radiation_source_value_roll_mean_6h",
    "weather_wind_speed",
]

def train_and_save():
    print(f"[1/4] Loading dataset from {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)

    weather_vars = [
        "weather_temperature_c", "weather_humidity_pct", "weather_rain_mm",
        "weather_wind_speed", "weather_radiation_source_value",
        "soil_temperature_surface_c", "soil_moisture_pct"
    ]

    print("[2/4] Calculating 6h rolling mean of past solar radiation (shift=1) ...")
    zone_dfs = []
    for z in [1, 2]:
        zdf = df[df["zone_id"] == z].copy().set_index("timestamp")
        past_vals = zdf[weather_vars].shift(1)
        rad_6h = past_vals["weather_radiation_source_value"].rolling("6h", min_periods=1).mean()
        rad_6h.name = "weather_radiation_source_value_roll_mean_6h"
        zdf = pd.concat([zdf, rad_6h], axis=1)
        zone_dfs.append(zdf.reset_index())

    df_all = pd.concat(zone_dfs).sort_values(["zone_id", "timestamp"]).reset_index(drop=True)

    train = df_all[df_all["split"] == "train"].copy()
    val   = df_all[df_all["split"] == "validation"].copy()
    test  = df_all[df_all["split"] == "test"].copy()

    y_vl_true = val["target_mean_soil_moisture_24h_pct"].values
    y_ts_true = test["target_mean_soil_moisture_24h_pct"].values

    # Clean steady dry-down training periods
    train_clean = train[(train["water_volume_past_4h_l"] == 0) & (train["weather_rain_mm"] == 0)].copy()
    dlt_clean = train_clean["target_mean_soil_moisture_24h_pct"] - train_clean["soil_moisture_pct"]
    train_clean = train_clean[dlt_clean.abs() < 1.5]
    y_tr_clean = train_clean["target_mean_soil_moisture_24h_pct"] - train_clean["soil_moisture_pct"]

    print(f"[3/4] Fitting Ridge(alpha=500.0) on {len(train_clean)} clean training rows ...")
    model = Ridge(alpha=500.0)
    model.fit(train_clean[FEATURES_7], y_tr_clean)

    p_vl = val["soil_moisture_pct"].values + model.predict(val[FEATURES_7])
    p_ts = test["soil_moisture_pct"].values + model.predict(test[FEATURES_7])

    val_mae = float(mean_absolute_error(y_vl_true, p_vl))
    val_rmse = float(math.sqrt(mean_squared_error(y_vl_true, p_vl)))
    val_r2 = float(r2_score(y_vl_true, p_vl))

    test_mae = float(mean_absolute_error(y_ts_true, p_ts))
    test_rmse = float(math.sqrt(mean_squared_error(y_ts_true, p_ts)))
    test_r2 = float(r2_score(y_ts_true, p_ts))

    print("Validation Metrics: MAE={:.4f}, RMSE={:.4f}, R2={:.4f}".format(val_mae, val_rmse, val_r2))
    print("Test Metrics:       MAE={:.4f}, RMSE={:.4f}, R2={:.4f}".format(test_mae, test_rmse, test_r2))

    # Save artifacts
    model_path = ARTIFACTS_DIR / "physics_ridge_model.joblib"
    feature_path = ARTIFACTS_DIR / "physics_feature_list.json"
    metrics_path = ARTIFACTS_DIR / "physics_metrics.json"

    print(f"[4/4] Saving artifacts to {ARTIFACTS_DIR} ...")
    joblib.dump(model, model_path)
    with open(feature_path, "w") as f:
        json.dump(FEATURES_7, f, indent=2)

    metrics_data = {
        "model_type": "Physics-Informed Ridge(alpha=500.0)",
        "features": FEATURES_7,
        "coefficients": {f: float(c) for f, c in zip(FEATURES_7, model.coef_)},
        "intercept": float(model.intercept_),
        "validation": {"mae": val_mae, "rmse": val_rmse, "r2": val_r2},
        "test": {"mae": test_mae, "rmse": test_rmse, "r2": test_r2},
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print("Successfully exported physics model artifacts!")
    return model, metrics_data

if __name__ == "__main__":
    train_and_save()
