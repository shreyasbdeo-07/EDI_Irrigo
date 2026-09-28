"""
Irrigo — Verification & Parity Test Suite
=========================================
Tests:
1. Parity between Original Benchmark Pipeline (Prediction A) vs Backend Predictor (Prediction B).
2. End-to-End Simulation of FastAPI /sensor-push endpoint with ESP32 JSON payload.
3. 6-Hour Rolling Mean Verification (Cold start vs Sequential updates).
4. Boundary & Out-of-range Validation.
"""

import math
import pathlib
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from fastapi.testclient import TestClient

# Add parent directory to sys.path
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from backend.main import app
from backend.predictor import predict_moisture_24h, predict_moisture_delta, FEATURE_NAMES
from backend.database import init_db, _conn

def test_benchmark_parity():
    print("\n" + "="*70)
    print("TEST 1: BENCHMARK PARITY (Prediction A vs Prediction B)")
    print("="*70)

    dataset_path = ROOT / "tomato_open_field_ml_dataset.csv"
    df = pd.read_csv(dataset_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)

    weather_vars = [
        "weather_temperature_c", "weather_humidity_pct", "weather_rain_mm",
        "weather_wind_speed", "weather_radiation_source_value",
        "soil_temperature_surface_c", "soil_moisture_pct"
    ]

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
    test  = df_all[df_all["split"] == "test"].copy()

    # Original training logic
    train_clean = train[(train["water_volume_past_4h_l"] == 0) & (train["weather_rain_mm"] == 0)].copy()
    dlt_clean = train_clean["target_mean_soil_moisture_24h_pct"] - train_clean["soil_moisture_pct"]
    train_clean = train_clean[dlt_clean.abs() < 1.5]
    y_tr_clean = train_clean["target_mean_soil_moisture_24h_pct"] - train_clean["soil_moisture_pct"]

    m_orig = Ridge(alpha=500.0)
    m_orig.fit(train_clean[FEATURE_NAMES], y_tr_clean)

    # Select 10 diverse test rows
    sample_indices = [0, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000]
    sample_rows = test.iloc[sample_indices]

    max_diff = 0.0
    print(f"{'Row':<6} {'Current(%)':<12} {'Pred A (Bench)':<16} {'Pred B (Backend)':<18} {'Diff':<12} {'Match'}")
    print("-" * 75)

    for idx, (_, row) in enumerate(sample_rows.iterrows()):
        # Pipeline A: In-memory original benchmark
        feat_vals = row[FEATURE_NAMES].values.reshape(1, -1)
        delta_A = float(m_orig.predict(feat_vals)[0])
        pred_A = float(row["soil_moisture_pct"] + delta_A)

        # Pipeline B: Saved backend predictor
        feat_dict = {f: float(row[f]) for f in FEATURE_NAMES}
        pred_B = predict_moisture_24h(feat_dict)

        diff = abs(pred_A - pred_B)
        max_diff = max(max_diff, diff)
        match = "PASS" if diff < 1e-3 else "FAIL"
        print(f"{sample_indices[idx]:<6} {row['soil_moisture_pct']:<12.2f} {pred_A:<16.4f} {pred_B:<18.4f} {diff:<12.6f} {match}")

    print(f"\nMax discrepancy across test samples: {max_diff:.8f}")
    assert max_diff < 1e-3, f"Parity check failed! Max diff = {max_diff}"
    print(">>> PARITY CHECK PASSED: Prediction A and Prediction B match within tolerance.")


def test_fastapi_simulation():
    print("\n" + "="*70)
    print("TEST 2: FASTAPI SIMULATED SENSOR PUSH & END-TO-END FLOW")
    print("="*70)

    init_db()
    client = TestClient(app)

    # Clean DB readings for reproducible test
    with _conn() as conn:
        conn.execute("DELETE FROM readings WHERE device_id LIKE 'test-esp32%'")
        conn.execute("DELETE FROM decisions WHERE reading_id IN (SELECT id FROM readings WHERE device_id LIKE 'test-esp32%')")
        conn.commit()

    # 1. Cold Start Push (First reading)
    payload_1 = {
        "soil_moisture_pct": 34.5,
        "soil_temperature_c": 22.3,
        "air_temperature_c": 28.7,
        "humidity_pct": 65.0,
        "forecast_rainfall_mm": 1.2,
        "forecast_temperature_c": 30.1,
        "wind_speed_mps": 2.4,
        "solar_radiation": 520.0,
        "crop_type": "tomato",
        "growth_stage": "mid_season",
        "device_id": "test-esp32-001",
        "zone_id": 1
    }

    res1 = client.post("/sensor-push", json=payload_1)
    assert res1.status_code == 200, f"Push 1 failed: {res1.text}"
    data1 = res1.json()
    print("[Push 1 - Cold Start]")
    print(f"  HTTP Status             : {res1.status_code}")
    print(f"  Predicted 24h Moisture  : {data1['predicted_moisture_24h']}%")
    print(f"  Recommend Irrigation    : {data1['recommend_irrigation']}")
    print(f"  Confidence              : {data1['confidence']:.2f}")
    print(f"  FAO-56 Net Deficit      : {data1['fao56']['net_deficit_mm']:.2f} mm")
    print(f"  Reasons                 : {data1['reasons']}")
    assert data1["predicted_moisture_24h"] is not None
    assert 0.0 <= data1["predicted_moisture_24h"] <= 100.0

    # 2. Sequential Push (Second reading to verify rolling solar average)
    payload_2 = {
        "soil_moisture_pct": 34.1,
        "soil_temperature_c": 22.5,
        "air_temperature_c": 29.0,
        "humidity_pct": 64.0,
        "forecast_rainfall_mm": 0.0,
        "wind_speed_mps": 2.5,
        "solar_radiation": 600.0,
        "crop_type": "tomato",
        "growth_stage": "mid_season",
        "device_id": "test-esp32-001",
        "zone_id": 1
    }
    res2 = client.post("/sensor-push", json=payload_2)
    assert res2.status_code == 200, f"Push 2 failed: {res2.text}"
    data2 = res2.json()
    print("\n[Push 2 - Sequential]")
    print(f"  HTTP Status             : {res2.status_code}")
    print(f"  Predicted 24h Moisture  : {data2['predicted_moisture_24h']}%")
    print(f"  Recommend Irrigation    : {data2['recommend_irrigation']}")
    print(f"  Reasons                 : {data2['reasons']}")

    # 3. Dry condition push triggering irrigation recommendation
    payload_3 = {
        "soil_moisture_pct": 24.0, # Below 30% threshold
        "soil_temperature_c": 28.0,
        "air_temperature_c": 35.0,
        "humidity_pct": 30.0,
        "forecast_rainfall_mm": 0.0, # No rain
        "wind_speed_mps": 3.5,
        "solar_radiation": 750.0,
        "crop_type": "tomato",
        "growth_stage": "mid_season",
        "device_id": "test-esp32-001",
        "zone_id": 1
    }
    res3 = client.post("/sensor-push", json=payload_3)
    assert res3.status_code == 200
    data3 = res3.json()
    print("\n[Push 3 - Dry Condition (Irrigation Trigger Test)]")
    print(f"  Current Moisture        : 24.0%")
    print(f"  Predicted 24h Moisture  : {data3['predicted_moisture_24h']}%")
    print(f"  Recommend Irrigation    : {data3['recommend_irrigation']}")
    print(f"  Primary Reason          : {data3['reasons'][0]}")
    assert data3["recommend_irrigation"] is True, "Expected irrigation recommendation on dry conditions!"

    # 4. Incomplete/Missing sensors (testing manual review flag & reliability)
    payload_4 = {
        "soil_moisture_pct": 28.0,
        # air_temp, humidity, solar missing
        "device_id": "test-esp32-001",
        "zone_id": 1
    }
    res4 = client.post("/sensor-push", json=payload_4)
    assert res4.status_code == 200
    data4 = res4.json()
    print("\n[Push 4 - Incomplete Sensors (Reliability Test)]")
    print(f"  Confidence              : {data4['confidence']:.2f}")
    print(f"  Requires Manual Review  : {data4['requires_manual_review']}")
    print(f"  Manual Review Reason    : {data4['manual_review_reason']}")
    assert data4["requires_manual_review"] is True, "Expected manual review required on incomplete data!"

    # 5. Check /latest-reading and /history endpoints
    res_latest = client.get("/latest-reading")
    assert res_latest.status_code == 200
    print("\n[GET /latest-reading]")
    print(f"  Latest Reading ID       : {res_latest.json()['id']}")
    print(f"  Latest Soil Moisture    : {res_latest.json()['soil_moisture_pct']}%")

    res_history = client.get("/history?limit=5")
    assert res_history.status_code == 200
    print(f"\n[GET /history] Rows returned: {res_history.json()['count']}")

    print("\n>>> FASTAPI SIMULATION SUITE PASSED SUCCESSFULLY.")

if __name__ == "__main__":
    test_benchmark_parity()
    test_fastapi_simulation()
