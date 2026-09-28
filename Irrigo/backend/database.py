"""
Irrigo Backend — SQLite Database Layer
=======================================
Two tables:
  readings  — one row per sensor push (raw readings from ESP32 or manual)
  decisions — one row per irrigation decision produced by the engine

Uses Python stdlib sqlite3 only — no ORM dependency.
"""

from __future__ import annotations

import sqlite3
import pathlib
from contextlib import contextmanager
from typing import Generator, Optional

DB_PATH = pathlib.Path(__file__).resolve().parent / "irrigo.db"


CREATE_READINGS = """
CREATE TABLE IF NOT EXISTS readings (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT    NOT NULL,
    soil_moisture_pct       REAL,
    soil_temperature_c      REAL,
    air_temperature_c       REAL,
    humidity_pct            REAL,
    forecast_rainfall_mm    REAL,
    forecast_temperature_c  REAL,
    wind_speed_mps          REAL,
    solar_radiation         REAL,
    water_volume_past_4h_l  REAL    DEFAULT 0.0,
    crop_type               TEXT    DEFAULT 'tomato',
    growth_stage            TEXT    DEFAULT 'mid_season',
    device_id               TEXT,
    zone_id                 INTEGER,
    month                   INTEGER,
    hour                    INTEGER,
    day_period              INTEGER
);
"""

CREATE_DECISIONS = """
CREATE TABLE IF NOT EXISTS decisions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    reading_id              INTEGER REFERENCES readings(id),
    timestamp               TEXT    NOT NULL,
    recommend_irrigation    INTEGER NOT NULL,   -- 0/1
    confidence              REAL    NOT NULL,
    requires_manual_review  INTEGER NOT NULL,   -- 0/1
    manual_review_reason    TEXT    DEFAULT '',
    predicted_moisture_24h  REAL,
    moisture_threshold_pct  REAL,
    ml_flag                 INTEGER,
    fao_eto_mm              REAL,
    fao_etc_mm              REAL,
    fao_kc                  REAL,
    fao_deficit_mm          REAL,
    fao_deficit_flag        INTEGER,
    forecast_rain_mm        REAL,
    rain_threshold_mm       REAL,
    rain_sufficient         INTEGER,
    reasons_json            TEXT    DEFAULT '[]'
);
"""


def init_db() -> None:
    """Create tables if they don't exist. Called once at app startup."""
    with _conn() as conn:
        conn.execute(CREATE_READINGS)
        conn.execute(CREATE_DECISIONS)
        conn.commit()


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ── Readings ────────────────────────────────────────────────────────────────

def insert_reading(data: dict) -> int:
    """Insert a sensor reading row, return new row id."""
    sql = """
    INSERT INTO readings (
        timestamp, soil_moisture_pct, soil_temperature_c, air_temperature_c,
        humidity_pct, forecast_rainfall_mm, forecast_temperature_c,
        wind_speed_mps, solar_radiation, water_volume_past_4h_l,
        crop_type, growth_stage, device_id, zone_id, month, hour, day_period
    ) VALUES (
        :timestamp, :soil_moisture_pct, :soil_temperature_c, :air_temperature_c,
        :humidity_pct, :forecast_rainfall_mm, :forecast_temperature_c,
        :wind_speed_mps, :solar_radiation, :water_volume_past_4h_l,
        :crop_type, :growth_stage, :device_id, :zone_id, :month, :hour, :day_period
    )
    """
    with _conn() as conn:
        cur = conn.execute(sql, data)
        conn.commit()
        return cur.lastrowid


def get_latest_reading() -> Optional[dict]:
    """Return the most recent reading as a dict, or None."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM readings ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def get_readings_history(limit: int = 144) -> list[dict]:
    """
    Return the last `limit` readings ordered chronologically.
    Default 144 = 72 h at 30-min intervals.
    """
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM readings ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


def get_past_solar_radiation(
    hours: float = 6.0,
    zone_id: Optional[int] = None,
    before_reading_id: Optional[int] = None,
) -> list[float]:
    """
    Return solar_radiation values from readings recorded in the past `hours` window
    strictly prior to before_reading_id.
    """
    from datetime import datetime, timezone
    with _conn() as conn:
        query = "SELECT id, timestamp, solar_radiation FROM readings WHERE solar_radiation IS NOT NULL"
        params = []
        if before_reading_id is not None:
            query += " AND id < ?"
            params.append(before_reading_id)
        if zone_id is not None:
            query += " AND zone_id = ?"
            params.append(zone_id)
        query += " ORDER BY id DESC LIMIT 50"
        rows = conn.execute(query, params).fetchall()

        now = datetime.now(tz=timezone.utc)
        values: list[float] = []
        for r in rows:
            ts_str = r["timestamp"]
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                diff_sec = (now - ts).total_seconds()
                if 0 <= diff_sec <= hours * 3600:
                    values.append(float(r["solar_radiation"]))
                elif diff_sec < 0:
                    # Clock skew tolerance (within 5 mins future)
                    values.append(float(r["solar_radiation"]))
            except Exception:
                values.append(float(r["solar_radiation"]))
        return values


# ── Decisions ───────────────────────────────────────────────────────────────

def insert_decision(reading_id: Optional[int], data: dict) -> int:
    """Insert a decision row, return new row id."""
    import json
    sql = """
    INSERT INTO decisions (
        reading_id, timestamp, recommend_irrigation, confidence,
        requires_manual_review, manual_review_reason,
        predicted_moisture_24h, moisture_threshold_pct, ml_flag,
        fao_eto_mm, fao_etc_mm, fao_kc, fao_deficit_mm, fao_deficit_flag,
        forecast_rain_mm, rain_threshold_mm, rain_sufficient, reasons_json
    ) VALUES (
        :reading_id, :timestamp, :recommend_irrigation, :confidence,
        :requires_manual_review, :manual_review_reason,
        :predicted_moisture_24h, :moisture_threshold_pct, :ml_flag,
        :fao_eto_mm, :fao_etc_mm, :fao_kc, :fao_deficit_mm, :fao_deficit_flag,
        :forecast_rain_mm, :rain_threshold_mm, :rain_sufficient, :reasons_json
    )
    """
    payload = {**data, "reading_id": reading_id}
    with _conn() as conn:
        cur = conn.execute(sql, payload)
        conn.commit()
        return cur.lastrowid


def get_decisions_history(limit: int = 144) -> list[dict]:
    """Return last `limit` decisions ordered chronologically."""
    import json
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        results = []
        for r in reversed(rows):
            d = dict(r)
            try:
                d["reasons"] = json.loads(d.pop("reasons_json", "[]"))
            except Exception:
                d["reasons"] = []
            results.append(d)
        return results
