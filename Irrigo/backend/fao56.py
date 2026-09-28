"""
Irrigo Backend — FAO-56 Penman-Monteith Module
================================================
Implements:
  - ETo (reference evapotranspiration) via Penman-Monteith (FAO-56, Eq. 6)
  - ETc = Kc × ETo using tomato crop coefficients (FAO-56 Table 12)
  - Water deficit = ETc − effective rainfall

References:
  Allen R.G. et al. (1998). Crop evapotranspiration — Guidelines for computing
  crop water requirements. FAO Irrigation and Drainage Paper 56. Rome.

Units throughout: mm/day, °C, m/s, MJ/m²/day, kPa
"""

from __future__ import annotations

import math
from typing import Optional

# ── Tomato crop coefficients (FAO-56 Table 12) ────────────────────────────
# Kc values for open-field tomato
TOMATO_KC: dict[str, float] = {
    "initial":     0.60,
    "development": 0.90,  # midpoint approximation for development stage
    "mid_season":  1.15,
    "late_season": 0.80,
}

# Minimum effective rainfall to consider "sufficient" to cover ETc deficit
RAIN_COVER_THRESHOLD_MM: float = 1.0  # mm/day


def saturation_vapour_pressure(T_c: float) -> float:
    """
    es (kPa) from air temperature T (°C).
    FAO-56 Eq. 11: es = 0.6108 × exp(17.27 T / (T + 237.3))
    """
    return 0.6108 * math.exp(17.27 * T_c / (T_c + 237.3))


def actual_vapour_pressure(T_c: float, rh_pct: float) -> float:
    """
    ea (kPa) from temperature and relative humidity.
    FAO-56 Eq. 17: ea = RH_mean/100 × es(T_mean)
    """
    return (rh_pct / 100.0) * saturation_vapour_pressure(T_c)


def psychrometric_constant(elev_m: float = 0.0) -> float:
    """
    γ (kPa/°C). FAO-56 Eq. 8.
    Atmospheric pressure P = 101.3 × ((293 - 0.0065 z)/293)^5.26 kPa
    γ = 0.000665 × P
    """
    P = 101.3 * ((293.0 - 0.0065 * elev_m) / 293.0) ** 5.26
    return 0.000665 * P


def slope_vapour_pressure_curve(T_c: float) -> float:
    """
    Δ (kPa/°C). FAO-56 Eq. 13.
    """
    return (4098.0 * saturation_vapour_pressure(T_c)) / ((T_c + 237.3) ** 2)


def net_radiation_from_solar(
    Rs_MJ: float,
    T_c: float,
    ea_kPa: float,
    doy: int,
    lat_deg: float,
    elev_m: float = 0.0,
) -> float:
    """
    Rn (MJ/m²/day) from incoming shortwave Rs.
    Rns = (1 - α) × Rs  with α = 0.23 (grass reference)
    Rnl estimated via FAO-56 Eq. 39.
    Ra estimated via FAO-56 Eq. 21.
    """
    # Net shortwave radiation
    alpha = 0.23
    Rns = (1.0 - alpha) * Rs_MJ

    # Extraterrestrial radiation Ra (FAO-56 Eq. 21)
    phi = math.radians(lat_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    delta = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    omega_s = math.acos(-math.tan(phi) * math.tan(delta))
    Ra = (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        omega_s * math.sin(phi) * math.sin(delta)
        + math.cos(phi) * math.cos(delta) * math.sin(omega_s)
    )

    # Clear-sky radiation Rso (FAO-56 Eq. 37)
    Rso = (0.75 + 2e-5 * elev_m) * Ra

    # Net longwave radiation Rnl (FAO-56 Eq. 39)
    T_K = T_c + 273.16
    ratio = min(Rs_MJ / Rso, 1.0) if Rso > 0 else 1.0
    Rnl = (4.903e-9 * T_K**4) * (0.34 - 0.14 * math.sqrt(ea_kPa)) * (
        1.35 * ratio - 0.35
    )

    return Rns - Rnl


def compute_eto(
    T_c: float,
    rh_pct: float,
    u2_mps: float,
    Rs_MJ: float,
    doy: int,
    lat_deg: float,
    elev_m: float = 0.0,
    G: float = 0.0,
) -> float:
    """
    FAO-56 Penman-Monteith ETo (mm/day).

    Parameters
    ----------
    T_c      : Mean daily air temperature (°C)
    rh_pct   : Mean relative humidity (%)
    u2_mps   : Wind speed at 2 m height (m/s)
    Rs_MJ    : Incoming solar radiation (MJ/m²/day)
               If given in W/m², divide by 11.574 first.
    doy      : Day of year (1–365)
    lat_deg  : Latitude (decimal degrees, positive N)
    elev_m   : Elevation above sea level (m)  [default 0]
    G        : Soil heat flux density (MJ/m²/day) [default 0, daily]
    """
    es  = saturation_vapour_pressure(T_c)
    ea  = actual_vapour_pressure(T_c, rh_pct)
    vpd = max(es - ea, 0.0)

    Delta = slope_vapour_pressure_curve(T_c)
    gamma = psychrometric_constant(elev_m)
    Rn    = net_radiation_from_solar(Rs_MJ, T_c, ea, doy, lat_deg, elev_m)

    # FAO-56 Eq. 6
    numerator = (0.408 * Delta * (Rn - G)
                 + gamma * (900.0 / (T_c + 273.0)) * u2_mps * vpd)
    denominator = Delta + gamma * (1.0 + 0.34 * u2_mps)

    eto = numerator / denominator if denominator > 0 else 0.0
    return max(eto, 0.0)


def get_kc(growth_stage: str) -> float:
    """Return FAO-56 Kc for the given growth stage key."""
    return TOMATO_KC.get(growth_stage.lower(), TOMATO_KC["mid_season"])


def compute_etc(eto: float, kc: float) -> float:
    """ETc = Kc × ETo (mm/day)."""
    return kc * eto


def water_deficit(
    etc_mm: float,
    rain_mm: float,
    deficit_threshold_mm: float = RAIN_COVER_THRESHOLD_MM,
) -> dict:
    """
    Compute net water deficit.

    Returns
    -------
    dict with keys:
      net_deficit_mm  : ETc − effective_rain  (positive = deficit)
      deficit_flag    : True if net_deficit > threshold
    """
    # Effective rainfall — simple coefficient (0.8 for field crops)
    effective_rain = rain_mm * 0.80
    net_deficit    = etc_mm - effective_rain
    return {
        "net_deficit_mm": round(net_deficit, 4),
        "deficit_flag":   net_deficit > deficit_threshold_mm,
    }


def run_fao56(
    T_c: float,
    rh_pct: float,
    u2_mps: float,
    Rs_MJ: float,
    doy: int,
    lat_deg: float,
    rain_mm: float,
    growth_stage: str,
    elev_m: float = 0.0,
) -> dict:
    """
    Convenience wrapper: compute ETo → ETc → deficit in one call.

    Returns a dict suitable for FAO56Result schema.
    """
    eto = compute_eto(T_c, rh_pct, u2_mps, Rs_MJ, doy, lat_deg, elev_m)
    kc  = get_kc(growth_stage)
    etc = compute_etc(eto, kc)
    deficit = water_deficit(etc, rain_mm)

    return {
        "eto_mm":         round(eto, 4),
        "etc_mm":         round(etc, 4),
        "kc":             round(kc,  4),
        "rainfall_mm":    round(rain_mm, 4),
        "net_deficit_mm": deficit["net_deficit_mm"],
        "deficit_flag":   deficit["deficit_flag"],
    }
