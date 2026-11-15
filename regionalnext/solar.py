"""Vectorized suncalc solar position features (matching suncalc-py / suncalc.js)."""
from __future__ import annotations

import numpy as np

_RAD = np.pi / 180.0
_DAY_S = 86400.0
_J1970 = 2440588.0
_J2000 = 2451545.0
_OBLIQ = _RAD * 23.4397


def sun_position(unix_s: np.ndarray | float, lon_deg: np.ndarray | float, lat_deg: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return (azimuth_rad, altitude_rad) matching `suncalc.get_position`."""
    d = (np.asarray(unix_s, dtype=np.float64) / _DAY_S) - 0.5 + _J1970 - _J2000
    lw = _RAD * (-np.asarray(lon_deg, dtype=np.float64))
    phi = _RAD * np.asarray(lat_deg, dtype=np.float64)

    m = _RAD * (357.5291 + 0.98560028 * d)
    c = _RAD * (1.9148 * np.sin(m) + 0.02 * np.sin(2.0 * m) + 0.0003 * np.sin(3.0 * m))
    l = m + c + _RAD * 102.9372 + np.pi

    dec = np.arcsin(np.sin(_OBLIQ) * np.sin(l))
    ra = np.arctan2(np.sin(l) * np.cos(_OBLIQ), np.cos(l))
    h = _RAD * (280.16 + 360.9856235 * d) - lw - ra

    az = np.arctan2(np.sin(h), np.cos(h) * np.sin(phi) - np.tan(dec) * np.cos(phi))
    alt = np.arcsin(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(h))
    return az, alt


def solar_cond_features(
    unix_s: np.ndarray | float,
    noon_unix_s: np.ndarray | float,
    lon_deg: np.ndarray | float,
    lat_deg: np.ndarray | float,
) -> np.ndarray:
    """Return 5D continuous solar FiLM features: [sin(az), cos(az), sin(alt), sin(noon_alt), lat_norm]."""
    az, alt = sun_position(unix_s, lon_deg, lat_deg)
    _, noon_alt = sun_position(noon_unix_s, lon_deg, lat_deg)
    lat_norm = np.asarray(lat_deg, dtype=np.float64) / 45.0 - 1.0
    feats = np.stack([np.sin(az), np.cos(az), np.sin(alt), np.sin(noon_alt), lat_norm], axis=-1)
    return feats.astype(np.float32)
