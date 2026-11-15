"""Shared meteorological conversions, HTTP utilities, and canonical station schema."""
import logging
import time
import urllib.request

import numpy as np
import pandas as pd

log = logging.getLogger("regionalnext.stations")

UA = "regionalnext/0.1 (open weather super-resolution dataset)"

OBS_COLS = [
    "timestamp_utc",
    "station_id",
    "source",
    "lat",
    "lon",
    "elevation_m",
    "t2m_c",
    "td2m_c",
    "u10_ms",
    "v10_ms",
    "wind_speed_ms",
    "wind_dir_deg",
]


def wind_to_uv(speed_ms: np.ndarray | pd.Series, dir_deg: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Convert meteorological wind (speed m/s, direction FROM deg clockwise from N) to (u, v)."""
    spd = np.asarray(speed_ms, dtype=np.float64)
    ang = np.asarray(dir_deg, dtype=np.float64)
    valid = np.isfinite(spd) & np.isfinite(ang) & (spd >= 0) & (ang >= 0) & (ang <= 360)
    u = np.full_like(spd, np.nan, dtype=np.float32)
    v = np.full_like(spd, np.nan, dtype=np.float32)
    rad = np.deg2rad(ang[valid])
    u[valid] = (-spd[valid] * np.sin(rad)).astype(np.float32)
    v[valid] = (-spd[valid] * np.cos(rad)).astype(np.float32)
    return u, v


def rh_to_dewpoint_c(temp_c: np.ndarray | pd.Series, rh_pct: np.ndarray | pd.Series) -> np.ndarray:
    """Alduchov-Eskridge Magnus approximation for dewpoint (°C) from T (°C) and RH (%)."""
    t = np.asarray(temp_c, dtype=np.float64)
    rh = np.asarray(rh_pct, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(rh) & (rh > 0.5) & (rh <= 105.0)
    td = np.full_like(t, np.nan, dtype=np.float32)
    a, b = 17.625, 243.04
    rh_c = np.clip(rh[valid], 0.5, 100.0)
    gamma = np.log(rh_c / 100.0) + (a * t[valid]) / (b + t[valid])
    td[valid] = ((b * gamma) / (a - gamma)).astype(np.float32)
    return td


def fetch_bytes(url: str, timeout: int = 120, retries: int = 3, sleep_s: float = 2.0) -> bytes:
    """Download URL bytes with simple exponential backoff."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(sleep_s * attempt)
    raise RuntimeError(f"Failed GET {url}: {last_err}")


def filter_bbox(df: pd.DataFrame, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> pd.DataFrame:
    return df[
        df["lat"].between(lat_min, lat_max)
        & df["lon"].between(lon_min, lon_max)
        & df["elevation_m"].notna()
    ].copy()
