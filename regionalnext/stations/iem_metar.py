"""IEM ASOS/METAR observations for surrounding European borders (ES, IT, BE, LU, UK, AT, NL).

First fetches each border network's GeoJSON station list to filter strictly to
stations inside the domain bbox, then downloads 1 year per request (only ~6
requests total, avoiding 429 rate-limits) and caches each year locally.
"""
import io
import json
import logging
import time
import urllib.parse

import numpy as np
import pandas as pd

from .. import config
from .common import OBS_COLS, fetch_bytes, wind_to_uv

log = logging.getLogger("regionalnext.stations.iem")

_KNOTS_TO_MS = 0.514444
_BORDER_NETWORKS = [
    "ES__ASOS",
    "IT__ASOS",
    "BE__ASOS",
    "LU__ASOS",
    "GB__ASOS",
    "AT__ASOS",
    "NL__ASOS",
]


def _stations_in_bbox(network: str) -> list[str]:
    url = f"https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
    data = json.loads(fetch_bytes(url, timeout=60).decode("utf-8"))
    sids = []
    for feat in data.get("features", []):
        coords = feat.get("geometry", {}).get("coordinates", [])
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        if config.LAT_MIN <= lat <= config.LAT_MAX and config.LON_MIN <= lon <= config.LON_MAX:
            sid = feat.get("properties", {}).get("sid")
            if sid:
                sids.append(str(sid))
    return sorted(set(sids))


def _download_year_stations(stations: list[str], year: int) -> str:
    pairs = [("station", s) for s in stations] + [
        ("data", "tmpc"),
        ("data", "dwpc"),
        ("data", "sknt"),
        ("data", "drct"),
        ("tz", "Etc/UTC"),
        ("year1", str(year)),
        ("month1", "1"),
        ("day1", "1"),
        ("year2", str(year + 1)),
        ("month2", "1"),
        ("day2", "1"),
        ("report_type", "3"),
        ("format", "comma"),
        ("latlon", "yes"),
        ("elev", "yes"),
    ]
    url = f"{config.IEM_ASOS_URL}?{urllib.parse.urlencode(pairs)}"
    return fetch_bytes(url, timeout=300, retries=3, sleep_s=10.0).decode("utf-8", errors="replace")


def _parse_csv(text: str, target_ts: set[pd.Timestamp]) -> pd.DataFrame | None:
    if "Too many requests" in text[:200]:
        raise RuntimeError("IEM rate limited (429)")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#") and not ln.startswith("DEBUG")]
    if len(lines) < 2:
        return None
    df = pd.read_csv(io.StringIO("\n".join(lines)), low_memory=False, na_values=["M", "null", ""])
    if df.empty or "valid" not in df.columns:
        return None

    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    df = df[
        df["lat"].between(config.LAT_MIN, config.LAT_MAX)
        & df["lon"].between(config.LON_MIN, config.LON_MAX)
    ].copy()
    if df.empty:
        return None

    vt = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    snapped = vt.dt.round("h")
    diff_min = (vt - snapped).abs().dt.total_seconds() / 60.0
    df = df[diff_min <= 20.0].copy()
    df["timestamp_utc"] = snapped.loc[df.index]
    df["_abs_diff"] = diff_min.loc[df.index]
    df = df[df["timestamp_utc"].isin(target_ts)].copy()
    if df.empty:
        return None

    df = df.sort_values("_abs_diff").drop_duplicates(subset=["station", "timestamp_utc"], keep="first")

    t2m = pd.to_numeric(df.get("tmpc"), errors="coerce").to_numpy(dtype=np.float32)
    td2m = pd.to_numeric(df.get("dwpc"), errors="coerce").to_numpy(dtype=np.float32)
    sknt = pd.to_numeric(df.get("sknt"), errors="coerce").to_numpy(dtype=np.float32)
    spd = np.where(np.isfinite(sknt), sknt * _KNOTS_TO_MS, np.nan).astype(np.float32)
    wdir = pd.to_numeric(df.get("drct"), errors="coerce").to_numpy(dtype=np.float32)
    u10, v10 = wind_to_uv(spd, wdir)
    elev = pd.to_numeric(df.get("elevation"), errors="coerce").to_numpy(dtype=np.float32)

    return pd.DataFrame({
        "timestamp_utc": df["timestamp_utc"],
        "station_id": "METAR_" + df["station"].astype(str).str.strip(),
        "source": "iem_metar",
        "lat": df["lat"].astype(np.float64),
        "lon": df["lon"].astype(np.float64),
        "elevation_m": elev,
        "t2m_c": t2m,
        "td2m_c": td2m,
        "u10_ms": u10,
        "v10_ms": v10,
        "wind_speed_ms": spd,
        "wind_dir_deg": wdir,
    })


def fetch_iem_metar(timestamps: pd.Series) -> pd.DataFrame:
    ts_utc = pd.to_datetime(timestamps, utc=True)
    target_ts = set(ts_utc)
    years = sorted(ts_utc.dt.year.unique().tolist())

    all_sids: list[str] = []
    for net in _BORDER_NETWORKS:
        try:
            sids = _stations_in_bbox(net)
            log.info("IEM %s: %d stations in bbox", net, len(sids))
            all_sids.extend(sids)
        except Exception as e:  # noqa: BLE001
            log.warning("IEM geojson failed %s: %s", net, e)

    all_sids = sorted(set(all_sids))
    log.info("IEM METAR: %d border stations x %d years", len(all_sids), len(years))

    cache_dir = config.RAW_DIR / "stations" / "iem_years"
    cache_dir.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []
    for y in years:
        y_cache = cache_dir / f"iem_{y}.parquet"
        if y_cache.exists():
            log.info("IEM METAR: using cached %s", y_cache)
            parts.append(pd.read_parquet(y_cache))
            continue
        log.info("IEM METAR: downloading year %d...", y)
        txt = _download_year_stations(all_sids, y)
        res = _parse_csv(txt, target_ts)
        if res is not None and not res.empty:
            res.to_parquet(y_cache, index=False)
            parts.append(res)
            log.info("IEM METAR %d: %d obs rows", y, len(res))
        time.sleep(1.0)

    if not parts:
        return pd.DataFrame(columns=OBS_COLS)
    df = pd.concat(parts, ignore_index=True)
    log.info("IEM METAR: %d obs rows across %d stations", len(df), df["station_id"].nunique())
    return df[OBS_COLS]
