"""Météo-France open hourly station observations (meteo.data.gouv.fr).

Dataset: `donnees-climatologiques-de-base-horaires` (Etalab Licence Ouverte 2.0).
Streams department `.csv.gz` archives (`previous-2020-2024` and `latest-2025-2026`),
keeping only rows matching our target timestamps.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import io
import json
import logging
import re

import numpy as np
import pandas as pd

from .. import config
from .common import OBS_COLS, fetch_bytes, filter_bbox, wind_to_uv

log = logging.getLogger("regionalnext.stations.meteofrance")

# Metropolitan France departments (01..95 incl. 2A/2B -> 20 in Météo-France files)
_METRO_DEPTS = {f"{d:02d}" for d in range(1, 96)}
_FILE_RE = re.compile(r"H_(\d{2})_(previous-2020-2024|latest-2025-2026)\.csv\.gz$")
_USECOLS = ["NUM_POSTE", "LAT", "LON", "ALTI", "AAAAMMJJHH", "T", "QT", "TD", "QTD", "FF", "QFF", "DD", "QDD"]


def _discover_urls(years: set[int]) -> list[str]:
    raw = fetch_bytes(config.MF_DATASET_API, timeout=60)
    meta = json.loads(raw.decode("utf-8"))
    need_prev = any(2020 <= y <= 2024 for y in years)
    need_latest = any(y >= 2025 for y in years)

    urls = []
    for res in meta.get("resources", []):
        url = res.get("url", "")
        m = _FILE_RE.search(url)
        if not m:
            continue
        dept, period = m.group(1), m.group(2)
        if dept not in _METRO_DEPTS:
            continue
        if period.startswith("previous") and need_prev:
            urls.append(url)
        elif period.startswith("latest") and need_latest:
            urls.append(url)
    return sorted(set(urls))


def _fetch_one_file(url: str, target_keys: set[str]) -> pd.DataFrame | None:
    raw_gz = fetch_bytes(url, timeout=180)
    with gzip.GzipFile(fileobj=io.BytesIO(raw_gz)) as gz:
        df = pd.read_csv(
            gz,
            sep=";",
            usecols=lambda c: c in _USECOLS,
            dtype={"NUM_POSTE": str, "AAAAMMJJHH": str},
            low_memory=False,
        )
    if df.empty or "AAAAMMJJHH" not in df.columns:
        return None
    df = df[df["AAAAMMJJHH"].isin(target_keys)].copy()
    if df.empty:
        return None

    # Quality codes in Météo-France: 1=validated, 9=raw/auto valid, 2=doubtful/erroneous
    for val_col, q_col in [("T", "QT"), ("TD", "QTD"), ("FF", "QFF"), ("DD", "QDD")]:
        if val_col not in df.columns:
            df[val_col] = np.nan
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        if q_col in df.columns:
            q = pd.to_numeric(df[q_col], errors="coerce")
            df.loc[q.isin([2]), val_col] = np.nan

    df["lat"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["lon"] = pd.to_numeric(df["LON"], errors="coerce")
    df["elevation_m"] = pd.to_numeric(df["ALTI"], errors="coerce")
    df["timestamp_utc"] = pd.to_datetime(df["AAAAMMJJHH"], format="%Y%m%d%H", utc=True)

    spd = df["FF"].to_numpy(dtype=np.float32)
    wdir = df["DD"].to_numpy(dtype=np.float32)
    u10, v10 = wind_to_uv(spd, wdir)

    out = pd.DataFrame({
        "timestamp_utc": df["timestamp_utc"],
        "station_id": "MF_" + df["NUM_POSTE"].str.strip(),
        "source": "meteofrance",
        "lat": df["lat"].astype(np.float64),
        "lon": df["lon"].astype(np.float64),
        "elevation_m": df["elevation_m"].astype(np.float32),
        "t2m_c": df["T"].astype(np.float32),
        "td2m_c": df["TD"].astype(np.float32),
        "u10_ms": u10,
        "v10_ms": v10,
        "wind_speed_ms": spd,
        "wind_dir_deg": wdir,
    })
    return filter_bbox(out, config.LAT_MIN, config.LAT_MAX, config.LON_MIN, config.LON_MAX)


def fetch_meteofrance(timestamps: pd.Series, workers: int = 8) -> pd.DataFrame:
    ts_utc = pd.to_datetime(timestamps, utc=True)
    target_keys = set(ts_utc.dt.strftime("%Y%m%d%H"))
    years = set(ts_utc.dt.year.unique().tolist())

    urls = _discover_urls(years)
    log.info("Météo-France: %d department files to scan for %d timestamps", len(urls), len(target_keys))

    parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one_file, u, target_keys): u for u in urls}
        for i, fut in enumerate(as_completed(futs), 1):
            u = futs[fut]
            try:
                res = fut.result()
                if res is not None and not res.empty:
                    parts.append(res)
            except Exception as e:  # noqa: BLE001
                log.warning("Météo-France failed %s: %s", u, e)
            if i % 30 == 0 or i == len(urls):
                log.info("Météo-France progress: %d/%d files", i, len(urls))

    if not parts:
        return pd.DataFrame(columns=OBS_COLS)
    df = pd.concat(parts, ignore_index=True)
    log.info("Météo-France: %d obs rows across %d stations", len(df), df["station_id"].nunique())
    return df[OBS_COLS]
