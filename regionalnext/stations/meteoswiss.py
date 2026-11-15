"""MeteoSwiss Open Government Data (ch.meteoschweiz.ogd-smn).

Fetches station metadata + per-station hourly CSVs (`historical_2020-2029` and
`recent`) from data.geo.admin.ch. Rich in high-Alpine stations (>2000-3500m).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import logging

import numpy as np
import pandas as pd

from .. import config
from .common import OBS_COLS, fetch_bytes, filter_bbox, wind_to_uv

log = logging.getLogger("regionalnext.stations.meteoswiss")


def _fetch_station(
    abbr: str,
    lat: float,
    lon: float,
    elev: float,
    target_ts: set[pd.Timestamp],
    need_recent: bool,
) -> pd.DataFrame | None:
    abbr_l = abbr.lower()
    urls = [f"https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/{abbr_l}/ogd-smn_{abbr_l}_h_historical_2020-2029.csv"]
    if need_recent:
        urls.append(f"https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/{abbr_l}/ogd-smn_{abbr_l}_h_recent.csv")

    frames = []
    usecols = {"station_abbr", "reference_timestamp", "tre200h0", "tde200h0", "fkl010h0", "dkl010h0"}
    for url in urls:
        try:
            raw = fetch_bytes(url, timeout=60, retries=2)
        except Exception:  # noqa: BLE001
            continue
        df = pd.read_csv(
            io.BytesIO(raw),
            sep=";",
            usecols=lambda c: c in usecols,
            low_memory=False,
        )
        if df.empty or "reference_timestamp" not in df.columns:
            continue
        df["timestamp_utc"] = pd.to_datetime(
            df["reference_timestamp"], format="%d.%m.%Y %H:%M", utc=True, errors="coerce"
        )
        df = df[df["timestamp_utc"].isin(target_ts)]
        if not df.empty:
            frames.append(df)

    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp_utc"], keep="last")

    t2m = pd.to_numeric(df.get("tre200h0"), errors="coerce").to_numpy(dtype=np.float32)
    td2m = pd.to_numeric(df.get("tde200h0"), errors="coerce").to_numpy(dtype=np.float32)
    spd = pd.to_numeric(df.get("fkl010h0"), errors="coerce").to_numpy(dtype=np.float32)
    wdir = pd.to_numeric(df.get("dkl010h0"), errors="coerce").to_numpy(dtype=np.float32)
    u10, v10 = wind_to_uv(spd, wdir)

    return pd.DataFrame({
        "timestamp_utc": df["timestamp_utc"],
        "station_id": f"MCH_{abbr.upper()}",
        "source": "meteoswiss",
        "lat": float(lat),
        "lon": float(lon),
        "elevation_m": np.float32(elev),
        "t2m_c": t2m,
        "td2m_c": td2m,
        "u10_ms": u10,
        "v10_ms": v10,
        "wind_speed_ms": spd,
        "wind_dir_deg": wdir,
    })


def fetch_meteoswiss(timestamps: pd.Series, workers: int = 10) -> pd.DataFrame:
    ts_utc = pd.to_datetime(timestamps, utc=True)
    target_ts = set(ts_utc)
    need_recent = ts_utc.max() >= pd.Timestamp("2026-01-01", tz="UTC")

    raw_meta = fetch_bytes(config.METEOSWISS_META_URL, timeout=60)
    meta = pd.read_csv(io.BytesIO(raw_meta), sep=";", encoding="latin1")
    meta = meta.rename(columns={
        "station_coordinates_wgs84_lat": "lat",
        "station_coordinates_wgs84_lon": "lon",
        "station_height_masl": "elevation_m",
    })
    meta["lat"] = pd.to_numeric(meta["lat"], errors="coerce")
    meta["lon"] = pd.to_numeric(meta["lon"], errors="coerce")
    meta["elevation_m"] = pd.to_numeric(meta["elevation_m"], errors="coerce")
    meta = filter_bbox(meta, config.LAT_MIN, config.LAT_MAX, config.LON_MIN, config.LON_MAX)

    log.info("MeteoSwiss: %d stations in bbox", len(meta))
    parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _fetch_station,
                str(row["station_abbr"]),
                float(row["lat"]),
                float(row["lon"]),
                float(row["elevation_m"]),
                target_ts,
                need_recent,
            )
            for _, row in meta.iterrows()
        ]
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None and not res.empty:
                parts.append(res)

    if not parts:
        return pd.DataFrame(columns=OBS_COLS)
    df = pd.concat(parts, ignore_index=True)
    log.info("MeteoSwiss: %d obs rows across %d stations", len(df), df["station_id"].nunique())
    return df[OBS_COLS]
