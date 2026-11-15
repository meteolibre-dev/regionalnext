"""DWD CDC hourly weather station observations (air_temperature + wind).

Covers SW Germany, Black Forest, Rhine Valley, and Bavarian Alps (when lon_max >= 13E).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import logging
import re
import zipfile

import numpy as np
import pandas as pd

from .. import config
from .common import OBS_COLS, fetch_bytes, filter_bbox, rh_to_dewpoint_c, wind_to_uv

log = logging.getLogger("regionalnext.stations.dwd")

_STATION_LINE_RE = re.compile(
    r"^\s*(\d{5})\s+(\d{8})\s+(\d{8})\s+(-?\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+"
)


def _parse_station_table(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines()[2:]:
        m = _STATION_LINE_RE.match(line)
        if not m:
            continue
        rows.append({
            "dwd_id": m.group(1),
            "von": int(m.group(2)),
            "bis": int(m.group(3)),
            "elevation_m": float(m.group(4)),
            "lat": float(m.group(5)),
            "lon": float(m.group(6)),
        })
    return pd.DataFrame(rows)


def _index_zip_urls(dir_url: str, prefix: str) -> dict[str, list[str]]:
    html = fetch_bytes(dir_url, timeout=60).decode("utf-8", errors="replace")
    pat = re.compile(rf'href="({re.escape(prefix)}_(\d{{5}})[^"]*\.zip)"')
    out: dict[str, list[str]] = {}
    for fname, sid in pat.findall(html):
        out.setdefault(sid, []).append(dir_url + fname)
    return out


def _read_zip_csv(url: str, prod_prefix: str, target_keys: set[str]) -> pd.DataFrame | None:
    raw = fetch_bytes(url, timeout=90, retries=2)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = [n for n in zf.namelist() if n.startswith(prod_prefix) and n.endswith(".txt")]
        if not names:
            return None
        with zf.open(names[0]) as fh:
            df = pd.read_csv(
                fh,
                sep=";",
                skipinitialspace=True,
                dtype={"MESS_DATUM": str},
                na_values=["-999", "-999.0"],
            )
    df.columns = [c.strip() for c in df.columns]
    if "MESS_DATUM" not in df.columns:
        return None
    df["MESS_DATUM"] = df["MESS_DATUM"].astype(str).str.strip()
    df = df[df["MESS_DATUM"].isin(target_keys)].copy()
    return None if df.empty else df


def _fetch_one_station(
    sid: str,
    lat: float,
    lon: float,
    elev: float,
    tu_urls: list[str],
    ff_urls: list[str],
    target_keys: set[str],
) -> pd.DataFrame | None:
    tu_parts = []
    for u in tu_urls:
        try:
            d = _read_zip_csv(u, "produkt_tu_stunde_", target_keys)
            if d is not None:
                tu_parts.append(d)
        except Exception:  # noqa: BLE001
            pass
    if not tu_parts:
        return None
    tu = pd.concat(tu_parts, ignore_index=True).drop_duplicates(subset=["MESS_DATUM"], keep="last")

    t2m = pd.to_numeric(tu.get("TT_TU"), errors="coerce").to_numpy(dtype=np.float32)
    rh = pd.to_numeric(tu.get("RF_TU"), errors="coerce").to_numpy(dtype=np.float32)
    td2m = rh_to_dewpoint_c(t2m, rh)

    out = pd.DataFrame({
        "MESS_DATUM": tu["MESS_DATUM"],
        "t2m_c": t2m,
        "td2m_c": td2m,
    })

    if ff_urls:
        ff_parts = []
        for u in ff_urls:
            try:
                d = _read_zip_csv(u, "produkt_ff_stunde_", target_keys)
                if d is not None:
                    ff_parts.append(d)
            except Exception:  # noqa: BLE001
                pass
        if ff_parts:
            ff = pd.concat(ff_parts, ignore_index=True).drop_duplicates(subset=["MESS_DATUM"], keep="last")
            ff = ff[["MESS_DATUM", "F", "D"]].copy()
            ff["F"] = pd.to_numeric(ff["F"], errors="coerce")
            ff["D"] = pd.to_numeric(ff["D"], errors="coerce")
            out = out.merge(ff, on="MESS_DATUM", how="left")
        else:
            out["F"] = np.nan
            out["D"] = np.nan
    else:
        out["F"] = np.nan
        out["D"] = np.nan

    spd = out["F"].to_numpy(dtype=np.float32)
    wdir = out["D"].to_numpy(dtype=np.float32)
    u10, v10 = wind_to_uv(spd, wdir)

    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime(out["MESS_DATUM"], format="%Y%m%d%H", utc=True),
        "station_id": f"DWD_{sid}",
        "source": "dwd",
        "lat": float(lat),
        "lon": float(lon),
        "elevation_m": np.float32(elev),
        "t2m_c": out["t2m_c"].astype(np.float32),
        "td2m_c": out["td2m_c"].astype(np.float32),
        "u10_ms": u10,
        "v10_ms": v10,
        "wind_speed_ms": spd,
        "wind_dir_deg": wdir,
    })


def fetch_dwd(timestamps: pd.Series, workers: int = 8) -> pd.DataFrame:
    ts_utc = pd.to_datetime(timestamps, utc=True)
    target_keys = set(ts_utc.dt.strftime("%Y%m%d%H"))
    min_ymd = int(ts_utc.min().strftime("%Y%m%d"))
    max_ymd = int(ts_utc.max().strftime("%Y%m%d"))
    need_recent = max_ymd >= 20260101

    tu_hist_dir = f"{config.DWD_BASE_URL}/air_temperature/historical/"
    tu_rec_dir = f"{config.DWD_BASE_URL}/air_temperature/recent/"
    ff_hist_dir = f"{config.DWD_BASE_URL}/wind/historical/"
    ff_rec_dir = f"{config.DWD_BASE_URL}/wind/recent/"

    st_txt = fetch_bytes(tu_hist_dir + "TU_Stundenwerte_Beschreibung_Stationen.txt", timeout=60).decode(
        "latin1", errors="replace"
    )
    meta = _parse_station_table(st_txt)
    meta = filter_bbox(meta, config.LAT_MIN, config.LAT_MAX, config.LON_MIN, config.LON_MAX)
    meta = meta[(meta["bis"] >= min_ymd) & (meta["von"] <= max_ymd)].copy()

    tu_urls_map = _index_zip_urls(tu_hist_dir, "stundenwerte_TU")
    ff_urls_map = _index_zip_urls(ff_hist_dir, "stundenwerte_FF")
    if need_recent:
        for sid, u in _index_zip_urls(tu_rec_dir, "stundenwerte_TU").items():
            tu_urls_map.setdefault(sid, []).extend(u)
        for sid, u in _index_zip_urls(ff_rec_dir, "stundenwerte_FF").items():
            ff_urls_map.setdefault(sid, []).extend(u)

    log.info("DWD: %d active stations in bbox", len(meta))
    parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _fetch_one_station,
                str(r["dwd_id"]),
                float(r["lat"]),
                float(r["lon"]),
                float(r["elevation_m"]),
                tu_urls_map.get(str(r["dwd_id"]), []),
                ff_urls_map.get(str(r["dwd_id"]), []),
                target_keys,
            )
            for _, r in meta.iterrows()
            if tu_urls_map.get(str(r["dwd_id"]))
        ]
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None and not res.empty:
                parts.append(res)

    if not parts:
        return pd.DataFrame(columns=OBS_COLS)
    df = pd.concat(parts, ignore_index=True)
    log.info("DWD: %d obs rows across %d stations", len(df), df["station_id"].nunique())
    return df[OBS_COLS]
