"""Unified station gatherer: Météo-France + MeteoSwiss + DWD + IEM METAR.

Builds:
  data/processed/stations.parquet      (unique stations + EPSG:3035 x/y + spatial CV fold)
  data/processed/observations.parquet  (QC'd hourly observations at sampled timestamps)
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from .. import config
from .dwd import fetch_dwd
from .iem_metar import fetch_iem_metar
from .meteofrance import fetch_meteofrance
from .meteoswiss import fetch_meteoswiss

log = logging.getLogger("regionalnext.stations.gather")


def _qc_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic physical plausibility checks (mask invalid fields to NaN)."""
    df = df.copy()
    bad_t = (df["t2m_c"] < -45.0) | (df["t2m_c"] > 50.0)
    df.loc[bad_t, "t2m_c"] = np.nan

    bad_td = (df["td2m_c"] < -55.0) | (df["td2m_c"] > 38.0) | (df["td2m_c"] > df["t2m_c"] + 1.5)
    df.loc[bad_td, "td2m_c"] = np.nan

    bad_w = (df["wind_speed_ms"] < 0.0) | (df["wind_speed_ms"] > 75.0)
    df.loc[bad_w, ["wind_speed_ms", "wind_dir_deg", "u10_ms", "v10_ms"]] = np.nan

    # Require at least temperature OR wind
    keep = df["t2m_c"].notna() | df["wind_speed_ms"].notna()
    df = df[keep].copy()

    # Deduplicate (station_id, timestamp_utc)
    df = df.drop_duplicates(subset=["station_id", "timestamp_utc"], keep="first")
    return df


def _dedupe_collocated_stations(stations: pd.DataFrame, obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """If an IEM METAR station is within 500m of a national-network station (MF/MCH/DWD),
    prefer the national-network station (higher precision than whole-°C METAR).
    """
    priority = {"meteofrance": 0, "meteoswiss": 0, "dwd": 0, "iem_metar": 1}
    stations = stations.copy()
    stations["_prio"] = stations["source"].map(priority).fillna(2)
    stations = stations.sort_values(["_prio", "n_obs"], ascending=[True, False]).reset_index(drop=True)

    coords = stations[["x_3035", "y_3035"]].to_numpy()
    keep_mask = np.ones(len(stations), dtype=bool)
    for i in range(len(stations)):
        if not keep_mask[i] or stations.loc[i, "_prio"] == 0:
            continue
        # Compare IEM METAR against already-kept higher-priority stations within 750 m
        nat_idx = np.where((stations["_prio"].to_numpy() == 0) & keep_mask)[0]
        if len(nat_idx) == 0:
            continue
        d2 = np.sum((coords[nat_idx] - coords[i]) ** 2, axis=1)
        if np.any(d2 < 750.0 ** 2):
            keep_mask[i] = False

    dropped = int((~keep_mask).sum())
    if dropped:
        log.info("Dropped %d IEM METAR stations collocated (<750m) with MF/MCH/DWD stations", dropped)
    kept = stations[keep_mask].drop(columns=["_prio"]).reset_index(drop=True)
    obs = obs[obs["station_id"].isin(set(kept["station_id"]))].copy()
    return kept, obs


def build_stations_and_obs(
    timestamps_path: Path = config.PROCESSED_DIR / "timestamps.parquet",
    out_dir: Path = config.PROCESSED_DIR,
    sources: tuple[str, ...] = ("meteofrance", "meteoswiss", "dwd", "iem_metar"),
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_df = pd.read_parquet(timestamps_path)
    ts = ts_df["timestamp_utc"]

    fetchers = {
        "meteofrance": fetch_meteofrance,
        "meteoswiss": fetch_meteoswiss,
        "dwd": fetch_dwd,
        "iem_metar": fetch_iem_metar,
    }
    cache_dir = config.RAW_DIR / "stations"
    cache_dir.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []
    for src in sources:
        cache_path = cache_dir / f"obs_{src}.parquet"
        if cache_path.exists() and not force:
            log.info("Using cached %s -> %s", src, cache_path)
            parts.append(pd.read_parquet(cache_path))
        else:
            df_src = fetchers[src](ts)
            df_src.to_parquet(cache_path, index=False)
            parts.append(df_src)

    raw = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    obs = _qc_observations(raw)

    # Build unique station registry with EPSG:3035 coordinates
    agg = (
        obs.groupby("station_id", as_index=False)
        .agg(
            source=("source", "first"),
            lat=("lat", "median"),
            lon=("lon", "median"),
            elevation_m=("elevation_m", "median"),
            n_obs=("t2m_c", "count"),
            n_wind=("u10_ms", "count"),
        )
    )
    transformer = Transformer.from_crs("EPSG:4326", config.TARGET_CRS, always_xy=True)
    x_m, y_m = transformer.transform(agg["lon"].to_numpy(), agg["lat"].to_numpy())
    agg["x_3035"] = x_m
    agg["y_3035"] = y_m

    stations, obs = _dedupe_collocated_stations(agg, obs)

    # Spatial block CV fold (0..4) based on 64 km x 64 km coarse grid blocks
    bx = np.floor(stations["x_3035"] / (config.PATCH_SIZE * config.RESOLUTION_M)).astype(np.int64)
    by = np.floor(stations["y_3035"] / (config.PATCH_SIZE * config.RESOLUTION_M)).astype(np.int64)
    stations["spatial_fold"] = ((bx * 73856093) ^ (by * 19349663)) % 5

    obs_clean = obs[
        ["timestamp_utc", "station_id", "source", "t2m_c", "td2m_c", "u10_ms", "v10_ms", "wind_speed_ms", "wind_dir_deg"]
    ].sort_values(["timestamp_utc", "station_id"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    st_path = out_dir / "stations.parquet"
    ob_path = out_dir / "observations.parquet"
    stations.to_parquet(st_path, index=False)
    obs_clean.to_parquet(ob_path, index=False)
    log.info("Saved %d stations -> %s | %d obs -> %s", len(stations), st_path, len(obs_clean), ob_path)
    return stations, obs_clean


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Gather multi-source European station observations")
    p.add_argument("--timestamps", type=Path, default=config.PROCESSED_DIR / "timestamps.parquet")
    p.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument(
        "--sources",
        nargs="+",
        default=["meteofrance", "meteoswiss", "dwd", "iem_metar"],
        choices=["meteofrance", "meteoswiss", "dwd", "iem_metar"],
    )
    p.add_argument("--force", action="store_true", help="Ignore cached per-source parquets")
    args = p.parse_args()

    stations, obs = build_stations_and_obs(args.timestamps, args.out_dir, tuple(args.sources), force=args.force)
    print("\n=== Stations by source ===")
    print(stations.groupby("source").agg(n=("station_id", "count"), max_elev=("elevation_m", "max")))
    print("\n=== Stations by elevation band ===")
    bins = pd.cut(stations["elevation_m"], [-50, 200, 500, 1000, 1500, 2500, 5000])
    print(stations.groupby(bins, observed=False)["station_id"].count())
    print(f"\nTotal observations: {len(obs):,} (t2m: {obs['t2m_c'].notna().sum():,}, wind: {obs['u10_ms'].notna().sum():,})")


if __name__ == "__main__":
    main()
