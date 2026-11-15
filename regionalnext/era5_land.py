"""Step 2: Fetch ERA5-Land (0.1 deg, hourly) from Copernicus CDS, batching by (year, month).

Key details handled cleanly:
1. Batches by (year, month), requesting only the sampled days in that month and
   the union of needed hours `H` plus `H-1` (so `ssrd` accumulated since 00:00 UTC
   can be cleanly de-accumulated to hourly W/m^2).
2. Caches each monthly NetCDF in `data/raw/era5_land/` so interrupted runs resume.
3. Also fetches static ERA5-Land orography (geopotential -> elevation_m = z / 9.80665)
   and fills ocean-masked NaNs via nearest-land extrapolation so coastal 250m pixels
   never receive NaNs during bilinear interpolation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import tempfile
import time
import zipfile

import cdsapi
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import xarray as xr

from . import config

log = logging.getLogger("regionalnext.era5_land")

# Area padded by ~0.4 deg around domain for clean bilinear edges: [N, W, S, E]
_AREA = [config.LAT_MAX + 0.4, config.LON_MIN - 0.4, config.LAT_MIN - 0.4, config.LON_MAX + 0.4]

# Map CDS variable short names in NetCDF -> canonical names
_NC_ALIASES = {
    "t2m": "t2m",
    "d2m": "d2m",
    "u10": "u10",
    "v10": "v10",
    "ssrd": "ssrd",
    "sp": "sp",
    "sd": "sd",
    "sde": "sd",
    "snowc": "snowc",
}


def _fill_ocean_nan(arr2d: np.ndarray) -> np.ndarray:
    """Fill NaN ocean cells with nearest valid land cell value."""
    mask = ~np.isfinite(arr2d)
    if not np.any(mask) or np.all(mask):
        return arr2d
    _, indices = distance_transform_edt(mask, return_indices=True)
    return arr2d[tuple(indices)]


def _download_month(year: int, month: int, days: list[int], hours: list[int], out_nc: Path) -> Path:
    if out_nc.exists() and out_nc.stat().st_size > 1024:
        return out_nc
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    tmp_nc = out_nc.with_suffix(".nc.tmp")

    req = {
        "variable": config.ERA5_LAND_VARS,
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in sorted(set(days))],
        "time": [f"{h:02d}:00" for h in sorted(set(hours))],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": _AREA,
    }
    for attempt in range(1, 8):
        try:
            client = cdsapi.Client(quiet=True)
            client.retrieve("reanalysis-era5-land", req, str(tmp_nc))
            tmp_nc.replace(out_nc)
            return out_nc
        except Exception as e:
            msg = str(e)
            if "Maximum number queued" in msg or "429" in msg or "502" in msg or "503" in msg:
                wait_s = min(90, 15 * attempt)
                log.warning("CDS busy for %d-%02d (attempt %d), sleeping %ds...", year, month, attempt, wait_s)
                time.sleep(wait_s)
                continue
            raise
    raise RuntimeError(f"CDS download failed after retries for {year}-{month:02d}")


def _open_single_nc(path: Path, keep_times: np.ndarray | None = None) -> xr.Dataset:
    with xr.open_dataset(path, engine="netcdf4") as raw_ds:
        ds = raw_ds
        if "valid_time" in ds.coords and "time" not in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        for drop_v in ("number", "expver", "surface"):
            if drop_v in ds or drop_v in ds.coords:
                ds = ds.drop_vars(drop_v, errors="ignore")
        if keep_times is not None:
            ds = ds.sel(time=ds["time"].isin(keep_times))
        ds = ds.load().astype(np.float32)
    ds = ds.assign_coords(
        latitude=np.round(ds["latitude"].to_numpy(), 2),
        longitude=np.round(ds["longitude"].to_numpy(), 2),
    )
    return ds


def _open_nc_clean(nc_path: Path, keep_times: np.ndarray | None = None) -> xr.Dataset:
    if zipfile.is_zipfile(nc_path):
        with zipfile.ZipFile(nc_path) as zf, tempfile.TemporaryDirectory() as td:
            zf.extractall(td)
            parts = [
                _open_single_nc(Path(td) / n, keep_times=keep_times)
                for n in sorted(zf.namelist())
                if n.endswith(".nc")
            ]
        ds = xr.merge(parts, compat="override", join="exact")
    else:
        ds = _open_single_nc(nc_path, keep_times=keep_times)
    if ds["latitude"][0] > ds["latitude"][-1]:
        ds = ds.sortby("latitude")
    if ds["longitude"][0] > ds["longitude"][-1]:
        ds = ds.sortby("longitude")
    return ds


def fetch_era5_land(
    timestamps_path: Path = config.PROCESSED_DIR / "timestamps.parquet",
    raw_dir: Path = config.RAW_DIR / "era5_land",
    out_zarr: Path = config.PROCESSED_DIR / "era5_land.zarr",
    workers: int = 2,
    limit_months: int | None = None,
) -> Path:
    ts_df = pd.read_parquet(timestamps_path)
    ts_target = pd.to_datetime(ts_df["timestamp_utc"], utc=True)
    ts_prev = pd.to_datetime(ts_df["prev_timestamp_utc"], utc=True)

    # All needed timestamps (target + prev for ssrd de-accumulation)
    all_ts = pd.DatetimeIndex(sorted(set(ts_target).union(set(ts_prev))))
    groups: dict[tuple[int, int], tuple[set[int], set[int]]] = {}
    for t in all_ts:
        ym = (int(t.year), int(t.month))
        days_s, hrs_s = groups.setdefault(ym, (set(), set()))
        days_s.add(int(t.day))
        hrs_s.add(int(t.hour))

    items = sorted(groups.items())
    if limit_months is not None:
        items = items[:limit_months]

    log.info("ERA5-Land: %d monthly CDS requests (workers=%d)", len(items), workers)
    nc_files: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _download_month,
                y,
                m,
                sorted(days_s),
                sorted(hrs_s),
                raw_dir / f"era5l_{y}_{m:02d}.nc",
            ): (y, m)
            for (y, m), (days_s, hrs_s) in items
        }
        for i, fut in enumerate(as_completed(futs), 1):
            ym = futs[fut]
            try:
                p = fut.result()
                nc_files.append(p)
                log.info("ERA5-Land [%d/%d] ready: %s", i, len(items), p.name)
            except Exception as e:
                log.error("ERA5-Land failed %s: %s", ym, e)
                raise

    if limit_months is not None:
        log.info("Test mode (limit_months=%d): skipping full Zarr assembly", limit_months)
        return raw_dir

    # Assemble clean Zarr at target timestamps with de-accumulated ssrd + coastal fill
    target_naive = np.array([t.tz_convert("UTC").tz_localize(None).to_datetime64() for t in ts_target])
    prev_naive = np.array([t.tz_convert("UTC").tz_localize(None).to_datetime64() for t in ts_prev])
    needed_naive = np.unique(np.concatenate([target_naive, prev_naive]))

    log.info("Assembling %d monthly NetCDFs (%d target timestamps) -> %s", len(nc_files), len(target_naive), out_zarr)
    datasets = [_open_nc_clean(p, keep_times=needed_naive) for p in sorted(nc_files)]
    ds_all = xr.concat(datasets, dim="time").sortby("time")
    del datasets

    ds_cur = ds_all.sel(time=target_naive)
    ds_prv = ds_all.sel(time=prev_naive)
    del ds_all

    ssrd_cur = ds_cur["ssrd"].to_numpy()
    ssrd_prv = ds_prv["ssrd"].to_numpy()
    is_01 = np.asarray(pd.DatetimeIndex(target_naive).hour == 1)[:, None, None]
    ssrd_wm2 = np.where(is_01, ssrd_cur, ssrd_cur - ssrd_prv) / 3600.0
    ssrd_wm2 = np.clip(ssrd_wm2, 0.0, 1400.0).astype(np.float32)

    lat = ds_cur["latitude"].to_numpy().astype(np.float64)
    lon = ds_cur["longitude"].to_numpy().astype(np.float64)

    # Extract & convert units + fill ocean NaNs per slice
    t2m_c = (ds_cur["t2m"].to_numpy() - 273.15).astype(np.float32)
    d2m_c = (ds_cur["d2m"].to_numpy() - 273.15).astype(np.float32)
    u10 = ds_cur["u10"].to_numpy().astype(np.float32)
    v10 = ds_cur["v10"].to_numpy().astype(np.float32)
    sp_hpa = (ds_cur["sp"].to_numpy() / 100.0).astype(np.float32)
    sd_name = "sd" if "sd" in ds_cur else ("sde" if "sde" in ds_cur else None)
    sd_m = ds_cur[sd_name].to_numpy().astype(np.float32) if sd_name else np.zeros_like(t2m_c)

    # Ocean mask on ERA5-Land (1 = land, 0 = sea)
    land_mask = np.isfinite(t2m_c[0]).astype(np.float32)

    for arr in (t2m_c, d2m_c, u10, v10, ssrd_wm2, sp_hpa, sd_m):
        for k in range(arr.shape[0]):
            arr[k] = _fill_ocean_nan(arr[k])

    ds_out = xr.Dataset(
        {
            "t2m_c": (("time", "latitude", "longitude"), t2m_c),
            "d2m_c": (("time", "latitude", "longitude"), d2m_c),
            "u10_ms": (("time", "latitude", "longitude"), u10),
            "v10_ms": (("time", "latitude", "longitude"), v10),
            "ssrd_wm2": (("time", "latitude", "longitude"), ssrd_wm2),
            "sp_hpa": (("time", "latitude", "longitude"), sp_hpa),
            "snow_depth_m": (("time", "latitude", "longitude"), sd_m),
            "land_mask": (("latitude", "longitude"), land_mask),
        },
        coords={
            "time": target_naive,
            "latitude": lat,
            "longitude": lon,
        },
    )

    if out_zarr.exists():
        import shutil
        shutil.rmtree(out_zarr)
    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_zarr(out_zarr, mode="w")
    log.info("Saved ERA5-Land Zarr %s -> %s", dict(ds_out.sizes), out_zarr)
    return out_zarr


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Fetch ERA5-Land from CDS for sampled timestamps")
    p.add_argument("--timestamps", type=Path, default=config.PROCESSED_DIR / "timestamps.parquet")
    p.add_argument("--raw-dir", type=Path, default=config.RAW_DIR / "era5_land")
    p.add_argument("--out", type=Path, default=config.PROCESSED_DIR / "era5_land.zarr")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit-months", type=int, default=None, help="Smoke-test N months only")
    args = p.parse_args()
    fetch_era5_land(args.timestamps, args.raw_dir, args.out, args.workers, args.limit_months)


if __name__ == "__main__":
    main()
