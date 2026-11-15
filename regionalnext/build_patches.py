"""Step 4: Build ML-ready (N, 256, 256, C) Zarr + COO sparse Parquet dataset.

Outputs under `data/processed/hf_dataset/`:
  patches_dense.zarr/     array `X` of shape (N, 256, 256, C), float32, chunked (8, 256, 256, C)
  patches_sparse.parquet  COO sparse targets (patch_id, y, x, t2m_c, td2m_c, u10_ms, v10_ms, masks...)
  patches_meta.parquet    Per-patch metadata (timestamp, split_time, tile coords, elev stats, n_stations)
  norm_stats.json         Per-channel mean/std computed on training split only
"""
import argparse
import json
import logging
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from rasterio.transform import Affine
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import zarr
from zarr.codecs import BloscCodec

from . import config

log = logging.getLogger("regionalnext.build_patches")

DYN_CHANNELS = [
    "t2m_c",
    "d2m_c",
    "u10_ms",
    "v10_ms",
    "ssrd_wm2",
    "sp_hpa",
    "snow_depth_m",
]

TIME_CHANNELS = [
    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy",
]

STATIC_CHANNELS = [
    "elevation_m",
    "elev_std_m",
    "slope_deg",
    "sin_aspect",
    "cos_aspect",
    "tpi_500m",
    "tpi_2km",
    "tpi_10km",
    "dist_coast_km",
    "land_mask",
    "frac_tree",
    "frac_grass_shrub",
    "frac_crop",
    "frac_urban",
    "frac_bare_snow",
    "frac_water",
]

ALL_CHANNELS = DYN_CHANNELS + TIME_CHANNELS + STATIC_CHANNELS


def _enumerate_tiles(
    ds_stat: xr.Dataset,
    stations: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Generate candidate 256x256 tiles and map stations inside each tile."""
    ny, nx = ds_stat.sizes["y"], ds_stat.sizes["x"]
    ps = config.PATCH_SIZE
    stride = config.PATCH_STRIDE
    land_mask = ds_stat["land_mask"].to_numpy()
    elev = ds_stat["elevation_m"].to_numpy()

    st_r = stations["grid_y"].to_numpy()
    st_c = stations["grid_x"].to_numpy()

    tiles = []
    tile_stations: dict[int, pd.DataFrame] = {}
    tid = 0
    for y0 in range(0, ny - ps + 1, stride):
        for x0 in range(0, nx - ps + 1, stride):
            lm_patch = land_mask[y0 : y0 + ps, x0 : x0 + ps]
            land_frac = float(lm_patch.mean())
            if land_frac < 0.15:
                continue
            in_tile = (st_r >= y0) & (st_r < y0 + ps) & (st_c >= x0) & (st_c < x0 + ps)
            sub_st = stations[in_tile]
            mean_elev = float(elev[y0 : y0 + ps, x0 : x0 + ps][lm_patch > 0.5].mean())
            min_req = config.MIN_STATIONS_MOUNTAIN if mean_elev >= config.MOUNTAIN_ELEV_M else config.MIN_STATIONS_PER_PATCH
            if len(sub_st) < min_req:
                continue

            max_elev = float(elev[y0 : y0 + ps, x0 : x0 + ps].max())
            p90_elev = float(np.percentile(elev[y0 : y0 + ps, x0 : x0 + ps], 90))
            tiles.append({
                "tile_id": tid,
                "y0": y0,
                "x0": x0,
                "land_frac": land_frac,
                "mean_elev_m": mean_elev,
                "p90_elev_m": p90_elev,
                "max_elev_m": max_elev,
                "n_stations_total": len(sub_st),
                "is_mountain": (mean_elev >= 500.0) or (p90_elev >= 1000.0),
            })
            st_copy = sub_st.copy()
            st_copy["local_y"] = (st_copy["grid_y"] - y0).astype(np.int16)
            st_copy["local_x"] = (st_copy["grid_x"] - x0).astype(np.int16)
            tile_stations[tid] = st_copy
            tid += 1

    return pd.DataFrame(tiles), tile_stations


def _interp_era5_patch(
    era5_slice: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    patch_lat: np.ndarray,
    patch_lon: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate (C_dyn, H_era, W_era) onto (256, 256, C_dyn)."""
    c_dyn = era5_slice.shape[0]
    pts = np.stack([patch_lat.ravel(), patch_lon.ravel()], axis=-1)
    out = np.empty((patch_lat.shape[0], patch_lat.shape[1], c_dyn), dtype=np.float32)
    for c in range(c_dyn):
        rgi = RegularGridInterpolator(
            (lat_grid, lon_grid),
            era5_slice[c],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        out[..., c] = rgi(pts).reshape(patch_lat.shape).astype(np.float32)
    return out


def build_patches(
    era5_zarr: Path = config.PROCESSED_DIR / "era5_land.zarr",
    static_zarr: Path = config.PROCESSED_DIR / "static_250m.zarr",
    timestamps_path: Path = config.PROCESSED_DIR / "timestamps.parquet",
    stations_path: Path = config.PROCESSED_DIR / "stations.parquet",
    obs_path: Path = config.PROCESSED_DIR / "observations.parquet",
    out_dir: Path = config.PROCESSED_DIR / "hf_dataset",
    n_patches: int = 5000,
    seed: int = config.RANDOM_SEED,
) -> Path:
    rng = np.random.default_rng(seed)

    ds_era = xr.open_zarr(era5_zarr)
    ds_stat = xr.open_zarr(static_zarr)
    ts_df = pd.read_parquet(timestamps_path)
    stations = pd.read_parquet(stations_path)
    obs = pd.read_parquet(obs_path)

    # Map stations onto 250m pixel indices
    t_affine = Affine(*ds_stat.attrs["transform"])
    cols, rows = ~t_affine * (stations["x_3035"].to_numpy(), stations["y_3035"].to_numpy())
    stations["grid_y"] = np.floor(rows).astype(np.int32)
    stations["grid_x"] = np.floor(cols).astype(np.int32)
    ny, nx = ds_stat.sizes["y"], ds_stat.sizes["x"]
    stations = stations[
        (stations["grid_y"] >= 0) & (stations["grid_y"] < ny)
        & (stations["grid_x"] >= 0) & (stations["grid_x"] < nx)
    ].copy()

    # Record DEM elevation at station pixel
    elev_2d = ds_stat["elevation_m"].to_numpy()
    stations["dem_elev_m"] = elev_2d[stations["grid_y"].to_numpy(), stations["grid_x"].to_numpy()]

    tiles_df, tile_stations = _enumerate_tiles(ds_stat, stations)
    mtn_tiles = tiles_df[tiles_df["is_mountain"]]["tile_id"].to_numpy()
    plain_tiles = tiles_df[~tiles_df["is_mountain"]]["tile_id"].to_numpy()
    log.info("Candidate 256x256 tiles: %d total (%d mountain, %d plain/coastal)",
             len(tiles_df), len(mtn_tiles), len(plain_tiles))

    # Index observations by timestamp_utc for fast lookup
    obs["timestamp_utc"] = pd.to_datetime(obs["timestamp_utc"], utc=True)
    obs_by_ts = {ts: grp.set_index("station_id") for ts, grp in obs.groupby("timestamp_utc")}

    # Preload static 250m arrays into RAM (~2 GB total) for fast patch slicing
    log.info("Loading 250m static layers into memory...")
    static_stack = np.stack([ds_stat[v].to_numpy() for v in STATIC_CHANNELS], axis=-1).astype(np.float32)
    lat_2d = ds_stat["lat_2d"].to_numpy()
    lon_2d = ds_stat["lon_2d"].to_numpy()

    # Preload ERA5-Land arrays into RAM (~250 MB total)
    log.info("Loading ERA5-Land array into memory...")
    era_times = pd.to_datetime(ds_era["time"].to_numpy(), utc=True)
    era_time_idx = {t: i for i, t in enumerate(era_times)}
    era_lat = ds_era["latitude"].to_numpy()
    era_lon = ds_era["longitude"].to_numpy()
    era_stack = np.stack([ds_era[v].to_numpy() for v in DYN_CHANNELS], axis=1).astype(np.float32)

    # Select ~n_patches (timestamp, tile) pairs (~50% mountain, ~50% plain/coast)
    per_ts = max(1, round(n_patches / len(ts_df)))
    selected_pairs: list[tuple[int, pd.Timestamp, str, int]] = []
    ts_df["timestamp_utc"] = pd.to_datetime(ts_df["timestamp_utc"], utc=True)

    for _, ts_row in ts_df.iterrows():
        ts = ts_row["timestamp_utc"]
        if ts not in era_time_idx or ts not in obs_by_ts:
            continue
        obs_t = obs_by_ts[ts]
        active_sids = set(obs_t.index)

        n_mtn = min(len(mtn_tiles), (per_ts + 1) // 2)
        n_pln = min(len(plain_tiles), per_ts - n_mtn)

        # Oversample pool slightly so after station-count check we hit ~per_ts
        cand_mtn = rng.choice(mtn_tiles, size=min(len(mtn_tiles), n_mtn * 2), replace=False)
        cand_pln = rng.choice(plain_tiles, size=min(len(plain_tiles), n_pln * 2), replace=False)

        chosen_mtn, chosen_pln = [], []
        for pool, target_list, target_k in [(cand_mtn, chosen_mtn, n_mtn), (cand_pln, chosen_pln, n_pln)]:
            for tid in pool:
                st_sub = tile_stations[int(tid)]
                n_act = int(st_sub["station_id"].isin(active_sids).sum())
                min_req = config.MIN_STATIONS_MOUNTAIN if tiles_df.loc[tid, "mean_elev_m"] >= config.MOUNTAIN_ELEV_M else config.MIN_STATIONS_PER_PATCH
                if n_act >= min_req:
                    target_list.append(int(tid))
                    if len(target_list) >= target_k:
                        break

        for tid in chosen_mtn + chosen_pln:
            selected_pairs.append((era_time_idx[ts], ts, str(ts_row["split_time"]), tid))

    N = len(selected_pairs)
    C = len(ALL_CHANNELS)
    ps = config.PATCH_SIZE
    log.info("Building %d patches of shape (%d, %d, %d)...", N, ps, ps, C)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = out_dir / "patches_dense.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    X_arr = root.create_array(
        name="X",
        shape=(N, ps, ps, C),
        chunks=(8, ps, ps, C),
        dtype="float32",
        compressors=BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle"),
    )
    root.attrs["channels"] = ALL_CHANNELS
    root.attrs["dyn_channels"] = DYN_CHANNELS
    root.attrs["time_channels"] = TIME_CHANNELS
    root.attrs["static_channels"] = STATIC_CHANNELS
    root.attrs["patch_size"] = ps
    root.attrs["resolution_m"] = config.RESOLUTION_M
    root.attrs["crs"] = config.TARGET_CRS

    meta_rows = []
    sparse_rows = []

    # Running sums on training patches for normalization stats
    train_count = 0
    train_sum = np.zeros(C, dtype=np.float64)
    train_sum2 = np.zeros(C, dtype=np.float64)

    batch_buf = np.empty((8, ps, ps, C), dtype=np.float32)
    buf_start = 0

    for pid, (t_idx, ts, split_time, tid) in enumerate(selected_pairs):
        t_row = tiles_df.iloc[tid]
        y0, x0 = int(t_row["y0"]), int(t_row["x0"])

        p_lat = lat_2d[y0 : y0 + ps, x0 : x0 + ps]
        p_lon = lon_2d[y0 : y0 + ps, x0 : x0 + ps]

        dyn_patch = _interp_era5_patch(era_stack[t_idx], era_lat, era_lon, p_lat, p_lon)
        stat_patch = static_stack[y0 : y0 + ps, x0 : x0 + ps]

        hour_rad = 2.0 * np.pi * (ts.hour / 24.0)
        doy_rad = 2.0 * np.pi * ((ts.day_of_year - 1) / 365.25)
        time_vec = np.array([np.sin(hour_rad), np.cos(hour_rad), np.sin(doy_rad), np.cos(doy_rad)], dtype=np.float32)
        time_patch = np.broadcast_to(time_vec, (ps, ps, 4))

        patch = np.concatenate([dyn_patch, time_patch, stat_patch], axis=-1)
        b_slot = pid - buf_start
        batch_buf[b_slot] = patch
        if b_slot == 7 or pid == N - 1:
            n_write = b_slot + 1
            X_arr[buf_start : buf_start + n_write] = batch_buf[:n_write]
            buf_start += n_write

        if split_time == "train":
            flat = patch.reshape(-1, C).astype(np.float64)
            train_count += flat.shape[0]
            train_sum += flat.sum(axis=0)
            train_sum2 += (flat ** 2).sum(axis=0)

        # Sparse station targets inside this patch
        obs_t = obs_by_ts[ts]
        st_sub = tile_stations[tid]
        matched = st_sub[st_sub["station_id"].isin(obs_t.index)]
        n_obs_t2m = 0
        n_obs_wind = 0
        for _, s in matched.iterrows():
            sid = s["station_id"]
            o = obs_t.loc[sid]
            if isinstance(o, pd.DataFrame):
                o = o.iloc[0]
            ly, lx = int(s["local_y"]), int(s["local_x"])
            t2m_val = float(o["t2m_c"])
            td2m_val = float(o["td2m_c"])
            u_val = float(o["u10_ms"])
            v_val = float(o["v10_ms"])
            spd_val = float(o["wind_speed_ms"])

            m_t = np.isfinite(t2m_val)
            m_td = np.isfinite(td2m_val)
            m_w = np.isfinite(u_val) and np.isfinite(v_val)
            if not (m_t or m_w):
                continue
            n_obs_t2m += int(m_t)
            n_obs_wind += int(m_w)

            sparse_rows.append({
                "patch_id": pid,
                "y": ly,
                "x": lx,
                "station_id": sid,
                "source": str(s["source"]),
                "spatial_fold": int(s["spatial_fold"]),
                "is_heldout_station": bool(s["spatial_fold"] == 4),
                "station_elev_m": float(s["elevation_m"]),
                "dem_elev_m": float(s["dem_elev_m"]),
                "t2m_c": np.float32(t2m_val),
                "td2m_c": np.float32(td2m_val),
                "u10_ms": np.float32(u_val),
                "v10_ms": np.float32(v_val),
                "wind_speed_ms": np.float32(spd_val),
                "mask_t2m": bool(m_t),
                "mask_td2m": bool(m_td),
                "mask_wind": bool(m_w),
                "era5l_t2m_c": np.float32(dyn_patch[ly, lx, 0]),
                "era5l_u10_ms": np.float32(dyn_patch[ly, lx, 2]),
                "era5l_v10_ms": np.float32(dyn_patch[ly, lx, 3]),
            })

        meta_rows.append({
            "patch_id": pid,
            "timestamp_utc": ts,
            "split_time": split_time,
            "tile_id": tid,
            "y0": y0,
            "x0": x0,
            "center_lat": float(p_lat[ps // 2, ps // 2]),
            "center_lon": float(p_lon[ps // 2, ps // 2]),
            "mean_elev_m": float(t_row["mean_elev_m"]),
            "max_elev_m": float(t_row["max_elev_m"]),
            "is_mountain": bool(t_row["is_mountain"]),
            "n_obs_t2m": n_obs_t2m,
            "n_obs_wind": n_obs_wind,
        })

        if (pid + 1) % 500 == 0 or pid + 1 == N:
            log.info("Patches built: %d/%d", pid + 1, N)

    meta_df = pd.DataFrame(meta_rows)
    sparse_df = pd.DataFrame(sparse_rows)
    meta_df.to_parquet(out_dir / "patches_meta.parquet", index=False)
    sparse_df.to_parquet(out_dir / "patches_sparse.parquet", index=False)

    mean = train_sum / max(train_count, 1)
    var = np.maximum(1e-8, (train_sum2 / max(train_count, 1)) - mean ** 2)
    std = np.sqrt(var)
    stats = {
        "channels": ALL_CHANNELS,
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "n_train_pixels": int(train_count),
    }
    (out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2))

    log.info(
        "Done! Dense Zarr: (%d, %d, %d, %d) | Sparse rows: %d | Meta: %d -> %s",
        N, ps, ps, C, len(sparse_df), len(meta_df), out_dir,
    )
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build ML-ready (N, 256, 256, C) Zarr + sparse Parquet")
    p.add_argument("--n-patches", type=int, default=5000)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR / "hf_dataset")
    args = p.parse_args()
    build_patches(out_dir=args.out_dir, n_patches=args.n_patches, seed=args.seed)


if __name__ == "__main__":
    main()
