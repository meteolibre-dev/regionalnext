"""Ready-to-use Dataset loaders for RegionalNext 250m super-resolution.

Provides two dataset classes returning identical tensor dicts:
  1. `RegionalNextPatchDataset`: loads pre-sliced `hf_dataset/` (`patches_dense.zarr` +
     `patches_sparse.parquet`), e.g. 5,040 curated 256x256 patches (~29 GB).
  2. `RegionalNextFullGridDataset`: loads compact `full_grid/` (`static_250m.zarr` +
     `era5_land.zarr` + `observations.parquet`, ~1.31 GB total) and dynamically extracts
     256x256 patches across ALL 297 non-overlapping station tiles x 720 timestamps
     (~210k patches) or arbitrary random crops on the fly (<1 ms/patch).
"""
import json
from pathlib import Path

from affine import Affine
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from .build_patches import ALL_CHANNELS, DYN_CHANNELS, STATIC_CHANNELS
from .solar import solar_cond_features


class RegionalNextPatchDataset:
    """Loads pre-extracted (N, 256, 256, C) dense Zarr + COO sparse Parquet.

    Each item returns a dict of numpy arrays (convert cleanly with `torch.from_numpy`):
      x:        (C, 256, 256) float32 normalized inputs (or raw if normalize=False)
      y:        (3, 256, 256) float32 sparse targets [t2m_c, u10_ms, v10_ms] (0 where masked)
      mask:     (3, 256, 256) bool    True where station observation is valid for this split
      coarse:   (3, 256, 256) float32 unnormalized ERA5-Land [t2m_c, u10_ms, v10_ms] for residual learning
      patch_id: ()            int64   patch index
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        normalize: bool = True,
        eval_heldout_only: bool = False,
    ) -> None:
        root = Path(dataset_dir)
        self.zarr_root = zarr.open_group(str(root / "patches_dense.zarr"), mode="r")
        self.X = self.zarr_root["X"]
        self.channels = list(self.zarr_root.attrs["channels"])

        meta = pd.read_parquet(root / "patches_meta.parquet")
        if split in ("train", "val", "test"):
            meta = meta[meta["split_time"] == split]
        self.patch_ids = meta["patch_id"].to_numpy(dtype=np.int64)

        ts_m = pd.to_datetime(meta["timestamp_utc"], utc=True)
        ts_m_noon = ts_m.dt.normalize() + pd.Timedelta(hours=12)
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        solar_all = solar_cond_features(
            (ts_m - epoch).dt.total_seconds().to_numpy(),
            (ts_m_noon - epoch).dt.total_seconds().to_numpy(),
            meta["center_lon"].to_numpy(),
            meta["center_lat"].to_numpy(),
        )
        self.solar_by_idx = solar_all  # (N_split, 5)

        sparse = pd.read_parquet(root / "patches_sparse.parquet")
        sparse = sparse[sparse["patch_id"].isin(set(self.patch_ids))].copy()
        if split == "train":
            # Strictly exclude spatial_fold==4 stations from training supervision
            sparse = sparse[~sparse["is_heldout_station"]]
        elif eval_heldout_only:
            sparse = sparse[sparse["is_heldout_station"]]

        self.sparse_by_pid = {int(pid): grp for pid, grp in sparse.groupby("patch_id")}

        stats = json.loads((root / "norm_stats.json").read_text())
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)
        self.normalize = normalize

        self.idx_t2m = self.channels.index("t2m_c")
        self.idx_u10 = self.channels.index("u10_ms")
        self.idx_v10 = self.channels.index("v10_ms")

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        pid = int(self.patch_ids[idx])
        raw = np.asarray(self.X[pid], dtype=np.float32)  # (256, 256, C)
        coarse = np.stack(
            [raw[..., self.idx_t2m], raw[..., self.idx_u10], raw[..., self.idx_v10]],
            axis=0,
        )

        if self.normalize:
            x = ((raw - self.mean) / self.std).transpose(2, 0, 1)
        else:
            x = raw.transpose(2, 0, 1)

        y = np.zeros((3, 256, 256), dtype=np.float32)
        mask = np.zeros((3, 256, 256), dtype=bool)

        grp = self.sparse_by_pid.get(pid)
        if grp is not None and not grp.empty:
            ys = grp["y"].to_numpy(dtype=np.int64)
            xs = grp["x"].to_numpy(dtype=np.int64)
            mt = grp["mask_t2m"].to_numpy(dtype=bool)
            mw = grp["mask_wind"].to_numpy(dtype=bool)

            if np.any(mt):
                y[0, ys[mt], xs[mt]] = grp["t2m_c"].to_numpy(dtype=np.float32)[mt]
                mask[0, ys[mt], xs[mt]] = True
            if np.any(mw):
                y[1, ys[mw], xs[mw]] = grp["u10_ms"].to_numpy(dtype=np.float32)[mw]
                y[2, ys[mw], xs[mw]] = grp["v10_ms"].to_numpy(dtype=np.float32)[mw]
                mask[1, ys[mw], xs[mw]] = True
                mask[2, ys[mw], xs[mw]] = True

        return {
            "x": x,
            "y": y,
            "mask": mask,
            "coarse": coarse,
            "solar": self.solar_by_idx[idx],
            "patch_id": np.int64(pid),
        }


class RegionalNextFullGridDataset:
    """Full-domain (~1.31 GB) dataset covering all non-overlapping 256x256 tiles or random crops.

    Reads `static_250m.zarr`, `era5_land.zarr`, `stations.parquet`, `observations.parquet`,
    `timestamps.parquet`, and `norm_stats.json` from `full_grid_dir` (e.g. `data/processed`
    or downloaded `full_grid/` from Hugging Face).

    Modes:
      - `random_crop=False` (default): deterministic enumeration of all non-overlapping
        256x256 tiles (`stride=256`, or configurable `tile_stride`) that have >=10% land
        and >=1 valid station observation for that split & timestamp (~100k+ train samples,
        ~25k+ val/test samples).
      - `random_crop=True` (recommended for training augmentation): for each sample, picks a
        timestamp + valid station in the split and extracts a randomly jittered 256x256
        crop containing that station.
    """

    def __init__(
        self,
        full_grid_dir: str | Path,
        split: str = "train",
        normalize: bool = True,
        eval_heldout_only: bool = False,
        patch_size: int = 256,
        tile_stride: int = 256,
        random_crop: bool = False,
        preload: bool = True,
        seed: int = 42,
    ) -> None:
        root = Path(full_grid_dir)
        self.channels = list(ALL_CHANNELS)
        self.patch_size = patch_size
        self.normalize = normalize
        self.random_crop = random_crop
        self.rng = np.random.default_rng(seed)

        # Normalization stats (broadcastable to (C, 1, 1))
        stats = json.loads((root / "norm_stats.json").read_text())
        self.mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]

        # Open Zarr stores
        ds_stat = xr.open_zarr(root / "static_250m.zarr")
        ds_era = xr.open_zarr(root / "era5_land.zarr")
        self.ny = int(ds_stat.sizes["y"])
        self.nx = int(ds_stat.sizes["x"])
        self.t_affine = Affine(*ds_stat.attrs["transform"])

        # Uniform 0.1 deg ERA5-Land coordinate mapping for fast map_coordinates
        era_lat = ds_era["latitude"].to_numpy()
        era_lon = ds_era["longitude"].to_numpy()
        self.lat0 = float(era_lat[0])
        self.dlat = float(era_lat[1] - era_lat[0])
        self.lon0 = float(era_lon[0])
        self.dlon = float(era_lon[1] - era_lon[0])
        self.era_h = len(era_lat)
        self.era_w = len(era_lon)

        # Timestamps for split
        ts_df = pd.read_parquet(root / "timestamps.parquet")
        ts_df["timestamp_utc"] = pd.to_datetime(ts_df["timestamp_utc"], utc=True)
        era_times = pd.to_datetime(ds_era["time"].to_numpy(), utc=True)
        era_idx_map = {t: i for i, t in enumerate(era_times)}
        ts_df["era_idx"] = ts_df["timestamp_utc"].map(era_idx_map)
        if split in ("train", "val", "test"):
            ts_df = ts_df[ts_df["split_time"] == split].copy()
        ts_df = ts_df.dropna(subset=["era_idx"]).reset_index(drop=True)

        self.ts_era_idx = ts_df["era_idx"].to_numpy(dtype=np.int32)
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        ts_noon = ts_df["timestamp_utc"].dt.normalize() + pd.Timedelta(hours=12)
        self.ts_unix_s = (ts_df["timestamp_utc"] - epoch).dt.total_seconds().to_numpy(dtype=np.float64)
        self.ts_noon_unix_s = (ts_noon - epoch).dt.total_seconds().to_numpy(dtype=np.float64)

        hours = ts_df["timestamp_utc"].dt.hour.to_numpy(dtype=np.float32)
        doys = ts_df["timestamp_utc"].dt.day_of_year.to_numpy(dtype=np.float32)
        h_rad = 2.0 * np.pi * (hours / 24.0)
        d_rad = 2.0 * np.pi * ((doys - 1.0) / 365.25)
        self.time_vec = np.stack(
            [np.sin(h_rad), np.cos(h_rad), np.sin(d_rad), np.cos(d_rad)], axis=1
        ).astype(np.float32)  # (T_split, 4)

        # Preload arrays into RAM (~2.5 GB total) or keep lazy Zarr arrays
        self.preload = preload
        if preload:
            self.static_arr = np.stack(
                [ds_stat[v].to_numpy() for v in STATIC_CHANNELS], axis=0
            ).astype(np.float32)  # (16, ny, nx)
            self.lat_2d = ds_stat["lat_2d"].to_numpy().astype(np.float32)
            self.lon_2d = ds_stat["lon_2d"].to_numpy().astype(np.float32)
            self.era_arr = np.stack(
                [ds_era[v].to_numpy() for v in DYN_CHANNELS], axis=1
            ).astype(np.float32)  # (T_all, 7, H_era, W_era)
            land_mask = self.static_arr[STATIC_CHANNELS.index("land_mask")]
        else:
            self.ds_stat = ds_stat
            self.ds_era = ds_era
            self.lat_2d = ds_stat["lat_2d"].to_numpy().astype(np.float32)
            self.lon_2d = ds_stat["lon_2d"].to_numpy().astype(np.float32)
            land_mask = ds_stat["land_mask"].to_numpy()

        # Stations & pixel coordinates
        stations = pd.read_parquet(root / "stations.parquet")
        cols, rows = ~self.t_affine * (stations["x_3035"].to_numpy(), stations["y_3035"].to_numpy())
        stations["grid_y"] = np.floor(rows).astype(np.int32)
        stations["grid_x"] = np.floor(cols).astype(np.int32)
        stations = stations[
            (stations["grid_y"] >= 0) & (stations["grid_y"] < self.ny)
            & (stations["grid_x"] >= 0) & (stations["grid_x"] < self.nx)
        ].copy()

        if split == "train":
            stations = stations[stations["spatial_fold"] != 4]
        elif eval_heldout_only:
            stations = stations[stations["spatial_fold"] == 4]

        # Observations merged with station pixel coordinates for this split
        obs = pd.read_parquet(root / "observations.parquet")
        obs["timestamp_utc"] = pd.to_datetime(obs["timestamp_utc"], utc=True)
        ts_to_local_idx = {t: i for i, t in enumerate(ts_df["timestamp_utc"])}
        obs["ts_idx"] = obs["timestamp_utc"].map(ts_to_local_idx)
        obs = obs.dropna(subset=["ts_idx"]).copy()
        obs["ts_idx"] = obs["ts_idx"].astype(np.int32)

        obs = obs.merge(
            stations[["station_id", "grid_y", "grid_x"]],
            on="station_id",
            how="inner",
        )
        obs["mask_t2m"] = obs["t2m_c"].notna()
        obs["mask_wind"] = obs["u10_ms"].notna() & obs["v10_ms"].notna()
        obs = obs[obs["mask_t2m"] | obs["mask_wind"]].reset_index(drop=True)

        # Compact per-timestamp observation arrays for O(1) slice filtering
        self.obs_by_ts: dict[int, tuple[np.ndarray, ...]] = {}
        for t_idx, grp in obs.groupby("ts_idx"):
            self.obs_by_ts[int(t_idx)] = (
                grp["grid_y"].to_numpy(dtype=np.int32),
                grp["grid_x"].to_numpy(dtype=np.int32),
                grp["t2m_c"].fillna(0.0).to_numpy(dtype=np.float32),
                grp["u10_ms"].fillna(0.0).to_numpy(dtype=np.float32),
                grp["v10_ms"].fillna(0.0).to_numpy(dtype=np.float32),
                grp["mask_t2m"].to_numpy(dtype=bool),
                grp["mask_wind"].to_numpy(dtype=bool),
            )

        # Enumerate tiles & (ts_idx, y0, x0) index pairs
        tiles: list[tuple[int, int]] = []
        for y0 in range(0, self.ny - patch_size + 1, tile_stride):
            for x0 in range(0, self.nx - patch_size + 1, tile_stride):
                if float(land_mask[y0 : y0 + patch_size, x0 : x0 + patch_size].mean()) >= 0.1:
                    tiles.append((y0, x0))

        self.samples: list[tuple[int, int, int]] = []
        if random_crop:
            # One entry per (ts_idx, valid_obs_anchor); __getitem__ picks random jitter around it
            for t_idx, (gy, gx, *_) in self.obs_by_ts.items():
                # Sample up to 64 anchor points per timestamp or all unique tile cells
                for k in range(len(gy)):
                    self.samples.append((t_idx, int(gy[k]), int(gx[k])))
        else:
            tile_arr = np.asarray(tiles, dtype=np.int32)
            ty0, tx0 = tile_arr[:, 0], tile_arr[:, 1]
            for t_idx in sorted(self.obs_by_ts):
                gy, gx = self.obs_by_ts[t_idx][0], self.obs_by_ts[t_idx][1]
                # Vectorized check: which tiles contain >= 1 active obs at this timestamp
                in_y = (gy[None, :] >= ty0[:, None]) & (gy[None, :] < ty0[:, None] + patch_size)
                in_x = (gx[None, :] >= tx0[:, None]) & (gx[None, :] < tx0[:, None] + patch_size)
                active_tiles = np.where(np.any(in_y & in_x, axis=1))[0]
                for tid in active_tiles:
                    self.samples.append((t_idx, int(ty0[tid]), int(tx0[tid])))

        self.idx_t2m = self.channels.index("t2m_c")
        self.idx_u10 = self.channels.index("u10_ms")
        self.idx_v10 = self.channels.index("v10_ms")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        t_idx, a, b = self.samples[idx]
        p = self.patch_size

        if self.random_crop:
            # a, b are anchor station coordinates (gy, gx); pick random crop containing (a, b)
            y_min = max(0, a - p + 1)
            y_max = min(self.ny - p, a)
            x_min = max(0, b - p + 1)
            x_max = min(self.nx - p, b)
            y0 = int(self.rng.integers(y_min, y_max + 1))
            x0 = int(self.rng.integers(x_min, x_max + 1))
        else:
            y0, x0 = a, b

        era_idx = int(self.ts_era_idx[t_idx])

        # 1. Vectorized bilinear interpolation of 7 ERA5-Land channels onto (p, p)
        patch_lat = self.lat_2d[y0 : y0 + p, x0 : x0 + p]
        patch_lon = self.lon_2d[y0 : y0 + p, x0 : x0 + p]
        fi = np.clip((patch_lat - self.lat0) / self.dlat, 0.0, float(self.era_h - 1))
        fj = np.clip((patch_lon - self.lon0) / self.dlon, 0.0, float(self.era_w - 1))
        i0 = np.minimum(fi.astype(np.int32), self.era_h - 2)
        j0 = np.minimum(fj.astype(np.int32), self.era_w - 2)
        di = fi - i0
        dj = fj - j0
        w00 = (1.0 - di) * (1.0 - dj)
        w01 = (1.0 - di) * dj
        w10 = di * (1.0 - dj)
        w11 = di * dj

        if self.preload:
            era_slice = self.era_arr[era_idx]
            stat_patch = self.static_arr[:, y0 : y0 + p, x0 : x0 + p]
        else:
            era_slice = np.stack(
                [self.ds_era[v].isel(time=era_idx).to_numpy() for v in DYN_CHANNELS], axis=0
            ).astype(np.float32)
            stat_patch = np.stack(
                [self.ds_stat[v].isel(y=slice(y0, y0 + p), x=slice(x0, x0 + p)).to_numpy()
                 for v in STATIC_CHANNELS],
                axis=0,
            ).astype(np.float32)

        dyn_patch = (
            era_slice[:, i0, j0] * w00
            + era_slice[:, i0, j0 + 1] * w01
            + era_slice[:, i0 + 1, j0] * w10
            + era_slice[:, i0 + 1, j0 + 1] * w11
        ).astype(np.float32)

        # 2. Time channels broadcast to (4, p, p)
        t_patch = np.broadcast_to(self.time_vec[t_idx][:, None, None], (4, p, p))

        # 3. Assemble raw (27, p, p) and extract coarse [t2m, u10, v10]
        raw = np.concatenate([dyn_patch, t_patch, stat_patch], axis=0)
        coarse = np.stack(
            [raw[self.idx_t2m], raw[self.idx_u10], raw[self.idx_v10]],
            axis=0,
        )

        x = (raw - self.mean) / self.std if self.normalize else raw

        # 4. Sparse target + mask tensors
        y = np.zeros((3, p, p), dtype=np.float32)
        mask = np.zeros((3, p, p), dtype=bool)
        gy, gx, t2m, u10, v10, mt, mw = self.obs_by_ts[t_idx]
        in_box = (gy >= y0) & (gy < y0 + p) & (gx >= x0) & (gx < x0 + p)
        if np.any(in_box):
            ly = gy[in_box] - y0
            lx = gx[in_box] - x0
            mt_b = mt[in_box]
            mw_b = mw[in_box]
            if np.any(mt_b):
                y[0, ly[mt_b], lx[mt_b]] = t2m[in_box][mt_b]
                mask[0, ly[mt_b], lx[mt_b]] = True
            if np.any(mw_b):
                y[1, ly[mw_b], lx[mw_b]] = u10[in_box][mw_b]
                y[2, ly[mw_b], lx[mw_b]] = v10[in_box][mw_b]
                mask[1, ly[mw_b], lx[mw_b]] = True
                mask[2, ly[mw_b], lx[mw_b]] = True

        center_lat = float(patch_lat[p // 2, p // 2])
        center_lon = float(patch_lon[p // 2, p // 2])
        solar = solar_cond_features(
            self.ts_unix_s[t_idx],
            self.ts_noon_unix_s[t_idx],
            center_lon,
            center_lat,
        )

        return {
            "x": x,
            "y": y,
            "mask": mask,
            "coarse": coarse,
            "solar": solar,
            "patch_id": np.int64(idx),
        }
