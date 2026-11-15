"""Generate 250m high-resolution visual comparisons (Paris, Alps, Amsterdam) with WN3-compat FiLM-UNet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from affine import Affine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
import torch
import xarray as xr

from . import config
from .build_patches import ALL_CHANNELS, DYN_CHANNELS, STATIC_CHANNELS
from .model import FiLMUNet
from .solar import solar_cond_features


def _bilinear_era(
    era_slice: np.ndarray,
    patch_lat: np.ndarray,
    patch_lon: np.ndarray,
    era_lat: np.ndarray,
    era_lon: np.ndarray,
) -> np.ndarray:
    lat0, dlat = float(era_lat[0]), float(era_lat[1] - era_lat[0])
    lon0, dlon = float(era_lon[0]), float(era_lon[1] - era_lon[0])
    eh, ew = len(era_lat), len(era_lon)

    fi = np.clip((patch_lat - lat0) / dlat, 0.0, float(eh - 1))
    fj = np.clip((patch_lon - lon0) / dlon, 0.0, float(ew - 1))
    i0 = np.minimum(fi.astype(np.int32), eh - 2)
    j0 = np.minimum(fj.astype(np.int32), ew - 2)
    di, dj = fi - i0, fj - j0

    return (
        era_slice[:, i0, j0] * ((1.0 - di) * (1.0 - dj))
        + era_slice[:, i0, j0 + 1] * ((1.0 - di) * dj)
        + era_slice[:, i0 + 1, j0] * (di * (1.0 - dj))
        + era_slice[:, i0 + 1, j0 + 1] * (di * dj)
    ).astype(np.float32)


def run_region_inference(
    model: FiLMUNet,
    ds_era: xr.Dataset,
    ds_stat: xr.Dataset,
    mean: np.ndarray,
    std: np.ndarray,
    ts_utc: pd.Timestamp,
    y0: int,
    x0: int,
    size: int = 512,
) -> dict[str, np.ndarray]:
    era_times = pd.to_datetime(ds_era["time"].to_numpy(), utc=True)
    era_idx = int(np.argmin(np.abs((era_times - ts_utc).total_seconds())))
    era_lat = ds_era["latitude"].to_numpy()
    era_lon = ds_era["longitude"].to_numpy()

    era_slice = np.stack(
        [ds_era[v].isel(time=era_idx).to_numpy() for v in DYN_CHANNELS], axis=0
    ).astype(np.float32)
    patch_lat = ds_stat["lat_2d"].isel(y=slice(y0, y0 + size), x=slice(x0, x0 + size)).to_numpy().astype(np.float32)
    patch_lon = ds_stat["lon_2d"].isel(y=slice(y0, y0 + size), x=slice(x0, x0 + size)).to_numpy().astype(np.float32)
    stat_patch = np.stack(
        [ds_stat[v].isel(y=slice(y0, y0 + size), x=slice(x0, x0 + size)).to_numpy() for v in STATIC_CHANNELS],
        axis=0,
    ).astype(np.float32)

    dyn_patch = _bilinear_era(era_slice, patch_lat, patch_lon, era_lat, era_lon)

    h_rad = 2.0 * np.pi * (ts_utc.hour / 24.0)
    d_rad = 2.0 * np.pi * ((ts_utc.day_of_year - 1.0) / 365.25)
    t_vec = np.array([np.sin(h_rad), np.cos(h_rad), np.sin(d_rad), np.cos(d_rad)], dtype=np.float32)
    t_patch = np.broadcast_to(t_vec[:, None, None], (4, size, size))

    raw = np.concatenate([dyn_patch, t_patch, stat_patch], axis=0)
    x_norm = (raw - mean[:, None, None]) / std[:, None, None]

    idx_t2m = ALL_CHANNELS.index("t2m_c")
    idx_u10 = ALL_CHANNELS.index("u10_ms")
    idx_v10 = ALL_CHANNELS.index("v10_ms")
    coarse = np.stack([raw[idx_t2m], raw[idx_u10], raw[idx_v10]], axis=0)

    center_lat = float(patch_lat[size // 2, size // 2])
    center_lon = float(patch_lon[size // 2, size // 2])
    ts_noon = ts_utc.normalize() + pd.Timedelta(hours=12)
    solar = solar_cond_features(ts_utc.timestamp(), ts_noon.timestamp(), center_lon, center_lat)

    with torch.no_grad():
        xt = torch.from_numpy(x_norm[None])
        ct = torch.from_numpy(coarse[None])
        st = torch.from_numpy(solar[None]) if model.use_solar_film else None
        pred = model(xt, ct, solar=st)[0].numpy()

    return {
        "elev": stat_patch[STATIC_CHANNELS.index("elevation_m")],
        "urban": stat_patch[STATIC_CHANNELS.index("frac_urban")],
        "water": stat_patch[STATIC_CHANNELS.index("frac_water")],
        "tree": stat_patch[STATIC_CHANNELS.index("frac_tree")],
        "ssrd": dyn_patch[DYN_CHANNELS.index("ssrd_wm2")],
        "coarse": coarse,
        "pred": pred,
    }


def _get_region_obs(
    stations: pd.DataFrame,
    obs: pd.DataFrame,
    t_affine: Affine,
    ts_utc: pd.Timestamp,
    y0: int,
    x0: int,
    size: int,
) -> pd.DataFrame:
    cols, rows = ~t_affine * (stations["x_3035"].to_numpy(), stations["y_3035"].to_numpy())
    st = stations.copy()
    st["ly"] = np.floor(rows).astype(int) - y0
    st["lx"] = np.floor(cols).astype(int) - x0
    st = st[(st["ly"] >= 0) & (st["ly"] < size) & (st["lx"] >= 0) & (st["lx"] < size)].copy()

    sub_obs = obs[obs["timestamp_utc"] == ts_utc]
    return st.merge(sub_obs, on="station_id", how="inner")


def _plot_context_panel(ax, fig, res: dict[str, np.ndarray], km_ext: list[float], show_landcover: bool) -> None:
    if show_landcover:
        # RGB composite: urban=red/orange, water=blue, tree=dark green, crop/grass=tan/olive
        r = np.clip(0.85 * res["urban"] + 0.55 * (1.0 - res["tree"] - res["water"] - res["urban"]), 0, 1)
        g = np.clip(0.45 * res["tree"] + 0.62 * (1.0 - res["tree"] - res["water"] - res["urban"]) + 0.25 * res["urban"], 0, 1)
        b = np.clip(0.90 * res["water"] + 0.25 * res["urban"] + 0.35 * (1.0 - res["tree"] - res["water"] - res["urban"]), 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
        ax.imshow(rgb, extent=km_ext)
        ax.contour(res["water"], levels=[0.5], colors="navy", linewidths=0.5, alpha=0.5, extent=km_ext, origin="upper")
        ax.set_title("ESA WorldCover 250m (Urban=Coral, Forest=Green, Water=Blue)", fontsize=10.5, fontweight="bold")
    else:
        elev = res["elev"]
        im0 = ax.imshow(elev, cmap="terrain", extent=km_ext, vmin=max(0.0, float(np.percentile(elev, 1))), vmax=float(np.percentile(elev, 99.5)))
        ax.contour(elev, levels=[1000, 2000, 3000], colors="k", linewidths=0.4, alpha=0.45, extent=km_ext, origin="upper")
        ax.set_title("Copernicus DEM 250m Elevation (m)", fontsize=11, fontweight="bold")
        fig.colorbar(im0, ax=ax, shrink=0.85, label="m")


def plot_t2m_case(
    res: dict[str, np.ndarray],
    st_obs: pd.DataFrame,
    title_prefix: str,
    out_path: Path,
    show_landcover: bool = False,
) -> None:
    elev = res["elev"]
    c_t2m = res["coarse"][0]
    p_t2m = res["pred"][0]
    diff = p_t2m - c_t2m
    km_ext = [0, elev.shape[1] * 0.25, elev.shape[0] * 0.25, 0]

    valid = st_obs[st_obs["t2m_c"].notna()].copy()
    if not valid.empty:
        ly, lx = valid["ly"].to_numpy(), valid["lx"].to_numpy()
        y_true = valid["t2m_c"].to_numpy()
        c_mae = float(np.mean(np.abs(c_t2m[ly, lx] - y_true)))
        p_mae = float(np.mean(np.abs(p_t2m[ly, lx] - y_true)))
        vmin = float(min(np.percentile(p_t2m, 1), np.percentile(c_t2m, 1), y_true.min()))
        vmax = float(max(np.percentile(p_t2m, 99), np.percentile(c_t2m, 99), y_true.max()))
        c_title = f"Coarse ERA5-Land (0.1° ≈ 9 km)\nStation MAE = {c_mae:.2f} °C (n={len(valid)})"
        pct = (1.0 - p_mae / max(c_mae, 1e-6)) * 100.0
        p_title = f"FiLM-UNet 250m Prediction (Obs Overlaid)\nStation MAE = {p_mae:.2f} °C ({-pct:+.0f}%)"
    else:
        vmin = float(min(np.percentile(p_t2m, 1), np.percentile(c_t2m, 1)))
        vmax = float(max(np.percentile(p_t2m, 99), np.percentile(c_t2m, 99)))
        c_title = "Coarse ERA5-Land (0.1° ≈ 9 km)"
        p_title = "FiLM-UNet 250m Prediction"

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.5), dpi=180, constrained_layout=True)

    _plot_context_panel(axes[0, 0], fig, res, km_ext, show_landcover)
    axes[0, 0].set_ylabel("km (N-S)")

    ax = axes[0, 1]
    im1 = ax.imshow(c_t2m, cmap="RdYlBu_r", extent=km_ext, vmin=vmin, vmax=vmax)
    ax.set_title(c_title, fontsize=11, fontweight="bold")
    fig.colorbar(im1, ax=ax, shrink=0.85, label="°C")

    ax = axes[1, 0]
    im2 = ax.imshow(p_t2m, cmap="RdYlBu_r", extent=km_ext, vmin=vmin, vmax=vmax)
    if not valid.empty:
        seen = valid[valid["spatial_fold"] != 4]
        held = valid[valid["spatial_fold"] == 4]
        if not seen.empty:
            ax.scatter(seen["lx"] * 0.25, seen["ly"] * 0.25, c=seen["t2m_c"], cmap="RdYlBu_r",
                       vmin=vmin, vmax=vmax, s=44, edgecolors="k", linewidths=0.85, marker="o", label="Seen stations", zorder=5)
        if not held.empty:
            ax.scatter(held["lx"] * 0.25, held["ly"] * 0.25, c=held["t2m_c"], cmap="RdYlBu_r",
                       vmin=vmin, vmax=vmax, s=60, edgecolors="white", linewidths=1.1, marker="^", label="Held-out stations", zorder=6)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.85)
    ax.set_title(p_title, fontsize=11, fontweight="bold")
    ax.set_xlabel("km (W-E)")
    ax.set_ylabel("km (N-S)")
    fig.colorbar(im2, ax=ax, shrink=0.85, label="°C")

    ax = axes[1, 1]
    dmax = max(1.5, float(np.percentile(np.abs(diff), 99.5)))
    im3 = ax.imshow(diff, cmap="RdBu_r", extent=km_ext, vmin=-dmax, vmax=dmax)
    if show_landcover:
        ax.contour(res["water"], levels=[0.5], colors="navy", linewidths=0.5, alpha=0.45, extent=km_ext, origin="upper")
        ax.contour(res["urban"], levels=[0.4], colors="black", linewidths=0.4, alpha=0.35, extent=km_ext, origin="upper")
    else:
        ax.contour(elev, levels=[1500, 2500, 3500], colors="k", linewidths=0.35, alpha=0.4, extent=km_ext, origin="upper")
    ax.set_title(f"Learned 250m Correction (Pred - Coarse)\nMean patch SSRD = {float(res['ssrd'].mean()):.0f} W/m²", fontsize=11, fontweight="bold")
    ax.set_xlabel("km (W-E)")
    fig.colorbar(im3, ax=ax, shrink=0.85, label="ΔT2m (°C)")

    fig.suptitle(title_prefix, fontsize=13.5, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_wind_case(
    res: dict[str, np.ndarray],
    st_obs: pd.DataFrame,
    title_prefix: str,
    out_path: Path,
    show_landcover: bool = False,
) -> None:
    elev = res["elev"]
    cu, cv = res["coarse"][1], res["coarse"][2]
    pu, pv = res["pred"][1], res["pred"][2]
    c_spd = np.sqrt(cu ** 2 + cv ** 2)
    p_spd = np.sqrt(pu ** 2 + pv ** 2)
    diff = p_spd - c_spd
    km_ext = [0, elev.shape[1] * 0.25, elev.shape[0] * 0.25, 0]

    valid = st_obs[st_obs["u10_ms"].notna() & st_obs["v10_ms"].notna()].copy()
    if not valid.empty:
        ly, lx = valid["ly"].to_numpy(), valid["lx"].to_numpy()
        yspd = np.sqrt(valid["u10_ms"].to_numpy() ** 2 + valid["v10_ms"].to_numpy() ** 2)
        c_mae = float(np.mean(np.abs(c_spd[ly, lx] - yspd)))
        p_mae = float(np.mean(np.abs(p_spd[ly, lx] - yspd)))
        vmax = float(max(np.percentile(p_spd, 99.5), np.percentile(c_spd, 99.5), yspd.max()))
        c_title = f"Coarse ERA5-Land 10m Wind\nStation Speed MAE = {c_mae:.2f} m/s (n={len(valid)})"
        pct = (1.0 - p_mae / max(c_mae, 1e-6)) * 100.0
        p_title = f"FiLM-UNet 250m Wind (Obs Overlaid)\nStation Speed MAE = {p_mae:.2f} m/s ({-pct:+.0f}%)"
    else:
        vmax = float(max(np.percentile(p_spd, 99.5), np.percentile(c_spd, 99.5)))
        c_title = "Coarse ERA5-Land 10m Wind"
        p_title = "FiLM-UNet 250m Wind"

    step = 24
    yy, xx = np.mgrid[step // 2 : elev.shape[0] : step, step // 2 : elev.shape[1] : step]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.5), dpi=180, constrained_layout=True)

    _plot_context_panel(axes[0, 0], fig, res, km_ext, show_landcover)
    axes[0, 0].set_ylabel("km (N-S)")

    ax = axes[0, 1]
    im1 = ax.imshow(c_spd, cmap="viridis", extent=km_ext, vmin=0, vmax=vmax)
    ax.quiver(xx * 0.25, yy * 0.25, cu[yy, xx], -cv[yy, xx], color="white", alpha=0.6, scale=140, width=0.0025)
    ax.set_title(c_title, fontsize=11, fontweight="bold")
    fig.colorbar(im1, ax=ax, shrink=0.85, label="m/s")

    ax = axes[1, 0]
    im2 = ax.imshow(p_spd, cmap="viridis", extent=km_ext, vmin=0, vmax=vmax)
    ax.quiver(xx * 0.25, yy * 0.25, pu[yy, xx], -pv[yy, xx], color="white", alpha=0.65, scale=140, width=0.0025)
    if not valid.empty:
        ax.scatter(lx * 0.25, ly * 0.25, c=yspd, cmap="viridis", vmin=0, vmax=vmax,
                   s=48, edgecolors="white", linewidths=1.0, zorder=5, label="Station speed")
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.85)
    ax.set_title(p_title, fontsize=11, fontweight="bold")
    ax.set_xlabel("km (W-E)")
    ax.set_ylabel("km (N-S)")
    fig.colorbar(im2, ax=ax, shrink=0.85, label="m/s")

    ax = axes[1, 1]
    dmax = max(1.5, float(np.percentile(np.abs(diff), 99)))
    im3 = ax.imshow(diff, cmap="PiYG", extent=km_ext, vmin=-dmax, vmax=dmax)
    if show_landcover:
        ax.contour(res["water"], levels=[0.5], colors="navy", linewidths=0.5, alpha=0.45, extent=km_ext, origin="upper")
    else:
        ax.contour(elev, levels=[1500, 2500, 3500], colors="k", linewidths=0.35, alpha=0.4, extent=km_ext, origin="upper")
    ax.set_title("Learned 250m Speedup / Sheltering (Pred - Coarse)", fontsize=11, fontweight="bold")
    ax.set_xlabel("km (W-E)")
    fig.colorbar(im3, ax=ax, shrink=0.85, label="ΔWind (m/s)")

    fig.suptitle(title_prefix, fontsize=13.5, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _center_box(ds_stat: xr.Dataset, t_affine: Affine, tr: Transformer, lon: float, lat: float, size: int) -> tuple[int, int]:
    cx, cy = tr.transform(lon, lat)
    col, row = ~t_affine * (cx, cy)
    y0 = max(0, min(ds_stat.sizes["y"] - size, int(row) - size // 2))
    x0 = max(0, min(ds_stat.sizes["x"] - size, int(col) - size // 2))
    return y0, x0


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize 250m FiLM-UNet predictions across regions")
    p.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--ckpt", type=Path, default=config.ROOT_DIR / "runs" / "film_unet_wn3" / "best.pt")
    p.add_argument("--out-dir", type=Path, default=config.ROOT_DIR / "outputs")
    p.add_argument("--size", type=int, default=512, help="Crop size in 250m pixels (512 = 128 km x 128 km)")
    args = p.parse_args()

    stats = json.loads((args.data_dir / "norm_stats.json").read_text())
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    base_ch = int(ckpt_args.get("base_ch", 64))
    cond_emb_dim = int(ckpt_args.get("cond_emb_dim", 128))
    wn3_compat = bool(ckpt_args.get("wn3_compat", False))
    use_solar_film = bool(ckpt_args.get("solar_film", False))
    model = FiLMUNet(
        base_ch=base_ch,
        cond_emb_dim=cond_emb_dim,
        wn3_compat=wn3_compat,
        use_solar_film=use_solar_film,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds_era = xr.open_zarr(args.data_dir / "era5_land.zarr")
    ds_stat = xr.open_zarr(args.data_dir / "static_250m.zarr")
    stations = pd.read_parquet(args.data_dir / "stations.parquet")
    obs = pd.read_parquet(args.data_dir / "observations.parquet")
    obs["timestamp_utc"] = pd.to_datetime(obs["timestamp_utc"], utc=True)

    t_affine = Affine(*ds_stat.attrs["transform"])
    tr = Transformer.from_crs("EPSG:4326", config.TARGET_CRS, always_xy=True)

    # 1. Paris region (Urban Heat Island night + afternoon heatwave)
    py0, px0 = _center_box(ds_stat, t_affine, tr, lon=2.35, lat=48.86, size=args.size)
    ts_paris_night = pd.Timestamp("2025-06-30 23:00:00", tz="UTC")
    plot_t2m_case(
        run_region_inference(model, ds_era, ds_stat, mean, std, ts_paris_night, py0, px0, args.size),
        _get_region_obs(stations, obs, t_affine, ts_paris_night, py0, px0, args.size),
        "Paris / Île-de-France (128x128 km @ 250m) — Test Set: 2025-06-30 23:00 UTC (Heatwave Night UHI)",
        args.out_dir / "paris_t2m_night_uhi.png",
        show_landcover=True,
    )
    ts_paris_day = pd.Timestamp("2025-07-01 14:00:00", tz="UTC")
    plot_t2m_case(
        run_region_inference(model, ds_era, ds_stat, mean, std, ts_paris_day, py0, px0, args.size),
        _get_region_obs(stations, obs, t_affine, ts_paris_day, py0, px0, args.size),
        "Paris / Île-de-France (128x128 km @ 250m) — Test Set: 2025-07-01 14:00 UTC (Heatwave Afternoon)",
        args.out_dir / "paris_t2m_afternoon.png",
        show_landcover=True,
    )

    # 2. Alpine region (Mont Blanc / Valais)
    ay0, ax0 = _center_box(ds_stat, t_affine, tr, lon=7.05, lat=45.95, size=args.size)
    ts_alp_sum = pd.Timestamp("2025-08-13 12:00:00", tz="UTC")
    plot_t2m_case(
        run_region_inference(model, ds_era, ds_stat, mean, std, ts_alp_sum, ay0, ax0, args.size),
        _get_region_obs(stations, obs, t_affine, ts_alp_sum, ay0, ax0, args.size),
        "Western Alps / Mont Blanc (128x128 km @ 250m, WN3-compat) — Test Set: 2025-08-13 12:00 UTC",
        args.out_dir / "alps_wn3_t2m_summer.png",
    )
    ts_alp_win = pd.Timestamp("2025-02-18 06:00:00", tz="UTC")
    plot_t2m_case(
        run_region_inference(model, ds_era, ds_stat, mean, std, ts_alp_win, ay0, ax0, args.size),
        _get_region_obs(stations, obs, t_affine, ts_alp_win, ay0, ax0, args.size),
        "Western Alps / Mont Blanc (128x128 km @ 250m, WN3-compat) — Test Set: 2025-02-18 06:00 UTC",
        args.out_dir / "alps_wn3_t2m_winter.png",
    )

    # 3. Amsterdam / Randstad / North Sea coast (top of grid y0=0)
    ny0, nx0 = _center_box(ds_stat, t_affine, tr, lon=4.90, lat=52.10, size=args.size)
    ts_ams = pd.Timestamp("2025-09-07 12:00:00", tz="UTC")
    res_ams = run_region_inference(model, ds_era, ds_stat, mean, std, ts_ams, ny0, nx0, args.size)
    obs_ams = _get_region_obs(stations, obs, t_affine, ts_ams, ny0, nx0, args.size)
    plot_t2m_case(
        res_ams,
        obs_ams,
        "Amsterdam / Randstad & North Sea (128x128 km @ 250m) — Test Set: 2025-09-07 12:00 UTC (T2m)",
        args.out_dir / "amsterdam_t2m.png",
        show_landcover=True,
    )
    plot_wind_case(
        res_ams,
        obs_ams,
        "Amsterdam / Randstad & North Sea (128x128 km @ 250m) — Test Set: 2025-09-07 12:00 UTC (10m Wind)",
        args.out_dir / "amsterdam_wind.png",
        show_landcover=True,
    )
    print("Saved 6 figures under", args.out_dir)


if __name__ == "__main__":
    main()
