"""Step 3: Build 250m static terrain + land-cover Zarr over the Europe/France domain.

Sources (free, zero auth):
- Copernicus DEM GLO-90 (90m COG tiles on S3): elevation_m, sub-pixel elev_std_m,
  slope_deg, sin_aspect, cos_aspect, tpi_500m, tpi_2km, tpi_10km
- ESA WorldCover 2021 v200 (10m COG tiles on S3, read at 40m overview=4):
  frac_tree, frac_grass_shrub, frac_crop, frac_urban, frac_bare_snow, frac_water
- Derived: land_mask, dist_coast_km, lat_2d, lon_2d

Output: `data/processed/static_250m.zarr` on EPSG:3035 (250m x 250m metric pixels).
"""
import argparse
import logging
from pathlib import Path
import shutil

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
import requests
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter
import xarray as xr

from . import config

log = logging.getLogger("regionalnext.static_250m")


def compute_target_grid() -> tuple[Affine, int, int, np.ndarray, np.ndarray]:
    """Compute snapped 250m EPSG:3035 grid covering [LAT_MIN..LAT_MAX, LON_MIN..LON_MAX]."""
    tr = Transformer.from_crs("EPSG:4326", config.TARGET_CRS, always_xy=True)
    # Sample boundary densely so curved lat/lon borders stay inside
    lons = np.linspace(config.LON_MIN, config.LON_MAX, 50)
    lats = np.linspace(config.LAT_MIN, config.LAT_MAX, 50)
    gx, gy = np.meshgrid(lons, lats)
    xs, ys = tr.transform(gx.ravel(), gy.ravel())
    res = float(config.RESOLUTION_M)
    xmin = float(np.floor(xs.min() / res) * res)
    xmax = float(np.ceil(xs.max() / res) * res)
    ymin = float(np.floor(ys.min() / res) * res)
    ymax = float(np.ceil(ys.max() / res) * res)
    nx = int(round((xmax - xmin) / res))
    ny = int(round((ymax - ymin) / res))
    # North-up rasterio Affine (y decreases with row index)
    transform = Affine(res, 0.0, xmin, 0.0, -res, ymax)
    x_centers = xmin + (np.arange(nx) + 0.5) * res
    y_centers = ymax - (np.arange(ny) + 0.5) * res
    return transform, ny, nx, y_centers, x_centers


def _copernicus_dem_url(lat_sw: int, lon_sw: int) -> str:
    ns = f"N{lat_sw:02d}" if lat_sw >= 0 else f"S{abs(lat_sw):02d}"
    ew = f"E{lon_sw:03d}" if lon_sw >= 0 else f"W{abs(lon_sw):03d}"
    name = f"Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM"
    return f"https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"


def _worldcover_url(lat_sw: int, lon_sw: int) -> str:
    ns = f"N{lat_sw:02d}" if lat_sw >= 0 else f"S{abs(lat_sw):02d}"
    ew = f"E{lon_sw:03d}" if lon_sw >= 0 else f"W{abs(lon_sw):03d}"
    return f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map.tif"


def _url_exists(url: str, session: requests.Session) -> bool:
    try:
        r = session.head(url, timeout=15)
        return r.status_code == 200
    except Exception:
        return False


def _build_dem_250m(transform: Affine, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """Warp Copernicus DEM 90m tiles onto 250m EPSG:3035 -> (elev_mean, elev_std)."""
    sum_z = np.zeros((ny, nx), dtype=np.float64)
    sum_z2 = np.zeros((ny, nx), dtype=np.float64)
    cnt = np.zeros((ny, nx), dtype=np.float64)

    lat_range = range(int(np.floor(config.LAT_MIN)) - 1, int(np.ceil(config.LAT_MAX)) + 1)
    lon_range = range(int(np.floor(config.LON_MIN)) - 1, int(np.ceil(config.LON_MAX)) + 1)
    tiles = [(la, lo) for la in lat_range for lo in lon_range]

    session = requests.Session()
    valid_urls = [u for la, lo in tiles if _url_exists(u := _copernicus_dem_url(la, lo), session)]
    log.info("Copernicus DEM GLO-90: %d land tiles found", len(valid_urls))

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        for idx, url in enumerate(valid_urls, 1):
            with rasterio.open(f"/vsicurl/{url}") as src:
                z = src.read(1).astype(np.float32)
                valid = np.isfinite(z) & (z > -500.0)
                z_clean = np.where(valid, z, 0.0).astype(np.float32)
                z2_clean = (z_clean ** 2).astype(np.float32)
                w = valid.astype(np.float32)

                dst_z = np.full((ny, nx), np.nan, dtype=np.float32)
                dst_z2 = np.full((ny, nx), np.nan, dtype=np.float32)
                dst_w = np.full((ny, nx), np.nan, dtype=np.float32)

                kw = dict(src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs=config.TARGET_CRS,
                          resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
                reproject(source=z_clean, destination=dst_z, **kw)
                reproject(source=z2_clean, destination=dst_z2, **kw)
                reproject(source=w, destination=dst_w, **kw)

                hit = np.isfinite(dst_w) & (dst_w > 0.05)
                sum_z[hit] += dst_z[hit]
                sum_z2[hit] += dst_z2[hit]
                cnt[hit] += dst_w[hit]
            if idx % 30 == 0 or idx == len(valid_urls):
                log.info("DEM progress: %d/%d tiles", idx, len(valid_urls))

    elev = np.where(cnt > 0, sum_z / np.maximum(cnt, 1e-6), 0.0).astype(np.float32)
    ez2 = np.where(cnt > 0, sum_z2 / np.maximum(cnt, 1e-6), 0.0)
    var = np.maximum(0.0, ez2 - elev.astype(np.float64) ** 2)
    elev_std = np.sqrt(var).astype(np.float32)
    return elev, elev_std


def _build_worldcover_250m(transform: Affine, ny: int, nx: int) -> dict[str, np.ndarray]:
    """Warp ESA WorldCover 2021 (40m overview=4) into 6 fractional land-cover layers."""
    # WorldCover classes:
    # 10=tree, 20=shrub, 30=grass, 40=crop, 50=urban, 60=bare, 70=snow/ice,
    # 80=water, 90=wetland, 95=mangrove, 100=moss/lichen
    groups = {
        "frac_tree": (10,),
        "frac_grass_shrub": (20, 30, 90, 100),
        "frac_crop": (40,),
        "frac_urban": (50,),
        "frac_bare_snow": (60, 70),
        "frac_water": (80,),
    }
    acc = {k: np.zeros((ny, nx), dtype=np.float64) for k in groups}
    weight_sum = np.zeros((ny, nx), dtype=np.float64)

    lat_starts = range(3 * (int(np.floor(config.LAT_MIN)) // 3), int(np.ceil(config.LAT_MAX)) + 1, 3)
    lon_starts = range(3 * (int(np.floor(config.LON_MIN)) // 3), int(np.ceil(config.LON_MAX)) + 1, 3)
    session = requests.Session()
    valid_urls = [
        u for la in lat_starts for lo in lon_starts if _url_exists(u := _worldcover_url(la, lo), session)
    ]
    log.info("ESA WorldCover 2021: %d tiles found", len(valid_urls))

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        for idx, url in enumerate(valid_urls, 1):
            with rasterio.open(f"/vsicurl/{url}") as src:
                # Overview level index 1 = 4x (40m pixels -> 9000x9000)
                ov_factor = 4
                out_h, out_w = src.height // ov_factor, src.width // ov_factor
                lc = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
                ov_transform = src.transform * Affine.scale(ov_factor, ov_factor)

                valid = (lc > 0).astype(np.float32)
                dst_w = np.full((ny, nx), np.nan, dtype=np.float32)
                reproject(
                    source=valid,
                    destination=dst_w,
                    src_transform=ov_transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=config.TARGET_CRS,
                    resampling=Resampling.average,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                )
                hit = np.isfinite(dst_w) & (dst_w > 0.05)
                weight_sum[hit] += dst_w[hit]

                for k, codes in groups.items():
                    mask = np.isin(lc, codes).astype(np.float32)
                    dst_m = np.full((ny, nx), np.nan, dtype=np.float32)
                    reproject(
                        source=mask,
                        destination=dst_m,
                        src_transform=ov_transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=config.TARGET_CRS,
                        resampling=Resampling.average,
                        src_nodata=np.nan,
                        dst_nodata=np.nan,
                    )
                    acc[k][hit] += dst_m[hit]
            log.info("WorldCover progress: %d/%d tiles", idx, len(valid_urls))

    out = {}
    denom = np.maximum(weight_sum, 1e-6)
    has_data = weight_sum > 0
    for k, arr in acc.items():
        frac = np.where(has_data, arr / denom, 1.0 if k == "frac_water" else 0.0)
        out[k] = np.clip(frac, 0.0, 1.0).astype(np.float32)
    return out


def build_static_250m(out_zarr: Path = config.PROCESSED_DIR / "static_250m.zarr") -> Path:
    transform, ny, nx, y_centers, x_centers = compute_target_grid()
    log.info("Target 250m grid: ny=%d, nx=%d (%s)", ny, nx, config.TARGET_CRS)

    elev, elev_std = _build_dem_250m(transform, ny, nx)
    wc = _build_worldcover_250m(transform, ny, nx)

    # Derived terrain features in metric 250m space
    res = float(config.RESOLUTION_M)
    # Note: y_centers decreases with row index -> negate dy for North-up aspect
    dy_neg, dx = np.gradient(elev.astype(np.float64), res)
    dy = -dy_neg
    slope_rad = np.arctan(np.hypot(dx, dy))
    slope_deg = np.rad2deg(slope_rad).astype(np.float32)
    # Aspect direction of steepest descent; zero out where flat
    mag = np.hypot(dx, dy)
    aspect_rad = np.arctan2(-dx, -dy)  # clockwise from North? -> use vector components directly
    sin_aspect = np.where(mag > 1e-4, np.sin(aspect_rad), 0.0).astype(np.float32)
    cos_aspect = np.where(mag > 1e-4, np.cos(aspect_rad), 0.0).astype(np.float32)

    tpi_500m = (elev - uniform_filter(elev, size=5)).astype(np.float32)
    tpi_2km = (elev - uniform_filter(elev, size=17)).astype(np.float32)
    tpi_10km = (elev - gaussian_filter(elev, sigma=40.0)).astype(np.float32)

    # Land mask & distance to sea (pixels where elev <= 0.1 and mostly water are ocean)
    land_mask = ((wc["frac_water"] < 0.85) | (elev > 1.0)).astype(np.float32)
    dist_coast_km = (distance_transform_edt(land_mask > 0.5) * (res / 1000.0)).astype(np.float32)

    # Precompute 2D WGS84 lat/lon for fast regridding & station alignment
    tr_inv = Transformer.from_crs(config.TARGET_CRS, "EPSG:4326", always_xy=True)
    gx, gy = np.meshgrid(x_centers, y_centers)
    lon2d, lat2d = tr_inv.transform(gx, gy)

    data_vars = {
        "elevation_m": (("y", "x"), elev),
        "elev_std_m": (("y", "x"), elev_std),
        "slope_deg": (("y", "x"), slope_deg),
        "sin_aspect": (("y", "x"), sin_aspect),
        "cos_aspect": (("y", "x"), cos_aspect),
        "tpi_500m": (("y", "x"), tpi_500m),
        "tpi_2km": (("y", "x"), tpi_2km),
        "tpi_10km": (("y", "x"), tpi_10km),
        "dist_coast_km": (("y", "x"), dist_coast_km),
        "land_mask": (("y", "x"), land_mask),
        "lat_2d": (("y", "x"), lat2d.astype(np.float32)),
        "lon_2d": (("y", "x"), lon2d.astype(np.float32)),
        **{k: (("y", "x"), v) for k, v in wc.items()},
    }

    ds = xr.Dataset(
        data_vars,
        coords={"y": y_centers, "x": x_centers},
        attrs={
            "crs": config.TARGET_CRS,
            "resolution_m": config.RESOLUTION_M,
            "transform": list(transform)[:6],
        },
    )

    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(out_zarr, mode="w")
    log.info("Saved static 250m Zarr %s -> %s", dict(ds.sizes), out_zarr)
    return out_zarr


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build 250m static DEM + WorldCover Zarr")
    p.add_argument("--out", type=Path, default=config.PROCESSED_DIR / "static_250m.zarr")
    args = p.parse_args()
    build_static_250m(args.out)


if __name__ == "__main__":
    main()
