"""Step 5: Push ML-ready Zarr + sparse Parquet dataset (and/or full_grid/) to Hugging Face Hub."""
import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import HfApi
import pandas as pd
import zarr

from . import config

log = logging.getLogger("regionalnext.push_hf")

FULL_GRID_PATTERNS = [
    "era5_land.zarr/**",
    "static_250m.zarr/**",
    "stations.parquet",
    "observations.parquet",
    "timestamps.parquet",
    "norm_stats.json",
]


def _build_dataset_card(dataset_dir: Path, repo_id: str) -> str:
    z = zarr.open_group(str(dataset_dir / "patches_dense.zarr"), mode="r")
    shape = z["X"].shape
    channels = list(z.attrs["channels"])
    meta = pd.read_parquet(dataset_dir / "patches_meta.parquet")
    sparse = pd.read_parquet(dataset_dir / "patches_sparse.parquet")
    stats = json.loads((dataset_dir / "norm_stats.json").read_text())

    return f"""---
license: cc-by-4.0
task_categories:
  - image-to-image
tags:
  - weather
  - downscaling
  - super-resolution
  - meteorology
  - era5-land
  - zarr
pretty_name: RegionalNext 250m Europe Super-Resolution Weather Dataset
---

# {repo_id}

ML-ready **250 m** resolution weather super-resolution dataset over France, the Alps, the Pyrenees, and bordering Europe (`lat [41.0, 51.8]`, `lon [-5.5, 13.5]`, `EPSG:3035`).

Designed for **sparse-supervised image-to-image downscaling**: dense `256x256` (`64 km x 64 km`) input patches from Copernicus **ERA5-Land (0.1°)** + **Copernicus DEM GLO-90** + **ESA WorldCover 2021** paired with **COO sparse ground-station observations** (Météo-France, MeteoSwiss, DWD, IEM METAR) for masked loss training.

## Two Ways to Load

### 1. Pre-sliced Curated Subset (root, ~29 GB)
- `patches_dense.zarr/` — Zarr v3 array `X` of shape **`{shape}`** (`float32`, zstd-compressed)
  - Channels (`C = {len(channels)}`): `{channels}`
- `patches_sparse.parquet` — **{len(sparse):,}** sparse station target pixels (`patch_id, y, x, t2m_c, td2m_c, u10_ms, v10_ms, mask_t2m, mask_wind, is_heldout_station, ...`)
- `patches_meta.parquet` — **{len(meta):,}** patch metadata rows (`split_time`: `train` 2020–2023, `val` 2024, `test` 2025)
- `norm_stats.json` — Channel mean/std computed on `{stats['n_train_pixels']:,}` training pixels

### 2. Full Domain Grid (`full_grid/`, ~1.95 GB — 100% of domain × 2,880 timestamps)
Contains the complete `5250 x 6382` (`250 m`) static grid and **2,880** ERA5-Land timestamps (`960` stratified days across `2020–2025`) + **6,490,387** station observations across `2,686` stations:
- `full_grid/static_250m.zarr/` (`~1.1 GB`)
- `full_grid/era5_land.zarr/` (`~808 MB`, `2880 x 117 x 199`)
- `full_grid/observations.parquet`, `stations.parquet`, `timestamps.parquet`, `norm_stats.json` (`~39 MB`)

Use `RegionalNextFullGridDataset` from `regionalnext/dataset.py` to sample **664,999 active non-overlapping 256x256 patches** (`434,637` train / `115,164` val / `115,198` test; `62,374` held-out test) or **3,258,431 random-crop training anchors** on the fly (~19 ms/patch on 1 CPU thread).

## Station Sources & Licenses

- Météo-France Open Hourly Data (`meteo.data.gouv.fr`, Etalab Licence Ouverte 2.0)
- MeteoSwiss OGD (`data.geo.admin.ch`, CC BY)
- DWD CDC Open Data (`opendata.dwd.de`)
- IEM ASOS/METAR border stations
- Copernicus ERA5-Land, Copernicus DEM GLO-90, ESA WorldCover 2021
"""


def push_to_hub(
    dataset_dir: Path,
    repo_id: str,
    private: bool = False,
    mode: str = "both",
    full_grid_dir: Path = config.PROCESSED_DIR,
) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    card = _build_dataset_card(dataset_dir, repo_id)
    (dataset_dir / "README.md").write_text(card)

    if mode in ("patches", "both"):
        log.info("Uploading patches %s -> hf://datasets/%s ...", dataset_dir, repo_id)
        api.upload_folder(
            folder_path=str(dataset_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload RegionalNext 250m ML-ready Zarr + sparse Parquet dataset",
        )

    if mode in ("full_grid", "both"):
        log.info("Uploading full_grid (~1.95 GB) %s -> hf://datasets/%s/full_grid ...", full_grid_dir, repo_id)
        api.upload_folder(
            folder_path=str(full_grid_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="full_grid",
            allow_patterns=FULL_GRID_PATTERNS,
            delete_patterns=["era5_land.zarr/**"],
            commit_message="Upload 4x (2,880 timestamps / 6.49M obs) full_grid Zarr + Parquet files",
        )
        # Also ensure root README.md reflects both modes
        api.upload_file(
            path_or_fileobj=str(dataset_dir / "README.md"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update README with full_grid dataset documentation",
        )

    log.info("Upload complete: https://huggingface.co/datasets/%s", repo_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Push ML-ready dataset to Hugging Face Hub")
    p.add_argument("--repo-id", default="meteolibre-dev/regionalnext", help="HF dataset repo id")
    p.add_argument("--dataset-dir", type=Path, default=config.PROCESSED_DIR / "hf_dataset")
    p.add_argument("--full-grid-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--mode", choices=["patches", "full_grid", "both"], default="both")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()
    push_to_hub(args.dataset_dir, args.repo_id, args.private, args.mode, args.full_grid_dir)


if __name__ == "__main__":
    main()
