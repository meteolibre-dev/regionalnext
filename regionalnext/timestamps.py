"""Step 1: Stratified timestamp sampling for ERA5-Land + station extraction.

Why sample days with multiple hours rather than 700 completely isolated hours?
1. CDS API queues per request: grouping ~240 days (3 hours/day = ~720 target
   timestamps) cuts CDS requests dramatically.
2. ERA5-Land `ssrd` is accumulated from 00:00 UTC (with 00:00 UTC holding the
   previous day's 24h accumulation). Fetching each target hour `H` together
   with `H - 1h` (and avoiding hour 00 UTC or including 23 UTC of D-1) makes
   de-accumulation exact.
3. ~25-30% of sampled days are placed in 2024-2025 so the exact same dates can
   later be paired with historical WeatherNext 3 forecasts.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("regionalnext.timestamps")


def generate_timestamps(
    start: str = config.DATE_START,
    end: str = config.DATE_END,
    n_days: int = config.N_SAMPLE_DAYS,
    hours_per_day: int = config.HOURS_PER_DAY,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Sample stratified dates across years & seasons, plus stratified hours/day.

    Stratification:
    - Year weights: slightly upweight 2024-2025 (~33% of days) for WN3 overlap.
    - Within each year: sample evenly across the 4 meteorological seasons (DJF/MAM/JJA/SON).
    - Within each day: pick 1 morning/inversion hour (04-08 UTC), 1 afternoon/Tmax
      hour (12-16 UTC), and 1 evening/night hour (18-23 UTC) so the diurnal cycle
      is balanced without 00 UTC cross-day ssrd wraparound issues.

    Returns DataFrame with columns:
      timestamp_utc, date, hour, prev_timestamp_utc, year, season, diurnal_bin, split_time
    """
    rng = np.random.default_rng(seed)
    all_days = pd.date_range(start, end, freq="D", tz="UTC")
    df_days = pd.DataFrame({"date": all_days})
    df_days["year"] = df_days["date"].dt.year
    month = df_days["date"].dt.month
    df_days["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["DJF", "MAM", "JJA"],
        default="SON",
    )

    years = sorted(df_days["year"].unique())
    # Allocate ~33% of days to 2024-2025 if present, rest split across earlier years
    recent_years = [y for y in years if y >= 2024]
    early_years = [y for y in years if y < 2024]
    year_quota: dict[int, int] = {}
    if recent_years and early_years:
        n_recent = round(n_days * 0.34)
        n_early = n_days - n_recent
        for i, y in enumerate(recent_years):
            year_quota[y] = n_recent // len(recent_years) + (1 if i < n_recent % len(recent_years) else 0)
        for i, y in enumerate(early_years):
            year_quota[y] = n_early // len(early_years) + (1 if i < n_early % len(early_years) else 0)
    else:
        for i, y in enumerate(years):
            year_quota[y] = n_days // len(years) + (1 if i < n_days % len(years) else 0)

    chosen_dates = []
    seasons = ["DJF", "MAM", "JJA", "SON"]
    for y in years:
        q = year_quota[y]
        sub_y = df_days[df_days["year"] == y]
        for i, s in enumerate(seasons):
            k = q // 4 + (1 if i < q % 4 else 0)
            pool = sub_y[sub_y["season"] == s]["date"].to_numpy()
            k = min(k, len(pool))
            idx = rng.choice(len(pool), size=k, replace=False)
            chosen_dates.extend(pool[idx])

    chosen_dates = sorted(pd.to_datetime(chosen_dates, utc=True))

    # Diurnal bins avoiding hour=00 (ssrd resets at 00 UTC)
    bins = [
        ("morning", np.arange(4, 9)),     # 04..08 UTC (cold-pool / inversion)
        ("afternoon", np.arange(12, 17)), # 12..16 UTC (Tmax window)
        ("night", np.arange(18, 24)),     # 18..23 UTC (evening cooling)
    ]

    rows = []
    for d in chosen_dates:
        selected_bins = bins[:hours_per_day] if hours_per_day <= 3 else bins
        for bname, hrs in selected_bins:
            h = int(rng.choice(hrs))
            ts = d + pd.Timedelta(hours=h)
            prev_ts = ts - pd.Timedelta(hours=1)
            rows.append({
                "timestamp_utc": ts,
                "prev_timestamp_utc": prev_ts,
                "date": d.strftime("%Y-%m-%d"),
                "hour": h,
                "year": int(d.year),
                "season": df_days.loc[df_days["date"] == d, "season"].iloc[0],
                "diurnal_bin": bname,
                # Temporal split: <=2023 train, 2024 val, 2025 test (WN3 eval!)
                "split_time": "train" if d.year <= 2023 else ("val" if d.year == 2024 else "test"),
            })

    out = pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)
    return out


def save_timestamps(out_path: Path | None = None, **kwargs) -> pd.DataFrame:
    df = generate_timestamps(**kwargs)
    out_path = out_path or (config.PROCESSED_DIR / "timestamps.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Saved %d timestamps (%d unique days) -> %s",
             len(df), df["date"].nunique(), out_path)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Sample stratified timestamps for regionalnext")
    p.add_argument("--n-days", type=int, default=config.N_SAMPLE_DAYS)
    p.add_argument("--hours-per-day", type=int, default=config.HOURS_PER_DAY)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--out", type=Path, default=config.PROCESSED_DIR / "timestamps.parquet")
    args = p.parse_args()

    df = save_timestamps(out_path=args.out, n_days=args.n_days,
                         hours_per_day=args.hours_per_day, seed=args.seed)
    print(df.groupby(["year", "season"]).size().unstack(fill_value=0))
    print("\nTemporal splits:\n", df["split_time"].value_counts())


if __name__ == "__main__":
    main()
