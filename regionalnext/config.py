"""Central configuration for regionalnext dataset & pipeline."""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Geographic Domain & Grid ----------------------------------------------
# France + Alps (CH/IT/DE/AT border) + Pyrenees (ES border) + Benelux/UK edge
LAT_MIN = 41.0
LAT_MAX = 51.8
LON_MIN = -5.5
LON_MAX = 13.5

# EPSG:3035 (ETRS89-extended / LAEA Europe): exact metric 250m x 250m pixels
TARGET_CRS = "EPSG:3035"
RESOLUTION_M = 250
PATCH_SIZE = 256            # 256 x 256 px @ 250m = 64 km x 64 km
PATCH_STRIDE = 128          # 50% overlap (32 km) for tile generation

# --- Temporal Sampling -----------------------------------------------------
DATE_START = "2020-01-01"
DATE_END = "2025-12-31"
# ~960 stratified days x 3 hours/day = ~2,880 timestamps (plus H-1 for ssrd de-accum)
N_SAMPLE_DAYS = 960
HOURS_PER_DAY = 3
RANDOM_SEED = 42

# --- Station Sources -------------------------------------------------------
MF_DATASET_API = "https://www.data.gouv.fr/api/1/datasets/donnees-climatologiques-de-base-horaires/"
METEOSWISS_STAC_COLLECTION = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.meteoschweiz.ogd-smn"
METEOSWISS_META_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_stations.csv"
DWD_BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# --- ERA5-Land -------------------------------------------------------------
# Note: in CDS reanalysis-era5-land, ssrd is accumulated since 00:00 UTC.
# We fetch H and H-1 so hourly W/m^2 = (ssrd(H) - ssrd(H-1)) / 3600.
ERA5_LAND_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "surface_pressure",
    "snow_depth",
]

# --- Patch Selection -------------------------------------------------------
MIN_STATIONS_PER_PATCH = 3
MIN_STATIONS_MOUNTAIN = 2    # relaxed threshold when mean_elev >= 800m
MOUNTAIN_ELEV_M = 800.0
