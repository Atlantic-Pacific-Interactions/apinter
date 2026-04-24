"""Path and grid constants used across apinter modules.

All paths are absolute and Perlmutter-specific. Override via environment
variables or by importing and reassigning before use:

    import apinter.config as cfg
    cfg.CMIP6_DIR = Path("/new/path")
"""
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Base data directories
# ---------------------------------------------------------------------------

DATA_DIR = Path("/pscratch/sd/y/yanxia/DATA")
CMIP6_DIR = Path("/pscratch/sd/y/yanxia/CMIP6")
ERA5_DIR = DATA_DIR / "ERA5"
LSMSK_PATH = DATA_DIR / "LandMask" / "lsmask.nc"

# ---------------------------------------------------------------------------
# Observational SST files
# ---------------------------------------------------------------------------

HADISST_PATH = DATA_DIR / "HadISST" / "HadISST.0-360.nc"
ERSST_PATH = DATA_DIR / "ERSST" / "ERSSTv5_sst.mnmean_185401_202412.nc"
COBE_PATH = DATA_DIR / "COBE" / "cobe.189101_202306.mon.mean.nc"

# ---------------------------------------------------------------------------
# ORAS5 ocean reanalysis (NEMO ORCA025, monthly per-file, from 1958-01)
# ---------------------------------------------------------------------------

ORAS5_DIR = DATA_DIR / "ORAS5"
ORAS5_START_YEAR = 1958

# ---------------------------------------------------------------------------
# ERA5 files (monthly, multi-decade, split by time range)
# ---------------------------------------------------------------------------

ERA5_U_PATHS = {
    "1940_2005": ERA5_DIR / "era5_u_component_1940_2005.nc",
    "2005_2025": ERA5_DIR / "era5_u_component_2005_2025.nc",
}
ERA5_V_PATHS = {
    "1940_2005": ERA5_DIR / "era5_v_component_1940_2005.nc",
    "2005_2025": ERA5_DIR / "era5_v_component_2005_2025.nc",
}

# ---------------------------------------------------------------------------
# E3SM simulation paths (from E3SMS/path_config.py)
# ---------------------------------------------------------------------------

E3SM_MMF_PATH = (
    "/global/cfs/cdirs/m3312/whannah/2023-CPL/"
    "E3SM.INCITE2023-CPL.ne30pg2_EC30to60E2r2.WCYCL20TR-MMF1/archive/atm/hist"
)

E3SM_V2_PATHS = {
    "0101": "/global/cfs/cdirs/m3312/whannah/e3smv2_historical/v2.LR.historical_0101/archive/atm/hist",
    "0151": "/global/cfs/cdirs/m3312/whannah/e3smv2_historical/v2.LR.historical_0151/archive/atm/hist",
    "0201": "/global/cfs/cdirs/m3312/whannah/e3smv2_historical/v2.LR.historical_0201/archive/atm/hist",
    "0251": "/global/cfs/cdirs/m3312/whannah/e3smv2_historical/v2.LR.historical_0251/archive/atm/hist",
    "0301": "/global/cfs/cdirs/m3312/whannah/e3smv2_historical/v2.LR.historical_0301/archive/atm/hist",
}

EXTRACTED_SIMULATIONS_PATH = "/pscratch/sd/y/yanxia/ENSO-CLOUD/Simulations"

EXTRACTED_FILES = {
    "E3SM_1950_2015": "E3SM_1950-2015_regrid180x360.nc",
    "E3SM_MMF_1950_2015": "E3SM-MMF_1950-2015_Regrid180x360.nc",
    "E3SM_CLDLOW": "e3sm_cldlow.nc",
    "E3SM_CLDTOT": "e3sm_remap_cldtot.nc",
    "MMF_CLDLOW": "mmf_cldlow.nc",
    "MMF_CLDTOT": "mmf_remap_cldtot.nc",
    "E3SM_PRECC_PRECL_1850_2015": "E3SM-PRECC,PRECL-1850-2015-Regrid180x360.nc",
    "E3SM_PRECC_PRECL_OUTPUT": "E3SM_PRECC,PRECL_output_file.nc",
}

# ---------------------------------------------------------------------------
# NERSC community mirrors (canonical ESGF DRS / LENS / datalake)
# ---------------------------------------------------------------------------
# These are read-only directories maintained by other NERSC projects. Paths
# work on Perlmutter only; loaders in apinter.io.nersc_cmip6 / apinter.io.lens
# skip gracefully when the directory isn't accessible.

NERSC_CMIP6_DIR        = Path("/global/cfs/cdirs/m3522/cmip6/CMIP6")
NERSC_CESM1_LENS_DIR   = Path("/global/cfs/cdirs/m2637/LENS")
NERSC_LOCA2_DIR        = Path("/global/cfs/cdirs/m3522/datalake/LOCA2")
NERSC_DATALAKE_DIR     = Path("/global/cfs/cdirs/m3522/datalake")

# ---------------------------------------------------------------------------
# Grid constants (used by apinter.circulation)
# ---------------------------------------------------------------------------

# 12 standard pressure levels for CMIP6 and MMM Walker/Hadley plots
CMIP6_PLEV = np.array(
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100],
    dtype=np.float64,
)
COMMON_PLEV = CMIP6_PLEV.copy()

# 1-degree longitude grid used by the Walker/velpot plots
COMMON_LON = np.arange(0.5, 360, 1.0)

# 1-degree latitude grid
COMMON_LAT = np.arange(-89.5, 90, 1.0)

# Physical constants
EARTH_RADIUS = 6.371e6   # m
GRAVITY = 9.81           # m/s^2
