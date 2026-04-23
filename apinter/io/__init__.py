"""Data loading and I/O.

Entry points:
  - load_cmip6(var, sim_time=..., models=..., level=...): generic CMIP6
    multi-model loader. Legacy load_cmip6_<var> names kept as wrappers.
  - load_obs_sst(source=...): HadISST / ERSST / COBE readers.
  - load_era5(var, level=..., region=..., sim_time=...): ERA5 reader.
  - load_and_concat(var, ssp, model, ...): historical + SSP concatenation.
  - load_joblib / save_joblib: serialization helpers.
"""
from .cmip6 import (
    CMIP6_VARS,
    get_cmip6_models,
    load_cmip6,
    # backward-compat wrappers
    load_cmip6_sst,
    load_cmip6_omega,
    load_cmip6_zg,
    load_cmip6_psl,
    load_cmip6_pr,
    load_cmip6_zos,
    load_cmip6_thetao,
    load_cmip6_tauu,
    load_cmip6_wind,
)
from .obs import OBS_SST_SOURCES, ERA5_VARS, load_obs_sst, load_era5
from .ssp import SSP_VARS, get_ssp_models, load_and_concat
from .joblib_io import load_joblib, save_joblib

__all__ = [
    # CMIP6
    "CMIP6_VARS",
    "get_cmip6_models",
    "load_cmip6",
    "load_cmip6_sst",
    "load_cmip6_omega",
    "load_cmip6_zg",
    "load_cmip6_psl",
    "load_cmip6_pr",
    "load_cmip6_zos",
    "load_cmip6_thetao",
    "load_cmip6_tauu",
    "load_cmip6_wind",
    # observations / reanalysis
    "OBS_SST_SOURCES",
    "ERA5_VARS",
    "load_obs_sst",
    "load_era5",
    # SSP
    "SSP_VARS",
    "get_ssp_models",
    "load_and_concat",
    # joblib
    "load_joblib",
    "save_joblib",
]
