"""Data loading and I/O.

Entry points:
  - load_cmip6(var, sim_time=..., models=..., level=...): generic CMIP6
    multi-model loader for the user's 1°-regridded zarr mirror. Legacy
    load_cmip6_<var> names kept as wrappers.
  - load_nersc_cmip6(variable_id, experiment_id, ...): native-grid loader
    for the NERSC ESGF CMIP6 mirror (/global/cfs/cdirs/m3522/cmip6/CMIP6);
    load_nersc_cmip6_ensemble for multi-member-single-model; data are
    **not** regridded — caller must regrid for MMM.
  - load_cesm1_lens(variable='ts'|'tas', member_ids=None): 40-member
    CESM1-CAM5 LENS at /global/cfs/cdirs/m2637/LENS.
  - load_loca2(model, variable, experiment, ...): LOCA2 statistically
    downscaled (~6 km) CMIP6 ensemble at /global/cfs/cdirs/m3522/datalake/LOCA2.
  - load_obs_sst(source=...): HadISST / ERSST / COBE readers. Added
    '*_nersc' variants for the m3522/datalake copies.
  - load_era5(var, level=..., region=..., sim_time=...): ERA5 reader
    covering u/v/q/z/pl_omega/omega500/slp/sst/tp/mtnlwrf/u10/v10/t2m/d2m.
  - load_oras5(var, sim_time=...): ORAS5 NEMO-grid reanalysis loader;
    regrid_to_equatorial_lon / load_oras5_equatorial for 1D-lon binning.
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
from .obs import OBS_SST_SOURCES, load_obs_sst
from .era5 import ERA5_VARS, load_era5, load_era5_flux
from .oras5 import (
    ORAS5_VARS,
    load_oras5,
    load_oras5_equatorial,
    regrid_to_equatorial_lon,
)
from .ssp import SSP_VARS, get_ssp_models, load_and_concat
from .nersc_cmip6 import (
    list_nersc_cmip6_models,
    load_nersc_cmip6,
    load_nersc_cmip6_ensemble,
)
from .lens import (
    CESM1_LENS_VARS,
    LOCA2_VARS,
    LOCA2_EXPERIMENTS,
    list_cesm1_lens_members,
    list_loca2_models,
    list_loca2_members,
    load_cesm1_lens,
    load_loca2,
)
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
    # observations
    "OBS_SST_SOURCES",
    "load_obs_sst",
    # ERA5 reanalysis
    "ERA5_VARS",
    "load_era5",
    "load_era5_flux",
    # ORAS5 ocean reanalysis
    "ORAS5_VARS",
    "load_oras5",
    "load_oras5_equatorial",
    "regrid_to_equatorial_lon",
    # SSP
    "SSP_VARS",
    "get_ssp_models",
    "load_and_concat",
    # NERSC ESGF CMIP6 mirror (native grid)
    "list_nersc_cmip6_models",
    "load_nersc_cmip6",
    "load_nersc_cmip6_ensemble",
    # Large Ensembles (CESM1-LE, LOCA2)
    "CESM1_LENS_VARS",
    "LOCA2_VARS",
    "LOCA2_EXPERIMENTS",
    "list_cesm1_lens_members",
    "list_loca2_models",
    "list_loca2_members",
    "load_cesm1_lens",
    "load_loca2",
    # joblib
    "load_joblib",
    "save_joblib",
]
