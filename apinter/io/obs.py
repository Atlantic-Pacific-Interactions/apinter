"""Observational data loaders: HadISST/ERSST/COBE SST and ERA5.

Replaces the scattered `read_data`, `load_raw_omega_obs`,
`load_and_process_obs_z200`, etc. functions with parameterized versions.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import xarray as xr

from apinter.config import (
    COBE_PATH,
    ERA5_DIR,
    ERA5_U_PATHS,
    ERA5_V_PATHS,
    ERSST_PATH,
    HADISST_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observational SST
# ---------------------------------------------------------------------------

OBS_SST_SOURCES: Dict[str, Dict] = {
    'hadisst': {'path': HADISST_PATH, 'var': 'sst', 'missing': -1000.0},
    'ersst':   {'path': ERSST_PATH,   'var': 'sst', 'missing': -1000.0},
    'cobesst': {'path': COBE_PATH,    'var': 'sst', 'missing': None},
}


def load_obs_sst(source: str = 'hadisst',
                 sim_time: slice = slice('1850', '2023')) -> xr.DataArray:
    """Load an observational SST dataset.

    Parameters
    ----------
    source : {'hadisst', 'ersst', 'cobesst'}
    sim_time : slice
        Time range passed to ``.sel(time=...)``.

    Returns
    -------
    xr.DataArray with dims (time, lat, lon). Missing values set to NaN.
    """
    if source not in OBS_SST_SOURCES:
        raise ValueError(
            f"Unknown source {source!r}. Choices: {sorted(OBS_SST_SOURCES)}"
        )
    spec = OBS_SST_SOURCES[source]

    ds = xr.open_dataset(spec['path'])
    sst = ds[spec['var']].sel(time=sim_time).squeeze()

    if 'latitude' in sst.dims:
        sst = sst.rename({'latitude': 'lat', 'longitude': 'lon'})

    if spec['missing'] is not None:
        sst = sst.where(sst != spec['missing'], np.nan)

    logger.info(f"Loaded {source} SST: shape={sst.shape}")
    return sst


# ---------------------------------------------------------------------------
# ERA5
# ---------------------------------------------------------------------------

# Map logical variable name -> (file-path dict, variable key inside file)
# Extend here when more ERA5 files are added.
ERA5_VARS: Dict[str, Dict] = {
    'u': {'paths': ERA5_U_PATHS, 'var': 'u'},
    'v': {'paths': ERA5_V_PATHS, 'var': 'v'},
}


def load_era5(var: str,
              level: Optional[int] = None,
              sim_time: Optional[slice] = None,
              region: Optional[Dict[str, tuple]] = None,
              era5_dir: Union[str, Path] = ERA5_DIR) -> xr.DataArray:
    """Load an ERA5 variable (concatenating across the split file set).

    Parameters
    ----------
    var : str
        Key of ``ERA5_VARS`` (e.g. 'u', 'v').
    level : int, optional
        Pressure level in hPa (uses ``pressure_level`` coord in ERA5 files).
    sim_time : slice, optional
        Time range to select. If None, all available.
    region : dict, optional
        {'lat': (latS, latN), 'lon': (lonW, lonE)} for spatial subsetting.
    era5_dir : Path-like
        Base ERA5 directory (default: apinter.config.ERA5_DIR).

    Returns
    -------
    xr.DataArray with coords renamed to time, lat, lon (plus level if 3D).
    """
    if var not in ERA5_VARS:
        raise ValueError(
            f"Unknown ERA5 var {var!r}. Choices: {sorted(ERA5_VARS)}"
        )
    spec = ERA5_VARS[var]

    parts: List[xr.DataArray] = []
    for key, path in spec['paths'].items():
        with xr.open_dataset(path) as ds:
            da = ds[spec['var']]
            if level is not None and 'pressure_level' in da.dims:
                da = da.sel(pressure_level=level)
            if 'number' in da.dims:
                da = da.isel(number=0)
            parts.append(da.load())

    # Concatenate on the ERA5 time coord (named 'valid_time')
    time_dim = 'valid_time' if 'valid_time' in parts[0].dims else 'time'
    combined = xr.concat(parts, dim=time_dim)

    rename_map = {}
    if time_dim != 'time':
        rename_map[time_dim] = 'time'
    if 'latitude' in combined.dims:
        rename_map['latitude'] = 'lat'
    if 'longitude' in combined.dims:
        rename_map['longitude'] = 'lon'
    combined = combined.rename(rename_map)

    if combined.lat[0] > combined.lat[-1]:
        combined = combined.sortby('lat')

    if sim_time is not None:
        combined = combined.sel(time=sim_time)

    if region is not None:
        if 'lat' in region:
            lo, hi = region['lat']
            combined = combined.sel(lat=slice(lo, hi))
        if 'lon' in region:
            lo, hi = region['lon']
            combined = combined.sel(lon=slice(lo, hi))

    logger.info(f"Loaded ERA5 {var}: shape={combined.shape}")
    return combined
