"""ERA5 reanalysis loaders.

Entry point: ``load_era5(var, level=, sim_time=, region=)``. Supported variables
are listed in ``ERA5_VARS``; each entry maps to one or more NetCDF files and
the in-file variable name.

Handles the quirks of the ERA5 files on disk:
  - Some files use ``valid_time`` + ``pressure_level``; others use
    ``time`` + ``level``. Both are normalized to ``time`` + ``level``.
  - The ``tp`` (total precipitation) file has a ``number`` ensemble dim —
    we select ``number=0``.
  - Latitude is often descending; sorted ascending on load.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import xarray as xr

from apinter.config import ERA5_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-variable spec
# ---------------------------------------------------------------------------
# Each entry:
#   paths : {label: Path}  (multi-entry → concat on time)
#   var   : str  (data_var name inside the NetCDF)

_E = ERA5_DIR

ERA5_VARS: Dict[str, Dict] = {
    # --- Pressure-level variables (3D) ---
    'u': {
        'paths': {
            '1940_2005': _E / 'era5_u_component_1940_2005.nc',
            '2005_2025': _E / 'era5_u_component_2005_2025.nc',
        },
        'var': 'u',
    },
    'v': {
        'paths': {
            '1940_2005': _E / 'era5_v_component_1940_2005.nc',
            '2005_2025': _E / 'era5_v_component_2005_2025.nc',
        },
        'var': 'v',
    },
    'q': {
        'paths': {
            '1940_2005': _E / 'era5_specific_humidity_1940_2005.nc',
            '2005_2025': _E / 'era5_specific_humidity_2005_2025.nc',
        },
        'var': 'q',
    },
    'z': {
        'paths': {
            '1950_2015': _E / 'era5_z_1950_2015.nc',
            '2016_2026': _E / 'era5_z_2016_2026.nc',
        },
        'var': 'z',
    },
    'pl_omega': {
        'paths': {'1958_2023': _E / 'era5_pl_omega.nc'},
        'var': 'w',
    },

    # --- Single-level / surface variables (2D) ---
    'omega500': {
        'paths': {'1950_2025': _E / 'era5_omega500_1950_2025.nc'},
        'var': 'w',
    },
    'slp': {
        'paths': {'1940_2025': _E / 'era5_slp_1940_2025.nc'},
        'var': 'msl',
    },
    'sst': {
        'paths': {'1940_2025': _E / 'era5_sst_1940_2025.nc'},
        'var': 'sst',
    },
    'tp': {
        'paths': {'1940_2025': _E / 'era5_total_precipitation_1940_2025.nc'},
        'var': 'tp',
    },
    'mtnlwrf': {
        'paths': {'1950_2025': _E / 'era5_mtnlwrf_1950_2025.nc'},
        'var': 'avg_tnlwrf',
    },

    # --- Surface bundle (4 variables share one NetCDF) ---
    'u10': {'paths': {'1940_2025': _E / 'era5_u10_v10_t2m_d2m_1940_2025.nc'}, 'var': 'u10'},
    'v10': {'paths': {'1940_2025': _E / 'era5_u10_v10_t2m_d2m_1940_2025.nc'}, 'var': 'v10'},
    't2m': {'paths': {'1940_2025': _E / 'era5_u10_v10_t2m_d2m_1940_2025.nc'}, 'var': 't2m'},
    'd2m': {'paths': {'1940_2025': _E / 'era5_u10_v10_t2m_d2m_1940_2025.nc'}, 'var': 'd2m'},
}


# ---------------------------------------------------------------------------
# Coordinate normalization
# ---------------------------------------------------------------------------

def _normalize_coords(da: xr.DataArray) -> xr.DataArray:
    """Standardize time/level/lat/lon names and drop ensemble dim if present."""
    rename: Dict[str, str] = {}
    if 'valid_time' in da.dims:
        rename['valid_time'] = 'time'
    if 'pressure_level' in da.dims:
        rename['pressure_level'] = 'level'
    if 'latitude' in da.dims:
        rename['latitude'] = 'lat'
    if 'longitude' in da.dims:
        rename['longitude'] = 'lon'
    if rename:
        da = da.rename(rename)
    if 'number' in da.dims:
        da = da.isel(number=0)
    if da.lat[0] > da.lat[-1]:
        da = da.sortby('lat')
    return da


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------

def load_era5(var: str,
              level: Optional[int] = None,
              sim_time: Optional[slice] = None,
              region: Optional[Dict[str, tuple]] = None,
              ) -> xr.DataArray:
    """Load an ERA5 variable, concatenating across the split files if needed.

    Selections (``level``, ``sim_time``, ``region``) are pushed down **into each
    per-file open** before ``.load()``, so asking for one month at one pressure
    level only reads that slab from disk — not the full multi-GB file.

    Parameters
    ----------
    var : str
        Key of ``ERA5_VARS`` — see module docstring for the full list.
    level : int, optional
        Pressure level in hPa (applies to 3D variables: u, v, q, z, pl_omega).
    sim_time : slice, optional
        Time range passed to ``.sel(time=...)``. None = all available.
    region : dict, optional
        {'lat': (latS, latN), 'lon': (lonW, lonE)} spatial subset.

    Returns
    -------
    xr.DataArray with coords (time, [level], lat, lon).
    """
    if var not in ERA5_VARS:
        raise ValueError(f"Unknown ERA5 var {var!r}. Choices: {sorted(ERA5_VARS)}")
    spec = ERA5_VARS[var]

    parts: List[xr.DataArray] = []
    for path in spec['paths'].values():
        with xr.open_dataset(path) as ds:
            if spec['var'] not in ds:
                raise KeyError(
                    f"Variable {spec['var']!r} not found in {path}; "
                    f"available: {list(ds.data_vars)}"
                )
            da = ds[spec['var']]

            # In-file time coord may be 'valid_time' or 'time'
            time_in_file = 'valid_time' if 'valid_time' in da.dims else 'time'
            level_in_file = ('pressure_level' if 'pressure_level' in da.dims
                             else 'level' if 'level' in da.dims else None)
            lat_in_file = 'latitude' if 'latitude' in da.dims else 'lat'
            lon_in_file = 'longitude' if 'longitude' in da.dims else 'lon'

            # Push selections down before .load()
            if level is not None:
                if level_in_file is None:
                    raise ValueError(
                        f"ERA5 var {var!r} is not a pressure-level field — "
                        "`level` not applicable."
                    )
                da = da.sel({level_in_file: level})

            if sim_time is not None:
                da = da.sel({time_in_file: sim_time})

            if region is not None:
                if 'lat' in region:
                    lo, hi = region['lat']
                    # File lat may be descending; use ascending-friendly slice
                    if da[lat_in_file][0] > da[lat_in_file][-1]:
                        da = da.sel({lat_in_file: slice(hi, lo)})
                    else:
                        da = da.sel({lat_in_file: slice(lo, hi)})
                if 'lon' in region:
                    lo, hi = region['lon']
                    da = da.sel({lon_in_file: slice(lo, hi)})

            if 'number' in da.dims:
                da = da.isel(number=0)

            # Only materialize after slicing
            if da.sizes.get(time_in_file, 1) == 0:
                # No overlap in this file after time subset — skip
                continue
            parts.append(da.load())

    if not parts:
        raise RuntimeError(
            f"No data returned for ERA5 {var!r} with "
            f"level={level}, sim_time={sim_time}, region={region}."
        )

    if len(parts) == 1:
        combined = parts[0]
    else:
        time_dim = 'valid_time' if 'valid_time' in parts[0].dims else 'time'
        combined = xr.concat(parts, dim=time_dim)

    combined = _normalize_coords(combined)
    logger.info(f"Loaded ERA5 {var}: shape={combined.shape}")
    return combined


# ---------------------------------------------------------------------------
# Surface heat flux GRIB → (qnet, qsw) in W/m²
# ---------------------------------------------------------------------------

def load_era5_flux(flux_file: Optional[Union[str, Path]] = None,
                   sim_time: Optional[slice] = None,
                   ) -> tuple[xr.DataArray, xr.DataArray]:
    """Load ERA5 surface-flux GRIB and return (qnet, qsw) in W/m².

    ERA5 monthly-mean GRIB files store fluxes as J/m² accumulated over the
    month. This function converts to W/m² by dividing by the number of
    seconds in each calendar month, then returns:
        qnet = ssr + str + slhf + sshf   (net surface heat flux, into ocean)
        qsw  = ssr                        (net shortwave at the surface)

    Parameters
    ----------
    flux_file : Path, optional
        Default: ``ERA5_DIR / 'era5_q_flux.grib'``.
    sim_time : slice, optional
        Passed to ``.sel(time=...)`` after coordinate normalization.

    Returns
    -------
    (qnet, qsw) : tuple of xr.DataArray (time, lat, lon).

    Notes
    -----
    Requires the ``cfgrib`` optional dependency:
        ``pip install 'apinter[heat_budget_io]'``
    """
    import calendar

    if flux_file is None:
        flux_file = ERA5_DIR / 'era5_q_flux.grib'

    ds = xr.open_dataset(flux_file, engine='cfgrib')

    rename: Dict[str, str] = {}
    if 'latitude' in ds.dims:
        rename['latitude'] = 'lat'
    if 'longitude' in ds.dims:
        rename['longitude'] = 'lon'
    if 'valid_time' in ds.dims:
        rename['valid_time'] = 'time'
    if rename:
        ds = ds.rename(rename)

    seconds_per_month = xr.DataArray(
        [calendar.monthrange(t.dt.year.item(), t.dt.month.item())[1] * 86400.0
         for t in ds.time],
        dims=['time'], coords={'time': ds.time},
    )

    ssr = ds['ssr'] / seconds_per_month
    str_ = ds['str'] / seconds_per_month
    slhf = ds['slhf'] / seconds_per_month
    sshf = ds['sshf'] / seconds_per_month

    qnet = ssr + str_ + slhf + sshf
    qsw = ssr

    if sim_time is not None:
        qnet = qnet.sel(time=sim_time)
        qsw = qsw.sel(time=sim_time)

    qnet.name = 'qnet'
    qsw.name = 'qsw'
    return qnet, qsw
