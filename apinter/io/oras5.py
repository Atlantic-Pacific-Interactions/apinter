"""ORAS5 ocean reanalysis loaders.

ORAS5 on Perlmutter lives as one NetCDF file per month per variable, on the
native NEMO ORCA025 curvilinear grid. Each file uses ``time_counter`` as the
time dimension; files start at 1958-01.

Canonical loading pattern (from Paper_1 notebook 36/42):

    ds = xr.open_mfdataset(pattern,
                           preprocess=lambda d: d.drop_vars(['time_counter_bnds'], errors='ignore'),
                           combine='nested',
                           concat_dim='time_counter',
                           chunks={'time_counter': 1},
                           parallel=False)
    ds = ds.rename({'time_counter': 'time', <orig>: <friendly>})
    ds = ds.assign_coords(time=pd.date_range('1958-01-01', periods=n, freq='MS'))

The grid is curvilinear (nav_lat, nav_lon are 2D). Paper_1 notebooks regrid
to a 1D longitude array by equatorial-band binning; that helper is exposed
here as ``regrid_to_equatorial_lon``.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from apinter.config import ORAS5_DIR, ORAS5_START_YEAR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-variable spec
# ---------------------------------------------------------------------------
# Each entry: {subdir, short_name, kind ('2D' or '3D'), friendly_name}
# ``short_name`` matches the NetCDF variable; ``friendly_name`` is the name
# the loader renames it to for downstream analysis.
ORAS5_VARS: Dict[str, Dict] = {
    'd20':       {'subdir': 'd20',                     'short_name': 'so20chgt', 'kind': '2D', 'friendly': 'd20'},
    'ssh':       {'subdir': 'ssh',                     'short_name': 'sossheig', 'kind': '2D', 'friendly': 'ssh'},
    'mld030':    {'subdir': 'mld030',                  'short_name': 'somxl030', 'kind': '2D', 'friendly': 'mld030'},
    'tauu':      {'subdir': 'zonal_wind_stress',       'short_name': 'sozotaux', 'kind': '2D', 'friendly': 'tauu'},
    'tauv':      {'subdir': 'meridional_wind_stress',  'short_name': 'sometauy', 'kind': '2D', 'friendly': 'tauv'},
    'hfds':      {'subdir': 'sohefldo',                'short_name': 'sohefldo', 'kind': '2D', 'friendly': 'hfds'},
    'thetao':    {'subdir': 'potential_temperature',   'short_name': 'votemper', 'kind': '3D', 'friendly': 'thetao'},
    'salinity':  {'subdir': 'salinity',                'short_name': 'vosaline', 'kind': '3D', 'friendly': 'salinity'},
    'uo':        {'subdir': 'vozocrtx',                'short_name': 'vozocrtx', 'kind': '3D', 'friendly': 'uo'},
    'vo':        {'subdir': 'vomecrty',                'short_name': 'vomecrty', 'kind': '3D', 'friendly': 'vo'},
    'wo':        {'subdir': 'wovecrtz',                'short_name': 'wovecrtz', 'kind': '3D', 'friendly': 'wo'},
}


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------

def load_oras5(var: str,
               sim_time: slice = slice('1958-01-01', '2014-12-31'),
               base_path: Union[str, Path] = ORAS5_DIR,
               pattern: Optional[str] = None) -> xr.DataArray:
    """Load an ORAS5 variable across monthly files as a single DataArray.

    Parameters
    ----------
    var : str
        Key of ``ORAS5_VARS`` (e.g. 'd20', 'ssh', 'tauu', 'thetao').
    sim_time : slice
        Time range applied after time-coord reconstruction.
    base_path : Path-like
        ORAS5 root directory (default: ``apinter.config.ORAS5_DIR``).
    pattern : str, optional
        Override file glob for non-standard layouts.

    Returns
    -------
    xr.DataArray
        The variable with ``time`` (monthly MS) plus native curvilinear dims
        (``y``, ``x``) and 2D ``nav_lat``, ``nav_lon`` coords. 3D variables
        also carry ``deptht`` (or similar depth coord).
    """
    if var not in ORAS5_VARS:
        raise ValueError(
            f"Unknown ORAS5 var {var!r}. Choices: {sorted(ORAS5_VARS)}"
        )
    spec = ORAS5_VARS[var]
    base_path = Path(base_path)

    if pattern is None:
        kind = spec['kind']
        pattern = str(
            base_path / spec['subdir']
            / f"{spec['short_name']}_control_monthly_highres_{kind}_*.nc"
        )

    ds = xr.open_mfdataset(
        pattern,
        preprocess=lambda d: d.drop_vars(['time_counter_bnds'], errors='ignore'),
        combine='nested',
        concat_dim='time_counter',
        chunks={'time_counter': 1},
        parallel=False,
    )
    ds = ds.rename({'time_counter': 'time', spec['short_name']: spec['friendly']})

    # Reconstruct monotonic monthly time coord (ORAS5 files omit a real one)
    n = len(ds.time)
    ds = ds.assign_coords(
        time=pd.date_range(f"{ORAS5_START_YEAR}-01-01", periods=n, freq='MS')
    )
    ds = ds.sel(time=sim_time)

    da = ds[spec['friendly']]
    logger.info(f"Loaded ORAS5 {var}: shape={da.shape}")
    return da


# ---------------------------------------------------------------------------
# Curvilinear -> 1D equatorial longitude regridding
# ---------------------------------------------------------------------------

def regrid_to_equatorial_lon(da: xr.DataArray,
                             lon_bounds: Tuple[float, float] = (120, 290),
                             lat_bounds: Tuple[float, float] = (-5.5, 5.5),
                             lon_step: float = 2.0) -> xr.DataArray:
    """Regrid curvilinear ORAS5 data onto a 1D longitude grid by
    equatorial-band masking + ``groupby_bins`` averaging.

    Matches the pattern used in Paper_1 notebooks 36 and 42 for Hovmoller
    analysis of tropical-Pacific observations.

    Parameters
    ----------
    da : xr.DataArray
        ORAS5 field with 2D ``nav_lat`` and ``nav_lon`` coords. Longitudes may
        be in [-180, 180] or [0, 360]; they are normalized to [0, 360].
    lon_bounds : (lonW, lonE)
        Longitude window in [0, 360].
    lat_bounds : (latS, latN)
        Latitude window; typically a narrow equatorial band.
    lon_step : float
        Output longitude bin width (degrees).

    Returns
    -------
    xr.DataArray with dims ``(time, lon)`` (3D inputs keep their depth coord).
    """
    if 'nav_lat' not in da.coords or 'nav_lon' not in da.coords:
        raise ValueError("Expected curvilinear ORAS5 DataArray with nav_lat/nav_lon coords.")

    nav_lat = da['nav_lat']
    nav_lon = da['nav_lon']
    if 'time' in nav_lat.dims:
        nav_lat = nav_lat.isel(time=0).drop_vars('time', errors='ignore')
        nav_lon = nav_lon.isel(time=0).drop_vars('time', errors='ignore')
    nav_lon_360 = xr.where(nav_lon < 0, nav_lon + 360, nav_lon)

    latS, latN = lat_bounds
    lonW, lonE = lon_bounds
    mask = ((nav_lat >= latS) & (nav_lat <= latN)
            & (nav_lon_360 >= lonW) & (nav_lon_360 <= lonE))
    masked = da.where(mask)

    lon_edges = np.arange(lonW - lon_step / 2, lonE + lon_step, lon_step)
    lon_centers = lon_edges[:-1] + lon_step / 2
    regrid = masked.groupby_bins(nav_lon_360, bins=lon_edges, labels=lon_centers).mean()

    # Rename the groupby_bins dim (its name varies with xarray version)
    bin_dim = [d for d in regrid.dims if d not in ('time',) and 'depth' not in d.lower()][-1]
    regrid = regrid.rename({bin_dim: 'lon'})

    logger.info(
        f"Regridded to 1D lon: {regrid.shape} (lat {lat_bounds}, lon {lon_bounds}, step {lon_step})"
    )
    return regrid


def load_oras5_equatorial(var: str,
                          lon_bounds: Tuple[float, float] = (120, 290),
                          lat_bounds: Tuple[float, float] = (-5.5, 5.5),
                          lon_step: float = 2.0,
                          sim_time: slice = slice('1958-01-01', '2014-12-31'),
                          base_path: Union[str, Path] = ORAS5_DIR,
                          ) -> xr.DataArray:
    """Convenience: ``load_oras5`` + ``regrid_to_equatorial_lon`` in one call."""
    da = load_oras5(var, sim_time=sim_time, base_path=base_path)
    return regrid_to_equatorial_lon(da, lon_bounds=lon_bounds,
                                     lat_bounds=lat_bounds, lon_step=lon_step)
