"""Horizontal lat/lon regridding via xesmf.

The canonical analysis grid for this project is **1°×1°, 0.5°-offset**:
``lon ∈ [0.5, 359.5]``, ``lat ∈ [-89.5, 89.5]`` — the same target grid the
batch CMIP6 regridder (``Paper_1/scripts/utils/batch_regrid_cmip6_var.py``)
writes via CDO. ``regrid_to_1deg(da)`` puts any DataArray onto that grid.

The legacy E3SMS scripts wrote one regridded zarr per model into
``/pscratch/sd/y/yanxia/CMIP6/<model>/1850-2015-atmos/<var>.zarr``; this
module is the in-memory equivalent — load a native-grid DataArray from
the NERSC ESGF mirror, regrid, use it directly. No disk roundtrip.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import xarray as xr

from apinter.config import COMMON_LAT, COMMON_LON

logger = logging.getLogger(__name__)


def _detect_lat_lon(da: xr.DataArray) -> tuple[str, str]:
    """Return (lat_name, lon_name) for ``da``. Tries common variants."""
    coord_names = list(da.coords)
    lat_name = next(
        (n for n in ('lat', 'latitude', 'nav_lat', 'rlat') if n in coord_names),
        None,
    )
    lon_name = next(
        (n for n in ('lon', 'longitude', 'nav_lon', 'rlon') if n in coord_names),
        None,
    )
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not find lat/lon coords on DataArray (coords: {coord_names})."
        )
    return lat_name, lon_name


def regrid_to_1deg(da: xr.DataArray,
                   method: str = 'bilinear',
                   target_lon: np.ndarray = COMMON_LON,
                   target_lat: np.ndarray = COMMON_LAT,
                   ) -> xr.DataArray:
    """Regrid ``da`` onto the canonical 1° lat/lon grid via xesmf.

    Parameters
    ----------
    da : xr.DataArray
        Input on any (rectilinear or curvilinear) lat/lon grid.
    method : {'bilinear', 'conservative', 'patch', 'nearest_s2d', 'nearest_d2s'}
        ``bilinear`` is the default — matches the CDO ``remapbil`` used by the
        legacy ``batch_regrid_cmip6_var.py``.
    target_lon, target_lat : 1D arrays, default ``COMMON_LON`` / ``COMMON_LAT``
        Standard 0.5°-offset 1°×1° grid.

    Returns
    -------
    xr.DataArray on the target grid with dims (..., lat, lon). Coord names
    are normalized to ``lat``/``lon`` regardless of the input convention.
    """
    import xesmf as xe

    lat_in, lon_in = _detect_lat_lon(da)

    # xesmf expects lat/lon coords. Rename if needed.
    rename = {}
    if lat_in != 'lat':
        rename[lat_in] = 'lat'
    if lon_in != 'lon':
        rename[lon_in] = 'lon'
    da_renamed = da.rename(rename) if rename else da

    target = xr.Dataset({
        'lat': (['lat'], np.asarray(target_lat, dtype=float)),
        'lon': (['lon'], np.asarray(target_lon, dtype=float)),
    })

    regridder = xe.Regridder(da_renamed, target, method, periodic=True)
    out = regridder(da_renamed, keep_attrs=True)
    logger.info(
        f"Regridded {da.name or '<unnamed>'} {da.shape} -> {out.shape} "
        f"via xesmf '{method}'"
    )
    return out


def regrid_dict_to_1deg(model_dict: dict[str, xr.DataArray],
                        method: str = 'bilinear',
                        target_lon: np.ndarray = COMMON_LON,
                        target_lat: np.ndarray = COMMON_LAT,
                        ) -> dict[str, xr.DataArray]:
    """Apply :func:`regrid_to_1deg` to every value of a model dict.

    Convenience for the ``{model_name: native_grid_da}`` shape returned by
    :func:`apinter.io.load_nersc_cmip6`. Models that fail to regrid are
    skipped with a warning.
    """
    out: dict[str, xr.DataArray] = {}
    for name, da in model_dict.items():
        try:
            out[name] = regrid_to_1deg(da, method=method,
                                       target_lon=target_lon,
                                       target_lat=target_lat)
        except Exception as e:
            logger.warning(f"regrid {name}: {e}")
    return out
