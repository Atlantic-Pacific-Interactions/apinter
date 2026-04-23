"""Detrending and anomaly utilities."""
import logging

import xarray as xr

logger = logging.getLogger(__name__)


def detrend_dim(da: xr.DataArray, dim: str = 'time', deg: int = 1) -> xr.DataArray:
    """Detrend data along a single dimension via polynomial fit."""
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit


def compute_anomaly(data: xr.DataArray,
                    detrend: bool = True,
                    complete_years_only: bool = False) -> xr.DataArray:
    """
    Monthly climatology anomaly with optional linear detrend.

    Pipeline: subtract monthly climatology -> optional linear detrend on time.
    Shape is preserved (no spatial averaging).

    Parameters
    ----------
    data : xr.DataArray
        Input with a 'time' dimension.
    detrend : bool
        Apply linear detrend on the time dim after anomaly (default True,
        matching the Paper_1 canonical pipeline).
    complete_years_only : bool
        If True, the monthly climatology is computed from the leading
        complete calendar years only (floor(n_months / 12) years). If False
        (default, Paper_1 canonical), climatology uses all months.
    """
    if complete_years_only:
        idx = data.time.shape[0] // 12
        clim = data[:idx * 12].groupby('time.month').mean('time')
    else:
        clim = data.groupby('time.month').mean('time')

    anom = data.groupby('time.month') - clim
    if detrend:
        anom = detrend_dim(anom, 'time')
    return anom
