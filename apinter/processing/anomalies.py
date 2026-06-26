"""Detrending and anomaly utilities."""
import logging

import xarray as xr

logger = logging.getLogger(__name__)


def standardize_time_to_month_start(da: xr.DataArray) -> xr.DataArray:
    """Align the time coordinate to month-start, for merging datasets whose
    time stamps land on different days of the month. Ported from
    E3SMS/scripts/moisture_budget/calc_moisture_budget_decomposition.py.

    Does not deduplicate — chain `.drop_duplicates(dim='time')` after if the
    input may have repeated months (e.g. concatenated overlapping files).
    """
    if 'time' not in da.coords:
        return da
    try:
        new_time = da.time.values.astype('datetime64[M]').astype('datetime64[ns]')
        return da.assign_coords(time=new_time)
    except Exception:
        return da


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
