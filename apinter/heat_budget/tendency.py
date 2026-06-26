"""Time tendency computation for the mixed-layer heat budget.

Port of Matlab code used in heat_budget_rd_lme.m. Handles both 'time'
and 'time_counter' dim names so the same helpers work for CMIP6 / ERA5
and for ORAS5 native files (which keep NEMO's 'time_counter').
"""
import xarray as xr

from .constants import CLIM_START, CLIM_END


def _get_time_dim(field):
    for name in ('time', 'time_counter'):
        if name in field.dims:
            return name
    raise ValueError(f"No time dimension found in {field.dims}")


def compute_tendency(field):
    """dT/dt via xarray central difference.

    Second-order central differences in the interior and first-order one-sided
    differences at boundaries. Returns units of [field_units]/s.
    """
    time_dim = _get_time_dim(field)
    return field.differentiate(time_dim, datetime_unit='s', edge_order=1)


def compute_anomaly_tendency(field, yrclim=None):
    """dT'/dt: subtract the monthly climatology over ``yrclim``, then differentiate."""
    if yrclim is None:
        yrclim = [CLIM_START, CLIM_END]

    time_dim = _get_time_dim(field)
    clim_slice = slice(str(yrclim[0]), str(yrclim[1]))

    clim = (field.sel({time_dim: clim_slice})
                 .groupby(f'{time_dim}.month').mean(time_dim))
    month = field[f'{time_dim}.month']
    clim_full = clim.sel(month=month).drop_vars('month')
    anomaly = field - clim_full

    return anomaly.differentiate(time_dim, datetime_unit='s', edge_order=1)
