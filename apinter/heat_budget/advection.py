"""Reynolds-decomposed horizontal advection for the mixed-layer heat budget.

Port of Matlab advection_ml_rd.m. Reference: Graham et al. (2014) Climate
Dynamics, 43:2399-2414.
"""
import numpy as np
import xarray as xr
from .constants import RE, CLIM_START, CLIM_END
from .tendency import compute_tendency, compute_anomaly_tendency


def _compute_gradients(field):
    """Compute spatial gradients on a regular lat-lon grid.

    dT/dx = dT/dlon_rad / (R_E * cos(lat))
    dT/dy = dT/dlat_rad / R_E

    Parameters
    ----------
    field : xr.DataArray with lat, lon coordinates (degrees)

    Returns
    -------
    dfdx, dfdy : xr.DataArray
        Gradients in physical units ([field_units] / m).
    """
    lat_rad = np.deg2rad(field.lat)
    cos_lat = np.cos(lat_rad)

    dfdx = field.differentiate('lon', edge_order=1) / (RE * cos_lat * np.deg2rad(1.0))

    if field.sizes['lat'] >= 2:
        dfdy = field.differentiate('lat', edge_order=1) / (RE * np.deg2rad(1.0))
    else:
        dfdy = xr.zeros_like(field)

    return dfdx, dfdy


def advection_ml_rd(Tmld, umld, vmld, yrclim=None):
    """Compute Reynolds-decomposed horizontal advection terms.

    Parameters
    ----------
    Tmld : xr.DataArray (time, lat, lon) - ML-averaged temperature (C).
    umld : xr.DataArray (time, lat, lon) - ML-averaged zonal velocity (m/s).
    vmld : xr.DataArray (time, lat, lon) - ML-averaged meridional velocity (m/s).
    yrclim : list [start_year, end_year], optional

    Returns
    -------
    dict of xr.DataArray
        Keys: dTdt, dTpdt, umdTmdx, updTmdx, umdTpdx, updTpdx,
              vmdTmdy, vpdTmdy, vmdTpdy, vpdTpdy, mnupdTpdx, mnvpdTpdy
    """
    if yrclim is None:
        yrclim = [CLIM_START, CLIM_END]

    clim_slice = slice(str(yrclim[0]), str(yrclim[1]))

    # --- Total gradients of T ---
    dTdx, dTdy = _compute_gradients(Tmld)

    # --- Monthly climatologies ---
    Tm = Tmld.sel(time=clim_slice).groupby('time.month').mean('time')
    um = umld.sel(time=clim_slice).groupby('time.month').mean('time')
    vm = vmld.sel(time=clim_slice).groupby('time.month').mean('time')

    # Map climatologies back to full time series
    month = Tmld['time.month']
    Tm_full = Tm.sel(month=month).drop_vars('month')
    um_full = um.sel(month=month).drop_vars('month')
    vm_full = vm.sel(month=month).drop_vars('month')

    # Gradients of climatological-mean temperature
    dTmdx, dTmdy = _compute_gradients(Tm_full)

    # --- Anomalies ---
    up = umld - um_full
    vp = vmld - vm_full
    dTpdx = dTdx - dTmdx
    dTpdy = dTdy - dTmdy

    # --- 4 Reynolds decomposition terms per direction ---
    umdTmdx = um_full * dTmdx
    updTmdx = up * dTmdx
    umdTpdx = um_full * dTpdx
    updTpdx = up * dTpdx

    vmdTmdy = vm_full * dTmdy
    vpdTmdy = vp * dTmdy
    vmdTpdy = vm_full * dTpdy
    vpdTpdy = vp * dTpdy

    # --- Climatological mean of eddy-eddy terms ---
    mnupdTpdx_clim = updTpdx.sel(time=clim_slice).groupby('time.month').mean('time')
    mnvpdTpdy_clim = vpdTpdy.sel(time=clim_slice).groupby('time.month').mean('time')
    mnupdTpdx = mnupdTpdx_clim.sel(month=month).drop_vars('month')
    mnvpdTpdy = mnvpdTpdy_clim.sel(month=month).drop_vars('month')

    # --- Time derivatives ---
    dTdt = compute_tendency(Tmld)
    dTpdt = compute_anomaly_tendency(Tmld, yrclim)

    return {
        'dTdt': dTdt,
        'dTpdt': dTpdt,
        'umdTmdx': umdTmdx,
        'updTmdx': updTmdx,
        'umdTpdx': umdTpdx,
        'updTpdx': updTpdx,
        'vmdTmdy': vmdTmdy,
        'vpdTmdy': vpdTmdy,
        'vmdTpdy': vmdTpdy,
        'vpdTpdy': vpdTpdy,
        'mnupdTpdx': mnupdTpdx,
        'mnvpdTpdy': mnvpdTpdy,
    }
