"""Reynolds-decomposed horizontal advection on the NEMO ORCA C-grid.

Uses exact grid metrics (e1u, e2v, ...) from mesh_mask.nc instead of the
spherical-coordinate approximation used on the regular lat/lon grid.

Reference: Graham et al. (2014) Climate Dynamics, 43:2399-2414.
"""
import xarray as xr

from .constants import CLIM_START, CLIM_END
from .tendency import compute_tendency, compute_anomaly_tendency


def advection_ml_rd(Tmld, umld, vmld, grid, yrclim=None):
    """Compute Reynolds-decomposed horizontal advection on NEMO C-grid.

    Parameters
    ----------
    Tmld : xr.DataArray (time, y, x)
        ML-averaged temperature at T-points (degC).
    umld : xr.DataArray (time, y, x)
        ML-averaged zonal velocity at U-points (m/s).
    vmld : xr.DataArray (time, y, x)
        ML-averaged meridional velocity at V-points (m/s).
    grid : NemoGrid
        Grid metrics object.
    yrclim : list [start_year, end_year], optional
        Climatology period.

    Returns
    -------
    dict of xr.DataArray
        Keys: dTdt, dTpdt,
              umdTmdx, updTmdx, umdTpdx, updTpdx,
              vmdTmdy, vpdTmdy, vmdTpdy, vpdTpdy,
              mnupdTpdx, mnvpdTpdy
    """
    if yrclim is None:
        yrclim = [CLIM_START, CLIM_END]

    clim_slice = slice(str(yrclim[0]), str(yrclim[1]))

    # --- Total advection: u·∂T/∂x and v·∂T/∂y at T-points ---
    uadv_total = grid.u_dot_grad_i(umld, Tmld)
    vadv_total = grid.v_dot_grad_j(vmld, Tmld)

    # --- Monthly climatologies ---
    Tm = Tmld.sel(time=clim_slice).groupby('time.month').mean('time')
    um = umld.sel(time=clim_slice).groupby('time.month').mean('time')
    vm = vmld.sel(time=clim_slice).groupby('time.month').mean('time')

    # Map climatology to full time series
    month = Tmld['time.month']
    Tm_full = Tm.sel(month=month).drop_vars('month')
    um_full = um.sel(month=month).drop_vars('month')
    vm_full = vm.sel(month=month).drop_vars('month')

    # --- Mean advection of mean gradient ---
    umdTmdx = grid.u_dot_grad_i(um_full, Tm_full)
    vmdTmdy = grid.v_dot_grad_j(vm_full, Tm_full)

    # --- Anomalies ---
    up = umld - um_full
    vp = vmld - vm_full

    # Anomalous advection of mean gradient
    updTmdx = grid.u_dot_grad_i(up, Tm_full)
    vpdTmdy = grid.v_dot_grad_j(vp, Tm_full)

    # Mean advection of anomalous gradient
    # Anomalous T gradient: total gradient - mean gradient
    # Total advection of T by um = um · grad(T)
    # Mean advection of mean gradient = um · grad(Tm)
    # Mean advection of anomalous gradient = um · grad(T') = um · grad(T) - um · grad(Tm)
    umdTpdx = grid.u_dot_grad_i(um_full, Tmld) - umdTmdx
    vmdTpdy = grid.v_dot_grad_j(vm_full, Tmld) - vmdTmdy

    # Anomalous advection of anomalous gradient
    # up · grad(T') = up · grad(T) - up · grad(Tm)
    updTpdx = grid.u_dot_grad_i(up, Tmld) - updTmdx
    vpdTpdy = grid.v_dot_grad_j(vp, Tmld) - vpdTmdy

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
