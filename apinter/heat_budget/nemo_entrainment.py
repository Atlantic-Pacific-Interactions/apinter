"""Reynolds-decomposed vertical advection (entrainment) on the NEMO C-grid.

Convention: w positive upward; w_sub > 0 means upwelling pushes water into
the ML, i.e. entrainment. Heaviside step is applied to the climatological-mean
entrainment velocity.

Reference: Graham et al. (2014) Climate Dynamics, 43:2399-2414.
"""
import numpy as np
import xarray as xr

from .constants import CLIM_START, CLIM_END, MLD_MIN
from .tendency import compute_tendency


def compute_w_from_continuity(u, v, grid):
    """Derive vertical velocity from continuity on the NEMO C-grid.

    Integrates dw/dz = -(du/dx + dv/dy) downward from the surface (w=0).
    Result is w at w-levels (bottom of each T-cell), positive upward.

    Parameters
    ----------
    u : xr.DataArray (time, z, y, x)
        Zonal velocity at U-points (m/s).
    v : xr.DataArray (time, z, y, x)
        Meridional velocity at V-points (m/s).
    grid : NemoGrid
        Grid metrics object.

    Returns
    -------
    w : xr.DataArray (time, z, y, x)
        Vertical velocity at the bottom of each T-cell (m/s).
        Positive upward.
    """
    # Horizontal divergence at T-points for each level
    # div_h = (1/e1t·e2t) * [u[i]*e2u[i] - u[i-1]*e2u[i-1]
    #                       + v[j]*e1v[j] - v[j-1]*e1v[j-1]]
    div = grid.div_h(u, v)  # (time, z, y, x)

    # Layer thicknesses — match the number of z-levels in the input
    nz = u.sizes['z']
    e3t = grid.e3t_0.isel(z=slice(0, nz))

    # w(k) = w(k-1) - div_h(k) * e3t(k)
    # w(surface) = 0, integrate downward
    # w positive upward: convergence (div < 0) → upwelling (w > 0)
    w_increment = -div * e3t
    w = w_increment.cumsum(dim='z')

    return w


def submld_w(mld, w, e3t):
    """Extract vertical velocity at the base of the mixed layer.

    Parameters
    ----------
    mld : xr.DataArray (time, y, x)
        Mixed layer depth (m).
    w : xr.DataArray (time, z, y, x)
        Vertical velocity (m/s), positive upward.
    e3t : xr.DataArray (z,)
        Layer thicknesses (m).

    Returns
    -------
    xr.DataArray (time, y, x)
        w at MLD base.
    """
    # w is at the bottom of each T-cell; depth of w-level = cumsum(e3t)
    z_bot = e3t.cumsum('z')

    nz = w.sizes['z']
    idx_arr = xr.DataArray(np.arange(nz), dims=['z'], coords={'z': w.z})
    idx_broadcast = idx_arr.broadcast_like(w)

    # Find first w-level at or below MLD
    condition = z_bot >= mld
    has_valid = condition.any(dim='z')
    masked_idx = xr.where(condition, idx_broadcast, nz)
    target_idx = masked_idx.min(dim='z').clip(0, nz - 1).astype(int)

    # Vectorized isel with a per-pixel indexer attaches the selected level's
    # own `z` coordinate value as a new (time, y, x) non-dim coordinate;
    # drop it — Tsub and wsub generally pick different levels per pixel, so
    # a shared `z` coordinate would conflict when bundled together.
    result = w.isel(z=target_idx).drop_vars('z', errors='ignore')
    result = xr.where(has_valid, result, np.nan)

    return result


def vertadv_ml_rd(mld, umld, vmld, Tmld, Tsub, wsub, grid, yrclim=None):
    """Compute Reynolds-decomposed vertical advection on NEMO C-grid.

    Entrainment velocity: w_e = dH/dt + u_T·∂H/∂x + v_T·∂H/∂y + w_sub
    where u_T, v_T are velocities interpolated to T-points, and
    w_sub is positive upward.

    Vertical advection = w_e * (T_ML - T_sub) / H  (with Heaviside on w_e)

    Parameters
    ----------
    mld : xr.DataArray (time, y, x)
        Mixed layer depth (m).
    umld : xr.DataArray (time, y, x)
        ML-averaged zonal velocity at T-points (m/s).
        Must already be interpolated from U-points to T-points.
    vmld : xr.DataArray (time, y, x)
        ML-averaged meridional velocity at T-points (m/s).
        Must already be interpolated from V-points to T-points.
    Tmld : xr.DataArray (time, y, x)
        ML-averaged temperature at T-points (degC).
    Tsub : xr.DataArray (time, y, x)
        Temperature just below ML at T-points (degC).
    wsub : xr.DataArray (time, y, x)
        Vertical velocity at MLD base (m/s), positive upward.
    grid : NemoGrid
        Grid metrics object.
    yrclim : list [start_year, end_year], optional
        Climatology period.

    Returns
    -------
    dict of xr.DataArray
        Keys: w_entr, wmdTmdz, wpdTmdz, wmdTpdz, wpdTpdz, mnwpdTpdz
    """
    if yrclim is None:
        yrclim = [CLIM_START, CLIM_END]

    clim_slice = slice(str(yrclim[0]), str(yrclim[1]))

    # --- MLD safety guard ---
    mld_invalid = (mld <= 0) | np.isnan(mld)
    mld_safe = mld.clip(min=MLD_MIN)

    # --- Time derivative of MLD ---
    dHdt = compute_tendency(mld_safe)

    # --- Spatial gradients of MLD at T-points (C-grid aware) ---
    dHdx = grid.grad_i_at_T(mld_safe)
    dHdy = grid.grad_j_at_T(mld_safe)

    # --- Entrainment velocity (positive = entraining from below) ---
    # w_sub positive upward means upwelling pushes water into ML → entrainment
    w_entr = dHdt + umld * dHdx + vmld * dHdy + wsub

    # --- Climatological means ---
    Tm_mld = Tmld.sel(time=clim_slice).groupby('time.month').mean('time')
    Tm_sub = Tsub.sel(time=clim_slice).groupby('time.month').mean('time')
    wm = w_entr.sel(time=clim_slice).groupby('time.month').mean('time')

    # Temperature difference
    deltaTm = Tm_mld - Tm_sub

    # Heaviside: only entrainment (w_e > 0, i.e., MLD deepening)
    wsgn = xr.where(wm > 0, 1.0, 0.0)

    # Map to full time series
    month = Tmld['time.month']
    Tm_mld_full = Tm_mld.sel(month=month).drop_vars('month')
    Tm_sub_full = Tm_sub.sel(month=month).drop_vars('month')
    wm_full = wm.sel(month=month).drop_vars('month')
    wsgn_full = wsgn.sel(month=month).drop_vars('month')
    deltaTm_full = deltaTm.sel(month=month).drop_vars('month')

    # --- Anomalies ---
    wp = w_entr - wm_full
    Tp_mld = Tmld - Tm_mld_full
    Tp_sub = Tsub - Tm_sub_full
    deltaTp = Tp_mld - Tp_sub

    # --- 4 Reynolds decomposition terms ---
    wmdTmdz = wsgn_full * wm_full * deltaTm_full / mld_safe
    wpdTmdz = wsgn_full * wp * deltaTm_full / mld_safe
    wmdTpdz = wsgn_full * wm_full * deltaTp / mld_safe
    wpdTpdz = wsgn_full * wp * deltaTp / mld_safe

    # --- Climatological mean of eddy-eddy term ---
    mnwpdTpdz_clim = wpdTpdz.sel(time=clim_slice).groupby('time.month').mean('time')
    mnwpdTpdz = mnwpdTpdz_clim.sel(month=month).drop_vars('month')

    # --- Apply MLD invalid mask ---
    for var in [wmdTmdz, wpdTmdz, wmdTpdz, wpdTpdz, mnwpdTpdz]:
        var = xr.where(mld_invalid, np.nan, var)

    return {
        'w_entr': w_entr,
        'wmdTmdz': wmdTmdz,
        'wpdTmdz': wpdTmdz,
        'wmdTpdz': wmdTpdz,
        'wpdTpdz': wpdTpdz,
        'mnwpdTpdz': mnwpdTpdz,
    }
