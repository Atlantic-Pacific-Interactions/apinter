"""Reynolds-decomposed vertical advection (entrainment) for the ML heat budget.

Port of Matlab vertadv_ml_rd.m. Reference: Graham et al. (2014) Climate
Dynamics, 43:2399-2414.
"""
import numpy as np
import xarray as xr
from .constants import RE, CLIM_START, CLIM_END, MLD_MIN
from .advection import _compute_gradients
from .tendency import compute_tendency


def compute_w_from_continuity(uvel, vvel):
    """Derive vertical velocity from horizontal velocities using continuity.

    Integrates the incompressibility condition dw/dz = -(du/dx + dv/dy)
    downward from the surface (w=0 at z=0), accumulating through each layer.

    This is needed because ORAS5 does not provide vertical velocity via CDS.
    Matches the approach in hb_cmip6.py:calc_w_no_w().

    Parameters
    ----------
    uvel : xr.DataArray (time, lev, lat, lon)
        Zonal velocity (m/s) on regular lat-lon grid.
    vvel : xr.DataArray (time, lev, lat, lon)
        Meridional velocity (m/s) on regular lat-lon grid.

    Returns
    -------
    w : xr.DataArray (time, lev, lat, lon)
        Vertical velocity (m/s) at the bottom of each layer.
        Positive = upward (out of the ocean).
    """
    lat_rad = np.deg2rad(uvel.lat)
    cos_lat = np.cos(lat_rad)

    # Horizontal divergence on spherical grid (correct spherical formula)
    # div = 1/(R cosφ) ∂u/∂λ + 1/(R cosφ) ∂(v cosφ)/∂φ
    dudx = uvel.differentiate('lon', edge_order=1) / (RE * cos_lat * np.deg2rad(1.0))
    # Meridional: must differentiate (v*cosφ) then divide by (R*cosφ)
    v_cos = vvel * cos_lat
    dvdy = v_cos.differentiate('lat', edge_order=1) / (RE * cos_lat * np.deg2rad(1.0))

    divergence = dudx + dvdy  # (time, lev, lat, lon)

    # Layer thicknesses from depth coordinate
    # Assume lev gives the center of each layer, surface at z=0
    z = uvel.lev.values
    layer_bot = np.zeros(len(z))
    for i in range(len(z)):
        layer_bot[i] = 2.0 * z[i] - (layer_bot[i - 1] if i > 0 else 0.0)
    thickness = np.diff(np.concatenate([[0.0], layer_bot]))

    # Integrate downward: w(k) = w(k-1) - divergence(k) * thickness(k)
    # w=0 at surface, w positive upward
    # From continuity: dw/dz = -(du/dx + dv/dy), so w = -∫ div_h dz
    thickness_da = xr.DataArray(thickness, dims=['lev'], coords={'lev': uvel.lev})

    # Cumulative sum of (-divergence * thickness) along depth
    w_increment = -divergence * thickness_da
    w = w_increment.cumsum(dim='lev')

    return w


def vertadv_ml_rd(mld, usub, vsub, Tmld, Tsub, wsub, yrclim=None):
    """Compute Reynolds-decomposed vertical advection terms.

    Computes entrainment velocity and decomposes w_e*(T_ml - T_sub)/H
    into 4 Reynolds terms. The Heaviside function is applied to the
    climatological mean entrainment velocity only.

    Parameters
    ----------
    mld : xr.DataArray (time, lat, lon) - Mixed layer depth (m).
    usub : xr.DataArray (time, lat, lon) - Zonal velocity at base of ML (m/s).
    vsub : xr.DataArray (time, lat, lon) - Meridional velocity at base of ML (m/s).
    Tmld : xr.DataArray (time, lat, lon) - ML-averaged temperature (C).
    Tsub : xr.DataArray (time, lat, lon) - Temperature just below ML (C).
    wsub : xr.DataArray (time, lat, lon) - Vertical velocity at base of ML (m/s).
    yrclim : list [start_year, end_year], optional

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

    # --- Spatial gradients of MLD ---
    dHdx, dHdy = _compute_gradients(mld_safe)

    # --- Entrainment velocity ---
    w_entr = dHdt + usub * dHdx + vsub * dHdy + wsub

    # --- Climatological means ---
    Tm_mld = Tmld.sel(time=clim_slice).groupby('time.month').mean('time')
    Tm_sub = Tsub.sel(time=clim_slice).groupby('time.month').mean('time')
    wm = w_entr.sel(time=clim_slice).groupby('time.month').mean('time')

    # Climatological temperature difference (raw, NOT divided by H)
    deltaTm = Tm_mld - Tm_sub

    # Map to full time series
    month = Tmld['time.month']
    Tm_mld_full = Tm_mld.sel(month=month).drop_vars('month')
    Tm_sub_full = Tm_sub.sel(month=month).drop_vars('month')
    wm_full = wm.sel(month=month).drop_vars('month')
    deltaTm_full = deltaTm.sel(month=month).drop_vars('month')

    # --- Heaviside function on climatological w_entr ---
    wsgn = xr.where(wm > 0, 1.0, 0.0)
    wsgn_full = wsgn.sel(month=month).drop_vars('month')

    # --- Anomalies ---
    wp = w_entr - wm_full
    Tp_mld = Tmld - Tm_mld_full
    Tp_sub = Tsub - Tm_sub_full
    deltaTp = Tp_mld - Tp_sub

    # --- 4 Reynolds decomposition terms (divided by time-varying MLD once) ---
    wmdTmdz = wsgn_full * wm_full * deltaTm_full / mld_safe
    wpdTmdz = wsgn_full * wp * deltaTm_full / mld_safe
    wmdTpdz = wsgn_full * wm_full * deltaTp / mld_safe
    wpdTpdz = wsgn_full * wp * deltaTp / mld_safe

    # --- Climatological mean of eddy-eddy term ---
    mnwpdTpdz_clim = wpdTpdz.sel(time=clim_slice).groupby('time.month').mean('time')
    mnwpdTpdz = mnwpdTpdz_clim.sel(month=month).drop_vars('month')

    # --- Apply MLD invalid mask ---
    wmdTmdz = xr.where(mld_invalid, np.nan, wmdTmdz)
    wpdTmdz = xr.where(mld_invalid, np.nan, wpdTmdz)
    wmdTpdz = xr.where(mld_invalid, np.nan, wmdTpdz)
    wpdTpdz = xr.where(mld_invalid, np.nan, wpdTpdz)
    mnwpdTpdz = xr.where(mld_invalid, np.nan, mnwpdTpdz)

    return {
        'w_entr': w_entr,
        'wmdTmdz': wmdTmdz,
        'wpdTmdz': wpdTmdz,
        'wmdTpdz': wmdTpdz,
        'wpdTpdz': wpdTpdz,
        'mnwpdTpdz': mnwpdTpdz,
    }
