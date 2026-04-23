"""Helmholtz decomposition of 2D horizontal winds via windspharm.

Returns the divergent (irrotational) component of the wind and/or the
velocity potential. Handles NaN input gracefully: cells that were NaN in
(u, v) are filled with the zonal mean before decomposition (which reduces
artificial divergence at coastlines) and restored to NaN on output.

Ported from ``E3SMS/scripts/circulation/plot_walker_streamfunction_v2.py``
and ``plot_hadley_cell_v2.py`` (scientific primitives).
"""
from typing import Tuple

import numpy as np
import xarray as xr


def _fill_nan_with_zonal_mean(u: xr.DataArray, v: xr.DataArray
                              ) -> Tuple[xr.DataArray, xr.DataArray, np.ndarray]:
    """Return (u_filled, v_filled, nan_mask) for feeding to VectorWind."""
    nan_mask = np.isnan(u.values) | np.isnan(v.values)
    u_filled = u.fillna(u.mean('lon')).fillna(0.0)
    v_filled = v.fillna(v.mean('lon')).fillna(0.0)
    return u_filled, v_filled, nan_mask


def get_divergent_u(u_clim: xr.DataArray, v_clim: xr.DataArray) -> np.ndarray:
    """Divergent (irrotational) zonal wind component from global (u, v).

    Parameters
    ----------
    u_clim, v_clim : xr.DataArray with dims (lat, lon).
        Climatological mean winds.

    Returns
    -------
    np.ndarray (lat, lon)
        Divergent u with NaN restored where input was NaN.
    """
    from windspharm.xarray import VectorWind
    u_filled, v_filled, nan_mask = _fill_nan_with_zonal_mean(u_clim, v_clim)
    w = VectorWind(u_filled, v_filled)
    ud, _ = w.irrotationalcomponent()
    ud_vals = ud.values if hasattr(ud, 'values') else np.asarray(ud)
    ud_vals[nan_mask] = np.nan
    return ud_vals


def get_divergent_v(u_clim: xr.DataArray, v_clim: xr.DataArray) -> np.ndarray:
    """Divergent (irrotational) meridional wind component from global (u, v).

    Parameters
    ----------
    u_clim, v_clim : xr.DataArray with dims (lat, lon).
        Climatological mean winds.

    Returns
    -------
    np.ndarray (lat, lon)
        Divergent v with NaN restored where input was NaN. The result is
        aligned to ``u_clim.lat`` ordering (windspharm otherwise returns
        lat descending regardless of input).
    """
    from windspharm.xarray import VectorWind
    u_filled, v_filled, nan_mask = _fill_nan_with_zonal_mean(u_clim, v_clim)
    w = VectorWind(u_filled, v_filled)
    _, vd = w.irrotationalcomponent()
    vd = vd.sel(lat=u_clim.lat)
    vd_vals = vd.values
    vd_vals[nan_mask] = np.nan
    return vd_vals


def compute_velpot(u_clim: xr.DataArray, v_clim: xr.DataArray):
    """Velocity potential and divergent wind components.

    Returns
    -------
    (chi, u_div, v_div) : tuple of xr.DataArray
        Velocity potential (m^2/s) and divergent wind components (m/s)
        on the same grid as the inputs.
    """
    from windspharm.xarray import VectorWind
    w = VectorWind(u_clim, v_clim)
    chi = w.velocitypotential()
    u_div, v_div = w.irrotationalcomponent()
    return chi, u_div, v_div
