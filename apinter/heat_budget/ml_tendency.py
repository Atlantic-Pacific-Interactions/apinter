"""Mixed-layer mean temperature with partial-cell interpolation.

Reference: Nnamchi et al. (2021, Nature Communications).
"""
import numpy as np
import xarray as xr


def _layer_bounds(z):
    """Compute layer top/bottom boundaries from level centers.

    Parameters
    ----------
    z : 1-D array
        Depth of level centers (m, positive downward), sorted ascending.

    Returns
    -------
    z_top, z_bot : 1-D arrays (nlev,)
        Top and bottom boundaries of each layer.
    """
    z = np.asarray(z, dtype=float)
    nlev = len(z)
    # Boundaries at midpoints between consecutive level centers
    z_mid = 0.5 * (z[:-1] + z[1:])  # (nlev-1,)
    z_top = np.empty(nlev)
    z_bot = np.empty(nlev)
    z_top[0] = 0.0
    z_top[1:] = z_mid
    z_bot[:-1] = z_mid
    z_bot[-1] = 2.0 * z[-1] - z_mid[-1]  # extrapolate last layer
    return z_top, z_bot


def ml_mean_temperature(temp, mld, z=None):
    """Compute mixed layer mean temperature with partial-cell interpolation.

    ⟨T⟩ = (1/h) [Σ_{full layers} T_k · Δz_k + T_partial · Δz_partial]

    For the layer where MLD falls between z_top and z_bot, only the
    fraction above MLD contributes, with temperature linearly interpolated.

    Parameters
    ----------
    temp : xr.DataArray (time, lev, lat, lon)
        3-D temperature field.
    mld : xr.DataArray (time, lat, lon)
        Mixed layer depth (m, positive).
    z : array-like, optional
        Depth levels (m). Defaults to temp.lev.

    Returns
    -------
    T_ml : xr.DataArray (time, lat, lon)
        Mixed layer mean temperature.
    """
    if z is None:
        z_vals = temp.lev.values.astype(float)
    else:
        z_vals = np.asarray(z, dtype=float)

    z_top, z_bot = _layer_bounds(z_vals)
    nlev = len(z_vals)

    # Convert to DataArrays for broadcasting
    z_top_da = xr.DataArray(z_top, dims=['lev'], coords={'lev': temp.lev})
    z_bot_da = xr.DataArray(z_bot, dims=['lev'], coords={'lev': temp.lev})

    # Effective thickness of each layer within the mixed layer:
    #   - full layers where z_bot <= mld
    #   - partial layer where z_top < mld < z_bot
    #   - zero where z_top >= mld
    eff_top = z_top_da  # always the layer top
    eff_bot = xr.where(z_bot_da <= mld, z_bot_da, mld)  # clip at MLD
    dz_eff = (eff_bot - eff_top).clip(min=0.0)

    # Weighted sum
    weighted_sum = (temp * dz_eff).sum(dim='lev')
    total_depth = dz_eff.sum(dim='lev')

    # Avoid division by zero (land/missing MLD)
    T_ml = xr.where(total_depth > 0, weighted_sum / total_depth, np.nan)
    T_ml.name = 'T_ml'

    return T_ml


def ml_tendency(T_ml, dt_seconds=None):
    """Compute ∂⟨T⟩/∂t using centered differencing on monthly data.

    ∂T/∂t(m) = (T(m+1) - T(m-1)) / (2·Δt)

    Forward/backward differencing at the endpoints.

    Parameters
    ----------
    T_ml : xr.DataArray (time, lat, lon)
        Mixed layer mean temperature (monthly).
    dt_seconds : float, optional
        Time step in seconds. Defaults to 30 * 86400 (30 days).

    Returns
    -------
    dTdt : xr.DataArray (time, lat, lon)
        Temperature tendency (degC/s).
    """
    if dt_seconds is None:
        dt_seconds = 30.0 * 86400.0

    nt = len(T_ml.time)
    T_vals = T_ml.values  # (time, lat, lon)

    dTdt_vals = np.full_like(T_vals, np.nan)

    # Centered differencing for interior
    dTdt_vals[1:-1] = (T_vals[2:] - T_vals[:-2]) / (2.0 * dt_seconds)

    # Forward/backward at endpoints
    dTdt_vals[0] = (T_vals[1] - T_vals[0]) / dt_seconds
    dTdt_vals[-1] = (T_vals[-1] - T_vals[-2]) / dt_seconds

    dTdt = xr.DataArray(dTdt_vals, dims=T_ml.dims, coords=T_ml.coords)
    dTdt.name = 'dTdt'

    return dTdt
