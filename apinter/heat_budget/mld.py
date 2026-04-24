"""Mixed-layer depth utilities on a regular lat/lon grid.

Port of Matlab functions mldavg_varytime.m, submld_varytime.m, botmld_varytime.m.
Reference: Stevenson et al. (2017) Climate Dynamics.
"""
import numpy as np
import xarray as xr


def mldavg_varytime(mld, field, z):
    """Depth-weighted average of a 4D field over the mixed layer.

    Weights each level by its layer thickness, computed from the lev coordinate
    (assuming lev gives layer centres, surface at z=0).

    Parameters
    ----------
    mld : xr.DataArray (time, lat, lon)
        Mixed layer depth in metres.
    field : xr.DataArray (time, lev, lat, lon)
        Field to average (e.g. temperature, velocity).
    z : xr.DataArray (lev,) or (lev, lat, lon)
        Depth levels in metres (positive downward).

    Returns
    -------
    xr.DataArray (time, lat, lon) — depth-weighted mean over the ML.
    """
    z_vals = z.values if z.ndim == 1 else z
    if z.ndim == 1:
        layer_bot = np.zeros(len(z_vals))
        for i in range(len(z_vals)):
            layer_bot[i] = 2.0 * z_vals[i] - (layer_bot[i - 1] if i > 0 else 0.0)
        thickness = np.diff(np.concatenate([[0.0], layer_bot]))
        dz = xr.DataArray(thickness, dims=['lev'], coords={'lev': field.lev})
    else:
        dz = z.diff('lev')
        dz = xr.concat([z.isel(lev=0), dz], dim='lev')

    mask = z <= mld
    weighted_sum = (field * dz).where(mask, 0.0).sum(dim='lev')
    total_weight = dz.where(mask, 0.0).sum(dim='lev')
    return xr.where(total_weight > 0, weighted_sum / total_weight, np.nan)


def submld_varytime(mld, field, z, search_type='first'):
    """Extract field value just below the mixed layer depth.

    Matches Matlab submld_varytime.m: ``find(z > mld, 1, 'first')``.
    Returns NaN where no sub-MLD level exists.

    Parameters
    ----------
    search_type : {'first', 'last'}
        'first' for CESM/POP/ORAS5 ordering, 'last' for ROMS.
    """
    if z.ndim == 1:
        z_broadcast = z.broadcast_like(field)
    else:
        z_broadcast = z

    diff = z_broadcast - mld
    nlev = field.sizes['lev']
    idx_arr = xr.DataArray(np.arange(nlev), dims=['lev'],
                           coords={'lev': field.lev})
    idx_broadcast = idx_arr.broadcast_like(field)

    if search_type == 'first':
        condition = diff > 0
        has_valid = condition.any(dim='lev')
        masked_idx = xr.where(condition, idx_broadcast, nlev)
        target_idx = masked_idx.min(dim='lev')
    elif search_type == 'last':
        condition = diff < 0
        has_valid = condition.any(dim='lev')
        masked_idx = xr.where(condition, idx_broadcast, -1)
        target_idx = masked_idx.max(dim='lev')
    else:
        raise ValueError("search_type must be 'first' or 'last'")

    target_idx = target_idx.clip(0, nlev - 1).astype(int)
    result = field.isel(lev=target_idx)
    return xr.where(has_valid, result, np.nan)


def botmld_varytime(mld, field, z, search_type='first'):
    """Extract field at the deepest level still inside the ML + layer thickness.

    Matches Matlab botmld_varytime.m: ``find(z < mld, 1, 'last')`` for CESM/POP.
    Returns (value, thickness).
    """
    if z.ndim == 1:
        z_broadcast = z.broadcast_like(field)
    else:
        z_broadcast = z

    diff = z_broadcast - mld
    nlev = field.sizes['lev']
    idx_arr = xr.DataArray(np.arange(nlev), dims=['lev'],
                           coords={'lev': field.lev})
    idx_broadcast = idx_arr.broadcast_like(field)

    condition = diff < 0
    has_valid = condition.any(dim='lev')
    masked_idx = xr.where(condition, idx_broadcast, -1)
    target_idx = masked_idx.max(dim='lev').clip(0, nlev - 1).astype(int)

    result = field.isel(lev=target_idx)
    result = xr.where(has_valid, result, np.nan)

    next_idx = (target_idx + 1).clip(0, nlev - 1).astype(int)
    if z.ndim == 1:
        z_at_target = z.values[target_idx.values]
        z_at_next = z.values[next_idx.values]
        thickness = xr.DataArray(
            z_at_next - z_at_target,
            dims=result.dims, coords=result.coords,
        )
    else:
        thickness = z_broadcast.isel(lev=next_idx) - z_broadcast.isel(lev=target_idx)
    thickness = xr.where(has_valid, thickness, np.nan)

    return result, thickness
