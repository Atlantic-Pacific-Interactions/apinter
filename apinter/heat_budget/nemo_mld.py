"""Depth-weighted MLD utilities on the NEMO ORCA C-grid.

Uses exact layer thicknesses e3t from mesh_mask.nc (not simple nanmean over
non-uniform levels). Reference: Stevenson et al. (2017) Climate Dynamics.
"""
import numpy as np
import xarray as xr


def mldavg_varytime(mld, field, e3t):
    """Depth-weighted average of a 3D field over the mixed layer.

    For each (time, y, x) point, averages field values at depths shallower
    than MLD, weighted by the layer thickness e3t.

    Parameters
    ----------
    mld : xr.DataArray (time, y, x)
        Mixed layer depth in meters.
    field : xr.DataArray (time, z, y, x)
        Field to average (e.g., temperature, velocity).
    e3t : xr.DataArray (z,)
        Layer thicknesses in meters.

    Returns
    -------
    xr.DataArray (time, y, x)
        Depth-weighted field average over the mixed layer.
    """
    # Depth of the center of each level
    z_center = e3t.cumsum('z') - 0.5 * e3t

    # Mask: keep only levels whose center is within the MLD
    mask = z_center <= mld

    # Weighted average: sum(field * e3t) / sum(e3t) over masked levels
    weighted_sum = (field * e3t).where(mask, 0.0).sum(dim='z')
    total_weight = e3t.where(mask, 0.0).sum(dim='z')

    # Avoid division by zero (land or very shallow points)
    result = xr.where(total_weight > 0, weighted_sum / total_weight, np.nan)

    return result


def submld_varytime(mld, field, e3t, search_type='first'):
    """Extract field value just below the mixed layer depth.

    Finds the first level whose center depth exceeds the MLD.

    Parameters
    ----------
    mld : xr.DataArray (time, y, x)
        Mixed layer depth in meters.
    field : xr.DataArray (time, z, y, x)
        Field to sample.
    e3t : xr.DataArray (z,)
        Layer thicknesses in meters.
    search_type : str
        'first' — first level below MLD (default, standard).

    Returns
    -------
    xr.DataArray (time, y, x)
        Field value at the first level below MLD.  NaN where no valid level.
    """
    # Depth of the center of each level
    z_center = e3t.cumsum('z') - 0.5 * e3t

    nz = field.sizes['z']
    idx_arr = xr.DataArray(np.arange(nz), dims=['z'],
                           coords={'z': field.z})
    idx_broadcast = idx_arr.broadcast_like(field)

    # Levels below MLD
    condition = z_center > mld
    has_valid = condition.any(dim='z')

    if search_type == 'first':
        masked_idx = xr.where(condition, idx_broadcast, nz)
        target_idx = masked_idx.min(dim='z')
    elif search_type == 'last':
        masked_idx = xr.where(~condition, idx_broadcast, -1)
        target_idx = masked_idx.max(dim='z')
    else:
        raise ValueError("search_type must be 'first' or 'last'")

    target_idx = target_idx.clip(0, nz - 1).astype(int)
    result = field.isel(z=target_idx)
    result = xr.where(has_valid, result, np.nan)

    return result
