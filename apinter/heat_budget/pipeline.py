"""End-to-end ML heat-budget pipeline.

Each function takes pre-loaded DataArrays and returns one xr.Dataset bundling
24 fields (5 intermediates + 12 advection + 6 entrainment + 1 surface flux).
The two backends are kept as separate functions because the staggering rules
differ enough that a unified version would lean on noisy `if backend ==`
branches.
"""
import xarray as xr

from .advection import advection_ml_rd
from .constants import CLIM_START, CLIM_END
from .entrainment import compute_w_from_continuity, vertadv_ml_rd
from .mld import mldavg_varytime, submld_varytime
from .nemo_advection import advection_ml_rd as nemo_advection_ml_rd
from .nemo_entrainment import (
    compute_w_from_continuity as nemo_compute_w,
    submld_w as nemo_submld_w,
    vertadv_ml_rd as nemo_vertadv_ml_rd,
)
from .nemo_mld import (
    mldavg_varytime as nemo_mldavg,
    submld_varytime as nemo_submld,
)
from .surface_flux import surface_heat_flux


_DESCRIPTION = 'Mixed-layer heat budget (Graham 2014, Stevenson 2017)'


def _bundle(intermediates, adv, ent, sfc, *, backend, yrclim):
    """Merge low-level dict outputs into one xr.Dataset with provenance attrs."""
    data_vars = {**intermediates, **adv, **ent, 'sfcflx': sfc}
    ds = xr.Dataset(data_vars)
    ds.attrs['backend'] = backend
    ds.attrs['yrclim'] = str(list(yrclim))
    ds.attrs['description'] = _DESCRIPTION
    return ds


def compute_budget_regular(thetao, uo, vo, mld, qnet, qsw, *,
                            z=None, yrclim=None):
    """End-to-end ML heat budget on a regular lat/lon grid.

    Parameters
    ----------
    thetao : xr.DataArray (time, lev, lat, lon) — potential temperature [°C].
    uo, vo : xr.DataArray (time, lev, lat, lon) — velocities at T-points [m/s].
    mld    : xr.DataArray (time, lat, lon)      — mixed-layer depth [m].
    qnet   : xr.DataArray (time, lat, lon)      — net surface heat flux [W/m²].
    qsw    : xr.DataArray (time, lat, lon)      — net shortwave at surface [W/m²].
    z      : xr.DataArray, optional             — depth coord; defaults to thetao.lev.
    yrclim : [start_year, end_year], optional   — defaults to [CLIM_START, CLIM_END].

    Returns
    -------
    xr.Dataset (time, lat, lon) with 24 data_vars.
    """
    if yrclim is None:
        yrclim = [CLIM_START, CLIM_END]
    if z is None:
        z = thetao.lev

    Tmld = mldavg_varytime(mld, thetao, z)
    Tsub = submld_varytime(mld, thetao, z)
    umld = mldavg_varytime(mld, uo, z)
    vmld = mldavg_varytime(mld, vo, z)
    usub = submld_varytime(mld, uo, z)
    vsub = submld_varytime(mld, vo, z)

    w3d = compute_w_from_continuity(uo, vo)
    wsub = submld_varytime(mld, w3d, z)

    adv = advection_ml_rd(Tmld, umld, vmld, yrclim=yrclim)
    ent = vertadv_ml_rd(mld, usub, vsub, Tmld, Tsub, wsub, yrclim=yrclim)
    sfc = surface_heat_flux(qnet, qsw, mld)

    intermediates = {'Tmld': Tmld, 'Tsub': Tsub,
                     'umld': umld, 'vmld': vmld, 'wsub': wsub}
    return _bundle(intermediates, adv, ent, sfc,
                   backend='regular', yrclim=yrclim)


def compute_budget_nemo(thetao, uo, vo, mld, qnet, qsw, grid, *, yrclim=None):
    """End-to-end ML heat budget on the NEMO ORCA C-grid.

    Parameters
    ----------
    thetao : xr.DataArray (time, z, y, x) at T-points [°C].
    uo     : xr.DataArray (time, z, y, x) at U-points [m/s].
    vo     : xr.DataArray (time, z, y, x) at V-points [m/s].
    mld    : xr.DataArray (time, y, x) at T-points [m].
    qnet, qsw : xr.DataArray (time, y, x) at T-points [W/m²].
    grid   : NemoGrid — wraps mesh_mask.nc.
    yrclim : [start_year, end_year], optional — defaults to [CLIM_START, CLIM_END].

    Returns
    -------
    xr.Dataset (time, y, x) with 24 data_vars.

    Notes
    -----
    Inputs must be eagerly materialised (call ``.load()`` on dask-backed arrays
    first). The NEMO backend's vectorised ``xarray.isel`` does not support
    dask-backed indexers.
    """
    if yrclim is None:
        yrclim = [CLIM_START, CLIM_END]

    e3t = grid.e3t_0.isel(z=slice(0, thetao.sizes['z']))

    Tmld = nemo_mldavg(mld, thetao, e3t)
    Tsub = nemo_submld(mld, thetao, e3t)
    umld_U = nemo_mldavg(mld, uo, e3t)
    vmld_V = nemo_mldavg(mld, vo, e3t)
    umld_T = grid.u_to_T(umld_U)
    vmld_T = grid.v_to_T(vmld_V)

    w3d = nemo_compute_w(uo, vo, grid)
    wsub = nemo_submld_w(mld, w3d, e3t)

    adv = nemo_advection_ml_rd(Tmld, umld_U, vmld_V, grid, yrclim=yrclim)
    ent = nemo_vertadv_ml_rd(mld, umld_T, vmld_T, Tmld, Tsub, wsub, grid,
                              yrclim=yrclim)
    sfc = surface_heat_flux(qnet, qsw, mld)

    intermediates = {'Tmld': Tmld, 'Tsub': Tsub,
                     'umld': umld_U, 'vmld': vmld_V, 'wsub': wsub}
    return _bundle(intermediates, adv, ent, sfc,
                   backend='nemo', yrclim=yrclim)
