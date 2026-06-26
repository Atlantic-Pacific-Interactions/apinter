"""Tests for Reynolds-decomposed horizontal advection on a regular grid."""
import numpy as np
import pandas as pd
import xarray as xr

from apinter.heat_budget.advection import advection_ml_rd


def _uniform_field(n_t=60):
    time = pd.date_range('1981-01-01', periods=n_t, freq='MS')
    lat = np.arange(-10.0, 11.0, 1.0)
    lon = np.arange(0.0, 360.0, 1.0)
    coords = {'time': time, 'lat': lat, 'lon': lon}
    dims = ['time', 'lat', 'lon']

    t_ix = np.arange(n_t)[:, None, None]
    lon_b = lon[None, None, :]
    season = 0.5 * np.sin(2 * np.pi * t_ix / 12)
    T = 20.0 + 0.01 * lon_b + season
    T = np.broadcast_to(T, (n_t, len(lat), len(lon))).copy()

    u = np.ones_like(T) * 1.0
    v = np.zeros_like(T)

    Tmld = xr.DataArray(T, coords=coords, dims=dims)
    umld = xr.DataArray(u, coords=coords, dims=dims)
    vmld = xr.DataArray(v, coords=coords, dims=dims)
    return Tmld, umld, vmld


def test_advection_ml_rd_returns_all_twelve_keys():
    Tmld, umld, vmld = _uniform_field()
    out = advection_ml_rd(Tmld, umld, vmld, yrclim=[1981, 1985])
    expected = {
        'dTdt', 'dTpdt',
        'umdTmdx', 'updTmdx', 'umdTpdx', 'updTpdx',
        'vmdTmdy', 'vpdTmdy', 'vmdTpdy', 'vpdTpdy',
        'mnupdTpdx', 'mnvpdTpdy',
    }
    assert set(out) == expected


def test_reynolds_decomposition_sums_to_total_u_grad_t():
    Tmld, umld, vmld = _uniform_field()
    out = advection_ml_rd(Tmld, umld, vmld, yrclim=[1981, 1985])

    from apinter.heat_budget.constants import RE
    lat_rad = np.deg2rad(Tmld.lat)
    cos_lat = np.cos(lat_rad)
    dTdx = Tmld.differentiate('lon', edge_order=1) / (RE * cos_lat * np.deg2rad(1.0))
    total = umld * dTdx

    sum_rd = out['umdTmdx'] + out['updTmdx'] + out['umdTpdx'] + out['updTpdx']
    mid = dict(lat=slice(2, -2), lon=slice(2, -2))
    assert np.allclose(sum_rd.isel(**mid).values,
                       total.isel(**mid).values, rtol=1e-6, atol=1e-12)
