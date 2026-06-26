"""Tests for the regular-grid entrainment module."""
import numpy as np
import pandas as pd
import xarray as xr

from apinter.heat_budget.entrainment import (
    compute_w_from_continuity, vertadv_ml_rd,
)


def test_compute_w_from_continuity_nondivergent_flow_gives_zero_w():
    n_t, n_lev, n_lat, n_lon = 1, 4, 5, 8
    time = pd.date_range('2000-01-01', periods=n_t, freq='MS')
    lev = np.array([5.0, 15.0, 25.0, 40.0])
    lat = np.linspace(-10.0, 10.0, n_lat)
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    coords = {'time': time, 'lev': lev, 'lat': lat, 'lon': lon}
    dims = ['time', 'lev', 'lat', 'lon']
    u = xr.DataArray(np.ones((n_t, n_lev, n_lat, n_lon)), coords=coords, dims=dims)
    v = xr.DataArray(np.zeros((n_t, n_lev, n_lat, n_lon)), coords=coords, dims=dims)

    w = compute_w_from_continuity(u, v)
    mid = dict(lat=slice(1, -1), lon=slice(1, -1))
    assert np.allclose(w.isel(**mid).values, 0.0, atol=1e-15)


def test_vertadv_ml_rd_returns_expected_keys():
    n_t, n_lat, n_lon = 60, 3, 4
    time = pd.date_range('1981-01-01', periods=n_t, freq='MS')
    lat = np.array([-5.0, 0.0, 5.0])
    lon = np.array([100.0, 120.0, 140.0, 160.0])
    coords = {'time': time, 'lat': lat, 'lon': lon}
    dims = ['time', 'lat', 'lon']
    shape = (n_t, n_lat, n_lon)

    rng = np.random.default_rng(0)

    def _da(val):
        if np.isscalar(val):
            arr = np.full(shape, val)
        else:
            arr = val
        return xr.DataArray(arr, coords=coords, dims=dims)

    mld_vals = (30.0
                + 5.0 * np.sin(2 * np.pi * np.arange(n_t)[:, None, None] / 12)
                + np.zeros((1, n_lat, n_lon)))
    mld = _da(np.broadcast_to(mld_vals, shape).copy())
    usub = _da(0.05 + rng.standard_normal(shape) * 0.01)
    vsub = _da(rng.standard_normal(shape) * 0.01)
    Tmld = _da(22.0 + rng.standard_normal(shape) * 0.2)
    Tsub = Tmld - _da(2.0)
    wsub = _da(rng.standard_normal(shape) * 1e-5)

    out = vertadv_ml_rd(mld, usub, vsub, Tmld, Tsub, wsub, yrclim=[1981, 1985])
    assert set(out) == {'w_entr', 'wmdTmdz', 'wpdTmdz', 'wmdTpdz', 'wpdTpdz',
                        'mnwpdTpdz'}
    for v in out.values():
        assert v.sizes['time'] == n_t


def test_vertadv_masks_invalid_mld():
    n_t, n_lat, n_lon = 60, 3, 3
    time = pd.date_range('1981-01-01', periods=n_t, freq='MS')
    lat = np.array([-5.0, 0.0, 5.0])
    lon = np.array([100.0, 110.0, 120.0])
    coords = {'time': time, 'lat': lat, 'lon': lon}
    dims = ['time', 'lat', 'lon']
    shape = (n_t, n_lat, n_lon)

    def _da(v):
        return xr.DataArray(np.full(shape, v), coords=coords, dims=dims)

    mld = _da(0.0)
    zero = _da(0.0)
    out = vertadv_ml_rd(mld, zero, zero, _da(22.0), _da(20.0), zero,
                        yrclim=[1981, 1985])
    assert np.all(np.isnan(out['wmdTmdz'].values))
