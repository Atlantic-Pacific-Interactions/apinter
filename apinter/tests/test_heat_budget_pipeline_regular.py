"""Synthetic test for compute_budget_regular (regular-grid pipeline)."""
import numpy as np
import pandas as pd
import xarray as xr

from apinter.heat_budget import compute_budget_regular


def _build_inputs():
    """Tiny in-memory regular-grid stack: 5 yrs × 4 lev × 5 lat × 8 lon."""
    n_t, n_lev, n_lat, n_lon = 60, 4, 5, 8
    time = pd.date_range('1981-01-01', periods=n_t, freq='MS')
    lev = np.array([5.0, 15.0, 30.0, 60.0])
    lat = np.linspace(-10.0, 10.0, n_lat)
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    coords4 = {'time': time, 'lev': lev, 'lat': lat, 'lon': lon}
    coords3 = {'time': time, 'lat': lat, 'lon': lon}

    rng = np.random.default_rng(0)
    t_ix = np.arange(n_t)[:, None, None, None]
    season4 = 0.5 * np.sin(2 * np.pi * t_ix / 12)
    thetao_arr = (
        25.0 - 0.1 * lev[None, :, None, None] + season4
        + rng.standard_normal((n_t, n_lev, n_lat, n_lon)) * 0.05
    )
    uo_arr = 0.05 + rng.standard_normal((n_t, n_lev, n_lat, n_lon)) * 0.005
    vo_arr = rng.standard_normal((n_t, n_lev, n_lat, n_lon)) * 0.005

    thetao = xr.DataArray(thetao_arr,
                          coords=coords4, dims=['time', 'lev', 'lat', 'lon'])
    uo = xr.DataArray(uo_arr, coords=coords4, dims=['time', 'lev', 'lat', 'lon'])
    vo = xr.DataArray(vo_arr, coords=coords4, dims=['time', 'lev', 'lat', 'lon'])

    mld = xr.DataArray(
        np.broadcast_to(
            (25.0 + 5.0 * np.sin(2 * np.pi * np.arange(n_t)[:, None, None] / 12)),
            (n_t, n_lat, n_lon)).copy(),
        coords=coords3, dims=['time', 'lat', 'lon'],
    )
    qnet = xr.DataArray(np.full((n_t, n_lat, n_lon), 100.0),
                        coords=coords3, dims=['time', 'lat', 'lon'])
    qsw = xr.DataArray(np.full((n_t, n_lat, n_lon), 200.0),
                       coords=coords3, dims=['time', 'lat', 'lon'])
    return thetao, uo, vo, mld, qnet, qsw


def test_compute_budget_regular_returns_dataset_with_24_vars():
    thetao, uo, vo, mld, qnet, qsw = _build_inputs()
    ds = compute_budget_regular(thetao, uo, vo, mld, qnet, qsw,
                                 yrclim=[1981, 1985])
    expected = {
        'Tmld', 'Tsub', 'umld', 'vmld', 'wsub',
        'dTdt', 'dTpdt',
        'umdTmdx', 'updTmdx', 'umdTpdx', 'updTpdx',
        'vmdTmdy', 'vpdTmdy', 'vmdTpdy', 'vpdTpdy',
        'mnupdTpdx', 'mnvpdTpdy',
        'w_entr', 'wmdTmdz', 'wpdTmdz', 'wmdTpdz', 'wpdTpdz', 'mnwpdTpdz',
        'sfcflx',
    }
    assert set(ds.data_vars) == expected
    for name in expected:
        assert ds[name].sizes['time'] == thetao.sizes['time']
        assert ds[name].sizes['lat'] == thetao.sizes['lat']
        assert ds[name].sizes['lon'] == thetao.sizes['lon']


def test_compute_budget_regular_attaches_provenance_attrs():
    thetao, uo, vo, mld, qnet, qsw = _build_inputs()
    ds = compute_budget_regular(thetao, uo, vo, mld, qnet, qsw,
                                 yrclim=[1981, 1985])
    assert ds.attrs['backend'] == 'regular'
    assert ds.attrs['yrclim'] == '[1981, 1985]'
    assert 'Graham 2014' in ds.attrs['description']
