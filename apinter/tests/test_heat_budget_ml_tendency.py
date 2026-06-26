"""Tests for partial-cell ML mean temperature and tendency."""
import numpy as np
import pandas as pd
import xarray as xr

from apinter.heat_budget.ml_tendency import ml_mean_temperature, ml_tendency


def _make_temp(mld_val):
    n_t, n_lev, n_lat, n_lon = 3, 5, 2, 2
    time = pd.date_range('2000-01-01', periods=n_t, freq='MS')
    lev = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
    lat = np.array([0.0, 5.0])
    lon = np.array([100.0, 110.0])
    temp = xr.DataArray(
        np.full((n_t, n_lev, n_lat, n_lon), 20.0),
        coords={'time': time, 'lev': lev, 'lat': lat, 'lon': lon},
        dims=['time', 'lev', 'lat', 'lon'],
    )
    mld = xr.DataArray(
        np.full((n_t, n_lat, n_lon), mld_val),
        coords={'time': time, 'lat': lat, 'lon': lon},
        dims=['time', 'lat', 'lon'],
    )
    return temp, mld


def test_ml_mean_temperature_uniform_field():
    temp, mld = _make_temp(mld_val=25.0)
    T_ml = ml_mean_temperature(temp, mld)
    assert np.allclose(T_ml.values, 20.0)


def test_ml_mean_temperature_partial_cell_interpolation():
    """Linear T(z) = 20 - 0.1 z → ⟨T⟩ over [0, H] = 20 - 0.05·H."""
    n_t, n_lev, n_lat, n_lon = 1, 10, 1, 1
    time = pd.date_range('2000-01-01', periods=n_t, freq='MS')
    lev = np.arange(2.5, 50.0, 5.0)
    T_profile = 20.0 - 0.1 * lev
    temp = xr.DataArray(
        np.broadcast_to(T_profile[None, :, None, None],
                        (n_t, n_lev, n_lat, n_lon)).copy(),
        coords={'time': time, 'lev': lev,
                'lat': [0.0], 'lon': [0.0]},
        dims=['time', 'lev', 'lat', 'lon'],
    )
    H = 22.5
    mld = xr.DataArray([[[H]]], coords={'time': time, 'lat': [0.0], 'lon': [0.0]},
                       dims=['time', 'lat', 'lon'])
    expected = 20.0 - 0.05 * H
    T_ml = ml_mean_temperature(temp, mld)
    assert np.isclose(T_ml.values[0, 0, 0], expected, atol=0.02)


def test_ml_tendency_linear_time_series():
    n_t = 12
    time = pd.date_range('2000-01-01', periods=n_t, freq='MS')
    t_days = np.arange(n_t) * 30.0
    T_ml = xr.DataArray(
        np.broadcast_to(t_days[:, None, None], (n_t, 1, 1)).copy(),
        coords={'time': time, 'lat': [0.0], 'lon': [0.0]},
        dims=['time', 'lat', 'lon'],
    )
    dTdt = ml_tendency(T_ml, dt_seconds=30.0 * 86400.0)
    mid = dTdt.isel(time=slice(1, -1)).values
    # Centered diff: (T[i+1] - T[i-1]) / (2·dt) = 60 / (2·30·86400) = 1/86400
    expected = 60.0 / (2.0 * 30.0 * 86400.0)
    assert np.allclose(mid, expected)
