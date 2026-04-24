"""Tests for compute_tendency and compute_anomaly_tendency (both time dims)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from apinter.heat_budget.tendency import compute_tendency, compute_anomaly_tendency


def _linear_ts(time_name='time'):
    """10-year monthly series that increases by 1 degC/year."""
    n = 120
    time = pd.date_range('1981-01-01', periods=n, freq='MS')
    values = np.arange(n, dtype=float) / 12.0
    return xr.DataArray(values, coords={time_name: time}, dims=[time_name], name='T')


def test_compute_tendency_linear_time_dim():
    da = _linear_ts('time')
    dTdt = compute_tendency(da)
    expected = 1.0 / (365.25 * 86400.0)
    mid = dTdt.isel(time=slice(10, -10)).values
    assert np.allclose(mid, expected, rtol=0.05), mid.mean()


def test_compute_tendency_time_counter_dim():
    da = _linear_ts('time_counter')
    dTdt = compute_tendency(da)
    expected = 1.0 / (365.25 * 86400.0)
    mid = dTdt.isel(time_counter=slice(10, -10)).values
    assert np.allclose(mid, expected, rtol=0.05)


def test_compute_anomaly_tendency_removes_seasonal_cycle():
    n = 120
    time = pd.date_range('1981-01-01', periods=n, freq='MS')
    season = 5.0 * np.sin(2 * np.pi * np.arange(n) / 12)
    da = xr.DataArray(season, coords={'time': time}, dims=['time'])
    dTpdt = compute_anomaly_tendency(da, yrclim=[1981, 1990])
    assert np.allclose(dTpdt.values, 0.0, atol=1e-10)


def test_compute_tendency_raises_without_time_dim():
    da = xr.DataArray([1.0, 2.0], dims=['x'])
    with pytest.raises(ValueError, match="No time dimension"):
        compute_tendency(da)
