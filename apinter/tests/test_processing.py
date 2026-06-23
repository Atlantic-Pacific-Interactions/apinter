"""Tests for apinter.processing — anomaly, filter, area-weighted mean.

Compares against the legacy Paper_1/src and ./src implementations for a
bit-level parity check where possible.
"""
import numpy as np
import pandas as pd
import xarray as xr


def test_detrend_dim_matches_legacy(ts_1d):
    from apinter.processing import detrend_dim as new
    from cal_index import detrend_dim as old
    np.testing.assert_allclose(new(ts_1d).values, old(ts_1d).values, rtol=1e-12)


def test_lanczos_lowpass_matches_legacy(ts_1d):
    from apinter.processing import lanczos_lowpass as new
    from data_processing import lanczos_lowpass as old
    np.testing.assert_allclose(new(ts_1d, 132).values,
                               old(ts_1d, 132).values, rtol=1e-12)


def test_compute_anomaly_all_months(ts_1d):
    """Default (all-months climatology + detrend) matches the Paper_1 sst_danmoly hand-code."""
    from apinter.processing import compute_anomaly, detrend_dim
    new = compute_anomaly(ts_1d, detrend=True, complete_years_only=False)
    clim = ts_1d.groupby('time.month').mean('time')
    anom = ts_1d.groupby('time.month') - clim
    expected = detrend_dim(anom, 'time')
    np.testing.assert_allclose(new.values, expected.values, rtol=1e-12)


def test_compute_anomaly_complete_years_matches_legacy(ts_1d):
    """complete_years_only=True reproduces the ./src/cal_index.compute_anomalies climatology."""
    from apinter.processing import compute_anomaly
    from cal_index import compute_anomalies as old
    new = compute_anomaly(ts_1d, detrend=False, complete_years_only=True)
    old_anom, _ = old(ts_1d)
    np.testing.assert_allclose(new.values, old_anom.values, rtol=1e-12)


def test_apply_lowpass_lanczos_matches_direct(ts_1d):
    from apinter.processing import apply_lowpass, lanczos_lowpass
    np.testing.assert_allclose(
        apply_lowpass(ts_1d, 132, method='lanczos').values,
        lanczos_lowpass(ts_1d, 132).values,
        rtol=1e-12,
    )


def test_apply_lowpass_running_mean(ts_1d):
    from apinter.processing import apply_lowpass
    out = apply_lowpass(ts_1d, 12, method='running_mean')
    expected = ts_1d.rolling(time=12, center=True, min_periods=1).mean()
    np.testing.assert_allclose(out.values, expected.values, rtol=1e-12)


def test_apply_lowpass_none_passthrough(ts_1d):
    from apinter.processing import apply_lowpass
    out = apply_lowpass(ts_1d, 132, method=None)
    np.testing.assert_array_equal(out.values, ts_1d.values)


def test_wgt_areaave_matches_legacy(field_2d):
    from apinter.processing import wgt_areaave as new
    from data_processing import wgt_areaave as old
    np.testing.assert_allclose(
        new(field_2d, -10, 10, 100, 200).values,
        old(field_2d, -10, 10, 100, 200).values,
        rtol=1e-12,
    )


def test_standardize_time_to_month_start_aligns_mid_month_timestamps():
    from apinter.processing import standardize_time_to_month_start
    time = pd.to_datetime(["1980-01-15", "1980-02-15", "1980-03-15"])
    da = xr.DataArray([1.0, 2.0, 3.0], coords={"time": time}, dims=["time"])
    out = standardize_time_to_month_start(da)
    expected = pd.to_datetime(["1980-01-01", "1980-02-01", "1980-03-01"])
    np.testing.assert_array_equal(out.time.values, expected.values)


def test_standardize_time_to_month_start_no_time_coord_passthrough():
    from apinter.processing import standardize_time_to_month_start
    da = xr.DataArray([1.0, 2.0, 3.0], dims=["x"])
    out = standardize_time_to_month_start(da)
    np.testing.assert_array_equal(out.values, da.values)
