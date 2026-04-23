"""Phase-1 parity tests: apinter.{processing, stats, indices} vs. legacy src/.

Runs small synthetic DataArrays through both the new apinter modules and the
original implementations in Paper_1/src and ./src, asserting numerical equality.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Legacy sources — kept until phase 5 cleanup
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Paper_1" / "src"))
sys.path.insert(0, str(REPO / "src"))


# ---------- Fixtures ----------

@pytest.fixture
def ts_1d():
    """Monthly time series 1980-01 .. 2014-12 (420 months) with trend+seasonal+noise."""
    n = 420
    time = pd.date_range("1980-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(42)
    t = np.arange(n)
    values = (
        0.01 * t                                      # trend
        + 2.0 * np.sin(2 * np.pi * t / 12)            # annual cycle
        + rng.standard_normal(n) * 0.5                # noise
    )
    return xr.DataArray(values, coords={"time": time}, dims=["time"], name="x")


@pytest.fixture
def field_2d():
    """Monthly (time, lat, lon) field on 5x6 grid."""
    n = 120
    time = pd.date_range("1980-01-01", periods=n, freq="MS")
    lat = np.array([-30., -15., 0., 15., 30.])
    lon = np.array([0., 60., 120., 180., 240., 300.])
    rng = np.random.default_rng(0)
    t = np.arange(n)
    data = (
        0.005 * t[:, None, None]
        + 1.0 * np.sin(2 * np.pi * t[:, None, None] / 12)
        + rng.standard_normal((n, 5, 6)) * 0.3
    )
    return xr.DataArray(data,
                        coords={"time": time, "lat": lat, "lon": lon},
                        dims=["time", "lat", "lon"], name="x")


# ---------- Processing ----------

def test_detrend_dim_matches_legacy(ts_1d):
    from apinter.processing import detrend_dim as new
    from cal_index import detrend_dim as old
    np.testing.assert_allclose(new(ts_1d).values, old(ts_1d).values, rtol=1e-12)


def test_lanczos_lowpass_matches_legacy(ts_1d):
    from apinter.processing import lanczos_lowpass as new
    from data_processing import lanczos_lowpass as old
    np.testing.assert_allclose(new(ts_1d, 132).values,
                               old(ts_1d, 132).values, rtol=1e-12)


def test_compute_anomalies_matches_legacy(ts_1d):
    from apinter.processing import compute_anomalies as new
    from cal_index import compute_anomalies as old
    a_new, n_new = new(ts_1d)
    a_old, n_old = old(ts_1d)
    np.testing.assert_allclose(a_new.values, a_old.values, rtol=1e-12)
    np.testing.assert_allclose(n_new.values, n_old.values, rtol=1e-12)


def test_calculate_anomalies_and_filter_matches_legacy(ts_1d):
    from apinter.processing import calculate_anomalies_and_filter as new
    from data_processing import calculate_anomalies_and_filter as old
    np.testing.assert_allclose(new(ts_1d).values, old(ts_1d).values, rtol=1e-12)


def test_wgt_areaave_matches_legacy(field_2d):
    from apinter.processing import wgt_areaave as new
    from data_processing import wgt_areaave as old
    np.testing.assert_allclose(
        new(field_2d, -10, 10, 100, 200).values,
        old(field_2d, -10, 10, 100, 200).values,
        rtol=1e-12,
    )


def test_extract_region_matches_legacy(field_2d):
    from apinter.processing import extract_region as new
    from cal_index import extract_region as old
    np.testing.assert_allclose(
        new(field_2d, (60, 240), (-15, 15)).values,
        old(field_2d, (60, 240), (-15, 15)).values,
        rtol=1e-12,
    )


# ---------- Stats: trends ----------

def test_spatial_trend_matches_legacy(field_2d):
    from apinter.stats import spatial_trend as new
    from linear_trend import spatial_trend as old
    np.testing.assert_allclose(new(field_2d).values, old(field_2d).values, rtol=1e-12)


def test_global_mean_trend_matches_legacy(field_2d):
    from apinter.stats import global_mean_trend as new
    from linear_trend import global_mean_trend as old
    # Legacy uses 'latitude/longitude'; supply a renamed copy.
    legacy_input = field_2d.rename({"lat": "latitude", "lon": "longitude"})
    np.testing.assert_allclose(new(field_2d).values,
                               old(legacy_input).values, rtol=1e-12)


def test_seasonal_trend_matches_legacy(field_2d):
    from apinter.stats import seasonal_trend as new
    from linear_trend import seasonal_trend as old
    np.testing.assert_allclose(new(field_2d, 'DJF').values,
                               old(field_2d, 'DJF').values, rtol=1e-12)


# ---------- Stats: regression / significance ----------

def test_linear_regression_vectorize_matches_legacy(ts_1d, field_2d):
    from apinter.stats import calculate_regression_vectorize as new
    from calculate_regression_vectorize import calculate_regression_vectorize as old
    # Align time coords
    index = ts_1d.isel(time=slice(0, field_2d.sizes["time"]))
    new_ds = new(index, field_2d)
    old_ds = old(index, field_2d)
    for v in ["slope", "intercept", "r_value", "p_value", "std_err"]:
        np.testing.assert_allclose(new_ds[v].values, old_ds[v].values, rtol=1e-10)


def test_effective_dof_matches_legacy(ts_1d):
    from apinter.stats import effective_degrees_of_freedom as new
    from statistical_utils import effective_degrees_of_freedom as old
    rng = np.random.default_rng(1)
    other = ts_1d + rng.standard_normal(ts_1d.size) * 0.1
    assert abs(new(ts_1d, other) - old(ts_1d, other)) < 1e-10


def test_autocorrelation_matches_legacy(ts_1d):
    from apinter.stats import autocorrelation_function as new
    from statistical_utils import autocorrelation_function as old
    np.testing.assert_allclose(new(ts_1d, 24), old(ts_1d, 24), rtol=1e-12)


# ---------- regression_lags: canonical regression entry point ----------

def test_regression_lags_concurrent_matches_lstsq(ts_1d, field_2d):
    """regression_lags(lags=[0]) returns the OLS slope per grid point."""
    from apinter.stats import regression_lags
    index = ts_1d.isel(time=slice(0, field_2d.sizes["time"]))
    ds = regression_lags(field_2d, index, lags=[0], compute_significance=False)
    # Hand-computed lstsq slope at each grid point
    y = field_2d.values.reshape(field_2d.shape[0], -1)
    X = np.column_stack([index.values, np.ones(len(index))])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    expected = b[0].reshape(field_2d.shape[1:])
    np.testing.assert_allclose(ds['beta'].isel(lag=0).values, expected, rtol=1e-12)


def test_regression_lags_shapes_and_coords(ts_1d, field_2d):
    """Output Dataset has the expected lag dim, spatial dims, and variables."""
    from apinter.stats import regression_lags
    index = ts_1d.isel(time=slice(0, field_2d.sizes["time"]))
    lags = [-24, 0, 12]
    ds = regression_lags(field_2d, index, lags=lags, compute_significance=True)
    assert list(ds.dims) == ['lag', 'lat', 'lon']
    assert list(ds['lag'].values) == lags
    assert set(ds.data_vars) == {'beta', 'p_value'}
    assert ds['p_value'].min() >= 0 and ds['p_value'].max() <= 1


def test_regression_lags_partial_with_confounder(ts_1d, field_2d):
    """With a confounder, output has target_beta and confounder_beta."""
    from apinter.stats import regression_lags
    index = ts_1d.isel(time=slice(0, field_2d.sizes["time"]))
    rng = np.random.default_rng(7)
    conf = index + rng.standard_normal(index.size) * 0.3
    conf = conf.rename('conf')
    ds = regression_lags(field_2d, index, lags=[-12, 0, 12], confounder=conf,
                         compute_significance=True)
    assert set(ds.data_vars) == {'target_beta', 'confounder_beta',
                                  'target_pval', 'confounder_pval'}


def test_partial_shim_delegates_to_regression_lags(ts_1d, field_2d):
    """Legacy shim returns the same coefficients as regression_lags."""
    from apinter.stats import calculate_partial_lead_lag_regression, regression_lags
    rng = np.random.default_rng(3)
    index1 = ts_1d.isel(time=slice(0, field_2d.sizes["time"]))
    index2 = index1 + rng.standard_normal(index1.size) * 0.2
    index2 = index2.rename('ts2')
    max_lag_years = 2
    shim = calculate_partial_lead_lag_regression(index1, index2, field_2d,
                                                 max_lag_years=max_lag_years)
    lags = list(np.arange(-max_lag_years, max_lag_years + 1) * 12)
    ds = regression_lags(field_2d, index1, lags=lags, confounder=index2,
                         compute_significance=False)
    np.testing.assert_allclose(shim['tamv_beta'], ds['target_beta'].values, rtol=1e-12)
    np.testing.assert_allclose(shim['tpdv_beta'], ds['confounder_beta'].values, rtol=1e-12)


# ---------- Indices ----------

def test_calculate_index_matches_paper1_canonical(field_2d):
    """Verify calculate_index matches Paper_1 notebook 05 hand-coded pipeline:
    anomaly -> detrend -> area-weighted mean -> Lanczos LPF -> standardize."""
    from apinter.indices import calculate_index
    from apinter.processing import detrend_dim, lanczos_lowpass, wgt_areaave

    sst = field_2d

    new = calculate_index(sst, (60, 240), (-15, 15))

    clim = sst.groupby('time.month').mean('time')
    anom = sst.groupby('time.month') - clim
    anom_d = detrend_dim(anom, 'time')
    regional = wgt_areaave(anom_d, -15, 15, 60, 240)
    filtered = lanczos_lowpass(regional, 132)
    standardized = filtered / filtered.std('time')

    np.testing.assert_allclose(new.values, standardized.values, rtol=1e-12)


def test_calculate_index_running_mean_method(field_2d):
    """method='running_mean' applies a centered rolling mean instead of Lanczos."""
    from apinter.indices import calculate_index
    from apinter.processing import detrend_dim, wgt_areaave

    sst = field_2d

    new = calculate_index(sst, (60, 240), (-15, 15), method='running_mean',
                          cutoff_period=12)

    clim = sst.groupby('time.month').mean('time')
    anom = sst.groupby('time.month') - clim
    anom_d = detrend_dim(anom, 'time')
    regional = wgt_areaave(anom_d, -15, 15, 60, 240)
    rolled = regional.rolling(time=12, center=True, min_periods=1).mean()
    standardized = rolled / rolled.std('time')

    np.testing.assert_allclose(new.values, standardized.values, rtol=1e-12)


def test_calculate_multiple_indices(field_2d):
    """calculate_multiple_indices computes anomaly once, returns one series per region."""
    from apinter.indices import calculate_index, calculate_multiple_indices

    regions = {
        'A': ((60, 240), (-15, 15)),
        'B': ((120, 300), (-30, 0)),
    }
    multi = calculate_multiple_indices(field_2d, regions)
    assert set(multi.keys()) == {'A', 'B'}

    for name, (lon_b, lat_b) in regions.items():
        single = calculate_index(field_2d, lon_b, lat_b)
        np.testing.assert_allclose(multi[name].values, single.values, rtol=1e-12)
