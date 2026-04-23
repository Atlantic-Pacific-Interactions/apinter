"""Tests for apinter.stats — trends, regression, correlation, significance."""
import numpy as np


# ---------- trends ----------

def test_spatial_trend_matches_legacy(field_2d):
    from apinter.stats import spatial_trend as new
    from linear_trend import spatial_trend as old
    np.testing.assert_allclose(new(field_2d).values, old(field_2d).values, rtol=1e-12)


def test_global_mean_trend_matches_legacy(field_2d):
    from apinter.stats import global_mean_trend as new
    from linear_trend import global_mean_trend as old
    legacy_input = field_2d.rename({"lat": "latitude", "lon": "longitude"})
    np.testing.assert_allclose(new(field_2d).values,
                               old(legacy_input).values, rtol=1e-12)


def test_seasonal_trend_matches_legacy(field_2d):
    from apinter.stats import seasonal_trend as new
    from linear_trend import seasonal_trend as old
    np.testing.assert_allclose(new(field_2d, 'DJF').values,
                               old(field_2d, 'DJF').values, rtol=1e-12)


# ---------- correlation / significance ----------

def test_correlation_lags_basic(ts_1d):
    """Self-correlation at lag 0 is exactly 1."""
    from apinter.stats import correlation_lags
    ds = correlation_lags(ts_1d, ts_1d, max_lag=12)
    assert list(ds.dims) == ['lag']
    assert ds.sizes['lag'] == 25
    np.testing.assert_allclose(ds['r'].sel(lag=0).item(), 1.0, atol=1e-12)


def test_correlation_lags_respects_caller_time_slicing(ts_1d):
    """No internal time slicing — caller controls the window."""
    from apinter.stats import correlation_lags
    sub = ts_1d.isel(time=slice(100, 300))
    ds = correlation_lags(sub, sub, max_lag=6)
    np.testing.assert_allclose(ds['r'].sel(lag=0).item(), 1.0, atol=1e-12)


def test_mmm_correlation_lags(ts_1d):
    """Multi-model mean over a dict of correlation_lags results; exclude respected."""
    from apinter.stats import correlation_lags, mmm_correlation_lags
    rng = np.random.default_rng(11)
    per_model = {
        'M1': correlation_lags(ts_1d, ts_1d + rng.standard_normal(ts_1d.size) * 0.1, max_lag=6),
        'M2': correlation_lags(ts_1d, ts_1d + rng.standard_normal(ts_1d.size) * 0.2, max_lag=6),
        'OBS': correlation_lags(ts_1d, ts_1d + rng.standard_normal(ts_1d.size) * 0.3, max_lag=6),
    }
    mmm = mmm_correlation_lags(per_model, exclude=['OBS'])
    assert int(mmm['n_models']) == 2
    assert mmm['r_mean'].sel(lag=0).item() > 0.9


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


# ---------- regression ----------

def test_regression_lags_concurrent_matches_lstsq(ts_1d, field_2d):
    """regression_lags(lags=[0]) returns the OLS slope per grid point."""
    from apinter.stats import regression_lags
    index = ts_1d.isel(time=slice(0, field_2d.sizes["time"]))
    ds = regression_lags(field_2d, index, lags=[0], compute_significance=False)
    y = field_2d.values.reshape(field_2d.shape[0], -1)
    X = np.column_stack([index.values, np.ones(len(index))])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    expected = b[0].reshape(field_2d.shape[1:])
    np.testing.assert_allclose(ds['beta'].isel(lag=0).values, expected, rtol=1e-12)


def test_regression_lags_shapes_and_coords(ts_1d, field_2d):
    from apinter.stats import regression_lags
    index = ts_1d.isel(time=slice(0, field_2d.sizes["time"]))
    lags = [-24, 0, 12]
    ds = regression_lags(field_2d, index, lags=lags, compute_significance=True)
    assert list(ds.dims) == ['lag', 'lat', 'lon']
    assert list(ds['lag'].values) == lags
    assert set(ds.data_vars) == {'beta', 'p_value'}
    assert ds['p_value'].min() >= 0 and ds['p_value'].max() <= 1


def test_regression_lags_partial_with_confounder(ts_1d, field_2d):
    from apinter.stats import regression_lags
    index = ts_1d.isel(time=slice(0, field_2d.sizes["time"]))
    rng = np.random.default_rng(7)
    conf = index + rng.standard_normal(index.size) * 0.3
    conf = conf.rename('conf')
    ds = regression_lags(field_2d, index, lags=[-12, 0, 12], confounder=conf,
                         compute_significance=True)
    assert set(ds.data_vars) == {'target_beta', 'confounder_beta',
                                  'target_pval', 'confounder_pval'}


def test_regression_lags_matches_notebook34_calc_partial_regression(ts_1d, field_2d):
    from apinter.stats import regression_lags
    rng = np.random.default_rng(3)
    index1 = ts_1d.isel(time=slice(0, field_2d.sizes["time"]))
    index2 = index1 + rng.standard_normal(index1.size) * 0.2
    index2 = index2.rename('ts2')
    lags = [-24, -12, 0, 12, 24]

    ds = regression_lags(field_2d, index1, lags=lags, confounder=index2,
                         compute_significance=False)

    x1 = index1.values
    x2 = index2.values
    y = field_2d.values.reshape(field_2d.shape[0], -1)
    X = np.column_stack([x1, x2, np.ones(len(x1))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    expected_b1 = beta[0].reshape(field_2d.shape[1:])
    expected_b2 = beta[1].reshape(field_2d.shape[1:])

    np.testing.assert_allclose(ds['target_beta'].sel(lag=0).values,
                               expected_b1, rtol=1e-12)
    np.testing.assert_allclose(ds['confounder_beta'].sel(lag=0).values,
                               expected_b2, rtol=1e-12)
