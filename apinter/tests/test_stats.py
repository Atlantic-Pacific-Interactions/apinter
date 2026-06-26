"""Tests for apinter.stats — trends, regression, correlation, significance."""
import numpy as np
import xarray as xr


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


def test_regression_lags_min_variance_skips_degenerate_lag(field_2d):
    """A lag whose (shifted) predictor window has near-zero variance is
    skipped entirely (NaN), matching a constant/degenerate index window."""
    from apinter.stats import regression_lags
    n = field_2d.sizes["time"]
    flat_index = xr.DataArray(
        np.full(n, 5.0), coords={"time": field_2d.time}, dims=["time"], name="flat",
    )
    ds = regression_lags(field_2d, flat_index, lags=[0], min_variance=1e-6,
                         compute_significance=False)
    assert np.all(np.isnan(ds['beta'].values))


# ---------- pointwise_regression ----------

def test_pointwise_regression_beta_matches_direct_cov_var(field_2d):
    """beta(pixel) = cov(x, y)(pixel) / var(x)(pixel), exactly, per pixel."""
    from apinter.stats import pointwise_regression
    rng = np.random.default_rng(11)
    y_field = 2.0 * field_2d + rng.standard_normal(field_2d.shape) * 0.5
    y_field = y_field.rename('y')

    ds = pointwise_regression(field_2d, y_field, compute_significance=False)

    x = field_2d.values.reshape(field_2d.shape[0], -1)
    y = y_field.values.reshape(y_field.shape[0], -1)
    x_dev = x - x.mean(axis=0)
    y_dev = y - y.mean(axis=0)
    expected_beta = (x_dev * y_dev).mean(axis=0) / (x_dev ** 2).mean(axis=0)

    np.testing.assert_allclose(ds['beta'].values.reshape(-1), expected_beta, rtol=1e-12)


def test_pointwise_regression_pvalue_in_unit_range(field_2d):
    from apinter.stats import pointwise_regression
    rng = np.random.default_rng(12)
    y_field = 2.0 * field_2d + rng.standard_normal(field_2d.shape) * 0.5
    y_field = y_field.rename('y')

    ds = pointwise_regression(field_2d, y_field, compute_significance=True)
    assert set(ds.data_vars) == {'beta', 'p_value'}
    pv = ds['p_value'].values
    assert np.nanmin(pv) >= 0 and np.nanmax(pv) <= 1


def test_pointwise_regression_min_variance_masks_degenerate_pixel(field_2d):
    """A pixel with ~zero variance in x (e.g. a CMIP6 regrid-seam artifact)
    is masked to NaN when min_variance is set; other pixels are unaffected."""
    from apinter.stats import pointwise_regression
    rng = np.random.default_rng(13)
    x_field = field_2d.copy()
    x_field.values[:, 0, 0] = 1.0 + rng.standard_normal(x_field.shape[0]) * 1e-9
    y_field = 2.0 * field_2d + rng.standard_normal(field_2d.shape) * 0.5
    y_field = y_field.rename('y')

    ds_unmasked = pointwise_regression(x_field, y_field, compute_significance=False)
    ds_masked = pointwise_regression(x_field, y_field, compute_significance=False,
                                     min_variance=1e-6)

    assert np.isnan(ds_masked['beta'].values[0, 0])
    assert not np.isnan(ds_unmasked['beta'].values[0, 0])
    other_pixels = np.s_[1:, 1:]
    np.testing.assert_allclose(ds_masked['beta'].values[other_pixels],
                               ds_unmasked['beta'].values[other_pixels], rtol=1e-12)


def test_pointwise_regression_uses_effective_dof_not_nominal(field_2d):
    """Effective-DOF p-values should be >= nominal-DOF p-values for
    autocorrelated series (effective N <= nominal N -> wider t-distribution)."""
    from apinter.stats import pointwise_regression
    from scipy import stats as sp_stats

    # Strongly autocorrelated x and y (shared smooth seasonal signal).
    n = field_2d.sizes['time']
    t = np.arange(n)
    rng = np.random.default_rng(14)
    smooth = np.sin(2 * np.pi * t / 12)[:, None, None]
    x_field = field_2d * 0 + smooth + rng.standard_normal(field_2d.shape) * 0.05
    y_field = (2.0 * x_field + rng.standard_normal(field_2d.shape) * 0.05).rename('y')

    ds = pointwise_regression(x_field, y_field, compute_significance=True)

    # Nominal-DOF p-value for the same pixel, by hand, for comparison.
    x = x_field.values.reshape(n, -1)
    y = y_field.values.reshape(n, -1)
    x_dev = x - x.mean(axis=0)
    y_dev = y - y.mean(axis=0)
    beta = (x_dev * y_dev).mean(axis=0) / (x_dev ** 2).mean(axis=0)
    ssr = ((y - x * beta) ** 2).sum(axis=0)
    mse = ssr / (n - 2)
    se_nom = np.sqrt(mse / ((x_dev ** 2).mean(axis=0) * n))
    t_nom = np.abs(beta) / se_nom
    p_nom = 2 * (1 - sp_stats.t.cdf(np.abs(t_nom), df=n - 2))

    assert np.all(ds['p_value'].values.reshape(-1) >= p_nom - 1e-9)


# ---------- calculate_neff_pointwise ----------

def test_calculate_neff_pointwise_white_noise_near_n(field_2d):
    """Near-white-noise series: Ne should be close to N (little autocorrelation)."""
    from apinter.stats import calculate_neff_pointwise
    rng = np.random.default_rng(15)
    n = field_2d.sizes['time']
    x = rng.standard_normal((n, 6))
    y = rng.standard_normal((n, 6))
    ne = calculate_neff_pointwise(x, y)
    assert np.all(ne <= n) and np.all(ne >= 2)
    assert np.nanmean(ne) > n * 0.5
