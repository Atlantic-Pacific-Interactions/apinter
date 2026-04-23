"""Tests for apinter.indices — calculate_index, calculate_multiple_indices, gridded_anomalies."""
import numpy as np


def test_calculate_index_matches_paper1_canonical(field_2d):
    """calculate_index matches Paper_1 notebook 05 hand-coded pipeline:
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
    from apinter.indices import calculate_index
    from apinter.processing import detrend_dim, wgt_areaave

    sst = field_2d
    new = calculate_index(sst, (60, 240), (-15, 15),
                          method='running_mean', cutoff_period=12)

    clim = sst.groupby('time.month').mean('time')
    anom = sst.groupby('time.month') - clim
    anom_d = detrend_dim(anom, 'time')
    regional = wgt_areaave(anom_d, -15, 15, 60, 240)
    rolled = regional.rolling(time=12, center=True, min_periods=1).mean()
    standardized = rolled / rolled.std('time')

    np.testing.assert_allclose(new.values, standardized.values, rtol=1e-12)


def test_calculate_multiple_indices(field_2d):
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


def test_gridded_anomalies_matches_pipeline(field_2d):
    """gridded_anomalies composes compute_anomaly + apply_lowpass + normalize."""
    from apinter.indices import gridded_anomalies
    from apinter.processing import compute_anomaly, apply_lowpass

    new = gridded_anomalies(field_2d, cutoff_period=24, method='lanczos', normalize=True)
    anom = compute_anomaly(field_2d, detrend=True)
    filt = apply_lowpass(anom, 24, method='lanczos')
    expected = filt / filt.std('time')
    np.testing.assert_allclose(new.values, expected.values, rtol=1e-12)
