"""End-to-end pipeline tests on the dev branch.

Proves the canonical Paper_1 analysis pipeline runs unchanged on
NERSC-mirror data. Once a DataArray is on the canonical 1° grid in
Celsius, every apinter downstream function works on it identically.

Each test loads a small slice (one model, a few years) so the suite
stays fast.
"""
import warnings

import numpy as np
import pytest
import xarray as xr

import apinter.config as cfg


pytest.importorskip("xesmf")


@pytest.fixture(scope="module")
def _has_nersc_cmip6():
    if not cfg.NERSC_CMIP6_DIR.is_dir():
        pytest.skip(f"{cfg.NERSC_CMIP6_DIR} not accessible")


def test_nersc_to_index_to_regression_pipeline(_has_nersc_cmip6):
    """Full canonical pipeline:
        load_nersc_cmip6 -> regrid_to_1deg -> K->C + outlier filter ->
        calculate_index -> gridded_anomalies -> regression_lags
    The downstream apinter functions don't care that the data came from
    the NERSC ESGF mirror instead of the user's pre-regridded zarr."""
    from apinter.indices import calculate_index, gridded_anomalies
    from apinter.io import list_nersc_cmip6_models, load_nersc_cmip6
    from apinter.processing import regrid_to_1deg
    from apinter.stats import regression_lags

    # 1. load native-grid CMIP6 ts for one model, ~5 years
    candidates = list_nersc_cmip6_models(
        experiment_id='historical', variable_id='ts',
        table_id='Amon', member_id='r1i1p1f1',
    )
    if not candidates:
        pytest.skip("no historical/ts/Amon/r1i1p1f1 models on NERSC mirror")

    model = candidates[0]
    native = load_nersc_cmip6(
        variable_id='ts', experiment_id='historical',
        source_ids=[model],
        sim_time=slice('2000-01-01', '2004-12-31'),
    )
    if model not in native:
        pytest.skip(f"{model} ts/historical/r1i1p1f1 failed to load")

    # 2. regrid to 1°
    sst_1deg = regrid_to_1deg(native[model])
    assert sst_1deg.sizes['lat'] == 180
    assert sst_1deg.sizes['lon'] == 360

    # 3. K->C + outlier filter (matches apinter.io.cmip6.load_cmip6 internals)
    sst_c = sst_1deg - 273.15
    sst_c = sst_c.where((sst_c > -10) & (sst_c < 40))

    # 4. compute TAMV + TPDV indices
    # Short series + short Lanczos cutoff just for the test.
    tamv = calculate_index(sst_c, lon_bounds=(280, 340),
                           lat_bounds=(0, 30), cutoff_period=12)
    tpdv = calculate_index(sst_c, lon_bounds=(180, 280),
                           lat_bounds=(-20, 20), cutoff_period=12)
    assert tamv.dims == ('time',)
    assert tpdv.dims == ('time',)
    # Filtered + standardized series should be O(1)
    assert -10 < float(tamv.min()) < float(tamv.max()) < 10
    assert -10 < float(tpdv.min()) < float(tpdv.max()) < 10

    # 5. gridded anomalies + concurrent partial regression with TPDV confounder
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssta = gridded_anomalies(sst_c, cutoff_period=12, normalize=True)
    reg = regression_lags(
        field=ssta,
        target_index=tamv,
        confounder=tpdv,
        lags=[0],
        compute_significance=True,
    )
    # Output shape from a partial regression
    assert set(reg.data_vars) == {
        'target_beta', 'confounder_beta',
        'target_pval', 'confounder_pval',
    }
    assert reg['target_beta'].sizes['lag'] == 1
    assert reg['target_beta'].sizes['lat'] == 180
    assert reg['target_beta'].sizes['lon'] == 360
    # p-values should be in [0, 1] (modulo NaN over land)
    pvals = reg['target_pval'].values
    finite = pvals[np.isfinite(pvals)]
    assert finite.size > 0
    assert 0 <= float(finite.min())
    assert float(finite.max()) <= 1
