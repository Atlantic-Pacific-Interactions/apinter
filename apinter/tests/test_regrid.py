"""Tests for apinter.processing.regrid (xesmf-based 1° regridder)."""
import numpy as np
import pytest
import xarray as xr

import apinter.config as cfg
from apinter.config import COMMON_LAT, COMMON_LON
from apinter.processing import regrid_dict_to_1deg, regrid_to_1deg


pytest.importorskip("xesmf")


# ---------- synthetic-data smoke ----------

def _synthetic(nlat: int = 36, nlon: int = 72) -> xr.DataArray:
    lat = np.linspace(-87.5, 87.5, nlat)
    lon = np.linspace(2.5, 357.5, nlon)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((nlat, nlon))
    return xr.DataArray(data, coords={'lat': lat, 'lon': lon},
                        dims=('lat', 'lon'), name='x')


def test_regrid_to_1deg_synthetic_output_shape():
    out = regrid_to_1deg(_synthetic())
    assert out.dims == ('lat', 'lon')
    assert out.sizes == {'lat': len(COMMON_LAT), 'lon': len(COMMON_LON)}
    np.testing.assert_array_equal(out.lat.values, COMMON_LAT)
    np.testing.assert_array_equal(out.lon.values, COMMON_LON)


def test_regrid_to_1deg_handles_latitude_longitude_naming():
    """Auto-renames latitude/longitude -> lat/lon."""
    da = _synthetic().rename({'lat': 'latitude', 'lon': 'longitude'})
    out = regrid_to_1deg(da)
    assert {'lat', 'lon'} <= set(out.dims)


# ---------- real NERSC CMIP6 round-trip ----------

@pytest.fixture(scope="module")
def _has_nersc_cmip6():
    if not cfg.NERSC_CMIP6_DIR.is_dir():
        pytest.skip(f"{cfg.NERSC_CMIP6_DIR} not accessible")


def test_regrid_nersc_cmip6_ts_one_model_one_year(_has_nersc_cmip6):
    """Load one CMIP6 model from the NERSC ESGF mirror, regrid to 1°,
    verify the result is on (180, 360) and roughly matches Kelvin SSTs."""
    from apinter.io import list_nersc_cmip6_models, load_nersc_cmip6

    models = list_nersc_cmip6_models(
        experiment_id='historical', variable_id='ts',
        table_id='Amon', member_id='r1i1p1f1',
    )
    if not models:
        pytest.skip("no models with ts/historical/r1i1p1f1")

    native = load_nersc_cmip6(
        variable_id='ts', experiment_id='historical',
        source_ids=models[:1],
        sim_time=slice('2000-01-01', '2000-01-31'),
    )
    if not native:
        pytest.skip("first model failed to load")

    model, da = next(iter(native.items()))
    da_1deg = regrid_to_1deg(da)

    # Spatial dims now exactly 180×360
    assert da_1deg.sizes['lat'] == 180
    assert da_1deg.sizes['lon'] == 360
    # Time preserved
    if 'time' in da.dims:
        assert da_1deg.sizes.get('time') == da.sizes['time']

    # ts is in Kelvin; ocean cells should be in a sane temperature range
    finite = da_1deg.values[np.isfinite(da_1deg.values)]
    assert finite.size > 0
    assert 200 < float(np.nanmean(finite)) < 320, (
        f"{model}: mean ts {np.nanmean(finite):.1f} K not plausible"
    )


def test_regrid_dict_to_1deg_smoke(_has_nersc_cmip6):
    """regrid_dict_to_1deg should return a dict with the same keys + uniform 180x360."""
    from apinter.io import list_nersc_cmip6_models, load_nersc_cmip6

    models = list_nersc_cmip6_models(
        experiment_id='historical', variable_id='ts',
    )
    if len(models) < 2:
        pytest.skip("need >=2 models for the dict test")

    native = load_nersc_cmip6(
        variable_id='ts', experiment_id='historical',
        source_ids=models[:2],
        sim_time=slice('2000-01-01', '2000-01-31'),
    )
    if len(native) < 2:
        pytest.skip("only one model loaded")

    out = regrid_dict_to_1deg(native)
    assert set(out) == set(native)
    for name, da in out.items():
        assert da.sizes['lat'] == 180
        assert da.sizes['lon'] == 360
