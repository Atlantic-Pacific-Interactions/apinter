"""Tests for apinter.processing.regrid (xesmf-based 1° regridder)."""
from typing import Optional

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


def test_regrid_nersc_cmip6_wap_4d(_has_nersc_cmip6):
    """Load a 4D (time, level, lat, lon) field and regrid. xesmf regrids the
    lat/lon plane only; level and time pass through untouched."""
    from apinter.io import list_nersc_cmip6_models, load_nersc_cmip6

    models = list_nersc_cmip6_models(
        experiment_id='historical', variable_id='wap',
        table_id='Amon', member_id='r1i1p1f1',
    )
    if not models:
        pytest.skip("no wap/historical/r1i1p1f1 models")

    native = load_nersc_cmip6(
        variable_id='wap', experiment_id='historical',
        source_ids=models[:1],
        sim_time=slice('2000-01-01', '2000-01-31'),
    )
    if not native:
        pytest.skip("wap load returned empty")

    model, da = next(iter(native.items()))
    vertical = next((d for d in ('plev', 'lev', 'level') if d in da.dims), None)
    assert vertical is not None, f"no vertical dim in {da.dims}"

    out = regrid_to_1deg(da)
    assert out.sizes['lat'] == 180
    assert out.sizes['lon'] == 360
    assert out.sizes.get('time') == da.sizes.get('time')
    assert out.sizes.get(vertical) == da.sizes.get(vertical)


# ---------- ocean variables (curvilinear native grids) ----------

def test_regrid_nersc_cmip6_tos_curvilinear_3d(_has_nersc_cmip6):
    """Ocean SST `tos` lives on a curvilinear grid (e.g., dims (time, nlat, nlon)
    with 2D lat(nlat, nlon) coords). xesmf handles this without special handling."""
    from apinter.io import list_nersc_cmip6_models, load_nersc_cmip6

    models = list_nersc_cmip6_models(
        experiment_id='historical', variable_id='tos',
        table_id='Omon', member_id='r1i1p1f1',
    )
    if not models:
        pytest.skip("no tos/Omon/historical models")

    native = load_nersc_cmip6(
        variable_id='tos', experiment_id='historical',
        table_id='Omon', source_ids=models[:1],
        sim_time=slice('2000-01-01', '2000-01-31'),
    )
    if not native:
        pytest.skip("tos load returned empty")

    model, da = next(iter(native.items()))
    out = regrid_to_1deg(da)
    assert out.sizes['lat'] == 180
    assert out.sizes['lon'] == 360
    finite = out.values[np.isfinite(out.values)]
    assert finite.size > 0
    # Ocean SST in °C; CMIP6 reports tos in degC by convention
    assert -3 < float(np.nanmean(finite)) < 35, (
        f"{model}: mean tos {np.nanmean(finite):.2f} °C not plausible"
    )


def test_regrid_nersc_cmip6_thetao_curvilinear_4d(_has_nersc_cmip6):
    """4D ocean (time, lev, nlat, nlon) thetao on a curvilinear grid. The
    depth dim and time dim pass through; only nlat/nlon -> lat/lon happens."""
    from apinter.io import list_nersc_cmip6_models, load_nersc_cmip6

    models = list_nersc_cmip6_models(
        experiment_id='historical', variable_id='thetao',
        table_id='Omon', member_id='r1i1p1f1',
    )
    if not models:
        pytest.skip("no thetao/Omon/historical models")

    # thetao files are big — load just one month then slice depth manually.
    from pathlib import Path
    import xarray as xr
    import apinter.config as cfg

    def _walk_to(model: str) -> Optional[Path]:
        for inst in (cfg.NERSC_CMIP6_DIR / 'CMIP').iterdir():
            cand = inst / model / 'historical' / 'r1i1p1f1' / 'Omon' / 'thetao' / 'gn'
            try:
                if cand.is_dir():
                    return cand
            except PermissionError:
                continue
        return None

    # Try models in order — some institutes restrict ocean subdirs.
    cand = None
    for m in models:
        cand = _walk_to(m)
        if cand is not None:
            break
    if cand is None:
        pytest.skip("no readable thetao dir found")
    ncs = sorted(cand.rglob('*.nc'))
    if not ncs:
        pytest.skip("no thetao .nc files")

    ds = xr.open_dataset(ncs[0]).isel(time=slice(0, 1))
    # Take just the first 3 depth levels to keep memory small
    depth_dim = next((d for d in ('lev', 'deptht', 'olevel', 'depth') if d in ds.dims), None)
    if depth_dim is None:
        pytest.skip(f"no recognized depth dim in {list(ds.dims)}")
    da = ds['thetao'].isel({depth_dim: slice(0, 3)})

    out = regrid_to_1deg(da)
    assert out.sizes['lat'] == 180
    assert out.sizes['lon'] == 360
    assert out.sizes.get(depth_dim) == 3
    assert out.sizes.get('time') == 1


# ---------- full-ensemble survey (opt-in, slow) ----------

@pytest.mark.slow
@pytest.mark.parametrize("variable,table_id", [
    ('ts',  'Amon'),   # 3D atmospheric (time, lat, lon)
    ('wap', 'Amon'),   # 4D atmospheric (time, plev, lat, lon)
    ('tos', 'Omon'),   # 3D ocean — curvilinear native grid
])
def test_regrid_all_nersc_cmip6_models(_has_nersc_cmip6, variable, table_id):
    """Load + regrid every available NERSC CMIP6 model for the given variable.

    Run with ``pytest -m slow`` — this takes ~2 minutes and exercises 30+
    models end-to-end (load native-grid -> xesmf bilinear -> 180x360).

    Two known-bad models are accepted (``ICON-ESM-LR`` has an icosahedral
    grid not representable as rectilinear lat/lon; ``IITM-ESM`` has an
    empty version directory on the NERSC mirror). All others must succeed.
    """
    from apinter.io import list_nersc_cmip6_models, load_nersc_cmip6

    models = list_nersc_cmip6_models(
        experiment_id='historical', variable_id=variable,
        table_id=table_id, member_id='r1i1p1f1',
    )
    if not models:
        pytest.skip(f"no {variable}/historical/r1i1p1f1 models")

    EXPECTED_FAILURES = {'ICON-ESM-LR', 'IITM-ESM'}
    succeeded, failed = [], []
    for m in models:
        try:
            native = load_nersc_cmip6(
                variable_id=variable, experiment_id='historical',
                table_id=table_id,
                source_ids=[m],
                sim_time=slice('2000-01-01', '2000-01-31'),
            )
            if m not in native:
                raise RuntimeError("load returned empty dict")
            out = regrid_to_1deg(native[m])
            assert out.sizes['lat'] == 180 and out.sizes['lon'] == 360
            succeeded.append(m)
        except Exception as e:
            failed.append((m, f"{type(e).__name__}: {str(e)[:80]}"))

    unexpected = [(m, e) for (m, e) in failed if m not in EXPECTED_FAILURES]
    assert not unexpected, (
        f"{variable}: {len(unexpected)} unexpected failure(s): " + "; ".join(
            f"{m} ({e})" for m, e in unexpected
        )
    )
    # Sanity: at least 80% of models should succeed
    assert len(succeeded) / len(models) > 0.80, (
        f"{variable}: only {len(succeeded)}/{len(models)} models succeeded"
    )
