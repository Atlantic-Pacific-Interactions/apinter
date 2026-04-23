"""Tests for apinter.io — CMIP6, observational SST, ERA5, ORAS5, SSP, joblib.

Loads small real subsets (one year or one month) to validate the loaders
against actual data. Meant to run on Perlmutter where the data exists.
Tests skip individually if a specific path is missing.
"""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import apinter.config as cfg
from apinter.io import (
    CMIP6_VARS,
    ERA5_VARS,
    OBS_SST_SOURCES,
    ORAS5_VARS,
    SSP_VARS,
    get_cmip6_models,
    get_ssp_models,
    load_and_concat,
    load_cmip6,
    load_cmip6_sst,
    load_cmip6_wind,
    load_era5,
    load_joblib,
    load_obs_sst,
    load_oras5,
    load_oras5_equatorial,
    regrid_to_equatorial_lon,
    save_joblib,
)


# =============================================================================
# API contracts (no I/O)
# =============================================================================

def test_cmip6_vars_spec_shape():
    expected = {'ts', 'wap', 'zg', 'psl', 'pr', 'zos', 'ua', 'va', 'thetao', 'tauu'}
    assert set(CMIP6_VARS) == expected
    for var, spec in CMIP6_VARS.items():
        assert 'subdir' in spec and 'ext' in spec
        assert spec['ext'] in ('zarr', 'nc')


def test_obs_sst_sources_shape():
    assert set(OBS_SST_SOURCES) == {'hadisst', 'ersst', 'cobesst'}


def test_ssp_vars_shape():
    assert set(SSP_VARS) >= {'sst', 'wap', 'pr'}


def test_era5_vars_shape():
    expected = {'u', 'v', 'q', 'z', 'pl_omega',
                'omega500', 'slp', 'sst', 'tp', 'mtnlwrf',
                'u10', 'v10', 't2m', 'd2m'}
    assert set(ERA5_VARS) == expected
    for v, spec in ERA5_VARS.items():
        assert 'paths' in spec and isinstance(spec['paths'], dict)
        assert 'var' in spec


def test_oras5_vars_shape():
    expected = {'d20', 'ssh', 'mld030', 'tauu', 'tauv', 'hfds',
                'thetao', 'salinity', 'uo', 'vo', 'wo'}
    assert set(ORAS5_VARS) == expected
    for var, spec in ORAS5_VARS.items():
        assert spec['kind'] in ('2D', '3D')
        for key in ('subdir', 'short_name', 'friendly'):
            assert key in spec


def test_load_cmip6_rejects_unknown_var():
    with pytest.raises(ValueError, match="Unknown variable"):
        load_cmip6('not_a_var')


def test_load_obs_sst_rejects_unknown_source():
    with pytest.raises(ValueError, match="Unknown source"):
        load_obs_sst(source='not_a_source')


# =============================================================================
# joblib round-trip
# =============================================================================

def test_joblib_roundtrip(tmp_path):
    data = {"a": 1, "b": np.arange(5)}
    fp = tmp_path / "nested" / "data.joblib"
    save_joblib(data, fp)
    assert fp.exists()
    loaded = load_joblib(fp)
    assert loaded["a"] == 1
    np.testing.assert_array_equal(loaded["b"], np.arange(5))


# =============================================================================
# Observational SST: load one year for each source
# =============================================================================

@pytest.mark.parametrize("source", ["hadisst", "ersst", "cobesst"])
def test_load_obs_sst_one_year(source):
    path = OBS_SST_SOURCES[source]['path']
    if not Path(path).exists():
        pytest.skip(f"{source}: {path} missing")

    sst = load_obs_sst(source, sim_time=slice('2000', '2000'))
    assert 'time' in sst.dims and 'lat' in sst.dims and 'lon' in sst.dims
    assert sst.sizes['time'] == 12, f"{source}: expected 12 months, got {sst.sizes['time']}"

    finite = sst.values[np.isfinite(sst.values)]
    assert finite.size > 0, f"{source}: entire year is NaN"
    # SSTs in Celsius, rough sanity range
    assert -5 <= finite.min() <= 40
    assert -5 <= finite.max() <= 40


# =============================================================================
# CMIP6 generic loader
# =============================================================================

@pytest.fixture(scope="module")
def _cmip6_models():
    if not cfg.CMIP6_DIR.exists():
        pytest.skip("CMIP6_DIR not present")
    models = get_cmip6_models()
    if not models:
        pytest.skip("no CMIP6 models found")
    return models


def test_get_cmip6_models_is_sorted(_cmip6_models):
    assert _cmip6_models == sorted(_cmip6_models)


def test_load_cmip6_ts_one_year(_cmip6_models):
    """Single-model SST: K->C + ocean mask + coord rename."""
    # Use the first model that actually has ts.zarr
    out = load_cmip6('ts', sim_time=slice('2000', '2000'), models=_cmip6_models[:3])
    assert out, "No ts.zarr found for first 3 models"

    model = next(iter(out))
    sst = out[model]
    assert 'lat' in sst.dims and 'lon' in sst.dims and 'time' in sst.dims
    assert sst.sizes['time'] == 12

    finite = sst.values[np.isfinite(sst.values)]
    # E3SMS/mean_state filter restricts SST to (-10, 40) °C by construction.
    assert -10 < finite.min() and finite.max() < 40


def test_load_cmip6_sst_compat_wrapper(_cmip6_models):
    """load_cmip6_sst wrapper matches load_cmip6('ts')."""
    a = load_cmip6('ts', sim_time=slice('2000', '2000'), models=_cmip6_models[:2])
    b = load_cmip6_sst(sim_time=slice('2000', '2000'), models=_cmip6_models[:2])
    assert set(a) == set(b)
    for m in a:
        xr.testing.assert_identical(a[m], b[m])


def test_load_cmip6_wap_one_year(_cmip6_models):
    """3D atmospheric variable: level coord present, no ocean mask, no K->C."""
    out = load_cmip6('wap', sim_time=slice('2000', '2000'), models=_cmip6_models[:3])
    if not out:
        pytest.skip("no wap.zarr in first 3 models")
    wap = next(iter(out.values()))
    assert 'level' in wap.dims and 'time' in wap.dims
    # Omega magnitudes in Pa/s are O(0.1) — not suspiciously huge (K) or tiny
    finite = wap.values[np.isfinite(wap.values)]
    assert np.nanmax(np.abs(finite)) < 10


def test_load_cmip6_ua_with_level(_cmip6_models):
    """Level selection returns a 2D (lat, lon) field without a level dim."""
    out = load_cmip6('ua', sim_time=slice('2000', '2000'),
                     level=850, models=_cmip6_models[:3])
    if not out:
        pytest.skip("no ua.nc in first 3 models")
    ua = next(iter(out.values()))
    assert 'level' not in ua.dims
    assert {'time', 'lat', 'lon'} <= set(ua.dims)


def test_load_cmip6_wind_compat(_cmip6_models):
    """load_cmip6_wind returns {model: {'ua': ..., 'va': ...}} at a single level."""
    out = load_cmip6_wind(target_level=850, sim_time=slice('2000', '2000'),
                          models=_cmip6_models[:3])
    if not out:
        pytest.skip("no ua/va in first 3 models")
    model = next(iter(out))
    assert set(out[model]) == {'ua', 'va'}


# =============================================================================
# ERA5 (one month / one year)
# =============================================================================

def _era5_files_present(var):
    return all(Path(p).exists() for p in ERA5_VARS[var]['paths'].values())


def test_load_era5_u_one_month_with_level():
    if not _era5_files_present('u'):
        pytest.skip("ERA5 u files not present")
    u = load_era5('u', level=850, sim_time=slice('2000-01-01', '2000-01-31'))
    assert 'level' not in u.dims
    assert {'time', 'lat', 'lon'} <= set(u.dims)
    assert u.sizes['time'] == 1
    finite = u.values[np.isfinite(u.values)]
    assert np.nanmax(np.abs(finite)) < 120  # m/s


def test_load_era5_omega500_single_file_variable():
    if not _era5_files_present('omega500'):
        pytest.skip("ERA5 omega500 not present")
    w = load_era5('omega500', sim_time=slice('2000-01-01', '2000-12-31'))
    assert 'time' in w.dims
    # omega (Pa/s) magnitude should be moderate
    finite = w.values[np.isfinite(w.values)]
    assert np.nanmax(np.abs(finite)) < 10


def test_load_era5_slp_surface_variable():
    if not _era5_files_present('slp'):
        pytest.skip("ERA5 SLP not present")
    msl = load_era5('slp', sim_time=slice('2000-01-01', '2000-01-31'))
    assert {'time', 'lat', 'lon'} <= set(msl.dims)
    # SLP around 100_000 Pa (sea-level pressure in Pa)
    finite = msl.values[np.isfinite(msl.values)]
    assert 90_000 < finite.mean() < 105_000


def test_load_era5_tp_strips_number_ensemble():
    if not _era5_files_present('tp'):
        pytest.skip("ERA5 tp not present")
    tp = load_era5('tp', sim_time=slice('2000-01-01', '2000-01-31'))
    assert 'number' not in tp.dims


def test_load_era5_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown ERA5 var"):
        load_era5('not_a_var')


def test_load_era5_level_on_surface_var_errors():
    if not _era5_files_present('slp'):
        pytest.skip("ERA5 SLP not present")
    with pytest.raises(ValueError, match="not a pressure-level field"):
        load_era5('slp', level=850)


# =============================================================================
# ORAS5 (curvilinear NEMO grid)
# =============================================================================

def test_load_oras5_ssh_one_year():
    """Load a year of ORAS5 SSH: 12 files, native curvilinear grid."""
    spec = ORAS5_VARS['ssh']
    pattern_dir = cfg.ORAS5_DIR / spec['subdir']
    if not pattern_dir.exists():
        pytest.skip(f"{pattern_dir} not present")

    ssh = load_oras5('ssh', sim_time=slice('2000-01-01', '2000-12-31'))
    assert ssh.sizes['time'] == 12
    assert 'nav_lat' in ssh.coords and 'nav_lon' in ssh.coords
    # SSH in meters, typically within [-3, 3]
    finite = ssh.values[np.isfinite(ssh.values)]
    assert finite.size > 0
    assert -5 < finite.min() and finite.max() < 5


def test_regrid_oras5_to_1d_lon_shape():
    """Equatorial binning produces a 1D lon grid, reducing the curvilinear (y, x)."""
    spec = ORAS5_VARS['ssh']
    pattern_dir = cfg.ORAS5_DIR / spec['subdir']
    if not pattern_dir.exists():
        pytest.skip(f"{pattern_dir} not present")

    ssh = load_oras5('ssh', sim_time=slice('2000-01-01', '2000-03-31'))
    regrid = regrid_to_equatorial_lon(ssh,
                                       lon_bounds=(120, 290),
                                       lat_bounds=(-5.5, 5.5),
                                       lon_step=2.0)
    assert 'lon' in regrid.dims
    assert 'time' in regrid.dims
    # Longitude centers every 2 deg from 120 to 290
    assert regrid.sizes['lon'] == int((290 - 120) / 2) + 1
    # No residual curvilinear dims (y, x)
    assert not any(d in regrid.dims for d in ('y', 'x'))


def test_load_oras5_equatorial_convenience():
    """Convenience wrapper yields the same result as load + regrid."""
    spec = ORAS5_VARS['ssh']
    if not (cfg.ORAS5_DIR / spec['subdir']).exists():
        pytest.skip("ORAS5 SSH not present")

    a = load_oras5_equatorial('ssh',
                              lon_bounds=(120, 290), lat_bounds=(-5.5, 5.5),
                              lon_step=2.0,
                              sim_time=slice('2000-01-01', '2000-03-31'))
    ssh = load_oras5('ssh', sim_time=slice('2000-01-01', '2000-03-31'))
    b = regrid_to_equatorial_lon(ssh, lon_bounds=(120, 290),
                                 lat_bounds=(-5.5, 5.5), lon_step=2.0)
    np.testing.assert_allclose(a.values, b.values, equal_nan=True)


# =============================================================================
# SSP historical+scenario concat (one model, one scenario)
# =============================================================================

@pytest.fixture(scope="module")
def _ssp_sst_model():
    if not cfg.CMIP6_DIR.exists():
        pytest.skip("CMIP6_DIR not present")
    for ssp in ('ssp245', 'ssp585', 'ssp370', 'ssp126'):
        models = get_ssp_models(ssp, var='sst')
        if models:
            return ssp, models[0]
    pytest.skip("no SSP SST models available")


def test_load_and_concat_sst(_ssp_sst_model):
    ssp, model = _ssp_sst_model
    da = load_and_concat('sst', ssp, model,
                         full_time=slice('2014-01-01', '2015-12-31'))
    assert 'time' in da.dims
    # 2014 (historical) + 2015 (SSP) = 24 months, should span the boundary
    years = np.unique(da.time.dt.year.values)
    assert 2014 in years and 2015 in years
    finite = da.values[np.isfinite(da.values)]
    # E3SMS filter restricts historical SST to (-10, 40); SSP `tos` is
    # native-ocean-only °C, so combined series stays within (-10, 40).
    assert -10 < finite.min() and finite.max() < 40
