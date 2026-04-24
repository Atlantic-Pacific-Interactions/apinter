"""Smoke tests for the NERSC-community-mirror loaders in apinter.io.

These tests load **small slices of real data** on Perlmutter. Each test
auto-skips when the backing directory isn't accessible (e.g., running off
NERSC or missing group permission), so the suite stays green elsewhere.
"""
import numpy as np
import pytest
import xarray as xr

import apinter.config as cfg
from apinter.io import (
    OBS_SST_SOURCES,
    list_cesm1_lens_members,
    list_loca2_members,
    list_loca2_models,
    list_nersc_cmip6_models,
    load_cesm1_lens,
    load_loca2,
    load_nersc_cmip6,
    load_nersc_cmip6_ensemble,
    load_obs_sst,
)


# =============================================================================
# NERSC ESGF CMIP6 mirror
# =============================================================================

@pytest.fixture(scope="module")
def _has_nersc_cmip6():
    if not cfg.NERSC_CMIP6_DIR.is_dir():
        pytest.skip(f"{cfg.NERSC_CMIP6_DIR} not accessible")


def test_list_nersc_cmip6_models_historical_ts(_has_nersc_cmip6):
    """The NERSC ESGF mirror should have >5 models with ts/historical/r1i1p1f1/Amon."""
    models = list_nersc_cmip6_models(
        experiment_id='historical', variable_id='ts',
        table_id='Amon', member_id='r1i1p1f1', grid_label='gn',
    )
    assert isinstance(models, list)
    assert len(models) > 5, f"Only found {len(models)} models — expected many"


def test_load_nersc_cmip6_single_model_one_slab(_has_nersc_cmip6):
    """Can we open and time-slice one model? Value validation is deferred
    to downstream regridding / unit normalization (not apinter's job here)."""
    models = list_nersc_cmip6_models(
        experiment_id='historical', variable_id='ts',
        table_id='Amon', member_id='r1i1p1f1',
    )
    if not models:
        pytest.skip("no models found for ts/historical/r1i1p1f1")

    out = load_nersc_cmip6(
        variable_id='ts', experiment_id='historical',
        source_ids=models[:1],
        sim_time=slice('2000-01-01', '2000-12-31'),
    )
    assert out, "Expected at least one DataArray back"
    model, da = next(iter(out.items()))
    # Readability check: the result is a time-indexed DataArray with
    # spatial dims (name may vary on native grids — lat/latitude, nlat, etc.)
    assert 'time' in da.dims
    assert da.ndim >= 2
    assert da.name == 'ts'


def test_load_nersc_cmip6_ensemble_shape(_has_nersc_cmip6):
    """Multi-member load for one model returns a dict keyed by member_id."""
    # Find any model with more than one member
    models = list_nersc_cmip6_models(
        experiment_id='historical', variable_id='ts',
        table_id='Amon', member_id='r1i1p1f1',
    )
    if not models:
        pytest.skip("no models with ts/historical/r1i1p1f1")

    # Try a couple of models to find one with >1 member
    for m in models[:10]:
        ensemble = load_nersc_cmip6_ensemble(
            variable_id='ts', experiment_id='historical', source_id=m,
            sim_time=slice('2000-01-01', '2000-01-31'),
        )
        if len(ensemble) >= 2:
            break
    else:
        pytest.skip("no multi-member model found in first 10")
    assert all('time' in da.dims for da in ensemble.values())


def test_load_nersc_cmip6_rejects_unknown_experiment(_has_nersc_cmip6):
    with pytest.raises(ValueError, match="Unknown experiment_id"):
        load_nersc_cmip6(variable_id='ts', experiment_id='not_an_experiment')


# =============================================================================
# CESM1-LE (m2637/LENS)
# =============================================================================

@pytest.fixture(scope="module")
def _has_cesm1_lens():
    if not cfg.NERSC_CESM1_LENS_DIR.is_dir():
        pytest.skip(f"{cfg.NERSC_CESM1_LENS_DIR} not accessible")


def test_list_cesm1_lens_members(_has_cesm1_lens):
    members = list_cesm1_lens_members('ts')
    assert isinstance(members, list)
    assert len(members) > 20, f"Expected 40 members of ts, got {len(members)}"
    assert all(m.startswith('r') for m in members)


def test_load_cesm1_lens_one_member_one_year(_has_cesm1_lens):
    members = list_cesm1_lens_members('ts')[:1]
    if not members:
        pytest.skip("no CESM1-LE members")
    out = load_cesm1_lens(
        variable='ts', member_ids=members,
        sim_time=slice('2000-01-01', '2000-12-31'),
    )
    assert len(out) == 1
    da = next(iter(out.values()))
    assert da.sizes.get('time', 0) == 12


def test_load_cesm1_lens_rejects_unknown_variable(_has_cesm1_lens):
    with pytest.raises(ValueError, match="Only.*on the NERSC mirror"):
        load_cesm1_lens(variable='pr')


# =============================================================================
# LOCA2
# =============================================================================

@pytest.fixture(scope="module")
def _has_loca2():
    if not cfg.NERSC_LOCA2_DIR.is_dir():
        pytest.skip(f"{cfg.NERSC_LOCA2_DIR} not accessible")


def test_list_loca2_models(_has_loca2):
    models = list_loca2_models()
    # Helper subdirs filtered out:
    assert 'monthly' not in models
    assert 'scripts' not in models
    assert 'training_data' not in models
    # Expect the CESM2-LENS entry the user specifically called out
    assert 'CESM2-LENS' in models


def test_list_loca2_members_cesm2_lens(_has_loca2):
    members = list_loca2_members('CESM2-LENS', resolution='0p0625deg')
    assert len(members) >= 10, f"Expected ≥10 members, got {len(members)}"


def test_load_loca2_cesm2_one_member_one_year(_has_loca2):
    members = list_loca2_members('CESM2-LENS')[:1]
    if not members:
        pytest.skip("no CESM2-LENS members")
    out = load_loca2(
        model='CESM2-LENS', variable='pr', experiment='historical',
        member_ids=members,
        sim_time=slice('2000-01-01', '2000-12-31'),
    )
    assert len(out) == 1
    da = next(iter(out.values()))
    assert {'time', 'lat', 'lon'} <= set(da.dims) or \
           {'time', 'latitude', 'longitude'} <= set(da.dims)


def test_load_loca2_rejects_unknown_variable(_has_loca2):
    with pytest.raises(ValueError, match="Unknown LOCA2 variable"):
        load_loca2(model='CESM2-LENS', variable='not_a_var',
                   experiment='historical')


def test_load_loca2_rejects_unknown_experiment(_has_loca2):
    with pytest.raises(ValueError, match="Unknown LOCA2 experiment"):
        load_loca2(model='CESM2-LENS', variable='pr',
                   experiment='not_an_experiment')


# =============================================================================
# Datalake obs SST sources
# =============================================================================

@pytest.mark.parametrize("source", ["hadisst_nersc", "cobe2_nersc", "oisst_v2_nersc"])
def test_load_obs_sst_datalake_one_year(source):
    spec = OBS_SST_SOURCES[source]
    if not spec['path'].exists():
        pytest.skip(f"{source}: {spec['path']} not accessible")
    sst = load_obs_sst(source, sim_time=slice('2000', '2000'))
    assert {'time', 'lat', 'lon'} <= set(sst.dims)
    # Monthly, one year
    assert sst.sizes['time'] >= 12
    finite = sst.values[np.isfinite(sst.values)]
    assert finite.size > 0
    # SST is in either Celsius or Kelvin; accept both rough ranges
    mu = finite.mean()
    assert (-5 < mu < 40) or (260 < mu < 310), \
        f"{source}: mean SST {mu:.2f} not a plausible temperature"
