"""Tests for the regular-grid MLD utilities."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from apinter.heat_budget.mld import (
    mldavg_varytime, submld_varytime, botmld_varytime,
)


def _build_temp_stack():
    """10 levels at 5 m spacing (2.5, 7.5, ..., 47.5 m); T = -lev (warm surface)."""
    n_t, ny, nx = 3, 2, 2
    lev = np.arange(2.5, 50, 5.0)
    time = pd.date_range('2000-01-01', periods=n_t, freq='MS')
    lat = np.array([-5.0, 5.0])
    lon = np.array([0.0, 180.0])
    temp3d = np.broadcast_to(-lev[None, :, None, None], (n_t, len(lev), ny, nx)).copy()
    temp = xr.DataArray(
        temp3d,
        coords={'time': time, 'lev': lev, 'lat': lat, 'lon': lon},
        dims=['time', 'lev', 'lat', 'lon'],
    )
    return temp, lev


def test_mldavg_weighted_average_matches_analytic():
    """MLD = 20 m: levels 0..3 (centres 2.5/7.5/12.5/17.5), T=-lev.
    ⟨T⟩ = mean(-2.5,-7.5,-12.5,-17.5) = -10.0"""
    temp, lev = _build_temp_stack()
    mld = xr.DataArray(
        np.full((3, 2, 2), 20.0),
        coords={'time': temp.time, 'lat': temp.lat, 'lon': temp.lon},
        dims=['time', 'lat', 'lon'],
    )
    Tmld = mldavg_varytime(mld, temp, z=temp.lev)
    assert np.allclose(Tmld.values, -10.0)


def test_submld_first_level_below_mld():
    temp, lev = _build_temp_stack()
    mld = xr.DataArray(
        np.full((3, 2, 2), 20.0),
        coords={'time': temp.time, 'lat': temp.lat, 'lon': temp.lon},
        dims=['time', 'lat', 'lon'],
    )
    Tsub = submld_varytime(mld, temp, z=temp.lev)
    assert np.allclose(Tsub.values, -22.5)


def test_submld_returns_nan_when_mld_below_deepest_level():
    temp, lev = _build_temp_stack()
    mld = xr.DataArray(
        np.full((3, 2, 2), 1000.0),
        coords={'time': temp.time, 'lat': temp.lat, 'lon': temp.lon},
        dims=['time', 'lat', 'lon'],
    )
    Tsub = submld_varytime(mld, temp, z=temp.lev)
    assert np.all(np.isnan(Tsub.values))


def test_botmld_returns_last_level_inside_ml():
    temp, lev = _build_temp_stack()
    mld = xr.DataArray(
        np.full((3, 2, 2), 20.0),
        coords={'time': temp.time, 'lat': temp.lat, 'lon': temp.lon},
        dims=['time', 'lat', 'lon'],
    )
    Tbot, thickness = botmld_varytime(mld, temp, z=temp.lev)
    assert np.allclose(Tbot.values, -17.5)
    assert np.allclose(thickness.values, 5.0)


def test_submld_rejects_unknown_search_type():
    temp, lev = _build_temp_stack()
    mld = xr.DataArray(np.full((3, 2, 2), 20.0),
                       coords={'time': temp.time, 'lat': temp.lat, 'lon': temp.lon},
                       dims=['time', 'lat', 'lon'])
    with pytest.raises(ValueError, match="search_type"):
        submld_varytime(mld, temp, z=temp.lev, search_type='banana')
