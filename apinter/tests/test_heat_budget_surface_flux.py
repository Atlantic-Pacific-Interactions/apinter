"""Tests for surface_heat_flux."""
import numpy as np
import pandas as pd
import xarray as xr

from apinter.heat_budget.constants import RHO, CP, SW_R, SW_D1, SW_D2
from apinter.heat_budget.surface_flux import surface_heat_flux


def _scalar_fields(qnet_val, qsw_val, mld_val):
    time = pd.date_range('2000-01-01', periods=1, freq='MS')
    coords = {'time': time, 'lat': [0.0], 'lon': [0.0]}
    dims = ['time', 'lat', 'lon']

    def _mk(x):
        return xr.DataArray([[[x]]], coords=coords, dims=dims)

    return _mk(qnet_val), _mk(qsw_val), _mk(mld_val)


def test_no_shortwave_gives_qnet_over_rho_cp_H():
    H, qnet = 50.0, 100.0
    qn, qs, mld = _scalar_fields(qnet, 0.0, H)
    sfc = surface_heat_flux(qn, qs, mld)
    expected = qnet / (RHO * CP * H)
    assert np.isclose(sfc.values[0, 0, 0], expected)


def test_shortwave_penetration_matches_paulson_simpson():
    H, qsw = 20.0, 200.0
    qn, qs, mld = _scalar_fields(qsw, qsw, H)
    sfc = surface_heat_flux(qn, qs, mld)
    qpen = qsw * (SW_R * np.exp(-H / SW_D1) + (1 - SW_R) * np.exp(-H / SW_D2))
    expected = (qsw - qpen) / (RHO * CP * H)
    assert np.isclose(sfc.values[0, 0, 0], expected)
