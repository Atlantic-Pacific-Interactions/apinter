"""Smoke tests for the NEMO-C-grid backend of apinter.heat_budget.

Each test auto-skips if ORAS5_MESH_PATH is not accessible. Tests stay
synthetic on the mesh (small constant/gradient fields) — we're exercising
the grid operators and depth-weighted MLD utilities, not the physics.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import apinter.config as cfg


@pytest.fixture(scope="module")
def _has_mesh():
    if not cfg.ORAS5_MESH_PATH.is_file():
        pytest.skip(f"{cfg.ORAS5_MESH_PATH} not accessible")


@pytest.fixture(scope="module")
def grid(_has_mesh):
    from apinter.heat_budget import NemoGrid
    return NemoGrid()


def test_nemo_grid_has_expected_metrics(grid):
    for name in ('e1t', 'e2t', 'e1u', 'e2u', 'e1v', 'e2v',
                 'e3t_0', 'e3w_0', 'gdept_0', 'gdepw_0',
                 'tmask', 'umask', 'vmask',
                 'glamt', 'gphit', 'ff'):
        assert hasattr(grid, name), f"NemoGrid missing {name!r}"
    assert grid.nz == 75
    assert grid.ny > 800 and grid.nx > 1000


def test_u_to_T_conserves_a_uniform_field(grid):
    """Interpolating a constant U-point field to T-points preserves the value
    (away from the x-wrap boundary)."""
    u = xr.ones_like(grid.e1u)
    uT = grid.u_to_T(u)
    interior = uT.isel(y=slice(1, -1), x=slice(1, -1))
    assert np.all(interior.values == 1.0)


def test_div_h_of_zero_flow_is_zero(grid):
    u = xr.zeros_like(grid.e1u)
    v = xr.zeros_like(grid.e1v)
    div = grid.div_h(u, v)
    assert np.all(div.values == 0.0)


def test_nemo_mldavg_depth_weighted_on_synthetic_column(grid):
    """Uniform T = 15 → ⟨T⟩ = 15 regardless of MLD, using grid.e3t_0."""
    from apinter.heat_budget import nemo_mldavg

    time = pd.date_range('2000-01-01', periods=1, freq='MS')
    nz = grid.nz
    T = xr.DataArray(
        np.full((1, nz, 1, 1), 15.0),
        coords={'time': time, 'z': np.arange(nz), 'y': [0], 'x': [0]},
        dims=['time', 'z', 'y', 'x'],
    )
    mld = xr.DataArray([[[50.0]]],
                       coords={'time': time, 'y': [0], 'x': [0]},
                       dims=['time', 'y', 'x'])
    e3t = xr.DataArray(grid.e3t_0.values, dims=['z'],
                       coords={'z': np.arange(nz)})
    Tmld = nemo_mldavg(mld, T, e3t)
    assert np.allclose(Tmld.values, 15.0)


def test_nemo_submld_returns_first_level_below_mld(grid):
    """Profile T(k) = k; first level with centre depth > MLD is returned."""
    from apinter.heat_budget import nemo_submld

    time = pd.date_range('2000-01-01', periods=1, freq='MS')
    nz = grid.nz
    T_profile = np.arange(nz, dtype=float)
    T = xr.DataArray(
        np.broadcast_to(T_profile[None, :, None, None], (1, nz, 1, 1)).copy(),
        coords={'time': time, 'z': np.arange(nz), 'y': [0], 'x': [0]},
        dims=['time', 'z', 'y', 'x'],
    )
    e3t = xr.DataArray(grid.e3t_0.values, dims=['z'],
                       coords={'z': np.arange(nz)})
    mld = xr.DataArray([[[100.0]]],
                       coords={'time': time, 'y': [0], 'x': [0]},
                       dims=['time', 'y', 'x'])
    Tsub = nemo_submld(mld, T, e3t)
    z_center = np.cumsum(grid.e3t_0.values) - 0.5 * grid.e3t_0.values
    expected_k = int(np.argmax(z_center > 100.0))
    assert np.isclose(Tsub.values[0, 0, 0], float(expected_k))
