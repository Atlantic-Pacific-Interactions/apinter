"""End-to-end ORAS5 smoke test for apinter.heat_budget.

Auto-skips if ORAS5 or mesh_mask.nc are not accessible (matches the
test_nersc_io.py auto-skip pattern). Loads a tiny 3-month slab, computes a
handful of budget terms on the NEMO C-grid backend, and verifies the
resulting DataArrays have the expected shape and are mostly finite.

The 3-month slab keeps peak memory under the login-node cgroup cap;
structural assertions are unchanged.
"""
import numpy as np
import pytest
import xarray as xr

import apinter.config as cfg


@pytest.fixture(scope='module')
def _has_oras5():
    if not cfg.ORAS5_DIR.is_dir():
        pytest.skip(f"{cfg.ORAS5_DIR} not accessible")
    if not cfg.ORAS5_MESH_PATH.is_file():
        pytest.skip(f"{cfg.ORAS5_MESH_PATH} not accessible")


def test_nemo_heat_budget_pipeline_two_years(_has_oras5):
    """Compute ML-averaged T + one advection term + one entrainment term from
    a 3-month ORAS5 slab. Structural assertions only — physics validation
    belongs in paper-specific budget-closure scripts. The 3-month slab keeps
    peak memory under the login-node cgroup cap; structural assertions are
    unchanged."""
    from apinter.io import load_oras5
    from apinter.heat_budget import (
        NemoGrid,
        nemo_mldavg, nemo_submld,
        nemo_advection_ml_rd, nemo_compute_w, nemo_submld_w,
        nemo_vertadv_ml_rd,
    )

    grid = NemoGrid()

    sim = slice('2000-01-01', '2000-03-31')
    # The NEMO-C-grid backend uses vectorised xarray.isel with 2-D indexers,
    # which xarray does not support for dask-backed arrays. Load the slab
    # into memory (matches the upstream workflow).
    thetao = load_oras5('thetao', sim_time=sim).load()
    uo = load_oras5('uo', sim_time=sim).load()
    vo = load_oras5('vo', sim_time=sim).load()
    mld = load_oras5('mld030', sim_time=sim).load()

    def _to_zyx(da):
        rename = {}
        for src in ('deptht', 'depthu', 'depthv'):
            if src in da.dims:
                rename[src] = 'z'
        return da.rename(rename) if rename else da

    thetao_z = _to_zyx(thetao)
    uo_z = _to_zyx(uo)
    vo_z = _to_zyx(vo)

    nz = thetao_z.sizes['z']
    e3t = grid.e3t_0.isel(z=slice(0, nz))

    Tmld = nemo_mldavg(mld, thetao_z, e3t)
    Tsub = nemo_submld(mld, thetao_z, e3t)
    umld_U = nemo_mldavg(mld, uo_z, e3t)
    vmld_V = nemo_mldavg(mld, vo_z, e3t)

    assert Tmld.sizes['time'] == thetao.sizes['time']
    assert Tmld.sizes['y'] == grid.ny
    assert Tmld.sizes['x'] == grid.nx

    finite_frac = float(np.isfinite(Tmld.isel(time=0).values).mean())
    assert finite_frac > 0.2, f"{finite_frac:.2%} finite — expected majority ocean"

    adv = nemo_advection_ml_rd(
        Tmld, umld_U, vmld_V, grid, yrclim=[2000, 2000],
    )
    assert set(adv) == {
        'dTdt', 'dTpdt',
        'umdTmdx', 'updTmdx', 'umdTpdx', 'updTpdx',
        'vmdTmdy', 'vpdTmdy', 'vmdTpdy', 'vpdTpdy',
        'mnupdTpdx', 'mnvpdTpdy',
    }

    w3d = nemo_compute_w(uo_z, vo_z, grid)
    wsub = nemo_submld_w(mld, w3d, e3t)
    umld_T = grid.u_to_T(umld_U)
    vmld_T = grid.v_to_T(vmld_V)

    ent = nemo_vertadv_ml_rd(
        mld, umld_T, vmld_T, Tmld, Tsub, wsub, grid, yrclim=[2000, 2000],
    )
    assert set(ent) == {'w_entr', 'wmdTmdz', 'wpdTmdz', 'wmdTpdz', 'wpdTpdz',
                        'mnwpdTpdz'}
