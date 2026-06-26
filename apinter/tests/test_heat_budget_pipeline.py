"""End-to-end ORAS5 smoke test for apinter.heat_budget.pipeline.compute_budget_nemo.

Auto-skips if ORAS5 or mesh_mask.nc are not accessible. Loads a tiny 3-month
slab and verifies the bundled Dataset has all 24 fields and the expected shape.
Physics validation belongs in paper-specific budget-closure scripts; this is
a structural smoke test only.
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


def test_compute_budget_nemo_pipeline_smoke(_has_oras5):
    from apinter.io import load_oras5
    from apinter.heat_budget import NemoGrid, compute_budget_nemo

    grid = NemoGrid()

    sim = slice('2000-01-01', '2000-03-31')
    # The NEMO-C-grid backend uses vectorised xarray.isel with 2-D indexers,
    # which xarray does not support for dask-backed arrays. Load the slab
    # into memory (matches the upstream workflow).
    thetao = load_oras5('thetao', sim_time=sim).load()
    uo = load_oras5('uo', sim_time=sim).load()
    vo = load_oras5('vo', sim_time=sim).load()
    mld = load_oras5('mld030', sim_time=sim).load()

    # ORAS5 uses 'deptht'/'depthu'/'depthv' for the depth dim; rename to 'z'
    # so it aligns with the NemoGrid mesh_mask 'z' axis.
    def _to_zyx(da):
        rename = {}
        for src in ('deptht', 'depthu', 'depthv'):
            if src in da.dims:
                rename[src] = 'z'
        return da.rename(rename) if rename else da

    thetao = _to_zyx(thetao)
    uo = _to_zyx(uo)
    vo = _to_zyx(vo)

    # Surface fluxes — the smoke test doesn't validate sfcflx physics, so use
    # zero-filled arrays of the right shape rather than loading ERA5 GRIB
    # (which adds a cfgrib dependency the smoke suite shouldn't require).
    qnet = xr.zeros_like(mld)
    qsw = xr.zeros_like(mld)

    ds = compute_budget_nemo(thetao, uo, vo, mld, qnet, qsw, grid,
                              yrclim=[2000, 2000])

    expected = {
        'Tmld', 'Tsub', 'umld', 'vmld', 'wsub',
        'dTdt', 'dTpdt',
        'umdTmdx', 'updTmdx', 'umdTpdx', 'updTpdx',
        'vmdTmdy', 'vpdTmdy', 'vmdTpdy', 'vpdTpdy',
        'mnupdTpdx', 'mnvpdTpdy',
        'w_entr', 'wmdTmdz', 'wpdTmdz', 'wmdTpdz', 'wpdTpdz', 'mnwpdTpdz',
        'sfcflx',
    }
    assert set(ds.data_vars) == expected
    assert ds.Tmld.sizes['time'] == thetao.sizes['time']
    assert ds.Tmld.sizes['y'] == grid.ny
    assert ds.Tmld.sizes['x'] == grid.nx

    finite_frac = float(np.isfinite(ds.Tmld.isel(time=0).values).mean())
    assert finite_frac > 0.2, f"{finite_frac:.2%} finite — expected majority ocean"

    assert ds.attrs['backend'] == 'nemo'
    assert ds.attrs['yrclim'] == '[2000, 2000]'
