# `apinter.heat_budget.pipeline` — end-to-end ML heat-budget convenience

**Status:** approved design, ready for implementation plan
**Branch:** `dev`
**Owner:** @yanxia
**Builds on:** `docs/superpowers/specs/2026-04-24-heat-budget-design.md` (the
just-completed port of the dual-backend `apinter.heat_budget` subpackage).

## Goal

Reduce the canonical mixed-layer heat-budget recipe from ~16 lines of
grid-juggling boilerplate to a single function call, while still exposing
every intermediate field for downstream inspection.

The current low-level API requires the user to:
- pick the right MLD utility per backend,
- correctly handle U-/V-/T-point staggering on the NEMO grid,
- thread a consistent `yrclim` window through three separate calls,
- bundle the dict outputs themselves.

A single pipeline call eliminates the most error-prone step (mixing up
U-grid `umld_U` vs T-grid `umld_T`) and bundles everything into one
`xr.Dataset` that is trivial to save (`ds.to_netcdf('budget.nc')`),
slice (`ds.sel(lat=...)`), and plot (`ds.umdTmdx.plot()`).

The pipeline is **pure math** — no I/O. Users load DataArrays via
existing `apinter.io.*` loaders and pass them in.

## API

Two functions, one per backend, in a new module `apinter/heat_budget/pipeline.py`.

### `compute_budget_regular`

```python
def compute_budget_regular(
    thetao: xr.DataArray,
    uo: xr.DataArray,
    vo: xr.DataArray,
    mld: xr.DataArray,
    qnet: xr.DataArray,
    qsw: xr.DataArray,
    *,
    z: xr.DataArray | None = None,
    yrclim: list[int] | None = None,
) -> xr.Dataset:
    """End-to-end ML heat budget on a regular lat/lon grid.

    Parameters
    ----------
    thetao : (time, lev, lat, lon) — potential temperature [°C].
    uo, vo : (time, lev, lat, lon) — velocities at T-points [m/s].
    mld    : (time, lat, lon)      — mixed-layer depth [m].
    qnet   : (time, lat, lon)      — net surface heat flux into ocean [W/m²].
    qsw    : (time, lat, lon)      — net shortwave at surface [W/m²].
    z      : optional depth coord — defaults to ``thetao.lev``.
    yrclim : optional [start, end] — defaults to [CLIM_START, CLIM_END].

    Returns
    -------
    xr.Dataset (time, lat, lon) bundling all 24 budget fields.
    """
```

### `compute_budget_nemo`

```python
def compute_budget_nemo(
    thetao: xr.DataArray,
    uo: xr.DataArray,
    vo: xr.DataArray,
    mld: xr.DataArray,
    qnet: xr.DataArray,
    qsw: xr.DataArray,
    grid: "NemoGrid",
    *,
    yrclim: list[int] | None = None,
) -> xr.Dataset:
    """End-to-end ML heat budget on the NEMO ORCA C-grid.

    Parameters
    ----------
    thetao : (time, z, y, x) at T-points [°C].
    uo     : (time, z, y, x) at U-points [m/s].
    vo     : (time, z, y, x) at V-points [m/s].
    mld    : (time, y, x) at T-points [m].
    qnet, qsw : (time, y, x) at T-points [W/m²].
    grid   : NemoGrid — wraps mesh_mask.nc.
    yrclim : optional [start, end] — defaults to [CLIM_START, CLIM_END].

    Returns
    -------
    xr.Dataset (time, y, x) bundling all 24 budget fields.

    Notes
    -----
    Inputs must be eagerly materialised (call ``.load()`` on dask-backed
    arrays first). The NEMO backend's vectorised ``xarray.isel`` does not
    support dask-backed indexers — see the parent heat_budget spec.
    """
```

## Internal flow

Both functions are short and structurally parallel. The NEMO version handles
the U-/V-/T-point staggering that the regular version does not need.

### Regular grid (`compute_budget_regular`)

```
1.  Resolve z (defaults to thetao.lev) and yrclim.
2.  Tmld = mldavg_varytime(mld, thetao, z)
    Tsub = submld_varytime(mld, thetao, z)
3.  umld = mldavg_varytime(mld, uo, z)
    vmld = mldavg_varytime(mld, vo, z)
    usub = submld_varytime(mld, uo, z)
    vsub = submld_varytime(mld, vo, z)
4.  w3d  = compute_w_from_continuity(uo, vo)
    wsub = submld_varytime(mld, w3d, z)
5.  adv = advection_ml_rd(Tmld, umld, vmld, yrclim=yrclim)
6.  ent = vertadv_ml_rd(mld, usub, vsub, Tmld, Tsub, wsub, yrclim=yrclim)
7.  sfc = surface_heat_flux(qnet, qsw, mld)
8.  Bundle into xr.Dataset (see "Returned Dataset" below).
```

### NEMO C-grid (`compute_budget_nemo`)

```
1.  Resolve yrclim.
2.  e3t = grid.e3t_0.isel(z=slice(0, thetao.sizes['z']))
    Tmld   = nemo_mldavg(mld, thetao, e3t)
    Tsub   = nemo_submld(mld, thetao, e3t)
3.  umld_U = nemo_mldavg(mld, uo, e3t)   # U-points (for advection)
    vmld_V = nemo_mldavg(mld, vo, e3t)   # V-points (for advection)
    umld_T = grid.u_to_T(umld_U)         # T-points (for entrainment)
    vmld_T = grid.v_to_T(vmld_V)
4.  w3d  = nemo_compute_w(uo, vo, grid)
    wsub = nemo_submld_w(mld, w3d, e3t)
5.  adv = nemo_advection_ml_rd(Tmld, umld_U, vmld_V, grid, yrclim=yrclim)
6.  ent = nemo_vertadv_ml_rd(mld, umld_T, vmld_T, Tmld, Tsub, wsub, grid,
                              yrclim=yrclim)
7.  sfc = surface_heat_flux(qnet, qsw, mld)
8.  Bundle into xr.Dataset.
```

The two functions are intentionally not factored into a shared helper
— the staggering rules differ enough that a unified implementation
would lean on ``if backend == 'nemo':`` branches that hurt clarity.
Total LOC: ~30 per function, ~60 in the module.

## Returned Dataset

Both backends return an `xr.Dataset` with **24 data_vars** and identical
naming so downstream code can be backend-agnostic:

| Group | Vars | Source |
|---|---|---|
| Intermediates (5) | `Tmld`, `Tsub`, `umld`, `vmld`, `wsub` | computed in steps 2-4 |
| Advection (12) | `dTdt`, `dTpdt`, `umdTmdx`, `updTmdx`, `umdTpdx`, `updTpdx`, `vmdTmdy`, `vpdTmdy`, `vmdTpdy`, `vpdTpdy`, `mnupdTpdx`, `mnvpdTpdy` | step 5 |
| Entrainment (6) | `w_entr`, `wmdTmdz`, `wpdTmdz`, `wmdTpdz`, `wpdTpdz`, `mnwpdTpdz` | step 6 |
| Surface flux (1) | `sfcflx` | step 7 |

For the NEMO backend, `umld` is at U-points and `vmld` is at V-points
(matching what the advection routine consumed). The T-point versions
used by entrainment are not separately exposed in the Dataset — they
are implementation detail of the entrainment computation.

`Dataset.attrs` carries provenance:

```python
{
    'backend': 'regular' | 'nemo',
    'yrclim': '[1981, 2010]',
    'description': 'Mixed-layer heat budget (Graham 2014, Stevenson 2017)',
}
```

## Public API

`apinter/heat_budget/__init__.py` adds:

```python
from .pipeline import compute_budget_regular, compute_budget_nemo
```

with both names appended to `__all__`. Existing low-level functions stay
exported and unchanged — the pipeline is purely additive.

## Usage

```python
from apinter.io import load_oras5, load_era5_flux
from apinter.heat_budget import NemoGrid, compute_budget_nemo

sim = slice('2000-01-01', '2009-12-31')

thetao = load_oras5('thetao', sim_time=sim).load()
uo     = load_oras5('uo',     sim_time=sim).load()
vo     = load_oras5('vo',     sim_time=sim).load()
mld    = load_oras5('mld030', sim_time=sim).load()
qnet, qsw = load_era5_flux(sim_time=sim)
grid = NemoGrid()

ds = compute_budget_nemo(thetao, uo, vo, mld, qnet, qsw, grid,
                          yrclim=[2000, 2009])
ds.to_netcdf('oras5_budget_2000s.nc')
ds.umdTmdx.sel(time='2005-06').plot()
```

For the regular-grid backend (e.g. with regridded CMIP6):

```python
from apinter.processing import regrid_to_1deg
from apinter.heat_budget import compute_budget_regular

thetao_1deg = regrid_to_1deg(load_cmip6('thetao'))
# ... uo, vo, mld likewise
ds = compute_budget_regular(thetao_1deg, uo_1deg, vo_1deg, mld_1deg,
                             qnet, qsw, yrclim=[1981, 2010])
```

## Tests

Add to `apinter/tests/`:

| File | Tests | Notes |
|---|---:|---|
| `test_heat_budget_pipeline_regular.py` | 2 | Smoke synthetic test that exercises the full pipeline on a tiny in-memory regular-grid stack; verifies the returned Dataset has all 24 data_vars and the expected dims; second test checks `attrs['backend']` and `attrs['yrclim']`. |
| `test_heat_budget_pipeline.py` (existing) | +1 | Replace the manual orchestration in the existing NEMO smoke test with a single `compute_budget_nemo(...)` call. Verify the returned Dataset has 24 data_vars and the expected `(time, y, x)` shape. The existing assertions on advection/entrainment key sets become assertions on `ds.data_vars`. |

The existing `test_heat_budget_pipeline.py` test file already auto-skips
when ORAS5 is absent and runs in ~5 min on Perlmutter; refactoring it to
use the pipeline keeps that timing.

## DEVLOG

Append a short note to the existing `## 2026-04-24 — apinter.heat_budget`
section under a new `### Added (later same day)` subheading, describing
the two new pipeline functions and pointing to this spec.

## Out of scope (explicit)

- I/O wrappers (e.g. `run_oras5_budget(sim_time)`) — see Q3 of the
  brainstorming dialogue: punted to a 3-line user-side wrapper.
- A unified function with a `backend=` switch — see Q1: separate
  signatures are clearer.
- Returning intermediate U/V-point velocity fields separately from the
  bundled `umld`/`vmld`. If a future caller needs that, expose them as
  additional data_vars then.
- Diffusion terms — the `diffusion.py` module from upstream is still
  intentionally skipped.
- A `xr.Dataset.heat_budget` accessor — adds API surface that's only
  cosmetic.
