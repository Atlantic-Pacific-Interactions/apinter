# `apinter.heat_budget` — mixed-layer heat budget subpackage

**Status:** approved design, ready for implementation plan
**Branch:** `dev`
**Owner:** @yanxia
**Source:** `/pscratch/sd/y/yanxia/Atlantic-Pacific-Interaction-MHW/src/heat_budget/`

## Goal

Port the mixed-layer heat budget code from the Atlantic-Pacific-Interaction-MHW
project into `apinter` as a first-class subpackage, so heat-budget diagnostics
can be computed by any user of `apinter` without reaching into the paper-specific
repo.

Two backends ship together:

1. **Regular lat/lon grid** — approximate spherical operators, works on any
   regridded field (CMIP6, ORAS5-regridded-to-1°, reanalysis).
2. **Native NEMO ORCA C-grid** — exact metrics from `mesh_mask.nc`, recommended
   for ORAS5 on its native curvilinear grid.

The source module `diffusion.py` (unverified horizontal Laplacian + vertical
Pacanowski–Philander eddy diffusion) is **intentionally skipped**.

## Package layout

```
apinter/heat_budget/
├── __init__.py              # flat public re-exports + consolidated references
├── constants.py             # SW_R, SW_D1, SW_D2, MLD_MIN  (subpackage-local only)
│                            #   Shared physics (EARTH_RADIUS→RE, SEAWATER_DENSITY→RHO,
│                            #   SEAWATER_HEAT_CAPACITY→CP, CLIM_START/END) are imported
│                            #   from apinter.config — see "Constants policy" below.
├── tendency.py              # compute_tendency, compute_anomaly_tendency
│                            #   (merged: handles both 'time' and 'time_counter')
├── mld.py                   # mldavg_varytime, submld_varytime, botmld_varytime
├── advection.py             # advection_ml_rd                   (regular grid)
├── entrainment.py           # vertadv_ml_rd, compute_w_from_continuity (regular)
├── surface_flux.py          # surface_heat_flux
├── ml_tendency.py           # ml_mean_temperature, ml_tendency (Nnamchi 2021)
├── nemo_grid.py             # NemoGrid class (ORCA C-grid metrics + operators)
├── nemo_mld.py              # mldavg_varytime, submld_varytime (NEMO, e3t-weighted)
├── nemo_advection.py        # advection_ml_rd                   (NEMO C-grid)
└── nemo_entrainment.py      # vertadv_ml_rd, compute_w_from_continuity, submld_w (NEMO)
```

**Module count:** 12 files total (11 modules + `__init__.py`). Source had
15 files (14 modules + `__init__.py`); this port drops `diffusion.py` (unverified)
and `io_utils.py` (I/O stays in `apinter.io`), and merges `tendency.py` +
`tendency_nemo.py` into a single `tendency.py` that handles both `time` and
`time_counter` dim names.

**Naming convention:** `nemo_*` prefix on filenames (not `_nemo` suffix) so
parallel modules group visually when listing the directory. Public names keep
the `nemo_` prefix too (`nemo_advection_ml_rd`, `nemo_mldavg`, etc.) — matches
the source repo's `__init__.py`.

## Public API

`apinter/heat_budget/__init__.py` re-exports everything flat, matching the
source repo's convention so users can do:

```python
from apinter.heat_budget import (
    # constants
    RHO, CP, RE, SW_R, SW_D1, SW_D2, CLIM_START, CLIM_END, MLD_MIN,
    # regular-grid core
    mldavg_varytime, submld_varytime, botmld_varytime,
    advection_ml_rd, vertadv_ml_rd, compute_w_from_continuity,
    surface_heat_flux, compute_tendency, compute_anomaly_tendency,
    ml_mean_temperature, ml_tendency,
    # NEMO C-grid backend
    NemoGrid,
    nemo_mldavg, nemo_submld,
    nemo_advection_ml_rd, nemo_vertadv_ml_rd,
    nemo_compute_w, nemo_submld_w,
)
```

## Constants policy (hybrid: shared in `config`, local in `heat_budget`)

Match the existing pattern: `apinter.circulation.{walker,hadley}` already
imports `EARTH_RADIUS`, `GRAVITY` from `apinter.config` rather than keeping
locals. `heat_budget/` follows the same rule.

**Rule of thumb:** a constant belongs in `apinter.config` if it could be
reused by another subpackage; otherwise it stays local to `heat_budget/`.

### Additions to `apinter/config.py`

```python
# Cross-subpackage physics (cited by apinter.heat_budget, possibly others later)
SEAWATER_DENSITY       = 1025.0   # kg/m³  (reference ρ, Graham 2014 / Matlab ref)
SEAWATER_HEAT_CAPACITY = 3993.0   # J/(kg·K)  (cp, Matlab ref)

# Project baseline climatology period
CLIM_START = 1981
CLIM_END   = 2010

# NEMO mesh for the ORAS5 backend of apinter.heat_budget
ORAS5_MESH_PATH = ORAS5_DIR / "mesh_mask.nc"
```

`EARTH_RADIUS` (already `6.371e6` in `config.py`) is reused; the Matlab
reference used `6378.0e3` (equatorial radius) — a 0.1 % difference, well
below ORAS5/ERA5 precision. Adopting the config value strictly improves
consistency with `apinter.circulation`.

### `apinter/heat_budget/constants.py`

```python
"""Heat-budget-specific constants.

Shared physics constants are imported from apinter.config:
    EARTH_RADIUS            (re-exported here as RE for brevity)
    SEAWATER_DENSITY        (re-exported as RHO)
    SEAWATER_HEAT_CAPACITY  (re-exported as CP)
    CLIM_START, CLIM_END    (project baseline climatology)
"""
from apinter.config import (
    EARTH_RADIUS as RE,
    SEAWATER_DENSITY as RHO,
    SEAWATER_HEAT_CAPACITY as CP,
    CLIM_START, CLIM_END,
)

# Paulson & Simpson (1977) Type I water — only used in surface_flux.py.
SW_R  = 0.58      # fraction in the visible band
SW_D1 = 0.35      # attenuation depth for visible [m]
SW_D2 = 23.0      # attenuation depth for UV/blue [m]

# Numerical guard against division-by-zero in per-MLD normalisations.
MLD_MIN = 1.0     # m
```

The short aliases (`RE`, `RHO`, `CP`) are re-exported for use inside the
subpackage (where they appear in nearly every formula) and for the public
API. Users importing `from apinter.heat_budget import RHO, CP, RE` get the
same objects as `from apinter.config import SEAWATER_DENSITY, ...`.

## I/O policy

`apinter.heat_budget` contains **no I/O code**. Users load DataArrays with the
existing `apinter.io.*` loaders (`load_oras5`, `load_era5`, `load_cmip6`,
`load_nersc_cmip6`, …) and pass them to the budget functions directly.

The source repo's `io_utils.py` is dissected as follows:

| Source item | Decision |
|---|---|
| `DATA_DIR`, `RESULTS_DIR` hardcoded paths | Dropped — paths belong in `apinter.config`. |
| `load_oras5_regridded` | Dropped — users go through `apinter.io.oras5.load_oras5` + `apinter.processing.regrid_to_1deg`. |
| `load_era5_flux` | **Moved to `apinter.io.era5.load_era5_flux`** — belongs with the other ERA5 loaders; returns `(qnet, qsw)` in W/m². Needs the `cfgrib` optional extra. |
| `regrid_oras5` | Dropped — duplicates `apinter.processing.regrid_to_1deg` (xesmf, handles curvilinear). |
| `save_budget_term` | Dropped — trivial `da.to_netcdf(...)` one-liner. |

`pyproject.toml` gains an optional extra:

```toml
[project.optional-dependencies]
heat_budget_io = ["cfgrib>=0.9"]
```

The core `apinter.heat_budget` subpackage itself adds no new required
dependencies beyond what `apinter` already requires (numpy, xarray).

## Usage examples

### Regular lat/lon grid

```python
from apinter.io import load_nersc_cmip6
from apinter.processing import regrid_to_1deg
from apinter.heat_budget import (
    mldavg_varytime, submld_varytime,
    advection_ml_rd, vertadv_ml_rd, compute_w_from_continuity,
    surface_heat_flux,
)

# User-supplied: 3D thetao/uo/vo + 2D mld + ERA5 qnet/qsw, all on a regular grid.
Tmld = mldavg_varytime(mld, thetao, z=thetao.lev)
umld = mldavg_varytime(mld, uo,     z=uo.lev)
vmld = mldavg_varytime(mld, vo,     z=vo.lev)
Tsub = submld_varytime(mld, thetao, z=thetao.lev)
w3d  = compute_w_from_continuity(uo, vo)
wsub = submld_varytime(mld, w3d, z=w3d.lev)
usub = submld_varytime(mld, uo,  z=uo.lev)
vsub = submld_varytime(mld, vo,  z=vo.lev)

adv = advection_ml_rd(Tmld, umld, vmld)                  # 12 Reynolds terms
ent = vertadv_ml_rd(mld, usub, vsub, Tmld, Tsub, wsub)   # 6 entrainment terms
sfc = surface_heat_flux(qnet, qsw, mld)
```

### NEMO C-grid backend (ORAS5 native grid)

```python
from apinter.io import load_oras5, load_era5_flux
from apinter.heat_budget import (
    NemoGrid,
    nemo_mldavg, nemo_submld,
    nemo_advection_ml_rd, nemo_vertadv_ml_rd,
    nemo_compute_w, nemo_submld_w,
)

grid = NemoGrid()                            # default = apinter.config.ORAS5_MESH_PATH
thetao = load_oras5('thetao')
uo, vo = load_oras5('uo'), load_oras5('vo')
mld    = load_oras5('mld030')
qnet, qsw = load_era5_flux()

Tmld  = nemo_mldavg(mld, thetao, grid.e3t_0)
Tsub  = nemo_submld(mld, thetao, grid.e3t_0)
umld_U = nemo_mldavg(mld, uo, grid.e3t_0)    # at U-points
vmld_V = nemo_mldavg(mld, vo, grid.e3t_0)    # at V-points
umld_T = grid.u_to_T(umld_U)                  # interpolate to T for entrainment
vmld_T = grid.v_to_T(vmld_V)

w3d  = nemo_compute_w(uo, vo, grid)
wsub = nemo_submld_w(mld, w3d, grid.e3t_0)

adv = nemo_advection_ml_rd(Tmld, umld_U, vmld_V, grid)
ent = nemo_vertadv_ml_rd(mld, umld_T, vmld_T, Tmld, Tsub, wsub, grid)
```

## Testing strategy

~26 new tests in `apinter/tests/`. Synthetic wherever possible; real-data
smoke tests auto-skip when their data paths are absent (matches
`test_nersc_io.py` pattern).

| File | Scope | Tests |
|---|---|---:|
| `test_heat_budget_constants.py` | physical constants sanity | 2 |
| `test_heat_budget_tendency.py` | central-difference + anomaly tendency on both `time` and `time_counter` | 4 |
| `test_heat_budget_mld.py` | `mldavg_varytime` weighted avg, `submld_varytime` / `botmld_varytime` first/last indexing | 5 |
| `test_heat_budget_advection.py` | `advection_ml_rd` on uniform-gradient synthetic field; total equals sum of 4 Reynolds terms | 2 |
| `test_heat_budget_entrainment.py` | `compute_w_from_continuity` conservation + `vertadv_ml_rd` sign convention | 3 |
| `test_heat_budget_surface_flux.py` | `surface_heat_flux` with SW penetration + Qnet/ρcpH | 2 |
| `test_heat_budget_ml_tendency.py` | `ml_mean_temperature` partial-cell and integer-MLD limits | 3 |
| `test_heat_budget_nemo.py` | `NemoGrid` load + C-grid operators; auto-skip if `ORAS5_MESH_PATH` absent | 4 |
| `test_heat_budget_pipeline.py` | end-to-end ORAS5 → budget terms smoke; auto-skip if ORAS5 absent | 1 |

Budget-closure tests on real ORAS5 (what the source repo's
`scripts/test/test_budget_closure.py` does) are intentionally **not** ported —
closure depends on data quality and belongs in the downstream paper pipeline,
not the package's unit tests.

## References cited in the package

Each module carries the relevant citation in its top docstring. The subpackage
`__init__.py` carries the consolidated block:

- **Graham, F.S. et al. (2014).** Effectiveness of the Bjerknes stability index
  in representing ocean dynamics. *Climate Dynamics*, 43, 2399–2414.
  → Reynolds decomposition of horizontal / vertical advection.
- **Stevenson, S. et al. (2017).** Spurious precipitation variability in the
  Last Millennium Ensemble. *Climate Dynamics*.
  → `mld_utils`: depth-weighted MLD averaging and sub-MLD sampling.
- **Nnamchi, H.C. et al. (2021).** Atlantic–Pacific link to the tropical
  climate. *Nature Communications*.
  → `ml_tendency`: partial-cell interpolation for ⟨T⟩ at MLD base.
- **Paulson, C.A. & Simpson, J.J. (1977).** Irradiance measurements in the
  upper ocean. *J. Phys. Oceanogr.*, 7, 952–956.
  → `surface_flux`: Type I water two-band shortwave penetration
  (SW_R = 0.58, SW_D1 = 0.35 m, SW_D2 = 23 m).
- **Pacanowski, R.C. & Philander, S.G.H. (1981).** Parameterization of
  vertical mixing in numerical models of tropical oceans. *J. Phys. Oceanogr.*
  → Not used in this port (`diffusion.py` skipped); kept here for reference
  in case the module is revisited later.
- **Madec, G. & the NEMO System Team (2019).** NEMO ocean engine.
  *Scientific Notes of Climate Modelling Center* 27, IPSL.
  → `nemo_grid`: Arakawa C-grid discretisation, metric tensors, mesh_mask.nc.

## DEVLOG entry

Append a 2026-04-24 section to `DEVLOG.md` describing:
- new `apinter.heat_budget` subpackage (12 modules, dual backend)
- `apinter.io.era5.load_era5_flux` (qnet/qsw from GRIB accumulations)
- `apinter.config.ORAS5_MESH_PATH`
- `pyproject.toml` optional extra `heat_budget_io = ["cfgrib>=0.9"]`
- 26 new tests
- example workflow snippets for both backends

## Out of scope (explicit)

- `diffusion.py` (unverified) — skipped.
- `regrid_oras5` helper — duplicates `apinter.processing.regrid_to_1deg`.
- Plotting helpers for budget terms — not in the source repo's subpackage.
- Budget-closure regression tests — depend on data quality, not package logic.
- `load_oras5_regridded` preprocessed-file loader — obsolete given
  `regrid_to_1deg` + `apinter.io.oras5.load_oras5`.
