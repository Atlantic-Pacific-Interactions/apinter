# `apinter` — Shared Package for Atlantic-Pacific Interaction Analysis

**Date:** 2026-04-23
**Branch:** `apinter-package`
**Status:** Draft — pending user review

## 1. Goal

Replace four scattered, partially-duplicated `src/` trees (`./src/`, `Paper_1/src/`,
`Paper_2/src/`, `SSP/src/`) and the `E3SMS/path_config.py` with a single installable
Python package, `apinter`, used by all folders (E3SM, Paper_1, Paper_2, SSP).

## 2. Current state (problem)

| Folder | Files | Lines | Role |
|---|---|---|---|
| `./src/` | 6 `.py` + `plot/` | ~960 | Loaders, trends, regression, indices, plotting |
| `Paper_1/src/` | 5 `.py` | ~2140 | Detrend/Lanczos, psi/phi (Li 2006), lead-lag stats, omega plotting |
| `Paper_2/src/` | 7 `.py` + `plot/` | ~1300 | ~90% duplicate of `./src/` + extended CMIP6 loaders |
| `SSP/src/` | 1 `.py` | 233 | Historical+SSP concat, land mask |
| `E3SMS/` | `path_config.py` + `scripts/circulation/*` | — | E3SM paths; Walker/Hadley computation (duplicated across 19 scripts) |

Problems:

1. **Duplication.** Top-level `src/` and `Paper_2/src/` are ~90% identical, with drift
   (`load_cmip6.py`: 240 vs 497 lines — Paper_2 newer/extended).
2. **Coordinate-rename conflict.** Top-level and Paper_1 rename `lat/lon → latitude/longitude`;
   Paper_2 does the **opposite**. Paper_1 imports from Paper_2 via `sys.path.append`, so
   Paper_1's runtime actually follows the Paper_2 convention.
3. **Name collision.** `linear_regression_correlation.py` exists in Paper_1 (149 lines, 3 fns)
   and Paper_2 (86 lines, 2 fns) with different content. Paper_1 version is a superset.
4. **Fragile cross-project `sys.path.append`.** E3SMS scripts and SSP scripts modify `sys.path`
   to reach `Paper_1/src` and `Paper_2/src`. Any move/rename breaks them silently.
5. **Per-variable CMIP6 loaders.** 11 near-identical functions (`load_cmip6_sst`,
   `load_cmip6_omega`, `load_cmip6_zg`, `load_cmip6_psl`, `load_cmip6_pr`, `load_cmip6_zos`,
   `load_cmip6_wind`, `load_cmip6_thetao`, `load_cmip6_tauu`, …) differing only in
   subdir/filename/post-processing.
6. **Circulation primitives copy-pasted.** `load_era5`, `load_e3sm`, `load_cmip6`,
   `interp_to_common`, `plot_psi`, etc. appear ~10× each across
   `E3SMS/scripts/circulation/*.py`. Scientific primitives (`calc_walker_sf`,
   `get_divergent_u`, `get_divergent_v`, `calc_streamfunction`, `compute_velpot`)
   are defined inside plot scripts rather than in a reusable module.

## 3. Design

### 3.1 Package layout

Installable via `pip install -e .` at the repo root. Import as `import apinter`.

```
Midlat-Atlantic-Pacific-Interactions/
├── pyproject.toml                  ← NEW
├── apinter/                        ← NEW
│   ├── __init__.py
│   ├── config.py                   # path constants + grid constants
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   ├── cmip6.py                # generic load_cmip6(var, …) + CMIP6_VARS spec
│   │   ├── obs.py                  # load_obs_sst(source=), load_era5(var, …)
│   │   ├── ssp.py                  # load_and_concat(var, ssp, model, …)
│   │   ├── climatology.py          # load_era5_uv_clim, load_e3sm_uv_clim, load_cmip6_uv_clim
│   │   └── joblib_io.py            # load_joblib / save_joblib
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── anomalies.py            # detrend_dim, compute_anomalies, calculate_anomalies_and_filter
│   │   ├── filters.py              # lanczos_lowpass
│   │   └── regions.py              # wgt_areaave, extract_region
│   │
│   ├── stats/
│   │   ├── __init__.py
│   │   ├── trends.py               # global/spatial/zonal/running/seasonal trends (7 fns)
│   │   ├── regression.py           # calculate_regression_vectorize + helpers
│   │   ├── leadlag.py              # calculate_lead_lag_regression, partial variants
│   │   └── significance.py         # autocorrelation, effective DOF
│   │
│   ├── indices.py                  # TAMV/TPDV/ENSO index calculation
│   │
│   ├── circulation/
│   │   ├── __init__.py
│   │   ├── helmholtz.py            # get_divergent_u, get_divergent_v, compute_velpot
│   │   ├── walker.py               # calc_walker_sf (v2), omega_to_streamfunction
│   │   ├── hadley.py               # calc_streamfunction (v2 regional), calc_streamfunction_global
│   │   ├── regions.py              # ep_mean_at_500, wep_mean_at_500, region_mean_at_500
│   │   ├── profiles.py             # zonal_mean_profile, compute_profiles_for_region, ocean mask
│   │   ├── regrid.py               # interp_to_common
│   │   ├── psi_phi.py              # Li et al. 2006 solver (from Paper_1)
│   │   └── plotting.py             # plot_walker, plot_psi, plot_panel (v2 layouts)
│   │
│   └── plotting/
│       ├── __init__.py
│       ├── index_plots.py          # from ./src/plot/plot_index.py
│       ├── regression_maps.py      # from ./src/plot/plot_regression_map.py
│       ├── trend_maps.py           # from ./src/plot/plot_trend_map.py
│       └── omega_regression.py     # from Paper_1/src/omega_reg_plotting.py
│
├── tests/                          ← NEW
│   └── …
│
├── Paper_1/                        # keeps notebook/, scripts/, results/; src/ removed after phase 5
├── Paper_2/                        # ditto
├── SSP/                            # ditto
└── E3SMS/                          # ditto (path_config.py → apinter.config)
```

### 3.2 Generalization: `apinter.io.cmip6`

The 11 per-variable CMIP6 loaders in `Paper_2/src/load_cmip6.py` reduce to one function
plus a spec dict:

```python
# apinter/io/cmip6.py
CMIP6_VARS = {
    'ts':     {'subdir': '1850-2015-atmos', 'ext': 'zarr', 'to_celsius': True,  'ocean_mask': True},
    'wap':    {'subdir': '1850-2015-atmos', 'ext': 'zarr'},
    'zg':     {'subdir': '1850-2015-atmos', 'ext': 'zarr'},
    'psl':    {'subdir': '1850-2015-atmos', 'ext': 'zarr'},
    'pr':     {'subdir': '1850-2015-atmos', 'ext': 'zarr'},
    'zos':    {'subdir': '1850-2015-atmos', 'ext': 'zarr', 'ocean_mask': True},
    'ua':     {'subdir': '1850-2015-atmos', 'ext': 'nc'},
    'va':     {'subdir': '1850-2015-atmos', 'ext': 'nc'},
    'thetao': {'subdir': 'ocean',           'ext': 'zarr'},
    'tauu':   {'subdir': 'atmos',           'ext': 'zarr'},
}

def load_cmip6(
    var: str,
    sim_time: slice = slice('1980-01-01', '2014-12-31'),
    models: list[str] | None = None,
    level: int | None = None,
    base_path: str | Path = CMIP6_DIR,
) -> dict[str, xr.DataArray]:
    """Generic CMIP6 multi-model loader. Replaces all load_cmip6_* functions."""
```

**Backward-compatibility wrappers** (keep existing notebooks working):

```python
# At the bottom of apinter/io/cmip6.py
def load_cmip6_sst(**kw):    return load_cmip6('ts',     **kw)
def load_cmip6_omega(**kw):  return load_cmip6('wap',    **kw)
def load_cmip6_zg(**kw):     return load_cmip6('zg',     **kw)
def load_cmip6_psl(**kw):    return load_cmip6('psl',    **kw)
def load_cmip6_pr(**kw):     return load_cmip6('pr',     **kw)
def load_cmip6_zos(**kw):    return load_cmip6('zos',    **kw)
def load_cmip6_thetao(**kw): return load_cmip6('thetao', **kw)
def load_cmip6_tauu(**kw):   return load_cmip6('tauu',   **kw)
def load_cmip6_wind(target_level=850, **kw):
    ua = load_cmip6('ua', level=target_level, **kw)
    va = load_cmip6('va', level=target_level, **kw)
    return {m: {'ua': ua[m], 'va': va.get(m)} for m in ua}
```

### 3.3 Generalization: observations

```python
# apinter/io/obs.py
OBS_SST_SOURCES = {
    'hadisst': {'path': HADISST_PATH, 'data_type': 1, 'missing': -1000},
    'ersst':   {'path': ERSST_PATH,   'data_type': 3, 'missing': -1000},
    'cobesst': {'path': COBESST_PATH, 'data_type': 2},
}

def load_obs_sst(source: str = 'hadisst',
                 sim_time: slice = slice('1850', '2023')) -> xr.DataArray:
    """Generic observational SST reader. Replaces read_data()."""


def load_era5(var: str,
              level: int | None = None,
              region: dict | None = None,
              sim_time: slice | None = None,
              process: str = 'raw') -> xr.DataArray:
    """Generic ERA5 reader. Replaces the 6 Paper_1 ERA5 helpers."""
```

### 3.4 Generalization: SSP

```python
# apinter/io/ssp.py
def load_and_concat(
    var: str,                 # 'sst' | 'wap' | 'pr' | …
    ssp: str,                 # 'ssp126' | 'ssp245' | 'ssp370' | 'ssp585'
    model: str,
    region: dict | None = None,
    full_time: slice = slice('1850-01-01', '2100-12-31'),
) -> xr.DataArray:
    """Concatenate historical (1850-2014) + SSP (2015-2100) for any variable."""
```

### 3.5 Circulation — using `_v2` implementations

Scientific primitives lifted from the `_v2.py` variants (confirmed by user as the
preferred versions):

- `calc_walker_sf` — from `plot_walker_streamfunction_v2.py`
- `calc_streamfunction` (regional Hadley) — from `plot_hadley_cell_v2.py`
- `plot_psi`, `plot_walker`, `plot_panel` — v2 signatures with `left_title`/`right_title`

The `_v1` scripts exist only for their slightly older plotting layout; v2 has the same
core math. After migration, both can be deleted and replaced by `apinter.circulation.*`.

### 3.6 Conflict resolution

**Coordinate-rename direction.** Package standardizes on `latitude/longitude → lat/lon`
(the Paper_2 direction), because:
- Short names are canonical in xarray climate workflows.
- Paper_1 notebooks already import Paper_2's `_rename_coordinates` via `sys.path.append`,
  so their runtime already uses `lat/lon`.
- Matches `load_sst.read_data` conventions.

Any `./src/` or `Paper_1/src/` internal code that assumed the opposite direction will
be corrected during phase 2/5.

**Duplicate `detrend_dim`.** Keep the `./src/cal_index.py` version (has `dim='time'`
default). Paper_1's `data_processing.detrend_dim` is a strict subset.

**`linear_regression_correlation.py`.** Paper_1's 3-function version wins — it's a
superset of Paper_2's 2-function version.

### 3.7 Function count summary

| Subpackage | # user-facing fns | Source |
|---|---:|---|
| `apinter.config` | 0 (constants) | E3SMS/path_config + SSP consts |
| `apinter.io.cmip6` | 1 generic + 9 compat wrappers | Paper_2/load_cmip6 |
| `apinter.io.obs` | 2 (load_obs_sst, load_era5) | Paper_2/load_sst + Paper_1 ERA5 helpers |
| `apinter.io.ssp` | 1 generic (+ compat wrappers) | SSP/ssp_data_loading |
| `apinter.io.climatology` | 3 | E3SMS circulation scripts |
| `apinter.io.joblib_io` | 2 | ./src/data_loader |
| `apinter.processing` | 6 | Paper_1/data_processing + ./src/cal_index |
| `apinter.stats` | 16 | ./src/linear_trend + Paper_1/{linear_regression_correlation, statistical_utils} |
| `apinter.indices` | 4 | ./src/cal_index |
| `apinter.circulation` | 33 | Paper_1/psi_phi + E3SMS/scripts/circulation |
| `apinter.plotting` | 15 | ./src/plot/* + Paper_1/omega_reg_plotting |
| **Total** | **~86 user-facing fns** | down from ~120+ duplicates |

## 4. Migration mapping (compat shims)

Existing notebooks do:

```python
sys.path.append('/pscratch/sd/y/yanxia/Midlat-Atlantic-Pacific-Interactions/Paper_2/src')
from load_cmip6 import load_cmip6_sst, load_cmip6_omega
```

After phase 2, both of these work:

```python
# Old style still works via compat shims in apinter.io.cmip6
from apinter.io.cmip6 import load_cmip6_sst, load_cmip6_omega

# New style (recommended)
from apinter.io import load_cmip6
sst = load_cmip6('ts')
omega = load_cmip6('wap')
```

During phase 5 we delete the `sys.path.append` lines folder by folder and switch imports
to `from apinter.* import …`.

## 5. Phased implementation plan

Each phase ends in a working, verifiable state and is a separate commit. All work on
branch `apinter-package`.

### Phase 0 — Scaffold
- Create `pyproject.toml` (setuptools, `apinter` as package name, Python ≥ 3.10)
- Create `apinter/` tree with empty `__init__.py` files in every subpackage
- `pip install -e .` from repo root
- **Verify:** `python -c "import apinter"` succeeds on Perlmutter

### Phase 1 — Pure-function modules (no I/O)
- `apinter.processing` (anomalies, filters, regions)
- `apinter.stats` (trends, regression, leadlag, significance)
- `apinter.indices`
- Tests in `tests/` comparing each function's output against the current `src/` implementation
- **Verify:** run existing notebook cells that use these, results bit-identical

### Phase 2 — I/O layer (the big consolidation)
- `apinter.config` (paths, grid constants)
- `apinter.io.joblib_io`
- `apinter.io.cmip6` (generic + 9 compat wrappers)
- `apinter.io.obs` (SST + ERA5 generic loaders)
- `apinter.io.ssp` (generic concat)
- **Verify:** for each variable, load via new generic vs old per-variable → assert equal

### Phase 3 — Circulation
- `apinter.circulation.{helmholtz, walker, hadley, regions, profiles, regrid}`
- `apinter.circulation.psi_phi` (Paper_1 Li 2006 solver)
- `apinter.circulation.plotting` (v2 layouts)
- `apinter.io.climatology` (consolidating duplicated `load_era5/e3sm/cmip6` climatology helpers)
- **Verify:** re-run one Walker SF figure + one regional Hadley figure, diff against
  committed PNGs in `E3SMS/figures/`

### Phase 4 — Remaining plotting
- `apinter.plotting.{index_plots, regression_maps, trend_maps, omega_regression}`
- **Verify:** re-render one figure per module

### Phase 5 — Migrate callers (one folder at a time)
Order: **Paper_2 → SSP → Paper_1 → E3SMS**. Each folder = one commit.

Per folder:
1. Replace `sys.path.append(...)` + `from <oldmod> import ...` with `from apinter.* import ...`
2. Smoke-run key notebooks/scripts
3. Commit
4. Once all callers migrated for a folder, delete that folder's `src/`

After all four folders migrate cleanly, delete top-level `./src/` and `E3SMS/path_config.py`.

## 6. Verification strategy

- **Phase 1 & 2:** unit tests in `tests/` that load small fixtures and compare
  new-generic-output to legacy-per-variable-output for equality.
- **Phase 3:** figure-regression — checked-in reference PNGs, diff after re-running.
  (Exact pixel match not required; small tolerance on numeric ψ arrays.)
- **Phase 5:** each folder's migration commit includes a short "smoke-test" block
  at the bottom of the commit message listing the notebooks/scripts I ran end-to-end.

## 7. Out of scope (for this spec)

- Documentation site / sphinx.
- Packaging for PyPI — local editable install only.
- Refactor of notebook contents beyond import lines.
- Adding new analysis functions — this spec is purely consolidation.
- Reformatting/renaming unrelated files in `E3SMS/` (many untracked notebooks exist there
  from other in-progress work; they stay as-is on this branch).

## 8. Open questions (resolved in brainstorming)

- [x] Package name → `apinter`
- [x] Keep backward-compat wrappers → yes
- [x] Use `_v2` circulation implementations → yes
- [x] Collapse 11 CMIP6 loaders into 1 generic → yes
- [x] Work on separate branch → `apinter-package`
- [x] Implementation cadence → phased (phase 0 → 5), one commit per phase
