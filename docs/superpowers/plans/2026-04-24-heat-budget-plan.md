# `apinter.heat_budget` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the mixed-layer heat budget code from `Atlantic-Pacific-Interaction-MHW/src/heat_budget/` into `apinter` as a first-class subpackage with dual backend (regular lat/lon + native NEMO ORCA C-grid). Diffusion module intentionally skipped.

**Architecture:** 11 modules under `apinter/heat_budget/` with flat re-exports from `__init__.py`, plus `load_era5_flux` added to `apinter.io.era5`, plus `ORAS5_MESH_PATH` added to `apinter.config`. The subpackage has no I/O: users pass DataArrays to budget functions. Ports are near-verbatim from the source repo (proven code); adaptations are merging `tendency.py` + `tendency_nemo.py` and dropping project-specific paths.

**Tech Stack:** Python ≥3.10, numpy, xarray, pytest. `cfgrib` becomes an optional extra for the ERA5 flux GRIB loader.

**Spec:** `docs/superpowers/specs/2026-04-24-heat-budget-design.md`
**Source reference:** `/pscratch/sd/y/yanxia/Atlantic-Pacific-Interaction-MHW/src/heat_budget/`

---

## File Structure

### Files to create

| Path | Responsibility |
|---|---|
| `apinter/heat_budget/__init__.py` | Flat public re-exports + consolidated reference block |
| `apinter/heat_budget/constants.py` | `RHO`, `CP`, `RE`, `SW_R`, `SW_D1`, `SW_D2`, `MLD_MIN`, `CLIM_START`, `CLIM_END` |
| `apinter/heat_budget/tendency.py` | `compute_tendency`, `compute_anomaly_tendency` (both `time` and `time_counter` dims) |
| `apinter/heat_budget/mld.py` | `mldavg_varytime`, `submld_varytime`, `botmld_varytime` (regular grid) |
| `apinter/heat_budget/advection.py` | `advection_ml_rd` (regular grid) |
| `apinter/heat_budget/entrainment.py` | `vertadv_ml_rd`, `compute_w_from_continuity` (regular grid) |
| `apinter/heat_budget/surface_flux.py` | `surface_heat_flux` |
| `apinter/heat_budget/ml_tendency.py` | `ml_mean_temperature`, `ml_tendency` (partial-cell interpolation) |
| `apinter/heat_budget/nemo_grid.py` | `NemoGrid` class wrapping `mesh_mask.nc` |
| `apinter/heat_budget/nemo_mld.py` | `mldavg_varytime`, `submld_varytime` (NEMO, e3t-weighted) |
| `apinter/heat_budget/nemo_advection.py` | `advection_ml_rd` on NEMO C-grid |
| `apinter/heat_budget/nemo_entrainment.py` | `vertadv_ml_rd`, `compute_w_from_continuity`, `submld_w` (NEMO) |
| `apinter/tests/test_heat_budget_constants.py` | Physical constants sanity |
| `apinter/tests/test_heat_budget_tendency.py` | Central-difference + anomaly tendency |
| `apinter/tests/test_heat_budget_mld.py` | MLD-weighted avg, sub/bot-MLD sampling |
| `apinter/tests/test_heat_budget_advection.py` | Reynolds decomposition closure |
| `apinter/tests/test_heat_budget_entrainment.py` | Continuity-w + entrainment sign |
| `apinter/tests/test_heat_budget_surface_flux.py` | SW penetration + Qnet/ρcpH |
| `apinter/tests/test_heat_budget_ml_tendency.py` | Partial-cell interpolation limits |
| `apinter/tests/test_heat_budget_nemo.py` | NemoGrid load + C-grid operators (auto-skip) |
| `apinter/tests/test_heat_budget_pipeline.py` | End-to-end ORAS5 smoke test (auto-skip) |

### Files to modify

| Path | Change |
|---|---|
| `apinter/config.py` | Add `ORAS5_MESH_PATH = ORAS5_DIR / "mesh_mask.nc"` |
| `apinter/io/era5.py` | Add `load_era5_flux` function returning `(qnet, qsw)` |
| `pyproject.toml` | Add `heat_budget_io = ["cfgrib>=0.9"]` optional extra |
| `DEVLOG.md` | Append 2026-04-24 entry describing the new subpackage |

---

## Task 1: Config + package skeleton + constants

**Files:**
- Modify: `apinter/config.py`
- Create: `apinter/heat_budget/__init__.py`
- Create: `apinter/heat_budget/constants.py`
- Create: `apinter/tests/test_heat_budget_constants.py`

- [ ] **Step 1.1: Write the failing test**

Create `apinter/tests/test_heat_budget_constants.py`:

```python
"""Sanity checks on heat-budget physical constants."""
import apinter.config as cfg
from apinter.heat_budget import (
    RHO, CP, RE, SW_R, SW_D1, SW_D2, MLD_MIN,
    CLIM_START, CLIM_END,
)


def test_physical_constants_in_expected_ranges():
    # Seawater reference density and heat capacity (Graham 2014 / Matlab ref)
    assert 1020.0 <= RHO <= 1030.0, f"ρ = {RHO} not a seawater density"
    assert 3900.0 <= CP <= 4100.0, f"cp = {CP} not a seawater heat capacity"
    # Earth radius in metres
    assert 6.0e6 <= RE <= 6.5e6, f"RE = {RE} not a plausible Earth radius"
    # Paulson & Simpson (1977) Type I water coefficients
    assert 0.0 < SW_R < 1.0
    assert SW_D1 < SW_D2
    assert SW_D1 > 0.0 and SW_D2 > 0.0
    # MLD safety guard must be a small positive depth
    assert 0.0 < MLD_MIN < 5.0


def test_climatology_period_default():
    # Project convention: 1981-2010 monthly climatology
    assert CLIM_START == 1981
    assert CLIM_END == 2010
    assert CLIM_START < CLIM_END


def test_config_has_oras5_mesh_path():
    # Spec: apinter.config must expose the mesh_mask path for NemoGrid().
    from apinter.config import ORAS5_MESH_PATH, ORAS5_DIR
    assert ORAS5_MESH_PATH == ORAS5_DIR / "mesh_mask.nc"
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `pytest apinter/tests/test_heat_budget_constants.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'apinter.heat_budget'`).

- [ ] **Step 1.3: Add `ORAS5_MESH_PATH` to config**

Edit `apinter/config.py`. Insert after the existing `ORAS5_START_YEAR = 1958` line:

```python
ORAS5_MESH_PATH = ORAS5_DIR / "mesh_mask.nc"
```

- [ ] **Step 1.4: Create `apinter/heat_budget/constants.py`**

```python
"""Physical constants for the mixed-layer heat budget.

Values match the Matlab reference code (heat_budget_rd_lme.m) used by
Graham et al. (2014) and Stevenson et al. (2017).
"""

# Seawater reference properties (Graham 2014 / Stevenson 2017)
RHO = 1025.0    # Reference density of seawater [kg/m³]
CP = 3993.0     # Specific heat of seawater [J/(kg·K)]

# Earth geometry (equatorial radius, matches Matlab reference)
RE = 6378.0e3   # [m]

# Shortwave penetration — Paulson & Simpson (1977), Type I water:
#   qpen = qsw * (SW_R * exp(-H/SW_D1) + (1-SW_R) * exp(-H/SW_D2))
SW_R = 0.58     # fraction for the visible band
SW_D1 = 0.35    # attenuation depth for visible [m]
SW_D2 = 23.0    # attenuation depth for UV/blue [m]

# Project baseline monthly climatology period
CLIM_START = 1981
CLIM_END = 2010

# Numerical guard against division-by-zero in per-MLD normalisations
MLD_MIN = 1.0   # [m]
```

- [ ] **Step 1.5: Create `apinter/heat_budget/__init__.py` (initial skeleton)**

```python
"""Mixed-layer heat budget with Reynolds decomposition.

Two backends:
  - Regular lat/lon grid (approximate spherical operators): mld, advection,
    entrainment, surface_flux, ml_tendency.
  - Native NEMO ORCA C-grid (exact metrics from mesh_mask.nc): nemo_grid,
    nemo_mld, nemo_advection, nemo_entrainment. Recommended for ORAS5.

References
----------
Graham, F.S. et al. (2014). Effectiveness of the Bjerknes stability index in
    representing ocean dynamics. Climate Dynamics, 43, 2399-2414.
    — Reynolds decomposition of horizontal / vertical advection.
Stevenson, S. et al. (2017). Spurious precipitation variability in the Last
    Millennium Ensemble. Climate Dynamics.
    — mld utilities: depth-weighted MLD averaging and sub-MLD sampling.
Nnamchi, H.C. et al. (2021). Atlantic-Pacific link to the tropical climate.
    Nature Communications.
    — ml_tendency: partial-cell interpolation for ⟨T⟩ at MLD base.
Paulson, C.A. & Simpson, J.J. (1977). Irradiance measurements in the upper
    ocean. J. Phys. Oceanogr., 7, 952-956.
    — surface_flux: Type I water two-band shortwave penetration.
Madec, G. & the NEMO System Team (2019). NEMO ocean engine. Scientific Notes
    of Climate Modelling Center 27, Institut Pierre-Simon Laplace.
    — nemo_grid: Arakawa C-grid discretisation, metric tensors, mesh_mask.nc.
"""
from .constants import (
    RHO, CP, RE,
    SW_R, SW_D1, SW_D2,
    MLD_MIN,
    CLIM_START, CLIM_END,
)

__all__ = [
    'RHO', 'CP', 'RE',
    'SW_R', 'SW_D1', 'SW_D2',
    'MLD_MIN',
    'CLIM_START', 'CLIM_END',
]
```

(Later tasks will extend `__init__.py`'s imports and `__all__` as modules are added.)

- [ ] **Step 1.6: Run test to verify it passes**

Run: `pytest apinter/tests/test_heat_budget_constants.py -v`
Expected: 3 PASSED.

- [ ] **Step 1.7: Commit**

```bash
git add apinter/config.py apinter/heat_budget/__init__.py \
        apinter/heat_budget/constants.py \
        apinter/tests/test_heat_budget_constants.py
git commit -m "dev: apinter.heat_budget — constants + config ORAS5_MESH_PATH"
```

---

## Task 2: `tendency.py` — dual-dim (`time` / `time_counter`) tendency helpers

**Files:**
- Create: `apinter/heat_budget/tendency.py`
- Create: `apinter/tests/test_heat_budget_tendency.py`
- Modify: `apinter/heat_budget/__init__.py` (add imports + `__all__` entries)

- [ ] **Step 2.1: Write the failing test**

Create `apinter/tests/test_heat_budget_tendency.py`:

```python
"""Tests for compute_tendency and compute_anomaly_tendency (both time dims)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from apinter.heat_budget.tendency import compute_tendency, compute_anomaly_tendency


def _linear_ts(time_name='time'):
    """10-year monthly series that increases by 1 degC/year (≈ 1/yr_s per second)."""
    n = 120
    time = pd.date_range('1981-01-01', periods=n, freq='MS')
    values = np.arange(n, dtype=float) / 12.0   # 1 degC per year
    return xr.DataArray(values, coords={time_name: time}, dims=[time_name], name='T')


def test_compute_tendency_linear_time_dim():
    da = _linear_ts('time')
    dTdt = compute_tendency(da)
    # Expect ≈ 1 degC/yr = 1/(365.25*86400) degC/s in the interior
    expected = 1.0 / (365.25 * 86400.0)
    mid = dTdt.isel(time=slice(10, -10)).values
    assert np.allclose(mid, expected, rtol=0.05), mid.mean()


def test_compute_tendency_time_counter_dim():
    da = _linear_ts('time_counter')
    dTdt = compute_tendency(da)
    expected = 1.0 / (365.25 * 86400.0)
    mid = dTdt.isel(time_counter=slice(10, -10)).values
    assert np.allclose(mid, expected, rtol=0.05)


def test_compute_anomaly_tendency_removes_seasonal_cycle():
    """A pure seasonal cycle has zero long-term tendency; its anomaly dT'/dt
    should also be zero (anomalies = 0 after deseasonalising)."""
    n = 120
    time = pd.date_range('1981-01-01', periods=n, freq='MS')
    season = 5.0 * np.sin(2 * np.pi * np.arange(n) / 12)
    da = xr.DataArray(season, coords={'time': time}, dims=['time'])
    dTpdt = compute_anomaly_tendency(da, yrclim=[1981, 1990])
    assert np.allclose(dTpdt.values, 0.0, atol=1e-10)


def test_compute_tendency_raises_without_time_dim():
    da = xr.DataArray([1.0, 2.0], dims=['x'])
    with pytest.raises(ValueError, match="No time dimension"):
        compute_tendency(da)
```

- [ ] **Step 2.2: Run to verify failure**

Run: `pytest apinter/tests/test_heat_budget_tendency.py -v`
Expected: FAIL (`ImportError` from `apinter.heat_budget.tendency`).

- [ ] **Step 2.3: Create `apinter/heat_budget/tendency.py`**

```python
"""Time tendency computation for the mixed-layer heat budget.

Port of Matlab code used in heat_budget_rd_lme.m. Handles both 'time'
and 'time_counter' dim names so the same helpers work for CMIP6 / ERA5
and for ORAS5 native files (which keep NEMO's 'time_counter').
"""
import xarray as xr

from .constants import CLIM_START, CLIM_END


def _get_time_dim(field):
    for name in ('time', 'time_counter'):
        if name in field.dims:
            return name
    raise ValueError(f"No time dimension found in {field.dims}")


def compute_tendency(field):
    """dT/dt via xarray central difference.

    Second-order central differences in the interior and first-order one-sided
    differences at boundaries. Returns units of [field_units]/s.
    """
    time_dim = _get_time_dim(field)
    return field.differentiate(time_dim, datetime_unit='s', edge_order=1)


def compute_anomaly_tendency(field, yrclim=None):
    """dT'/dt: subtract the monthly climatology over ``yrclim``, then differentiate."""
    if yrclim is None:
        yrclim = [CLIM_START, CLIM_END]

    time_dim = _get_time_dim(field)
    clim_slice = slice(str(yrclim[0]), str(yrclim[1]))

    clim = (field.sel({time_dim: clim_slice})
                 .groupby(f'{time_dim}.month').mean(time_dim))
    month = field[f'{time_dim}.month']
    clim_full = clim.sel(month=month).drop_vars('month')
    anomaly = field - clim_full

    return anomaly.differentiate(time_dim, datetime_unit='s', edge_order=1)
```

- [ ] **Step 2.4: Extend `apinter/heat_budget/__init__.py`**

After the existing `from .constants import ...` block, add:

```python
from .tendency import compute_tendency, compute_anomaly_tendency
```

In `__all__`, append:

```python
    'compute_tendency', 'compute_anomaly_tendency',
```

- [ ] **Step 2.5: Run to verify pass**

Run: `pytest apinter/tests/test_heat_budget_tendency.py -v`
Expected: 4 PASSED.

- [ ] **Step 2.6: Commit**

```bash
git add apinter/heat_budget/tendency.py apinter/heat_budget/__init__.py \
        apinter/tests/test_heat_budget_tendency.py
git commit -m "dev: apinter.heat_budget — tendency helpers (time + time_counter)"
```

---

## Task 3: `mld.py` — regular-grid MLD utilities

**Files:**
- Create: `apinter/heat_budget/mld.py`
- Create: `apinter/tests/test_heat_budget_mld.py`
- Modify: `apinter/heat_budget/__init__.py`

- [ ] **Step 3.1: Write the failing test**

Create `apinter/tests/test_heat_budget_mld.py`:

```python
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
    # T decreases linearly from 0 at surface; values: +2.5 at surface, -47.5 at bottom
    temp3d = np.broadcast_to(-lev[None, :, None, None], (n_t, len(lev), ny, nx)).copy()
    temp = xr.DataArray(
        temp3d,
        coords={'time': time, 'lev': lev, 'lat': lat, 'lon': lon},
        dims=['time', 'lev', 'lat', 'lon'],
    )
    return temp, lev


def test_mldavg_weighted_average_matches_analytic():
    """With lev centers at 2.5, 7.5, ... and MLD = 20 m, ML contains levels 0..3
    (centers 2.5/7.5/12.5/17.5). T=-lev; layer thickness = 5 m each.
    ⟨T⟩ = mean(-2.5, -7.5, -12.5, -17.5) = -10.0"""
    temp, lev = _build_temp_stack()
    mld = xr.DataArray(
        np.full((3, 2, 2), 20.0),
        coords={'time': temp.time, 'lat': temp.lat, 'lon': temp.lon},
        dims=['time', 'lat', 'lon'],
    )
    Tmld = mldavg_varytime(mld, temp, z=temp.lev)
    assert np.allclose(Tmld.values, -10.0)


def test_submld_first_level_below_mld():
    """MLD = 20 m → first level below at 22.5 m → T = -22.5."""
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
        np.full((3, 2, 2), 1000.0),  # deeper than any level
        coords={'time': temp.time, 'lat': temp.lat, 'lon': temp.lon},
        dims=['time', 'lat', 'lon'],
    )
    Tsub = submld_varytime(mld, temp, z=temp.lev)
    assert np.all(np.isnan(Tsub.values))


def test_botmld_returns_last_level_inside_ml():
    """MLD = 20 m → last level inside ML at 17.5 m → T = -17.5."""
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
```

- [ ] **Step 3.2: Run to verify failure**

Run: `pytest apinter/tests/test_heat_budget_mld.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3.3: Create `apinter/heat_budget/mld.py`**

Port `src/heat_budget/mld_utils.py` verbatim, renaming the module. The only change is the module docstring:

```python
"""Mixed-layer depth utilities on a regular lat/lon grid.

Port of Matlab functions mldavg_varytime.m, submld_varytime.m, botmld_varytime.m.
Reference: Stevenson et al. (2017) Climate Dynamics.
"""
import numpy as np
import xarray as xr


def mldavg_varytime(mld, field, z):
    """Depth-weighted average of a 4D field over the mixed layer.

    Weights each level by its layer thickness, computed from the lev coordinate
    (assuming lev gives layer centres, surface at z=0).

    Parameters
    ----------
    mld : xr.DataArray (time, lat, lon)
        Mixed layer depth in metres.
    field : xr.DataArray (time, lev, lat, lon)
        Field to average (e.g. temperature, velocity).
    z : xr.DataArray (lev,) or (lev, lat, lon)
        Depth levels in metres (positive downward).

    Returns
    -------
    xr.DataArray (time, lat, lon) — depth-weighted mean over the ML.
    """
    z_vals = z.values if z.ndim == 1 else z
    if z.ndim == 1:
        layer_bot = np.zeros(len(z_vals))
        for i in range(len(z_vals)):
            layer_bot[i] = 2.0 * z_vals[i] - (layer_bot[i - 1] if i > 0 else 0.0)
        thickness = np.diff(np.concatenate([[0.0], layer_bot]))
        dz = xr.DataArray(thickness, dims=['lev'], coords={'lev': field.lev})
    else:
        dz = z.diff('lev')
        dz = xr.concat([z.isel(lev=0), dz], dim='lev')

    mask = z <= mld
    weighted_sum = (field * dz).where(mask, 0.0).sum(dim='lev')
    total_weight = dz.where(mask, 0.0).sum(dim='lev')
    return xr.where(total_weight > 0, weighted_sum / total_weight, np.nan)


def submld_varytime(mld, field, z, search_type='first'):
    """Extract field value just below the mixed layer depth.

    Matches Matlab submld_varytime.m: ``find(z > mld, 1, 'first')``.
    Returns NaN where no sub-MLD level exists.

    Parameters
    ----------
    search_type : {'first', 'last'}
        'first' for CESM/POP/ORAS5 ordering, 'last' for ROMS.
    """
    if z.ndim == 1:
        z_broadcast = z.broadcast_like(field)
    else:
        z_broadcast = z

    diff = z_broadcast - mld
    nlev = field.sizes['lev']
    idx_arr = xr.DataArray(np.arange(nlev), dims=['lev'],
                           coords={'lev': field.lev})
    idx_broadcast = idx_arr.broadcast_like(field)

    if search_type == 'first':
        condition = diff > 0
        has_valid = condition.any(dim='lev')
        masked_idx = xr.where(condition, idx_broadcast, nlev)
        target_idx = masked_idx.min(dim='lev')
    elif search_type == 'last':
        condition = diff < 0
        has_valid = condition.any(dim='lev')
        masked_idx = xr.where(condition, idx_broadcast, -1)
        target_idx = masked_idx.max(dim='lev')
    else:
        raise ValueError("search_type must be 'first' or 'last'")

    target_idx = target_idx.clip(0, nlev - 1).astype(int)
    result = field.isel(lev=target_idx)
    return xr.where(has_valid, result, np.nan)


def botmld_varytime(mld, field, z, search_type='first'):
    """Extract field at the deepest level still inside the ML + layer thickness.

    Matches Matlab botmld_varytime.m: ``find(z < mld, 1, 'last')`` for CESM/POP.
    Returns (value, thickness).
    """
    if z.ndim == 1:
        z_broadcast = z.broadcast_like(field)
    else:
        z_broadcast = z

    diff = z_broadcast - mld
    nlev = field.sizes['lev']
    idx_arr = xr.DataArray(np.arange(nlev), dims=['lev'],
                           coords={'lev': field.lev})
    idx_broadcast = idx_arr.broadcast_like(field)

    condition = diff < 0
    has_valid = condition.any(dim='lev')
    masked_idx = xr.where(condition, idx_broadcast, -1)
    target_idx = masked_idx.max(dim='lev').clip(0, nlev - 1).astype(int)

    result = field.isel(lev=target_idx)
    result = xr.where(has_valid, result, np.nan)

    next_idx = (target_idx + 1).clip(0, nlev - 1).astype(int)
    if z.ndim == 1:
        z_at_target = z.values[target_idx.values]
        z_at_next = z.values[next_idx.values]
        thickness = xr.DataArray(
            z_at_next - z_at_target,
            dims=result.dims, coords=result.coords,
        )
    else:
        thickness = z_broadcast.isel(lev=next_idx) - z_broadcast.isel(lev=target_idx)
    thickness = xr.where(has_valid, thickness, np.nan)

    return result, thickness
```

- [ ] **Step 3.4: Extend `apinter/heat_budget/__init__.py`**

Add import:

```python
from .mld import mldavg_varytime, submld_varytime, botmld_varytime
```

Append to `__all__`:

```python
    'mldavg_varytime', 'submld_varytime', 'botmld_varytime',
```

- [ ] **Step 3.5: Run to verify pass**

Run: `pytest apinter/tests/test_heat_budget_mld.py -v`
Expected: 5 PASSED.

- [ ] **Step 3.6: Commit**

```bash
git add apinter/heat_budget/mld.py apinter/heat_budget/__init__.py \
        apinter/tests/test_heat_budget_mld.py
git commit -m "dev: apinter.heat_budget — MLD utilities (regular grid)"
```

---

## Task 4: `surface_flux.py`

**Files:**
- Create: `apinter/heat_budget/surface_flux.py`
- Create: `apinter/tests/test_heat_budget_surface_flux.py`
- Modify: `apinter/heat_budget/__init__.py`

- [ ] **Step 4.1: Write the failing test**

Create `apinter/tests/test_heat_budget_surface_flux.py`:

```python
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
    """When qsw = 0, qpen = 0 and sfcflx = qnet / (ρ cp H)."""
    H = 50.0
    qnet = 100.0
    qn, qs, mld = _scalar_fields(qnet, 0.0, H)
    sfc = surface_heat_flux(qn, qs, mld)
    expected = qnet / (RHO * CP * H)
    assert np.isclose(sfc.values[0, 0, 0], expected)


def test_shortwave_penetration_matches_paulson_simpson():
    """With qnet = qsw, qpen comes out of the ML; sfcflx = (qsw - qpen) / (ρ cp H).
    qpen = qsw * (SW_R * exp(-H/SW_D1) + (1-SW_R) * exp(-H/SW_D2))."""
    H = 20.0
    qsw = 200.0
    qn, qs, mld = _scalar_fields(qsw, qsw, H)
    sfc = surface_heat_flux(qn, qs, mld)
    qpen = qsw * (SW_R * np.exp(-H / SW_D1) + (1 - SW_R) * np.exp(-H / SW_D2))
    expected = (qsw - qpen) / (RHO * CP * H)
    assert np.isclose(sfc.values[0, 0, 0], expected)
```

- [ ] **Step 4.2: Run to verify failure**

Run: `pytest apinter/tests/test_heat_budget_surface_flux.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 4.3: Create `apinter/heat_budget/surface_flux.py`**

Port the source verbatim:

```python
"""Surface heat flux term for the mixed-layer heat budget.

sfcflx = (Qnet - Qpen) / (ρ cp H),
where Qpen is the shortwave that penetrates below the ML via the
Paulson & Simpson (1977) Type I water two-band formula:
    Qpen = Qsw * (SW_R * exp(-H/SW_D1) + (1 - SW_R) * exp(-H/SW_D2))

Port of Matlab heat_budget_rd_lme.m lines 98-105.
"""
import numpy as np
import xarray as xr

from .constants import RHO, CP, SW_R, SW_D1, SW_D2, MLD_MIN


def surface_heat_flux(qnet, qsw, mld):
    """Surface-flux contribution to the ML temperature tendency (°C/s).

    Parameters
    ----------
    qnet : xr.DataArray (time, lat, lon)
        Net surface heat flux (W/m²), positive into ocean.
    qsw : xr.DataArray (time, lat, lon)
        Net shortwave at the surface (W/m²).
    mld : xr.DataArray (time, lat, lon)
        Mixed layer depth (m).
    """
    mld_invalid = (mld <= 0) | np.isnan(mld)
    mld_safe = mld.clip(min=MLD_MIN)

    qpen = qsw * (SW_R * np.exp(-mld_safe / SW_D1)
                  + (1 - SW_R) * np.exp(-mld_safe / SW_D2))

    sfcflx = (qnet - qpen) / (RHO * CP * mld_safe)
    return xr.where(mld_invalid, np.nan, sfcflx)
```

- [ ] **Step 4.4: Extend `__init__.py`**

```python
from .surface_flux import surface_heat_flux
```

Append to `__all__`: `'surface_heat_flux',`.

- [ ] **Step 4.5: Run**

Run: `pytest apinter/tests/test_heat_budget_surface_flux.py -v`
Expected: 2 PASSED.

- [ ] **Step 4.6: Commit**

```bash
git add apinter/heat_budget/surface_flux.py apinter/heat_budget/__init__.py \
        apinter/tests/test_heat_budget_surface_flux.py
git commit -m "dev: apinter.heat_budget — surface heat flux with SW penetration"
```

---

## Task 5: `advection.py` — regular-grid horizontal advection (Reynolds decomposition)

**Files:**
- Create: `apinter/heat_budget/advection.py`
- Create: `apinter/tests/test_heat_budget_advection.py`
- Modify: `apinter/heat_budget/__init__.py`

- [ ] **Step 5.1: Write the failing test**

Create `apinter/tests/test_heat_budget_advection.py`:

```python
"""Tests for Reynolds-decomposed horizontal advection on a regular grid."""
import numpy as np
import pandas as pd
import xarray as xr

from apinter.heat_budget.advection import advection_ml_rd


def _uniform_field(n_t=60):
    time = pd.date_range('1981-01-01', periods=n_t, freq='MS')
    lat = np.arange(-10.0, 11.0, 1.0)
    lon = np.arange(0.0, 360.0, 1.0)
    coords = {'time': time, 'lat': lat, 'lon': lon}
    dims = ['time', 'lat', 'lon']

    # T varies linearly in lon (zonal gradient only); seasonal cycle added in time.
    t_ix = np.arange(n_t)[:, None, None]
    lon_b = lon[None, None, :]
    season = 0.5 * np.sin(2 * np.pi * t_ix / 12)
    T = 20.0 + 0.01 * lon_b + season
    T = np.broadcast_to(T, (n_t, len(lat), len(lon))).copy()

    # Constant westward advection u = 1 m/s; v = 0.
    u = np.ones_like(T) * 1.0
    v = np.zeros_like(T)

    Tmld = xr.DataArray(T, coords=coords, dims=dims)
    umld = xr.DataArray(u, coords=coords, dims=dims)
    vmld = xr.DataArray(v, coords=coords, dims=dims)
    return Tmld, umld, vmld


def test_advection_ml_rd_returns_all_twelve_keys():
    Tmld, umld, vmld = _uniform_field()
    out = advection_ml_rd(Tmld, umld, vmld, yrclim=[1981, 1985])
    expected = {
        'dTdt', 'dTpdt',
        'umdTmdx', 'updTmdx', 'umdTpdx', 'updTpdx',
        'vmdTmdy', 'vpdTmdy', 'vmdTpdy', 'vpdTpdy',
        'mnupdTpdx', 'mnvpdTpdy',
    }
    assert set(out) == expected


def test_reynolds_decomposition_sums_to_total_u_grad_t():
    """umdTmdx + updTmdx + umdTpdx + updTpdx should equal u·∂T/∂x."""
    Tmld, umld, vmld = _uniform_field()
    out = advection_ml_rd(Tmld, umld, vmld, yrclim=[1981, 1985])

    # Direct total using the same spherical formula
    from apinter.heat_budget.constants import RE
    lat_rad = np.deg2rad(Tmld.lat)
    cos_lat = np.cos(lat_rad)
    dTdx = Tmld.differentiate('lon', edge_order=1) / (RE * cos_lat * np.deg2rad(1.0))
    total = umld * dTdx

    sum_rd = out['umdTmdx'] + out['updTmdx'] + out['umdTpdx'] + out['updTpdx']
    # Compare away from the lon wrap boundary to avoid edge effects
    mid = dict(lat=slice(2, -2), lon=slice(2, -2))
    assert np.allclose(sum_rd.isel(**mid).values,
                       total.isel(**mid).values, rtol=1e-6, atol=1e-12)
```

- [ ] **Step 5.2: Run to verify failure**

Run: `pytest apinter/tests/test_heat_budget_advection.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 5.3: Create `apinter/heat_budget/advection.py`**

Port `src/heat_budget/horizontal_advection.py` verbatim. The `_compute_gradients` helper stays private; it is also used by `entrainment.py` in the next task (import it from this module).

```python
"""Reynolds-decomposed horizontal advection for the mixed-layer heat budget.

Port of Matlab advection_ml_rd.m. Reference: Graham et al. (2014) Climate
Dynamics, 43:2399-2414.
"""
import numpy as np
import xarray as xr

from .constants import RE, CLIM_START, CLIM_END
from .tendency import compute_tendency, compute_anomaly_tendency


def _compute_gradients(field):
    """Spherical gradients on a regular lat/lon grid (radians under the hood).

    dT/dx = dT/dlon_rad / (R_E cos φ)
    dT/dy = dT/dlat_rad / R_E
    """
    lat_rad = np.deg2rad(field.lat)
    cos_lat = np.cos(lat_rad)

    dfdx = field.differentiate('lon', edge_order=1) / (RE * cos_lat * np.deg2rad(1.0))
    if field.sizes['lat'] >= 2:
        dfdy = field.differentiate('lat', edge_order=1) / (RE * np.deg2rad(1.0))
    else:
        dfdy = xr.zeros_like(field)

    return dfdx, dfdy


def advection_ml_rd(Tmld, umld, vmld, yrclim=None):
    """Compute the 4 Reynolds-decomposed horizontal advection terms per direction.

    Returns a dict with keys:
        dTdt, dTpdt,
        umdTmdx, updTmdx, umdTpdx, updTpdx,
        vmdTmdy, vpdTmdy, vmdTpdy, vpdTpdy,
        mnupdTpdx, mnvpdTpdy.
    """
    if yrclim is None:
        yrclim = [CLIM_START, CLIM_END]

    clim_slice = slice(str(yrclim[0]), str(yrclim[1]))

    dTdx, dTdy = _compute_gradients(Tmld)

    Tm = Tmld.sel(time=clim_slice).groupby('time.month').mean('time')
    um = umld.sel(time=clim_slice).groupby('time.month').mean('time')
    vm = vmld.sel(time=clim_slice).groupby('time.month').mean('time')

    month = Tmld['time.month']
    Tm_full = Tm.sel(month=month).drop_vars('month')
    um_full = um.sel(month=month).drop_vars('month')
    vm_full = vm.sel(month=month).drop_vars('month')

    dTmdx, dTmdy = _compute_gradients(Tm_full)

    up = umld - um_full
    vp = vmld - vm_full
    dTpdx = dTdx - dTmdx
    dTpdy = dTdy - dTmdy

    umdTmdx = um_full * dTmdx
    updTmdx = up * dTmdx
    umdTpdx = um_full * dTpdx
    updTpdx = up * dTpdx

    vmdTmdy = vm_full * dTmdy
    vpdTmdy = vp * dTmdy
    vmdTpdy = vm_full * dTpdy
    vpdTpdy = vp * dTpdy

    mnupdTpdx_clim = updTpdx.sel(time=clim_slice).groupby('time.month').mean('time')
    mnvpdTpdy_clim = vpdTpdy.sel(time=clim_slice).groupby('time.month').mean('time')
    mnupdTpdx = mnupdTpdx_clim.sel(month=month).drop_vars('month')
    mnvpdTpdy = mnvpdTpdy_clim.sel(month=month).drop_vars('month')

    return {
        'dTdt': compute_tendency(Tmld),
        'dTpdt': compute_anomaly_tendency(Tmld, yrclim),
        'umdTmdx': umdTmdx, 'updTmdx': updTmdx,
        'umdTpdx': umdTpdx, 'updTpdx': updTpdx,
        'vmdTmdy': vmdTmdy, 'vpdTmdy': vpdTmdy,
        'vmdTpdy': vmdTpdy, 'vpdTpdy': vpdTpdy,
        'mnupdTpdx': mnupdTpdx, 'mnvpdTpdy': mnvpdTpdy,
    }
```

- [ ] **Step 5.4: Extend `__init__.py`**

```python
from .advection import advection_ml_rd
```

Append `'advection_ml_rd',` to `__all__`.

- [ ] **Step 5.5: Run**

Run: `pytest apinter/tests/test_heat_budget_advection.py -v`
Expected: 2 PASSED.

- [ ] **Step 5.6: Commit**

```bash
git add apinter/heat_budget/advection.py apinter/heat_budget/__init__.py \
        apinter/tests/test_heat_budget_advection.py
git commit -m "dev: apinter.heat_budget — Reynolds-decomposed horizontal advection (regular grid)"
```

---

## Task 6: `entrainment.py` — continuity-w + Reynolds-decomposed vertical advection

**Files:**
- Create: `apinter/heat_budget/entrainment.py`
- Create: `apinter/tests/test_heat_budget_entrainment.py`
- Modify: `apinter/heat_budget/__init__.py`

- [ ] **Step 6.1: Write the failing test**

Create `apinter/tests/test_heat_budget_entrainment.py`:

```python
"""Tests for the regular-grid entrainment module."""
import numpy as np
import pandas as pd
import xarray as xr

from apinter.heat_budget.entrainment import (
    compute_w_from_continuity, vertadv_ml_rd,
)


def test_compute_w_from_continuity_nondivergent_flow_gives_zero_w():
    """For a purely zonal, lon-independent flow: du/dx = 0, dv/dy = 0 → w = 0."""
    n_t, n_lev, n_lat, n_lon = 1, 4, 5, 8
    time = pd.date_range('2000-01-01', periods=n_t, freq='MS')
    lev = np.array([5.0, 15.0, 25.0, 40.0])
    lat = np.linspace(-10.0, 10.0, n_lat)
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    coords = {'time': time, 'lev': lev, 'lat': lat, 'lon': lon}
    dims = ['time', 'lev', 'lat', 'lon']
    u = xr.DataArray(np.ones((n_t, n_lev, n_lat, n_lon)), coords=coords, dims=dims)
    v = xr.DataArray(np.zeros((n_t, n_lev, n_lat, n_lon)), coords=coords, dims=dims)

    w = compute_w_from_continuity(u, v)
    mid = dict(lat=slice(1, -1), lon=slice(1, -1))
    assert np.allclose(w.isel(**mid).values, 0.0, atol=1e-15)


def test_vertadv_ml_rd_returns_expected_keys():
    n_t, n_lat, n_lon = 60, 3, 4
    time = pd.date_range('1981-01-01', periods=n_t, freq='MS')
    lat = np.array([-5.0, 0.0, 5.0])
    lon = np.array([100.0, 120.0, 140.0, 160.0])
    coords = {'time': time, 'lat': lat, 'lon': lon}
    dims = ['time', 'lat', 'lon']
    shape = (n_t, n_lat, n_lon)

    rng = np.random.default_rng(0)

    def _da(val):
        if np.isscalar(val):
            arr = np.full(shape, val)
        else:
            arr = val
        return xr.DataArray(arr, coords=coords, dims=dims)

    mld = _da(30.0 + 5.0 * np.sin(2 * np.pi * np.arange(n_t)[:, None, None] / 12)
              + np.broadcast_to(np.zeros((1, n_lat, n_lon)), shape))
    usub = _da(0.05 + rng.standard_normal(shape) * 0.01)
    vsub = _da(rng.standard_normal(shape) * 0.01)
    Tmld = _da(22.0 + rng.standard_normal(shape) * 0.2)
    Tsub = Tmld - _da(2.0)
    wsub = _da(rng.standard_normal(shape) * 1e-5)

    out = vertadv_ml_rd(mld, usub, vsub, Tmld, Tsub, wsub, yrclim=[1981, 1985])
    assert set(out) == {'w_entr', 'wmdTmdz', 'wpdTmdz', 'wmdTpdz', 'wpdTpdz',
                        'mnwpdTpdz'}
    # w_entr is a full-time field; others are too.
    for v in out.values():
        assert v.sizes['time'] == n_t


def test_vertadv_masks_invalid_mld():
    """Points with MLD <= 0 return NaN for all normalised terms."""
    n_t = 60
    time = pd.date_range('1981-01-01', periods=n_t, freq='MS')
    lat = np.array([0.0])
    lon = np.array([0.0])
    coords = {'time': time, 'lat': lat, 'lon': lon}
    dims = ['time', 'lat', 'lon']

    def _da(v):
        return xr.DataArray(np.full((n_t, 1, 1), v), coords=coords, dims=dims)

    mld = _da(0.0)   # invalid
    zero = _da(0.0)
    one = _da(0.1)
    out = vertadv_ml_rd(mld, zero, zero, _da(22.0), _da(20.0), zero,
                        yrclim=[1981, 1985])
    assert np.all(np.isnan(out['wmdTmdz'].values))
```

- [ ] **Step 6.2: Run to verify failure**

Run: `pytest apinter/tests/test_heat_budget_entrainment.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 6.3: Create `apinter/heat_budget/entrainment.py`**

Port `src/heat_budget/vertical_advection.py`:

```python
"""Reynolds-decomposed vertical advection (entrainment) for the ML heat budget.

Port of Matlab vertadv_ml_rd.m. Reference: Graham et al. (2014) Climate
Dynamics, 43:2399-2414.
"""
import numpy as np
import xarray as xr

from .advection import _compute_gradients
from .constants import RE, CLIM_START, CLIM_END, MLD_MIN
from .tendency import compute_tendency


def compute_w_from_continuity(uvel, vvel):
    """Derive w from incompressibility on a regular lat/lon grid.

    Integrates dw/dz = -(du/dx + dv/dy) downward from the surface (w=0),
    with w positive upward. Needed when w is not directly available
    (e.g., from CDS ORAS5 downloads).

    Parameters
    ----------
    uvel, vvel : xr.DataArray (time, lev, lat, lon) — m/s.

    Returns
    -------
    w : xr.DataArray (time, lev, lat, lon) at the bottom of each layer, m/s,
        positive upward.
    """
    lat_rad = np.deg2rad(uvel.lat)
    cos_lat = np.cos(lat_rad)

    dudx = uvel.differentiate('lon', edge_order=1) / (RE * cos_lat * np.deg2rad(1.0))
    v_cos = vvel * cos_lat
    dvdy = v_cos.differentiate('lat', edge_order=1) / (RE * cos_lat * np.deg2rad(1.0))

    divergence = dudx + dvdy

    z = uvel.lev.values
    layer_bot = np.zeros(len(z))
    for i in range(len(z)):
        layer_bot[i] = 2.0 * z[i] - (layer_bot[i - 1] if i > 0 else 0.0)
    thickness = np.diff(np.concatenate([[0.0], layer_bot]))
    thickness_da = xr.DataArray(thickness, dims=['lev'], coords={'lev': uvel.lev})

    return (-divergence * thickness_da).cumsum(dim='lev')


def vertadv_ml_rd(mld, usub, vsub, Tmld, Tsub, wsub, yrclim=None):
    """Reynolds-decomposed vertical advection (entrainment).

    Returns a dict with keys w_entr, wmdTmdz, wpdTmdz, wmdTpdz, wpdTpdz, mnwpdTpdz.
    The Heaviside step is applied to the climatological-mean entrainment
    velocity (w_m > 0 → entrainment active), following Graham (2014).
    """
    if yrclim is None:
        yrclim = [CLIM_START, CLIM_END]

    clim_slice = slice(str(yrclim[0]), str(yrclim[1]))

    mld_invalid = (mld <= 0) | np.isnan(mld)
    mld_safe = mld.clip(min=MLD_MIN)

    dHdt = compute_tendency(mld_safe)
    dHdx, dHdy = _compute_gradients(mld_safe)

    w_entr = dHdt + usub * dHdx + vsub * dHdy + wsub

    Tm_mld = Tmld.sel(time=clim_slice).groupby('time.month').mean('time')
    Tm_sub = Tsub.sel(time=clim_slice).groupby('time.month').mean('time')
    wm = w_entr.sel(time=clim_slice).groupby('time.month').mean('time')

    deltaTm = Tm_mld - Tm_sub

    month = Tmld['time.month']
    Tm_mld_full = Tm_mld.sel(month=month).drop_vars('month')
    Tm_sub_full = Tm_sub.sel(month=month).drop_vars('month')
    wm_full = wm.sel(month=month).drop_vars('month')
    deltaTm_full = deltaTm.sel(month=month).drop_vars('month')

    wsgn = xr.where(wm > 0, 1.0, 0.0)
    wsgn_full = wsgn.sel(month=month).drop_vars('month')

    wp = w_entr - wm_full
    Tp_mld = Tmld - Tm_mld_full
    Tp_sub = Tsub - Tm_sub_full
    deltaTp = Tp_mld - Tp_sub

    wmdTmdz = wsgn_full * wm_full * deltaTm_full / mld_safe
    wpdTmdz = wsgn_full * wp * deltaTm_full / mld_safe
    wmdTpdz = wsgn_full * wm_full * deltaTp / mld_safe
    wpdTpdz = wsgn_full * wp * deltaTp / mld_safe

    mnwpdTpdz_clim = wpdTpdz.sel(time=clim_slice).groupby('time.month').mean('time')
    mnwpdTpdz = mnwpdTpdz_clim.sel(month=month).drop_vars('month')

    wmdTmdz = xr.where(mld_invalid, np.nan, wmdTmdz)
    wpdTmdz = xr.where(mld_invalid, np.nan, wpdTmdz)
    wmdTpdz = xr.where(mld_invalid, np.nan, wmdTpdz)
    wpdTpdz = xr.where(mld_invalid, np.nan, wpdTpdz)
    mnwpdTpdz = xr.where(mld_invalid, np.nan, mnwpdTpdz)

    return {
        'w_entr': w_entr,
        'wmdTmdz': wmdTmdz, 'wpdTmdz': wpdTmdz,
        'wmdTpdz': wmdTpdz, 'wpdTpdz': wpdTpdz,
        'mnwpdTpdz': mnwpdTpdz,
    }
```

- [ ] **Step 6.4: Extend `__init__.py`**

```python
from .entrainment import compute_w_from_continuity, vertadv_ml_rd
```

Append `'compute_w_from_continuity', 'vertadv_ml_rd',` to `__all__`.

- [ ] **Step 6.5: Run**

Run: `pytest apinter/tests/test_heat_budget_entrainment.py -v`
Expected: 3 PASSED.

- [ ] **Step 6.6: Commit**

```bash
git add apinter/heat_budget/entrainment.py apinter/heat_budget/__init__.py \
        apinter/tests/test_heat_budget_entrainment.py
git commit -m "dev: apinter.heat_budget — entrainment + continuity-w (regular grid)"
```

---

## Task 7: `ml_tendency.py` — partial-cell ML mean temperature

**Files:**
- Create: `apinter/heat_budget/ml_tendency.py`
- Create: `apinter/tests/test_heat_budget_ml_tendency.py`
- Modify: `apinter/heat_budget/__init__.py`

- [ ] **Step 7.1: Write the failing test**

Create `apinter/tests/test_heat_budget_ml_tendency.py`:

```python
"""Tests for partial-cell ML mean temperature and tendency."""
import numpy as np
import pandas as pd
import xarray as xr

from apinter.heat_budget.ml_tendency import ml_mean_temperature, ml_tendency


def _make_temp(mld_val):
    """Uniform temperature = 20 °C → ⟨T⟩ should equal 20 regardless of MLD."""
    n_t, n_lev, n_lat, n_lon = 3, 5, 2, 2
    time = pd.date_range('2000-01-01', periods=n_t, freq='MS')
    lev = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
    lat = np.array([0.0, 5.0])
    lon = np.array([100.0, 110.0])
    temp = xr.DataArray(
        np.full((n_t, n_lev, n_lat, n_lon), 20.0),
        coords={'time': time, 'lev': lev, 'lat': lat, 'lon': lon},
        dims=['time', 'lev', 'lat', 'lon'],
    )
    mld = xr.DataArray(
        np.full((n_t, n_lat, n_lon), mld_val),
        coords={'time': time, 'lat': lat, 'lon': lon},
        dims=['time', 'lat', 'lon'],
    )
    return temp, mld


def test_ml_mean_temperature_uniform_field():
    temp, mld = _make_temp(mld_val=25.0)
    T_ml = ml_mean_temperature(temp, mld)
    assert np.allclose(T_ml.values, 20.0)


def test_ml_mean_temperature_partial_cell_interpolation():
    """Linear T(z) = 20 - 0.1 z → ⟨T⟩ over [0, H] should be 20 - 0.05·H."""
    n_t, n_lev, n_lat, n_lon = 1, 10, 1, 1
    time = pd.date_range('2000-01-01', periods=n_t, freq='MS')
    lev = np.arange(2.5, 50.0, 5.0)   # centres 2.5, 7.5, ..., 47.5
    T_profile = 20.0 - 0.1 * lev      # linear in depth
    temp = xr.DataArray(
        np.broadcast_to(T_profile[None, :, None, None],
                        (n_t, n_lev, n_lat, n_lon)).copy(),
        coords={'time': time, 'lev': lev,
                'lat': [0.0], 'lon': [0.0]},
        dims=['time', 'lev', 'lat', 'lon'],
    )
    H = 22.5   # falls inside the level whose centre is 22.5 (layer [20, 25])
    mld = xr.DataArray([[[H]]], coords={'time': time, 'lat': [0.0], 'lon': [0.0]},
                       dims=['time', 'lat', 'lon'])
    # Analytic integral (1/H) ∫₀^H (20 - 0.1 z) dz = 20 - 0.05 H
    expected = 20.0 - 0.05 * H
    T_ml = ml_mean_temperature(temp, mld)
    assert np.isclose(T_ml.values[0, 0, 0], expected, atol=0.02)


def test_ml_tendency_linear_time_series():
    """T_ml = t days → dT/dt = 1/86400 degC/s in the interior."""
    n_t = 12
    time = pd.date_range('2000-01-01', periods=n_t, freq='MS')
    t_days = np.arange(n_t) * 30.0   # 30-day months
    T_ml = xr.DataArray(
        np.broadcast_to(t_days[:, None, None], (n_t, 1, 1)).copy(),
        coords={'time': time, 'lat': [0.0], 'lon': [0.0]},
        dims=['time', 'lat', 'lon'],
    )
    dTdt = ml_tendency(T_ml, dt_seconds=30.0 * 86400.0)
    mid = dTdt.isel(time=slice(1, -1)).values
    # expected = Δ(30 days) / (2 × 30 × 86400 s) = 30 / 60 / 86400 = 1/(2·86400)
    expected = 30.0 / (2.0 * 30.0 * 86400.0)
    assert np.allclose(mid, expected)
```

- [ ] **Step 7.2: Run to verify failure**

Run: `pytest apinter/tests/test_heat_budget_ml_tendency.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 7.3: Create `apinter/heat_budget/ml_tendency.py`**

Port `src/heat_budget/ml_tendency.py` verbatim:

```python
"""Mixed-layer mean temperature with partial-cell interpolation.

Reference: Nnamchi et al. (2021, Nature Communications).
"""
import numpy as np
import xarray as xr


def _layer_bounds(z):
    """Layer top/bottom boundaries from level centres."""
    z = np.asarray(z, dtype=float)
    nlev = len(z)
    z_mid = 0.5 * (z[:-1] + z[1:])
    z_top = np.empty(nlev)
    z_bot = np.empty(nlev)
    z_top[0] = 0.0
    z_top[1:] = z_mid
    z_bot[:-1] = z_mid
    z_bot[-1] = 2.0 * z[-1] - z_mid[-1]
    return z_top, z_bot


def ml_mean_temperature(temp, mld, z=None):
    """⟨T⟩ = (1/h) ∫₀ʰ T dz with partial-cell interpolation at the MLD base."""
    if z is None:
        z_vals = temp.lev.values.astype(float)
    else:
        z_vals = np.asarray(z, dtype=float)

    z_top, z_bot = _layer_bounds(z_vals)

    z_top_da = xr.DataArray(z_top, dims=['lev'], coords={'lev': temp.lev})
    z_bot_da = xr.DataArray(z_bot, dims=['lev'], coords={'lev': temp.lev})

    eff_top = z_top_da
    eff_bot = xr.where(z_bot_da <= mld, z_bot_da, mld)
    dz_eff = (eff_bot - eff_top).clip(min=0.0)

    weighted_sum = (temp * dz_eff).sum(dim='lev')
    total_depth = dz_eff.sum(dim='lev')
    T_ml = xr.where(total_depth > 0, weighted_sum / total_depth, np.nan)
    T_ml.name = 'T_ml'
    return T_ml


def ml_tendency(T_ml, dt_seconds=None):
    """∂⟨T⟩/∂t via centered differencing on monthly data."""
    if dt_seconds is None:
        dt_seconds = 30.0 * 86400.0

    T_vals = T_ml.values
    dTdt_vals = np.full_like(T_vals, np.nan)
    dTdt_vals[1:-1] = (T_vals[2:] - T_vals[:-2]) / (2.0 * dt_seconds)
    dTdt_vals[0] = (T_vals[1] - T_vals[0]) / dt_seconds
    dTdt_vals[-1] = (T_vals[-1] - T_vals[-2]) / dt_seconds

    dTdt = xr.DataArray(dTdt_vals, dims=T_ml.dims, coords=T_ml.coords)
    dTdt.name = 'dTdt'
    return dTdt
```

- [ ] **Step 7.4: Extend `__init__.py`**

```python
from .ml_tendency import ml_mean_temperature, ml_tendency
```

Append `'ml_mean_temperature', 'ml_tendency',` to `__all__`.

- [ ] **Step 7.5: Run**

Run: `pytest apinter/tests/test_heat_budget_ml_tendency.py -v`
Expected: 3 PASSED.

- [ ] **Step 7.6: Commit**

```bash
git add apinter/heat_budget/ml_tendency.py apinter/heat_budget/__init__.py \
        apinter/tests/test_heat_budget_ml_tendency.py
git commit -m "dev: apinter.heat_budget — partial-cell ML mean temperature"
```

---

## Task 8: `nemo_grid.py` — NemoGrid class (ORCA C-grid)

**Files:**
- Create: `apinter/heat_budget/nemo_grid.py`
- Create: `apinter/tests/test_heat_budget_nemo.py` (first tests only — extended in later tasks)
- Modify: `apinter/heat_budget/__init__.py`

- [ ] **Step 8.1: Write the failing test**

Create `apinter/tests/test_heat_budget_nemo.py`:

```python
"""Smoke + unit tests for the NEMO-C-grid backend.

Every test auto-skips if ORAS5_MESH_PATH is not accessible (matches the
test_nersc_io.py pattern).
"""
import numpy as np
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
    # ORCA025 grid has ny ~ 1021, nx ~ 1442, nz = 75
    assert grid.nz == 75
    assert grid.ny > 800 and grid.nx > 1000


def test_u_to_T_conserves_a_uniform_field(grid):
    """Interpolating a constant U-point field to T-points should preserve the value."""
    u = xr.ones_like(grid.e1u)
    uT = grid.u_to_T(u)
    # Interior should be exactly 1; boundary roll can create edge artefacts.
    interior = uT.isel(y=slice(1, -1), x=slice(1, -1))
    assert np.all(interior.values == 1.0)


def test_div_h_of_zero_flow_is_zero(grid):
    u = xr.zeros_like(grid.e1u)
    v = xr.zeros_like(grid.e1v)
    div = grid.div_h(u, v)
    assert np.all(div.values == 0.0)
```

- [ ] **Step 8.2: Run to verify failure**

Run: `pytest apinter/tests/test_heat_budget_nemo.py -v`
Expected: FAIL (ImportError) if mesh is present; otherwise tests auto-skip (still green).

- [ ] **Step 8.3: Create `apinter/heat_budget/nemo_grid.py`**

Port `src/heat_budget/nemo_grid.py` verbatim with the default path pulled from `apinter.config`:

```python
"""NEMO ORCA025 C-grid operations for the mixed-layer heat budget.

Loads mesh_mask.nc and provides finite-difference operators on the native
Arakawa C-grid (T, U, V staggering).

Grid layout (Arakawa C)::

        +------V[j,i]------+
        |                   |
      U[j,i-1]   T[j,i]  U[j,i]
        |                   |
        +----V[j-1,i]------+

Reference: Madec, G. & the NEMO System Team (2019). NEMO ocean engine. Chapter 4.
"""
from pathlib import Path

import xarray as xr

from apinter.config import ORAS5_MESH_PATH


class NemoGrid:
    """Container for NEMO ORCA025 grid metrics, masks, and C-grid operators.

    Parameters
    ----------
    mesh_path : str or Path, optional
        Path to ``mesh_mask.nc``. Defaults to ``apinter.config.ORAS5_MESH_PATH``.
    """

    def __init__(self, mesh_path=None):
        if mesh_path is None:
            mesh_path = ORAS5_MESH_PATH
        self._load(Path(mesh_path))

    def _load(self, mesh_path):
        ds = xr.open_dataset(mesh_path)

        self.e1t = ds['e1t'].squeeze(drop=True)
        self.e2t = ds['e2t'].squeeze(drop=True)
        self.e1u = ds['e1u'].squeeze(drop=True)
        self.e2u = ds['e2u'].squeeze(drop=True)
        self.e1v = ds['e1v'].squeeze(drop=True)
        self.e2v = ds['e2v'].squeeze(drop=True)

        self.e3t_0 = ds['e3t_0'].squeeze(drop=True)
        self.e3w_0 = ds['e3w_0'].squeeze(drop=True)
        self.gdept_0 = ds['gdept_0'].squeeze(drop=True)
        self.gdepw_0 = ds['gdepw_0'].squeeze(drop=True)

        self.tmask = ds['tmask'].squeeze(drop=True)
        self.umask = ds['umask'].squeeze(drop=True)
        self.vmask = ds['vmask'].squeeze(drop=True)

        self.tmaskutil = ds['tmaskutil'].squeeze(drop=True)
        self.umaskutil = ds['umaskutil'].squeeze(drop=True)
        self.vmaskutil = ds['vmaskutil'].squeeze(drop=True)

        self.glamt = ds['glamt'].squeeze(drop=True)
        self.gphit = ds['gphit'].squeeze(drop=True)
        self.glamu = ds['glamu'].squeeze(drop=True)
        self.gphiu = ds['gphiu'].squeeze(drop=True)
        self.glamv = ds['glamv'].squeeze(drop=True)
        self.gphiv = ds['gphiv'].squeeze(drop=True)

        self.ny = ds.sizes['y']
        self.nx = ds.sizes['x']
        self.nz = ds.sizes['z']

        self.ff = ds['ff'].squeeze(drop=True)

        ds.close()

    # ---- gradient operators (T → flux-point) ----

    def grad_i(self, field):
        """Zonal gradient of a T-point field → U-point."""
        dfield = field.roll(x=-1, roll_coords=False) - field
        return dfield / self.e1u

    def grad_j(self, field):
        """Meridional gradient of a T-point field → V-point."""
        dfield = field.roll(y=-1, roll_coords=False) - field
        return dfield / self.e2v

    # ---- advection at T-point ----

    def u_dot_grad_i(self, u, scalar):
        """u·∂scalar/∂x at T-point, averaging east/west U-point fluxes."""
        flux_east = u * self.grad_i(scalar)
        flux_west = flux_east.roll(x=1, roll_coords=False)
        return 0.5 * (flux_east + flux_west)

    def v_dot_grad_j(self, v, scalar):
        """v·∂scalar/∂y at T-point, averaging north/south V-point fluxes."""
        flux_north = v * self.grad_j(scalar)
        flux_south = flux_north.roll(y=1, roll_coords=False)
        return 0.5 * (flux_north + flux_south)

    # ---- divergence at T-point (continuity) ----

    def div_h(self, u, v):
        """Horizontal divergence at T-points."""
        u_flux = u * self.e2u
        du = u_flux - u_flux.roll(x=1, roll_coords=False)
        v_flux = v * self.e1v
        dv = v_flux - v_flux.roll(y=1, roll_coords=False)
        area = self.e1t * self.e2t
        return (du + dv) / area

    # ---- T-point gradient of a 2D scalar ----

    def grad_i_at_T(self, field):
        grad_u = self.grad_i(field)
        return 0.5 * (grad_u + grad_u.roll(x=1, roll_coords=False))

    def grad_j_at_T(self, field):
        grad_v = self.grad_j(field)
        return 0.5 * (grad_v + grad_v.roll(y=1, roll_coords=False))

    # ---- U/V → T interpolation ----

    def u_to_T(self, u):
        return 0.5 * (u + u.roll(x=1, roll_coords=False))

    def v_to_T(self, v):
        return 0.5 * (v + v.roll(y=1, roll_coords=False))
```

- [ ] **Step 8.4: Extend `__init__.py`**

```python
from .nemo_grid import NemoGrid
```

Append `'NemoGrid',` to `__all__`.

- [ ] **Step 8.5: Run**

Run: `pytest apinter/tests/test_heat_budget_nemo.py -v`
Expected: PASSED on Perlmutter (mesh present); SKIPPED elsewhere.

- [ ] **Step 8.6: Commit**

```bash
git add apinter/heat_budget/nemo_grid.py apinter/heat_budget/__init__.py \
        apinter/tests/test_heat_budget_nemo.py
git commit -m "dev: apinter.heat_budget — NemoGrid C-grid operators"
```

---

## Task 9: `nemo_mld.py` — depth-weighted MLD utilities on NEMO grid

**Files:**
- Create: `apinter/heat_budget/nemo_mld.py`
- Modify: `apinter/tests/test_heat_budget_nemo.py`
- Modify: `apinter/heat_budget/__init__.py`

- [ ] **Step 9.1: Extend the NEMO test file**

Append to `apinter/tests/test_heat_budget_nemo.py`:

```python
def test_nemo_mldavg_depth_weighted_on_synthetic_profile(grid):
    """Uniform T = 15 → ⟨T⟩ = 15 regardless of MLD over grid.e3t_0."""
    from apinter.heat_budget import nemo_mldavg

    # One-time, one-column constant temperature. We mimic the full 3D layout
    # (time, z, y, x) but with ny=nx=1 for speed.
    import pandas as pd
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
    """Profile T(k) = k; first level with centre > MLD should be the returned value."""
    from apinter.heat_budget import nemo_submld
    import pandas as pd

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
    mld = xr.DataArray([[[100.0]]],  # somewhere in the middle of the ORCA column
                       coords={'time': time, 'y': [0], 'x': [0]},
                       dims=['time', 'y', 'x'])
    Tsub = nemo_submld(mld, T, e3t)
    # z_center = cumsum(e3t) - 0.5 e3t; first k where z_center > 100 m
    z_center = np.cumsum(grid.e3t_0.values) - 0.5 * grid.e3t_0.values
    expected_k = int(np.argmax(z_center > 100.0))
    assert np.isclose(Tsub.values[0, 0, 0], float(expected_k))
```

- [ ] **Step 9.2: Run to verify failure**

Run: `pytest apinter/tests/test_heat_budget_nemo.py -v`
Expected: new tests FAIL (ImportError for `nemo_mldavg`, `nemo_submld`) on Perlmutter; SKIPPED elsewhere.

- [ ] **Step 9.3: Create `apinter/heat_budget/nemo_mld.py`**

Port `src/heat_budget/mld_utils_nemo.py`:

```python
"""Depth-weighted MLD utilities on the NEMO ORCA C-grid.

Uses exact layer thicknesses e3t from mesh_mask.nc (not simple nanmean over
non-uniform levels). Reference: Stevenson et al. (2017) Climate Dynamics.
"""
import numpy as np
import xarray as xr


def mldavg_varytime(mld, field, e3t):
    """Depth-weighted average of a 3D field over the mixed layer.

    Parameters
    ----------
    mld : xr.DataArray (time, y, x) — metres.
    field : xr.DataArray (time, z, y, x).
    e3t : xr.DataArray (z,) — layer thicknesses in metres.
    """
    z_center = e3t.cumsum('z') - 0.5 * e3t
    mask = z_center <= mld
    weighted_sum = (field * e3t).where(mask, 0.0).sum(dim='z')
    total_weight = e3t.where(mask, 0.0).sum(dim='z')
    return xr.where(total_weight > 0, weighted_sum / total_weight, np.nan)


def submld_varytime(mld, field, e3t, search_type='first'):
    """Field value at the first level with centre depth > MLD."""
    z_center = e3t.cumsum('z') - 0.5 * e3t
    nz = field.sizes['z']
    idx_arr = xr.DataArray(np.arange(nz), dims=['z'], coords={'z': field.z})
    idx_broadcast = idx_arr.broadcast_like(field)

    condition = z_center > mld
    has_valid = condition.any(dim='z')

    if search_type == 'first':
        masked_idx = xr.where(condition, idx_broadcast, nz)
        target_idx = masked_idx.min(dim='z')
    elif search_type == 'last':
        masked_idx = xr.where(~condition, idx_broadcast, -1)
        target_idx = masked_idx.max(dim='z')
    else:
        raise ValueError("search_type must be 'first' or 'last'")

    target_idx = target_idx.clip(0, nz - 1).astype(int)
    result = field.isel(z=target_idx)
    return xr.where(has_valid, result, np.nan)
```

- [ ] **Step 9.4: Extend `__init__.py`**

Add aliases matching the source repo's naming:

```python
from .nemo_mld import (
    mldavg_varytime as nemo_mldavg,
    submld_varytime as nemo_submld,
)
```

Append `'nemo_mldavg', 'nemo_submld',` to `__all__`.

- [ ] **Step 9.5: Run**

Run: `pytest apinter/tests/test_heat_budget_nemo.py -v`
Expected: 5 PASSED on Perlmutter; SKIPPED elsewhere.

- [ ] **Step 9.6: Commit**

```bash
git add apinter/heat_budget/nemo_mld.py apinter/heat_budget/__init__.py \
        apinter/tests/test_heat_budget_nemo.py
git commit -m "dev: apinter.heat_budget — NEMO-grid MLD utilities"
```

---

## Task 10: `nemo_advection.py` — advection on NEMO C-grid

**Files:**
- Create: `apinter/heat_budget/nemo_advection.py`
- Modify: `apinter/heat_budget/__init__.py`
- (No new tests — covered by the existing NEMO smoke tests + Task 14's pipeline test.)

- [ ] **Step 10.1: Create `apinter/heat_budget/nemo_advection.py`**

Port `src/heat_budget/horizontal_advection_nemo.py` verbatim:

```python
"""Reynolds-decomposed horizontal advection on the NEMO ORCA C-grid.

Uses exact grid metrics (e1u, e2v, ...) from mesh_mask.nc instead of the
spherical-coordinate approximation used on the regular lat/lon grid.

Reference: Graham et al. (2014) Climate Dynamics, 43:2399-2414.
"""
import xarray as xr

from .constants import CLIM_START, CLIM_END
from .tendency import compute_tendency, compute_anomaly_tendency


def advection_ml_rd(Tmld, umld, vmld, grid, yrclim=None):
    """Reynolds-decomposed horizontal advection on the NEMO C-grid.

    Parameters
    ----------
    Tmld : xr.DataArray (time, y, x) — ML-averaged T at T-points (degC).
    umld : xr.DataArray (time, y, x) — ML-averaged zonal velocity at U-points (m/s).
    vmld : xr.DataArray (time, y, x) — ML-averaged meridional velocity at V-points (m/s).
    grid : NemoGrid
    yrclim : [start_year, end_year], optional.
    """
    if yrclim is None:
        yrclim = [CLIM_START, CLIM_END]

    clim_slice = slice(str(yrclim[0]), str(yrclim[1]))

    Tm = Tmld.sel(time=clim_slice).groupby('time.month').mean('time')
    um = umld.sel(time=clim_slice).groupby('time.month').mean('time')
    vm = vmld.sel(time=clim_slice).groupby('time.month').mean('time')

    month = Tmld['time.month']
    Tm_full = Tm.sel(month=month).drop_vars('month')
    um_full = um.sel(month=month).drop_vars('month')
    vm_full = vm.sel(month=month).drop_vars('month')

    umdTmdx = grid.u_dot_grad_i(um_full, Tm_full)
    vmdTmdy = grid.v_dot_grad_j(vm_full, Tm_full)

    up = umld - um_full
    vp = vmld - vm_full

    updTmdx = grid.u_dot_grad_i(up, Tm_full)
    vpdTmdy = grid.v_dot_grad_j(vp, Tm_full)

    umdTpdx = grid.u_dot_grad_i(um_full, Tmld) - umdTmdx
    vmdTpdy = grid.v_dot_grad_j(vm_full, Tmld) - vmdTmdy

    updTpdx = grid.u_dot_grad_i(up, Tmld) - updTmdx
    vpdTpdy = grid.v_dot_grad_j(vp, Tmld) - vpdTmdy

    mnupdTpdx_clim = updTpdx.sel(time=clim_slice).groupby('time.month').mean('time')
    mnvpdTpdy_clim = vpdTpdy.sel(time=clim_slice).groupby('time.month').mean('time')
    mnupdTpdx = mnupdTpdx_clim.sel(month=month).drop_vars('month')
    mnvpdTpdy = mnvpdTpdy_clim.sel(month=month).drop_vars('month')

    return {
        'dTdt': compute_tendency(Tmld),
        'dTpdt': compute_anomaly_tendency(Tmld, yrclim),
        'umdTmdx': umdTmdx, 'updTmdx': updTmdx,
        'umdTpdx': umdTpdx, 'updTpdx': updTpdx,
        'vmdTmdy': vmdTmdy, 'vpdTmdy': vpdTmdy,
        'vmdTpdy': vmdTpdy, 'vpdTpdy': vpdTpdy,
        'mnupdTpdx': mnupdTpdx, 'mnvpdTpdy': mnvpdTpdy,
    }
```

- [ ] **Step 10.2: Extend `__init__.py`**

```python
from .nemo_advection import advection_ml_rd as nemo_advection_ml_rd
```

Append `'nemo_advection_ml_rd',` to `__all__`.

- [ ] **Step 10.3: Run the existing heat-budget test suite to make sure nothing regressed**

Run: `pytest apinter/tests/test_heat_budget*.py -v`
Expected: all prior tests still PASS (NEMO tests SKIP off-Perlmutter; PASS on).

- [ ] **Step 10.4: Commit**

```bash
git add apinter/heat_budget/nemo_advection.py apinter/heat_budget/__init__.py
git commit -m "dev: apinter.heat_budget — Reynolds-decomposed advection on NEMO C-grid"
```

---

## Task 11: `nemo_entrainment.py` — entrainment on NEMO C-grid

**Files:**
- Create: `apinter/heat_budget/nemo_entrainment.py`
- Modify: `apinter/heat_budget/__init__.py`

- [ ] **Step 11.1: Create `apinter/heat_budget/nemo_entrainment.py`**

Port `src/heat_budget/vertical_advection_nemo.py` verbatim:

```python
"""Reynolds-decomposed vertical advection (entrainment) on the NEMO C-grid.

Convention: w positive upward; w_sub > 0 means upwelling pushes water into
the ML, i.e. entrainment. Heaviside step is applied to the climatological-mean
entrainment velocity.

Reference: Graham et al. (2014) Climate Dynamics, 43:2399-2414.
"""
import numpy as np
import xarray as xr

from .constants import CLIM_START, CLIM_END, MLD_MIN
from .tendency import compute_tendency


def compute_w_from_continuity(u, v, grid):
    """w at w-levels (bottom of each T-cell), positive upward.

    Integrates dw/dz = -div_h(u, v) downward from w=0 at the surface.
    """
    div = grid.div_h(u, v)
    nz = u.sizes['z']
    e3t = grid.e3t_0.isel(z=slice(0, nz))
    return (-div * e3t).cumsum(dim='z')


def submld_w(mld, w, e3t):
    """w at the MLD base, from w at w-levels (positive upward)."""
    z_bot = e3t.cumsum('z')
    nz = w.sizes['z']
    idx_arr = xr.DataArray(np.arange(nz), dims=['z'], coords={'z': w.z})
    idx_broadcast = idx_arr.broadcast_like(w)
    condition = z_bot >= mld
    has_valid = condition.any(dim='z')
    masked_idx = xr.where(condition, idx_broadcast, nz)
    target_idx = masked_idx.min(dim='z').clip(0, nz - 1).astype(int)
    result = w.isel(z=target_idx)
    return xr.where(has_valid, result, np.nan)


def vertadv_ml_rd(mld, umld, vmld, Tmld, Tsub, wsub, grid, yrclim=None):
    """Reynolds-decomposed entrainment on the NEMO C-grid.

    Parameters
    ----------
    mld : xr.DataArray (time, y, x) — MLD (m).
    umld, vmld : xr.DataArray (time, y, x) — ML-averaged velocities, already
        interpolated to T-points.
    Tmld, Tsub : xr.DataArray (time, y, x) — ML mean T and T just below ML.
    wsub : xr.DataArray (time, y, x) — w at MLD base, positive upward.
    grid : NemoGrid.
    yrclim : [start_year, end_year], optional.
    """
    if yrclim is None:
        yrclim = [CLIM_START, CLIM_END]

    clim_slice = slice(str(yrclim[0]), str(yrclim[1]))

    mld_invalid = (mld <= 0) | np.isnan(mld)
    mld_safe = mld.clip(min=MLD_MIN)

    dHdt = compute_tendency(mld_safe)
    dHdx = grid.grad_i_at_T(mld_safe)
    dHdy = grid.grad_j_at_T(mld_safe)

    w_entr = dHdt + umld * dHdx + vmld * dHdy + wsub

    Tm_mld = Tmld.sel(time=clim_slice).groupby('time.month').mean('time')
    Tm_sub = Tsub.sel(time=clim_slice).groupby('time.month').mean('time')
    wm = w_entr.sel(time=clim_slice).groupby('time.month').mean('time')

    deltaTm = Tm_mld - Tm_sub
    wsgn = xr.where(wm > 0, 1.0, 0.0)

    month = Tmld['time.month']
    Tm_mld_full = Tm_mld.sel(month=month).drop_vars('month')
    Tm_sub_full = Tm_sub.sel(month=month).drop_vars('month')
    wm_full = wm.sel(month=month).drop_vars('month')
    wsgn_full = wsgn.sel(month=month).drop_vars('month')
    deltaTm_full = deltaTm.sel(month=month).drop_vars('month')

    wp = w_entr - wm_full
    Tp_mld = Tmld - Tm_mld_full
    Tp_sub = Tsub - Tm_sub_full
    deltaTp = Tp_mld - Tp_sub

    wmdTmdz = wsgn_full * wm_full * deltaTm_full / mld_safe
    wpdTmdz = wsgn_full * wp * deltaTm_full / mld_safe
    wmdTpdz = wsgn_full * wm_full * deltaTp / mld_safe
    wpdTpdz = wsgn_full * wp * deltaTp / mld_safe

    mnwpdTpdz_clim = wpdTpdz.sel(time=clim_slice).groupby('time.month').mean('time')
    mnwpdTpdz = mnwpdTpdz_clim.sel(month=month).drop_vars('month')

    wmdTmdz = xr.where(mld_invalid, np.nan, wmdTmdz)
    wpdTmdz = xr.where(mld_invalid, np.nan, wpdTmdz)
    wmdTpdz = xr.where(mld_invalid, np.nan, wmdTpdz)
    wpdTpdz = xr.where(mld_invalid, np.nan, wpdTpdz)
    mnwpdTpdz = xr.where(mld_invalid, np.nan, mnwpdTpdz)

    return {
        'w_entr': w_entr,
        'wmdTmdz': wmdTmdz, 'wpdTmdz': wpdTmdz,
        'wmdTpdz': wmdTpdz, 'wpdTpdz': wpdTpdz,
        'mnwpdTpdz': mnwpdTpdz,
    }
```

- [ ] **Step 11.2: Extend `__init__.py`**

```python
from .nemo_entrainment import (
    vertadv_ml_rd as nemo_vertadv_ml_rd,
    compute_w_from_continuity as nemo_compute_w,
    submld_w as nemo_submld_w,
)
```

Append to `__all__`:

```python
    'nemo_vertadv_ml_rd', 'nemo_compute_w', 'nemo_submld_w',
```

- [ ] **Step 11.3: Run the suite**

Run: `pytest apinter/tests/test_heat_budget*.py -v`
Expected: all prior PASS; NEMO tests SKIP off-Perlmutter.

- [ ] **Step 11.4: Commit**

```bash
git add apinter/heat_budget/nemo_entrainment.py apinter/heat_budget/__init__.py
git commit -m "dev: apinter.heat_budget — NEMO-grid entrainment + continuity-w"
```

---

## Task 12: `load_era5_flux` + `cfgrib` optional extra + pyproject

**Files:**
- Modify: `apinter/io/era5.py`
- Modify: `pyproject.toml`
- Modify: `apinter/io/__init__.py` (re-export)

- [ ] **Step 12.1: Add optional extra to `pyproject.toml`**

Edit `pyproject.toml`. Inside `[project.optional-dependencies]` (currently has `circulation`, `regrid`, `dev`), add:

```toml
heat_budget_io = ["cfgrib>=0.9"]
```

- [ ] **Step 12.2: Check the existing `apinter.io` __init__ to see where to re-export**

Run: `cat apinter/io/__init__.py`. Note the existing exports; we'll append `load_era5_flux`.

- [ ] **Step 12.3: Add `load_era5_flux` to `apinter/io/era5.py`**

Append this function at the bottom of the file (after `load_era5`):

```python
# ---------------------------------------------------------------------------
# Surface heat flux GRIB → (qnet, qsw) in W/m²
# ---------------------------------------------------------------------------

def load_era5_flux(flux_file: Optional[Union[str, Path]] = None,
                   sim_time: Optional[slice] = None,
                   ) -> tuple[xr.DataArray, xr.DataArray]:
    """Load ERA5 surface-flux GRIB and return (qnet, qsw) in W/m².

    ERA5 monthly-mean GRIB files store fluxes as J/m² accumulated over the
    month. This function converts to W/m² by dividing by the number of
    seconds in each calendar month, then returns:
        qnet = ssr + str + slhf + sshf   (net surface heat flux, into ocean)
        qsw  = ssr                        (net shortwave at the surface)

    Parameters
    ----------
    flux_file : Path, optional
        Default: ``ERA5_DIR / 'era5_q_flux.grib'``.
    sim_time : slice, optional
        Passed to ``.sel(time=...)`` after coordinate normalization.

    Returns
    -------
    (qnet, qsw) : tuple of xr.DataArray (time, lat, lon).

    Notes
    -----
    Requires the ``cfgrib`` optional dependency:
        ``pip install 'apinter[heat_budget_io]'``
    """
    import calendar

    if flux_file is None:
        flux_file = ERA5_DIR / 'era5_q_flux.grib'

    ds = xr.open_dataset(flux_file, engine='cfgrib')

    rename: Dict[str, str] = {}
    if 'latitude' in ds.dims:
        rename['latitude'] = 'lat'
    if 'longitude' in ds.dims:
        rename['longitude'] = 'lon'
    if 'valid_time' in ds.dims:
        rename['valid_time'] = 'time'
    if rename:
        ds = ds.rename(rename)

    seconds_per_month = xr.DataArray(
        [calendar.monthrange(t.dt.year.item(), t.dt.month.item())[1] * 86400.0
         for t in ds.time],
        dims=['time'], coords={'time': ds.time},
    )

    ssr = ds['ssr'] / seconds_per_month
    str_ = ds['str'] / seconds_per_month
    slhf = ds['slhf'] / seconds_per_month
    sshf = ds['sshf'] / seconds_per_month

    qnet = ssr + str_ + slhf + sshf
    qsw = ssr

    if sim_time is not None:
        qnet = qnet.sel(time=sim_time)
        qsw = qsw.sel(time=sim_time)

    qnet.name = 'qnet'
    qsw.name = 'qsw'
    return qnet, qsw
```

- [ ] **Step 12.4: Re-export from `apinter/io/__init__.py`**

Add `load_era5_flux` to the existing `from .era5 import ...` line (and to `__all__` if one exists).

- [ ] **Step 12.5: Run the existing era5 tests**

Run: `pytest apinter/tests -k era5 -v`
Expected: existing era5 tests still PASS (we added a new function; nothing broken).

- [ ] **Step 12.6: Commit**

```bash
git add apinter/io/era5.py apinter/io/__init__.py pyproject.toml
git commit -m "dev: apinter.io.era5.load_era5_flux + heat_budget_io optional extra"
```

---

## Task 13: Pipeline smoke test (end-to-end ORAS5)

**Files:**
- Create: `apinter/tests/test_heat_budget_pipeline.py`

- [ ] **Step 13.1: Write the smoke test**

Create `apinter/tests/test_heat_budget_pipeline.py`:

```python
"""End-to-end ORAS5 smoke test for apinter.heat_budget.

Auto-skips if ORAS5 or mesh_mask.nc are not accessible (matches the
test_nersc_io.py auto-skip pattern). Loads a tiny 2-year slab, computes a
handful of budget terms on the NEMO C-grid backend, and verifies the
resulting DataArrays have the expected shape and are mostly finite.
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
    a two-year ORAS5 slab. The assertion is structural (shape, finite fraction)
    — physics validation belongs in paper-specific budget-closure scripts."""
    from apinter.io import load_oras5
    from apinter.heat_budget import (
        NemoGrid,
        nemo_mldavg, nemo_submld,
        nemo_advection_ml_rd, nemo_compute_w, nemo_submld_w,
        nemo_vertadv_ml_rd,
    )

    grid = NemoGrid()

    sim = slice('2000-01-01', '2001-12-31')
    thetao = load_oras5('thetao', sim_time=sim)
    uo = load_oras5('uo', sim_time=sim)
    vo = load_oras5('vo', sim_time=sim)
    mld = load_oras5('mld030', sim_time=sim)

    # 3D → ML-averaged. ORAS5 uses 'deptht' for 3D T; need to align to grid.e3t_0.
    # Rename the depth dim to 'z' and the time dim to match expectations.
    def _to_zyx(da):
        rename = {}
        if 'deptht' in da.dims:
            rename['deptht'] = 'z'
        if 'depthu' in da.dims:
            rename['depthu'] = 'z'
        if 'depthv' in da.dims:
            rename['depthv'] = 'z'
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

    # Structural checks
    assert Tmld.sizes['time'] == thetao.sizes['time']
    assert Tmld.sizes['y'] == grid.ny
    assert Tmld.sizes['x'] == grid.nx

    # Ocean fraction should be mostly finite (>20% of the globe is ocean).
    finite_frac = float(np.isfinite(Tmld.isel(time=0).values).mean())
    assert finite_frac > 0.2, f"{finite_frac:.2%} finite — expected majority ocean"

    # One advection call on a tiny 2-year climatology window
    adv = nemo_advection_ml_rd(
        Tmld, umld_U, vmld_V, grid, yrclim=[2000, 2001],
    )
    assert set(adv) == {
        'dTdt', 'dTpdt',
        'umdTmdx', 'updTmdx', 'umdTpdx', 'updTpdx',
        'vmdTmdy', 'vpdTmdy', 'vmdTpdy', 'vpdTpdy',
        'mnupdTpdx', 'mnvpdTpdy',
    }

    # Continuity-w + submld_w + one entrainment call
    w3d = nemo_compute_w(uo_z, vo_z, grid)
    wsub = nemo_submld_w(mld, w3d, e3t)
    umld_T = grid.u_to_T(umld_U)
    vmld_T = grid.v_to_T(vmld_V)

    ent = nemo_vertadv_ml_rd(
        mld, umld_T, vmld_T, Tmld, Tsub, wsub, grid, yrclim=[2000, 2001],
    )
    assert set(ent) == {'w_entr', 'wmdTmdz', 'wpdTmdz', 'wmdTpdz', 'wpdTpdz',
                        'mnwpdTpdz'}
```

- [ ] **Step 13.2: Run**

Run: `pytest apinter/tests/test_heat_budget_pipeline.py -v`
Expected: PASS on Perlmutter; SKIPPED elsewhere.

- [ ] **Step 13.3: Commit**

```bash
git add apinter/tests/test_heat_budget_pipeline.py
git commit -m "dev: apinter.heat_budget — end-to-end ORAS5 pipeline smoke test"
```

---

## Task 14: Full-suite regression check + DEVLOG entry

**Files:**
- Modify: `DEVLOG.md`

- [ ] **Step 14.1: Run the full regular suite**

Run: `pytest --pyargs apinter.tests -v`
Expected: all new `test_heat_budget_*` tests PASS (26 new tests). Previous tests remain untouched. NEMO tests SKIP off-Perlmutter. The suite should stay under ~2 minutes.

If any previous test fails, **stop** and investigate — we should not have touched those code paths.

- [ ] **Step 14.2: Append a 2026-04-24 entry to `DEVLOG.md`**

Insert immediately after the header and before the existing `## 2026-04-23 — NERSC central-mirror access + xesmf regrid + ocean coverage` section:

```markdown
## 2026-04-24 — apinter.heat_budget subpackage (MLD heat budget, dual backend)

### Added

- **`apinter.heat_budget`** — new subpackage porting the mixed-layer heat
  budget code from the `Atlantic-Pacific-Interaction-MHW` project. Two
  backends:
  - **Regular lat/lon grid** (approximate spherical operators) for any
    regridded field — `mldavg_varytime`, `submld_varytime`, `botmld_varytime`,
    `advection_ml_rd`, `vertadv_ml_rd`, `compute_w_from_continuity`,
    `surface_heat_flux`, `ml_mean_temperature`, `ml_tendency`,
    `compute_tendency`, `compute_anomaly_tendency`.
  - **Native NEMO ORCA C-grid** (exact metrics from `mesh_mask.nc`) for
    ORAS5 — `NemoGrid`, `nemo_mldavg`, `nemo_submld`,
    `nemo_advection_ml_rd`, `nemo_vertadv_ml_rd`, `nemo_compute_w`,
    `nemo_submld_w`. Implements Reynolds decomposition following
    Graham et al. (2014) and depth-weighted MLD averaging following
    Stevenson et al. (2017).
  - **Diffusion module is intentionally skipped** — the upstream
    `diffusion.py` (horizontal Laplacian + Pacanowski-Philander vertical
    diffusion) was not verified and is not part of this port.
- **`apinter.io.era5.load_era5_flux`** — reads an ERA5 surface-flux GRIB
  and returns `(qnet, qsw)` in W/m², converting monthly-accumulated
  J/m² via each month's calendar day count.
- **`apinter.config.ORAS5_MESH_PATH`** — default path for NEMO grid
  metrics, used by `NemoGrid()`. Explicit override remains supported.

### Tests added (`apinter/tests/`)

| File | Tests | Notes |
|---|---:|---|
| `test_heat_budget_constants.py` | 3 | Physical constants in expected ranges; config mesh path exposed. |
| `test_heat_budget_tendency.py` | 4 | Central difference on `time` and `time_counter`; anomaly removes seasonal cycle. |
| `test_heat_budget_mld.py` | 5 | Weighted avg, sub-/bot-MLD sampling, NaN outside ML, `search_type` validation. |
| `test_heat_budget_advection.py` | 2 | Returns all 12 keys; Reynolds terms sum to `u·∂T/∂x`. |
| `test_heat_budget_entrainment.py` | 3 | Non-divergent flow → w = 0; returns 6 keys; masks invalid MLD. |
| `test_heat_budget_surface_flux.py` | 2 | No SW → `qnet/(ρcpH)`; SW-penetration matches Paulson–Simpson. |
| `test_heat_budget_ml_tendency.py` | 3 | Uniform T, linear T(z), monthly centered tendency. |
| `test_heat_budget_nemo.py` | 5 | `NemoGrid` metric load + operators + MLD utilities on the mesh (auto-skip). |
| `test_heat_budget_pipeline.py` | 1 | End-to-end ORAS5 → ML-averaged T + advection + entrainment (auto-skip). |

28 new tests total; synthetic tests run in a few hundred milliseconds.
NEMO and pipeline smoke tests auto-skip when
`apinter.config.ORAS5_DIR` / `ORAS5_MESH_PATH` are not accessible.

### Packaging

- **`pyproject.toml`** — adds `heat_budget_io = ["cfgrib>=0.9"]` optional
  extra. The subpackage's core math adds no new required dependencies.

### Workflow sketch — ORAS5 native grid (C-grid backend)

```python
from apinter.io import load_oras5, load_era5_flux
from apinter.heat_budget import (
    NemoGrid,
    nemo_mldavg, nemo_submld,
    nemo_advection_ml_rd, nemo_vertadv_ml_rd,
    nemo_compute_w, nemo_submld_w,
)

grid = NemoGrid()                                  # default mesh_mask path
thetao = load_oras5('thetao')
uo, vo = load_oras5('uo'), load_oras5('vo')
mld    = load_oras5('mld030')
qnet, qsw = load_era5_flux()

Tmld  = nemo_mldavg(mld, thetao, grid.e3t_0)
Tsub  = nemo_submld(mld, thetao, grid.e3t_0)
umld_U = nemo_mldavg(mld, uo, grid.e3t_0)          # at U-points
vmld_V = nemo_mldavg(mld, vo, grid.e3t_0)
umld_T = grid.u_to_T(umld_U)                       # interpolate to T
vmld_T = grid.v_to_T(vmld_V)
w3d   = nemo_compute_w(uo, vo, grid)
wsub  = nemo_submld_w(mld, w3d, grid.e3t_0)

adv = nemo_advection_ml_rd(Tmld, umld_U, vmld_V, grid)
ent = nemo_vertadv_ml_rd(mld, umld_T, vmld_T, Tmld, Tsub, wsub, grid)
```
```

- [ ] **Step 14.3: Commit**

```bash
git add DEVLOG.md
git commit -m "dev: DEVLOG entry for apinter.heat_budget subpackage"
```

- [ ] **Step 14.4: Final sanity check**

Run: `pytest --pyargs apinter.tests` (no `-v`) to confirm the final pass count. Also run a quick import smoke:

```bash
python -c "from apinter.heat_budget import (
    RHO, CP, RE, SW_R, SW_D1, SW_D2, MLD_MIN, CLIM_START, CLIM_END,
    mldavg_varytime, submld_varytime, botmld_varytime,
    advection_ml_rd, vertadv_ml_rd, compute_w_from_continuity,
    surface_heat_flux, compute_tendency, compute_anomaly_tendency,
    ml_mean_temperature, ml_tendency,
    NemoGrid,
    nemo_mldavg, nemo_submld,
    nemo_advection_ml_rd, nemo_vertadv_ml_rd,
    nemo_compute_w, nemo_submld_w,
); print('OK')"
```

Expected output: `OK`.

---

## Self-review checklist (run before handoff)

1. **Spec coverage** — every section of the spec has a task:
   - Package layout (12 files) → Tasks 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 (each creates a module and extends `__init__.py`).
   - Config additions → Task 1.
   - I/O policy (`load_era5_flux` + optional extra) → Task 12.
   - Public API flat re-export → Tasks 1-11 extend `__init__.py` incrementally.
   - References block → Task 1 (initial) + Tasks 2-11 extend imports; references are already in the Task 1 docstring.
   - Testing strategy (26 tests) → Tasks 1-7 (synthetic), 8-9 + 13 (NEMO + pipeline smoke with auto-skip). Count in the DEVLOG is 28 (includes 3 for constants) — matches the tests actually written.
   - DEVLOG entry → Task 14.
   - `diffusion.py` intentionally skipped → omitted from all tasks; called out in DEVLOG.

2. **Placeholder scan** — no "TBD", no "implement later", no "add appropriate error handling", no "similar to Task N without code". Every code-change step shows the code in full.

3. **Type/name consistency:**
   - `compute_tendency` / `compute_anomaly_tendency` — same signature in Tasks 2, 5, 6, 10, 11.
   - `_compute_gradients` — private helper in `advection.py` (Task 5); imported by `entrainment.py` (Task 6).
   - `NemoGrid` attributes (`e1u`, `e2v`, `e3t_0`, …) — consistent across Tasks 8, 9, 10, 11.
   - Public aliases (`nemo_mldavg`, `nemo_submld`, `nemo_advection_ml_rd`, `nemo_vertadv_ml_rd`, `nemo_compute_w`, `nemo_submld_w`) — every name registered in `__all__` when it's first introduced.

4. **Ordering:** tasks 2→6 have real dependencies (`tendency` → `advection` → `entrainment`). Tasks 8→11 depend on `NemoGrid`. Tasks 12-14 finalize.

No issues found in the self-review.
