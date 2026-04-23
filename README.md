# apinter

Shared analysis library for Atlantic-Pacific climate interaction research.

Consolidates the previously scattered `src/` trees (`Paper_1/src`, `Paper_2/src`,
`SSP/src`, `./src`, `E3SMS/path_config.py`) into one installable Python
package used across all project folders.

## Install

From the repository root:

```bash
pip install -e .
```

Then from anywhere:

```python
from apinter.indices import calculate_index
from apinter.stats import regression_lags, correlation_lags
from apinter.processing import compute_anomaly, lanczos_lowpass, wgt_areaave
```

## Quick example

Compute a TAMV index from HadISST and regress global SST onto it at
several lead-lag months:

```python
import xarray as xr
from apinter.indices import calculate_index
from apinter.stats import regression_lags

sst = xr.open_dataset("/path/to/hadisst.nc")["sst"].sel(time=slice("1891", "2014"))

tamv = calculate_index(sst, lon_bounds=(280, 340), lat_bounds=(0, 30))
# default pipeline: anomaly -> detrend -> area-mean -> 11-yr Lanczos -> standardize

reg = regression_lags(
    field=sst,
    target_index=tamv,
    lags=[-60, -24, 0, 24, 60],     # months; [0] = concurrent
    compute_significance=True,       # Bretherton effective DoF t-test
)
# reg is xr.Dataset with dims (lag, lat, lon), vars (beta, p_value)
```

## Modules

| Subpackage | Purpose | Status |
|---|---|---|
| `apinter.processing` | detrend, anomaly, filters, area-weighted mean | ✅ phase 1 |
| `apinter.stats` | trends, regression, lead-lag correlation, significance | ✅ phase 1 |
| `apinter.indices` | climate index calculation (canonical Paper_1 pipeline) | ✅ phase 1 |
| `apinter.config` | paths, grid constants (CMIP6_DIR, ERA5_DIR, COMMON_PLEV, …) | ⬜ phase 2 |
| `apinter.io` | generic loaders for CMIP6, ERA5, obs SST, SSP, joblib | ⬜ phase 2 |
| `apinter.circulation` | Walker, Hadley, Helmholtz, velocity potential | ⬜ phase 3 |
| `apinter.plotting` | regression maps, index plots, trend maps, omega regression | ⬜ phase 4 |

## Documentation

- **[docs/usage.md](docs/usage.md)** — task-oriented recipes with code examples
- **[docs/migration.md](docs/migration.md)** — mapping from old `sys.path.append` imports to `apinter.*`
- **[design spec](docs/superpowers/specs/2026-04-23-apinter-shared-package-design.md)** — architecture, function inventory, phased plan

## Testing

```bash
pytest tests/
```

Phase-1 tests compare each new function to the legacy `Paper_1/src` and
`./src` implementations and hand-coded Paper_1 notebook pipelines.
