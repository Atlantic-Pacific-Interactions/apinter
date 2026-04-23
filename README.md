# apinter

Shared analysis library for Atlantic-Pacific climate interaction research.

Built on top of xarray, this package consolidates the data loaders, anomaly /
regression / Walker-Hadley diagnostics, and plotting helpers used across the
group's Paper_1, Paper_2, SSP, and E3SMS analyses into one installable,
tested library.

## Install

From GitHub (recommended):

```bash
pip install git+https://github.com/Atlantic-Pacific-Interactions/apinter.git
# or a frozen version
pip install git+https://github.com/Atlantic-Pacific-Interactions/apinter.git@v0.1.0
```

From a local clone (development):

```bash
git clone https://github.com/Atlantic-Pacific-Interactions/apinter.git
cd apinter
pip install -e '.[circulation,dev]'
```

Then from anywhere:

```python
from apinter.io import load_cmip6, load_obs_sst, load_era5
from apinter.indices import calculate_index
from apinter.stats import regression_lags, correlation_lags
from apinter.circulation import calc_walker_sf, calc_streamfunction
from apinter.processing import compute_anomaly, lanczos_lowpass, wgt_areaave
```

## Quick example

Compute a TAMV index from HadISST and regress global SST onto it at
several lead-lag months:

```python
from apinter.io import load_obs_sst
from apinter.indices import calculate_index
from apinter.stats import regression_lags

sst = load_obs_sst("hadisst", sim_time=slice("1891", "2014"))

tamv = calculate_index(sst, lon_bounds=(280, 340), lat_bounds=(0, 30))
# default pipeline: anomaly -> detrend -> area-mean -> 11-yr Lanczos -> standardize

reg = regression_lags(
    field=sst,
    target_index=tamv,
    lags=[-60, -24, 0, 24, 60],     # months; [0] = concurrent
    compute_significance=True,       # Bretherton effective-DoF t-test
)
# reg is xr.Dataset with dims (lag, lat, lon) and vars (beta, p_value)
```

## Modules

| Subpackage | Purpose |
|---|---|
| `apinter.config` | Paths and grid constants (CMIP6_DIR, ERA5_DIR, ORAS5_DIR, COMMON_PLEV, …) |
| `apinter.io` | Loaders for CMIP6, observational SST, ERA5, ORAS5, SSP; joblib I/O |
| `apinter.processing` | Detrending, monthly anomalies, Lanczos low-pass, area-weighted mean |
| `apinter.stats` | Linear trends, lead-lag regression, 1-D correlation, significance |
| `apinter.indices` | Canonical Paper_1 climate-index pipeline (TAMV, TPDV, etc.) |
| `apinter.circulation` | Walker / Hadley ψ, Helmholtz decomposition, velocity potential, Li-2006 ψ/φ solver |
| `apinter.plotting` | Index time series, regression / trend maps, Walker-Hadley panels, omega profiles |

## Documentation

- [docs/usage.md](docs/usage.md) — task-oriented recipes with runnable snippets
- [docs/migration.md](docs/migration.md) — map from legacy `sys.path.append` imports to `apinter.*`
- [design spec](docs/superpowers/specs/2026-04-23-apinter-shared-package-design.md) — architecture and phased implementation plan

## Testing

Tests ship with the package. Run them against an installed apinter:

```bash
pytest --pyargs apinter.tests
```

Or, from a local clone:

```bash
pytest apinter/tests/
```

Tests that compare apinter functions to the legacy `Paper_1/src` and
`./src` implementations auto-skip when those folders are absent (i.e.,
outside the `Midlat-Atlantic-Pacific-Interactions` mono-repo). Set
`APINTER_LEGACY_ROOT` if you have the mono-repo in a non-standard
location and want the parity tests to run.

## License

MIT — see [LICENSE](LICENSE).
