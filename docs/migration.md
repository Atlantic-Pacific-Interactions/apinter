# Migrating notebooks from legacy `sys.path.append` imports to `apinter`

This cheat sheet lists every legacy import encountered in `Paper_1/`, `Paper_2/`,
`SSP/`, and `E3SMS/` notebooks/scripts, and its new `apinter.*` equivalent.
Phase 5 of the consolidation will apply these mechanically, folder by folder.

Until phase 5, legacy imports from `Paper_1/src`, `Paper_2/src`, `./src`,
and `SSP/src` still work via `sys.path.append`. New code should import from
`apinter` directly.

---

## Remove the sys.path hacks

**Legacy:**

```python
sys.path.append('/pscratch/sd/y/yanxia/Midlat-Atlantic-Pacific-Interactions/Paper_2/src')
sys.path.append('/pscratch/sd/y/yanxia/Midlat-Atlantic-Pacific-Interactions/Paper_1/src')

from load_cmip6 import load_cmip6_sst, load_cmip6_omega
from data_processing import calculate_anomalies_and_filter, lanczos_lowpass
from statistical_utils import calculate_lead_lag_regression
```

**New:**

```python
from apinter.io import load_cmip6                    # (phase 2)
from apinter.processing import apply_lowpass, lanczos_lowpass
from apinter.indices import gridded_anomalies
from apinter.stats import regression_lags
```

---

## Import map — phase 1 (available now)

| Legacy location | Legacy name | New location |
|---|---|---|
| `Paper_1/src/data_processing` | `lanczos_lowpass` | `apinter.processing.lanczos_lowpass` |
| `Paper_1/src/data_processing` | `calculate_anomalies_and_filter` | `apinter.indices.gridded_anomalies` (or compose `compute_anomaly` + `apply_lowpass`) |
| `Paper_1/src/data_processing` | `wgt_areaave` | `apinter.processing.wgt_areaave` |
| `Paper_1/src/linear_regression_correlation` | `calculate_partial_lead_lag_regression` | `apinter.stats.regression_lags(field, ts1, confounder=ts2, lags=...)` |
| `Paper_1/src/statistical_utils` | `calculate_lead_lag_regression` | `apinter.stats.regression_lags(field, index, lags=...)` |
| `Paper_1/src/statistical_utils` | `autocorrelation_function` | `apinter.stats.autocorrelation_function` |
| `Paper_1/src/statistical_utils` | `effective_degrees_of_freedom` | `apinter.stats.effective_degrees_of_freedom` |
| `./src/cal_index` | `detrend_dim` | `apinter.processing.detrend_dim` |
| `./src/cal_index` | `compute_anomalies` | `apinter.processing.compute_anomaly(..., complete_years_only=True)` for exact parity; default drops the tuple return |
| `./src/cal_index` | `calculate_index` | `apinter.indices.calculate_index` (new canonical: Lanczos + standardize, returns single DataArray) |
| `./src/cal_index` | `calculate_multiple_indices` | `apinter.indices.calculate_multiple_indices` |
| `./src/cal_index` | `calculate_gridded_anomalies` | `apinter.indices.gridded_anomalies` |
| `./src/cal_index` | `extract_region` | `apinter.processing.wgt_areaave` (drop the tuple signature) |
| `./src/linear_trend` | `global_mean_trend`, `spatial_trend`, `zonal_mean_trend`, `running_trend`, `assign_season`, `seasonal_trend`, `all_seasonal_trends` | `apinter.stats.<same name>` |
| `./src/calculate_regression_vectorize` | `calculate_regression_vectorize` | `apinter.stats.regression_lags(field, index, lags=[0])` (r_value via `xr.corr`) |

---

## Breaking changes worth calling out

### `calculate_index` return type

**Legacy** (`./src/cal_index.calculate_index`) returned `(anomaly, normalized_anomaly)` tuple with running-mean smoothing.

**New** (`apinter.indices.calculate_index`) returns a single `xr.DataArray` — the Paper_1 canonical Lanczos-filtered, standardized index.

```python
# Legacy
anom, norm = calculate_index(sst, (280, 340), (0, 30))

# New (usually what you want)
index = calculate_index(sst, (280, 340), (0, 30))   # already standardized
```

### Coordinate convention

All `apinter` functions expect `lat`/`lon` coordinate names (short form). If
loading raw ERA5 or CMIP6 data with `latitude`/`longitude`, rename first:

```python
da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
```

After phase 2, `apinter.io.*` loaders do this automatically.

### `regression_lags` output

Replaces three old functions with different output shapes. Always returns an
`xr.Dataset`:

- **No confounder:** `beta`, `p_value` along `(lag, *spatial)`.
- **With confounder (partial):** `target_beta`, `confounder_beta`, `target_pval`,
  `confounder_pval`.

See `docs/usage.md` for examples.

### Hardcoded time periods and model lists removed

- `calculate_correlation` (legacy) hardcoded `slice('1955', '2009')`. The new
  `correlation_lags(ts1, ts2, max_lag=12)` does no internal slicing — the caller
  pre-selects the time window.
- `calculate_multi_model_mean_correlation` (legacy) hardcoded
  `exclude = {'HadISST', 'ERSST', 'E3SM-MMF', 'E3SMv2'}`. The new
  `mmm_correlation_lags(per_model_results, exclude=...)` requires the caller to
  supply the exclude list.

---

## Phase 2+ import map (to be filled in as phases land)

| Legacy name | New location | Phase |
|---|---|---|
| `load_cmip6_sst`, `load_cmip6_omega`, `load_cmip6_pr`, `load_cmip6_zg`, `load_cmip6_psl`, `load_cmip6_zos`, `load_cmip6_wind`, `load_cmip6_thetao`, `load_cmip6_tauu` | `apinter.io.load_cmip6(var, ...)` or compat wrappers | 2 |
| `load_sst.read_data` | `apinter.io.load_obs_sst(source=...)` | 2 |
| `load_and_process_z200`, `load_and_process_omega_5s5n`, etc. (Paper_1) | `apinter.io.load_era5(var, level=, region=, process=...)` | 2 |
| `ssp_data_loading.load_and_concat_sst / _omega_5s5n / _precip` | `apinter.io.load_and_concat(var, ssp, model, ...)` | 2 |
| `E3SMS/path_config.py` constants | `apinter.config.*` | 2 |
| `calc_walker_sf`, `get_divergent_u`, `calc_streamfunction`, `compute_velpot`, `interp_to_common` | `apinter.circulation.*` | 3 |
| `psi_lietal` (Li et al. 2006) | `apinter.circulation.psi_phi` | 3 |
| `Paper_1/src/omega_reg_plotting` | `apinter.plotting.omega_regression` | 4 |
| `./src/plot/plot_index`, `plot_regression_map`, `plot_trend_map` | `apinter.plotting.*` | 4 |
