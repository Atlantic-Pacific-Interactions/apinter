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
from apinter.io import load_cmip6
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

## Phase 2 — I/O layer (available now)

### CMIP6 multi-model loaders

| Legacy (Paper_2/src/load_cmip6) | New canonical | Legacy shim kept? |
|---|---|---|
| `load_cmip6_sst()` | `load_cmip6('ts', ...)` | ✅ |
| `load_cmip6_omega()` | `load_cmip6('wap', ...)` | ✅ |
| `load_cmip6_zg()` | `load_cmip6('zg', ...)` | ✅ |
| `load_cmip6_psl()` | `load_cmip6('psl', ...)` | ✅ |
| `load_cmip6_pr()` | `load_cmip6('pr', ...)` | ✅ |
| `load_cmip6_zos()` | `load_cmip6('zos', ...)` | ✅ |
| `load_cmip6_thetao()` | `load_cmip6('thetao', ...)` | ✅ |
| `load_cmip6_tauu()` | `load_cmip6('tauu', ...)` | ✅ |
| `load_cmip6_wind(target_level=...)` | `load_cmip6('ua', level=...)` + `load_cmip6('va', level=...)` | ✅ (returns `{model:{'ua','va'}}`) |
| `get_cmip6_models()` | `apinter.io.get_cmip6_models()` | same name |

### Observational SST

| Legacy (`./src/load_sst.read_data`) | New canonical |
|---|---|
| `read_data(path, data_type=1)` (HadISST) | `load_obs_sst('hadisst', sim_time=...)` |
| `read_data(path, data_type=2)` (COBE) | `load_obs_sst('cobesst', sim_time=...)` |
| `read_data(path, data_type=3)` (ERSST) | `load_obs_sst('ersst', sim_time=...)` |

No separate paths to hardcode — the path lives in `apinter.config`.

### ERA5 reanalysis

| Legacy | New canonical |
|---|---|
| `Paper_1/data_processing.load_and_process_z200()` | `load_era5('z', level=200, ...)` + `gridded_anomalies(...)` |
| `Paper_1/data_processing.load_and_process_omega_5s5n()` | `load_era5('pl_omega', region={'lat':(-5,5)}, ...).mean('lat')` + `gridded_anomalies(...)` |
| `Paper_1/data_processing.load_and_process_obs_omega_5s5n()` | Same as above — Paper_1 had duplicated obs/model helpers; now one path. |
| `Paper_1/data_processing.load_and_process_obs_z200()` | `load_era5('z', level=200, ...)` + `gridded_anomalies(...)` |
| `Paper_1/data_processing.load_raw_omega_obs()` | `load_era5('pl_omega', ...)` |
| `Paper_1/data_processing.load_era5_pr()` | `load_era5('tp', ...)` |
| inline `xr.open_mfdataset('era5_u_component_*')` + rename | `load_era5('u', level=..., sim_time=..., region=...)` |
| inline `xr.open_mfdataset('era5_v_component_*')` + rename | `load_era5('v', level=..., ...)` |

All selections (`level`, `sim_time`, `region`) are pushed into each per-file
open *before* `.load()`, so small slices no longer read full multi-GB files.

### ORAS5 ocean reanalysis

| Legacy (inline in Paper_1 notebooks 36/42) | New canonical |
|---|---|
| `xr.open_mfdataset(...).rename({'time_counter':'time', '<var>':<friendly>})` + `assign_coords(time=pd.date_range('1958-01-01', ...))` | `load_oras5('<var>', sim_time=...)` |
| `groupby_bins(nav_lon_360, bins=lon_bins, labels=...)` equatorial regrid | `regrid_to_equatorial_lon(da, lon_bounds, lat_bounds, lon_step)` |
| Same, all in one step | `load_oras5_equatorial('<var>', ...)` |

### SSP historical + scenario concat

| Legacy (`SSP/src/ssp_data_loading`) | New canonical |
|---|---|
| `load_and_concat_sst(model, ssp, full_time)` | `load_and_concat('sst', ssp, model, full_time=...)` |
| `load_and_concat_omega_5s5n(model, ssp, full_time)` | `load_and_concat('wap', ssp, model, full_time=...)` (5S-5N mean baked into SSP_VARS spec) |
| `load_and_concat_precip(model, ssp, var='pr', full_time)` | `load_and_concat('pr', ssp, model, full_time=...)` |
| `get_ssp_models(ssp, variable='sst')` | `get_ssp_models(ssp, var='sst')` (param renamed) |

### Config / paths

| Legacy | New canonical |
|---|---|
| `E3SMS/path_config.py :: E3SM_MMF_PATH` | `apinter.config.E3SM_MMF_PATH` |
| `E3SMS/path_config.py :: E3SM_V2_PATHS` | `apinter.config.E3SM_V2_PATHS` |
| `E3SMS/path_config.py :: EXTRACTED_FILES` | `apinter.config.EXTRACTED_FILES` |
| Hardcoded `/pscratch/sd/y/yanxia/CMIP6` | `apinter.config.CMIP6_DIR` |
| Hardcoded `/pscratch/sd/y/yanxia/DATA/ERA5` | `apinter.config.ERA5_DIR` |
| Hardcoded `/pscratch/sd/y/yanxia/DATA/HadISST/...` | `apinter.config.HADISST_PATH` |
| Hardcoded `/pscratch/sd/y/yanxia/DATA/ERSST/...` | `apinter.config.ERSST_PATH` |
| Hardcoded `/pscratch/sd/y/yanxia/DATA/COBE/...` | `apinter.config.COBE_PATH` |
| Hardcoded `/pscratch/sd/y/yanxia/DATA/ORAS5` | `apinter.config.ORAS5_DIR` |
| Hardcoded land-mask path | `apinter.config.LSMSK_PATH` |

### joblib

| Legacy (`./src/data_loader`) | New canonical |
|---|---|
| `load_joblib(path)` | `apinter.io.load_joblib(path)` |
| `save_joblib(obj, path, compress=3)` | `apinter.io.save_joblib(obj, path, compress=3)` |

---

## Phase 2 breaking changes

### SST now filtered, not land-masked

`load_cmip6('ts', ...)` applies `(sst > -10) & (sst < 40)` °C rather than the
explicit land-sea mask. Rationale (from `E3SMS/calc_mean_state_sst_wind.py`):
this removes land, sea-ice surface temperatures, and 0-K regridding artifacts
in a single step, matching the existing Paper_1 / E3SMS convention.

### Coordinate convention enforced

Every loader returns `lat` / `lon` (short form). If code downstream still
expects `latitude` / `longitude`, rename back explicitly:

```python
da = da.rename({'lat': 'latitude', 'lon': 'longitude'})
```

### ERA5 time coord always `time`

ERA5 files use `valid_time` on some vars and `time` on others; `load_era5`
always returns `time`. Downstream `.sel(valid_time=...)` must be rewritten as
`.sel(time=...)`.

### `tp` ensemble dim stripped

`load_era5('tp', ...)` auto-selects `number=0` and drops the `number` dim.

---

## Phase 3+ import map (to be filled in as phases land)

| Legacy name | New location | Phase |
|---|---|---|
| `calc_walker_sf`, `get_divergent_u`, `calc_streamfunction`, `compute_velpot`, `interp_to_common` | `apinter.circulation.*` | 3 |
| `psi_lietal` (Li et al. 2006) | `apinter.circulation.psi_phi` | 3 |
| `Paper_1/src/omega_reg_plotting` | `apinter.plotting.omega_regression` | 4 |
| `./src/plot/plot_index`, `plot_regression_map`, `plot_trend_map` | `apinter.plotting.*` | 4 |
