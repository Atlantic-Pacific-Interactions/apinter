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

## Phase 3 — circulation (available now)

All Walker/Hadley primitives were extracted verbatim from the E3SMS `_v2`
scripts and now live in `apinter.circulation`. The scripts themselves (that
combined loading + science + plotting) will be deleted in phase 5; the
primitives below are all that's portable.

### Helmholtz decomposition

| Legacy (defined inline in many E3SMS `plot_*.py` files) | New canonical |
|---|---|
| `get_divergent_u(u_clim, v_clim)` | `apinter.circulation.get_divergent_u` |
| `get_divergent_v(u_clim, v_clim)` | `apinter.circulation.get_divergent_v` |
| `compute_velpot(u_clim, v_clim)` | `apinter.circulation.compute_velpot` |

All three now handle NaN input cleanly (fill with zonal mean for the
transform, restore NaN on output).

### Walker stream function

| Legacy (`plot_walker_streamfunction_v2.py`) | New canonical |
|---|---|
| `calc_walker_sf(ud_eq, plev_hpa)` | `apinter.circulation.calc_walker_sf(ud_eq, plev_hpa, phi0_deg=0.0)` |
| `omega_to_streamfunction(omega, lon, plev)` (from `plot_walker_circulation.py`) | `apinter.circulation.omega_to_streamfunction(omega, lon, plev_hpa, phi0_deg=0, band_half_width_deg=5)` |

Output units unchanged: 10^11 kg/s (u-based) or 10^10 kg/s (ω-based).

### Hadley stream function

| Legacy (`plot_hadley_cell_v2.py`) | New canonical |
|---|---|
| `calc_streamfunction(vd_sector, lat, plev_pa)` (hardcoded Atlantic LON_W/LON_E) | `apinter.circulation.calc_streamfunction(vd_sector, lat, plev_pa, lon_w, lon_e)` — sector is now a parameter, so the same call covers Atlantic, Pacific, or any user-chosen box |
| `calc_streamfunction_global(v_mean, lat, plev_pa)` (from `plot_hadley_cell_global.py`) | `apinter.circulation.calc_streamfunction_global(v_mean, lat, plev_pa)` |

### Common-grid interpolation (for multi-model mean)

| Legacy | New canonical |
|---|---|
| `interp_to_common(data, lon, plev, ...)` in Walker scripts (cyclic-lon wrap) | `apinter.circulation.interp_to_common_lon` |
| `interp_to_common(psi, lat, plev, ...)` in Hadley scripts (no wrap) | `apinter.circulation.interp_to_common_lat` |

The two were same-named but behaved differently; now they have distinct names.

### Li et al. (2006) ψ/φ solver

| Legacy (`Paper_1/src/psi_phi.py`) | New canonical |
|---|---|
| `psi_lietal`, `ja`, `grad_ja`, `derive_ax`, `derive_adj`, `v_zonal_integration`, `v_meridional_integration`, `dx_from_dlon`, `dy_from_dlat`, `periodify`, `integ`, `uv2psiphi` | `from apinter.circulation.psi_phi import <same_name>` — submodule port, no API change |

### Dropped (not ported)

The following pieces of the E3SMS circulation scripts were intentionally
NOT ported:

| Dropped | Why |
|---|---|
| `load_era5() / load_e3sm() / load_cmip6()` defined inside every `plot_*.py` | Replaced by generic `apinter.io.load_era5` / `load_cmip6` / direct E3SM paths in `apinter.config` |
| `plot_walker() / plot_psi() / plot_panel() / prep_data()` | Figure layout is notebook-specific; scripts will inline or own them |
| `ep_mean_at_500`, `wep_mean_at_500`, `region_mean_at_500`, `both_regions` | One-liners on the ψ array (`np.nanmean(psi[k, mask])`); per-script where used |
| `load_ocean_mask / zonal_mean_profile / compute_profiles_for_region` from `calc_meridional_profiles.py` | Narrow to that script; not reused elsewhere |

If any of these turn out to be reused across folders during phase-5
migration, they can be promoted back into the package then.

---

## Phase 4 — plotting (available now)

All ported functions return the Figure/axes/contour instead of calling
``plt.show()``, and accept ``savepath=...`` (single figure) or
``output_dir=...`` (multi-file) rather than hardcoded output paths.

### Index time series

| Legacy (`/pscratch/.../backup/ENSO-CLOUD/CMIP/cmip_plot.py`) | New canonical |
|---|---|
| `plot_nat_ts(data, ax, left_title, center_title, right_title)` | `apinter.plotting.plot_index_ts(data, ax, ...)` — generalized (any standardized index, not just NAT) |
| `plot_cmip_ts_index(ssta_ds_dict, ncols, index, rolling_time)` — hardcoded save path | `apinter.plotting.plot_index_grid(indices_dict, ncols=3, ..., savepath=None)` |

### Spatial maps — regression & trend

| Legacy (`./src/plot/`) | New canonical |
|---|---|
| `plot_regression_map.plot_regression_map(ax, data, variable='slope', ...)` | `apinter.plotting.plot_regression_map(ax, data, variable='beta', ...)` — default variable renamed to match `regression_lags` output; `lat/lon` convention enforced |
| `plot_regression_map.plot_multiple_regression_maps(datasets, ..., output_path='/pscratch/...figures')` | `apinter.plotting.plot_multiple_regression_maps(datasets, ..., savepath=None)` — hardcoded output path removed |
| `plot_trend_map.plot_trend_map(figsize, data, vmin, vmax, save_fig=True, fig_name=None, title, label)` | `apinter.plotting.plot_trend_map(data, vmin, vmax, figsize=..., title=..., colorbar_label=..., savepath=None)` — signature reordered (``data`` first), `savepath` replaces `save_fig`+`fig_name` |
| `plot_trend_map.plot_cmip6_trends(trend_data, vmin, vmax, title, label, save_fig, fig_name, ncols=4)` | `apinter.plotting.plot_cmip6_trends(trend_data, vmin, vmax, title='', colorbar_label='', ncols=4, savepath=None)` |

### Walker / Hadley / velocity-potential panels

These were defined inline in each E3SMS `plot_*_v2.py` (tightly coupled with
the surrounding figure layout). The reusable single-axes bodies are now in
`apinter.plotting.circulation`.

| Legacy (inline in E3SMS `_v2` scripts) | New canonical |
|---|---|
| `plot_panel(ax, psi, lon, plev, left_title, right_title, levels, smooth_sigma=1.5)` from `plot_walker_streamfunction_v2.py` | `apinter.plotting.plot_walker_section(ax, psi, lon, plev_hpa, levels, ...)` |
| `plot_psi(ax, psi, lat, plev_hpa, left_title, right_title, levels)` from `plot_hadley_cell_v2.py` | `apinter.plotting.plot_hadley_section(ax, psi, lat, plev_hpa, levels, ...)` |
| `plot_panel(ax, chi, u_div, v_div, ...)` from `plot_velpot_200_global.py` | `apinter.plotting.plot_velpot_panel(ax, chi, u_div, v_div, lat, lon, levels, ...)` |

The countour-label step now snaps requested label values to the nearest
actually-drawn contour level instead of failing on near-duplicate floats,
which makes the same call work with any ``levels=`` array.

### Omega lead-lag profiles (Paper_1)

| Legacy (`Paper_1/src/omega_reg_plotting.py`) | New canonical |
|---|---|
| `plot_omega_lead_lag_profile(data_dict, model_name, index_name, var, vmin, vmax, interval, save_fig, thinning_factor_lon, output_dir, figsize)` | `apinter.plotting.plot_omega_lead_lag_profile(data_dict, model_name, ..., savepath=None)` — `save_fig`+`output_dir` consolidated into `savepath` |
| `create_omega_regression_plots(model_results, model_name, output_dir=None, vmin, vmax, interval)` | `apinter.plotting.create_omega_regression_plots(model_results, model_name, output_dir=None, ...)` — API preserved |
| `convert_to_hPa` (legacy module-level helper) | inlined as the private `_convert_to_hPa` in `apinter.plotting.omega_regression` |

### Dropped — not ported

From the actual-usage audit (0 calls across all notebooks):

| Dropped function | Legacy file |
|---|---|
| `fix_cftime` | `./src/plot/plot_index.py` |
| `plot_single_index_subplot` | `./src/plot/plot_index.py` |
| `plot_indices` | `./src/plot/plot_index.py` |
| `plot_multiple_datasets` | `./src/plot/plot_index.py` |
| `plot_combined_indices` | `./src/plot/plot_index.py` (1 call → inline or migrate to `plot_index_grid`) |
| `plot_multiple_combined_indices` | `./src/plot/plot_index.py` |
| `plot_single_regression_map` | `./src/plot/plot_regression_map.py` |
| `plot_all_omega_indices` | `Paper_1/src/omega_reg_plotting.py` (wrapper that `create_omega_regression_plots` subsumes) |

Total removed: ~800 lines of legacy plotting code that nothing imported.

---

All five phases are now available. Phase 5 migrates the notebook callers
folder-by-folder (Paper_2 → SSP → Paper_1 → E3SMS) and removes the legacy
`src/` trees once the last caller is updated.
