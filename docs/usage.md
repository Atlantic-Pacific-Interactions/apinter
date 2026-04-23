# apinter — Usage

Task-oriented reference with small, runnable examples. For full parameter
docs and edge cases, read the docstring: e.g. `help(apinter.stats.regression_lags)`.

## The canonical pipeline

Every analysis in Paper_1/2 follows this shape:

```
load -> compute_anomaly -> (regional mean) -> lowpass filter -> standardize
                       \-> linear trend map
                       \-> regression_lags / correlation_lags
```

`apinter.processing` provides each stage as a reusable primitive.
`apinter.indices` composes them for common index and gridded-anomaly workflows.
`apinter.io` handles all reading from disk (CMIP6, observations, reanalyses).

---

## I/O — loading data

Single-variable spec dicts drive every loader. The full list of supported
variables is the keys of each `*_VARS` dict:

```python
from apinter.io import CMIP6_VARS, ERA5_VARS, OBS_SST_SOURCES, ORAS5_VARS, SSP_VARS
```

### CMIP6 multi-model load

Generic entry point replaces the 9 legacy per-variable functions:

```python
from apinter.io import load_cmip6

# SST: K->C + outlier filter (-10 < SST < 40 °C, E3SMS convention).
# Returns {model_name: xr.DataArray}. Missing files silently skipped.
sst = load_cmip6('ts', sim_time=slice('1980', '2014'))

# Multi-model omega, specific subset of models:
wap = load_cmip6('wap', models=['ACCESS-CM2', 'CESM2'])

# 3D wind field, sliced to a single pressure level on load:
ua_850 = load_cmip6('ua', level=850)

# Legacy names kept as thin wrappers — existing code still works:
from apinter.io import load_cmip6_sst, load_cmip6_wind
sst = load_cmip6_sst(sim_time=slice('1980', '2014'))
winds = load_cmip6_wind(target_level=850)   # {model: {'ua': ..., 'va': ...}}
```

Available variables:

```python
>>> sorted(CMIP6_VARS)
['pr', 'psl', 'ts', 'ua', 'va', 'wap', 'zg', 'tauu', 'thetao', 'zos']
```

`ts` gets K→C + the SST outlier filter; `zos` gets the explicit land-sea
mask; all other variables are passed through with only the coord rename
(`latitude/longitude → lat/lon`, `plev → level`).

### Observational SST

```python
from apinter.io import load_obs_sst

hadisst = load_obs_sst('hadisst', sim_time=slice('1891', '2014'))
ersst   = load_obs_sst('ersst',   sim_time=slice('1891', '2014'))
cobe    = load_obs_sst('cobesst', sim_time=slice('1891', '2014'))
```

Sources are in `OBS_SST_SOURCES`; missing-value sentinels (e.g. HadISST's
−1000) are already converted to NaN.

### ERA5 reanalysis

One function covers every ERA5 file on disk:

```python
from apinter.io import load_era5

# 3D variable, single pressure level, one-month smoke load:
u850 = load_era5('u', level=850, sim_time=slice('2000-01', '2000-01'))

# Surface variable with regional subset:
slp_atl = load_era5('slp',
                    sim_time=slice('1990', '2014'),
                    region={'lat': (-30, 30), 'lon': (280, 360)})

# Surface-bundle variables (u10/v10/t2m/d2m share one NetCDF):
t2m = load_era5('t2m', sim_time=slice('2000', '2005'))
```

Supported `var` keys:

| Kind | Variables |
|---|---|
| Pressure-level (3D) | `u`, `v`, `q`, `z`, `pl_omega` |
| Single-level | `omega500`, `slp`, `sst`, `tp`, `mtnlwrf` |
| Surface bundle | `u10`, `v10`, `t2m`, `d2m` |

**Performance note:** selections (`level`, `sim_time`, `region`) are pushed
down into each file open *before* `.load()`, so small requests don't read
the full multi-GB file. Always pass these when you can.

### ORAS5 ocean reanalysis

NEMO ORCA025 per-month files, starting 1958-01:

```python
from apinter.io import load_oras5, regrid_to_equatorial_lon, load_oras5_equatorial

# Native curvilinear grid (keeps nav_lat, nav_lon 2D coords):
ssh = load_oras5('ssh', sim_time=slice('2000', '2014'))

# Equatorial-band Hovmoller regrid (matches Paper_1 nb 36/42):
ssh_eq = regrid_to_equatorial_lon(ssh,
                                   lon_bounds=(120, 290),
                                   lat_bounds=(-5.5, 5.5),
                                   lon_step=2.0)

# Or the one-shot convenience:
tauu_eq = load_oras5_equatorial('tauu', lon_bounds=(120, 290),
                                lat_bounds=(-5.5, 5.5), lon_step=2.0,
                                sim_time=slice('1958', '2014'))
```

Supported `var` keys: `d20`, `ssh`, `mld030`, `tauu`, `tauv`, `hfds`,
`thetao`, `salinity`, `uo`, `vo`, `wo`.

### SSP historical + scenario concat

One function for every SSP variable — handles K→C, SST outlier filter,
grid alignment, and the historical→SSP variable name change automatically:

```python
from apinter.io import load_and_concat, get_ssp_models

models = get_ssp_models('ssp245', var='sst')       # models with both halves
sst_series = load_and_concat('sst', ssp='ssp245', model=models[0],
                             full_time=slice('1850', '2100'))
# Continuous time axis from 1850-01 to 2100-12, °C, ocean-only.
```

Supported `var` keys: `sst`, `wap`, `pr`, `prc`.

### joblib I/O

```python
from apinter.io import load_joblib, save_joblib

save_joblib(obj, '/path/to/result.joblib')    # auto-creates parent dir
data = load_joblib('/path/to/result.joblib')
```

---

## Processing

### Monthly anomaly (with optional detrend)

```python
from apinter.processing import compute_anomaly

anom = compute_anomaly(sst, detrend=True)
# default: all-months climatology + linear detrend on time (Paper_1 canonical)

anom_raw = compute_anomaly(sst, detrend=False)
# just the climatology subtraction

anom_legacy = compute_anomaly(sst, complete_years_only=True)
# climatology from floor(n_months/12) complete years (legacy ./src/cal_index behavior)
```

### Low-pass filtering

```python
from apinter.processing import lanczos_lowpass, apply_lowpass

filt = lanczos_lowpass(ts, cutoff_period=132)              # 11-year cutoff (monthly data)

filt = apply_lowpass(ts, cutoff_period=132, method='lanczos')       # same thing
filt = apply_lowpass(ts, cutoff_period=12,  method='running_mean')  # 12-month rolling mean
filt = apply_lowpass(ts, cutoff_period=132, method=None)            # pass-through
```

### Area-weighted regional mean

```python
from apinter.processing import wgt_areaave

# latS, latN, lonW, lonE  (positional order)
tamv_box = wgt_areaave(anom, 0, 30, 280, 340)
```

Requires `lat`/`lon` coord names (package-wide convention).

### Linear detrend along any dimension

```python
from apinter.processing import detrend_dim

detrended = detrend_dim(da, dim='time', deg=1)
```

---

## Indices

### One regional index

```python
from apinter.indices import calculate_index

tamv = calculate_index(
    sst,
    lon_bounds=(280, 340),
    lat_bounds=(0, 30),
    detrend=True,
    cutoff_period=132,       # months
    method='lanczos',        # or 'running_mean', None
    normalize=True,
)
# Single xr.DataArray along time
```

### Multiple indices from one SST field (efficient)

Computes the gridded anomaly once, reuses it for every region:

```python
from apinter.indices import calculate_multiple_indices

regions = {
    'tamv': ((280, 340), (0, 30)),
    'tpdv': ((180, 280), (-20, 20)),
}
indices = calculate_multiple_indices(sst, regions)
# {'tamv': DataArray, 'tpdv': DataArray}
```

### Gridded low-pass-filtered anomalies

Keeps spatial dimensions — use this for SSTA maps and lead-lag regression inputs:

```python
from apinter.indices import gridded_anomalies

lpf_ssta = gridded_anomalies(
    sst,
    cutoff_period=132,
    method='lanczos',
    normalize=True,      # divide by temporal std at each grid point
)
# Same dims as sst
```

Pass `lon_bounds` / `lat_bounds` to crop first, or `cutoff_period=None` to skip filtering.

---

## Regression

### Concurrent regression map

```python
from apinter.stats import regression_lags

reg = regression_lags(field=ssta, target_index=tamv, lags=[0])
# reg vars: beta, p_value — dims (lag, lat, lon) with size-1 lag
```

### Lead-lag regression map

```python
lags_months = [-180, -120, -60, 0]    # TAMV leads SSTA by up to 15 years
reg = regression_lags(field=ssta, target_index=tamv, lags=lags_months)
```

### Partial (two-index) lead-lag regression

Matches `calc_partial_regression` from Paper_1 notebook 34:

```python
reg = regression_lags(
    field=ssta,
    target_index=tamv,
    confounder=tpdv,
    lags=[-180, -120, -60, 0],
    compute_significance=True,
)
# With confounder, variables are:
#   target_beta, confounder_beta, target_pval, confounder_pval
```

Significance uses the Bretherton et al. (1999) integral-time-scale
effective DoF with spatial-vectorized lstsq.

---

## Correlation

### 1D-on-1D lead-lag correlation

```python
from apinter.stats import correlation_lags

c = correlation_lags(tamv, tpdv, max_lag=120)   # months
# c has dim `lag` (length 2*max_lag + 1), vars `r`, `p_value`
```

No internal time-range selection — slice `tamv`, `tpdv` to your window first.

### Multi-model ensemble mean

```python
from apinter.stats import mmm_correlation_lags

per_model = {model: correlation_lags(tamv[model], tpdv[model], max_lag=120)
             for model in cmip6_models}
mmm = mmm_correlation_lags(per_model, exclude=['HadISST', 'ERSST'])
# vars: r_mean, r_std, p_mean, n_models
```

Caller supplies the `exclude` list — no hardcoded observational-dataset filtering.

---

## Trends

```python
from apinter.stats import (
    global_mean_trend,       # area-weighted global mean, then trend
    spatial_trend,           # linear trend per grid cell
    zonal_mean_trend,        # zonal mean then trend
    running_trend,           # rolling-window trend
    seasonal_trend,          # trend for one season (DJF/MAM/JJA/SON)
    all_seasonal_trends,     # Dataset with all four
    assign_season,           # attach a 'season' coord
)

# trend_period scales the output: 12 = per year, 120 = per decade
trend_per_decade = spatial_trend(sst, trend_period=120)
```

---

## Circulation

### Walker stream function (longitude × pressure)

Two constructions. Pick whichever matches your inputs.

From equatorial divergent U (Helmholtz-decomposed winds):

```python
from apinter.io import load_era5
from apinter.circulation import get_divergent_u, calc_walker_sf

# Per pressure level, compute divergent U on a global grid, then
# average to a 5°S–5°N equatorial band.
ud_levels = []
for lev in [100, 200, 300, 500, 700, 850, 925, 1000]:
    u = load_era5('u', level=lev, sim_time=slice('1958', '2014')).mean('time')
    v = load_era5('v', level=lev, sim_time=slice('1958', '2014')).mean('time')
    ud = get_divergent_u(u, v)                       # np.ndarray (lat, lon)
    ud_eq = np.nanmean(ud[(u.lat >= -5) & (u.lat <= 5), :], axis=0)
    ud_levels.append(ud_eq)

ud_eq = np.stack(ud_levels)                          # (nlev, nlon)
psi = calc_walker_sf(ud_eq, plev_hpa=np.array([100, 200, 300, 500, 700, 850, 925, 1000]))
# psi in units of 10^11 kg/s
```

Alternative — from ω integrated zonally (older method):

```python
from apinter.circulation import omega_to_streamfunction

omega = ...              # (nlev, nlon), Pa/s, already equatorial-band-averaged
psi = omega_to_streamfunction(omega, lon=..., plev_hpa=...,
                              phi0_deg=0.0, band_half_width_deg=5.0)
# psi in units of 10^10 kg/s
```

### Hadley stream function (latitude × pressure)

Regional sector (e.g. Atlantic 280°–360°E):

```python
from apinter.circulation import get_divergent_v, calc_streamfunction

vd_sector = ...              # (nlev, nlat) divergent V averaged over lon_w..lon_e, m/s
psi = calc_streamfunction(vd_sector, lat=..., plev_pa=...,
                          lon_w=280, lon_e=360)
# psi in units of 10^10 kg/s
```

Global zonal-mean:

```python
from apinter.circulation import calc_streamfunction_global

v_mean = v.mean('lon').mean('time').values        # (nlev, nlat)
psi = calc_streamfunction_global(v_mean, lat=..., plev_pa=...)
```

### Velocity potential (200 hPa overlay)

```python
from apinter.circulation import compute_velpot

u200 = load_era5('u', level=200, sim_time=slice('1958','2014')).mean('time')
v200 = load_era5('v', level=200, sim_time=slice('1958','2014')).mean('time')
chi, u_div, v_div = compute_velpot(u200, v200)
```

### Multi-model mean on a common grid

`interp_to_common_lon` / `interp_to_common_lat` put each model's ψ onto the
canonical pressure × longitude (or × latitude) grid so they can be averaged:

```python
from apinter.circulation import interp_to_common_lon
from apinter.config import COMMON_LON, COMMON_PLEV

psi_common = interp_to_common_lon(psi_model, lon=model_lon, plev_hpa=model_plev)
# psi_common is on (COMMON_PLEV, COMMON_LON) — averageable across models
```

### Li et al. (2006) ψ/φ minimization solver (Paper_1)

For non-rectangular domains with complex coastlines. Different algorithm from
the windspharm-based Helmholtz decomposition — minimizes an L² functional.

```python
from apinter.circulation.psi_phi import uv2psiphi

psi, phi = uv2psiphi(LON, LAT, U, V,
                     ZBC='closed', MBC='closed',
                     ALPHA=1.0e-14, fac=111195, period=False)
```

---

## Plotting

All plotting functions return the Figure/axes (or contour mesh) instead of
calling `plt.show()`. Save with `savepath=...` or `output_dir=...`; if you
don't pass those, nothing is written to disk.

### Index time series

```python
import matplotlib.pyplot as plt
from apinter.plotting import plot_index_ts, plot_index_grid

# Single panel
fig, ax = plt.subplots(figsize=(14, 4))
plot_index_ts(tamv, ax,
              left_title='HadISST', center_title='TAMV',
              right_title='11-yr LPF', ylim=(-3, 3))

# Grid of panels, one per dataset
fig = plot_index_grid(
    {'HadISST': tamv_had, 'ERSST': tamv_ers, 'COBE': tamv_cobe},
    ncols=1, center_title='TAMV', right_title='11-yr LPF',
    ylim=(-3, 3), savepath='figures/tamv_obs.png',
)
```

### Spatial maps — regression & trend

Regression map on a single axes (works with `regression_lags` output):

```python
import cartopy.crs as ccrs
from apinter.plotting import plot_regression_map

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=180)})
im = plot_regression_map(
    ax, reg_dataset, variable='beta',     # 'beta' matches regression_lags naming
    title='TAMV → SSTA (concurrent)',
    vmin=-0.8, vmax=0.8,
    add_stippling=True, significance_alpha=0.1,
    add_box=True, box_coords=(280, 0, 60, 30),   # TAMV box
)
fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8)
```

Grid of regression maps:

```python
from apinter.plotting import plot_multiple_regression_maps

fig = plot_multiple_regression_maps(
    datasets=[reg_hadisst, reg_ersst, reg_cobe, reg_cmip6_mmm],
    variable='beta',
    titles=['HadISST', 'ERSST', 'COBE', 'CMIP6 MMM'],
    suptitle='TAMV → SSTA regression',
    nrows=2, ncols=2, vmin=-0.8, vmax=0.8,
    savepath='figures/tamv_ssta_reg.png',
)
```

Trend map (single field):

```python
from apinter.plotting import plot_trend_map

fig = plot_trend_map(
    trend_da, vmin=-0.5, vmax=0.5,
    title='HadISST trend 1958–2014',
    colorbar_label='°C/decade',
)
```

Grid of per-model trends:

```python
from apinter.plotting import plot_cmip6_trends

fig = plot_cmip6_trends(
    {m: spatial_trend(sst_dict[m]) for m in sst_dict},
    vmin=-0.5, vmax=0.5, ncols=4,
    title='CMIP6 SST trends 1958–2014',
    colorbar_label='°C/decade',
    savepath='figures/cmip6_sst_trends.png',
)
```

### Walker / Hadley / velocity-potential panels

Each panel renders onto a caller-supplied axes; you build the figure and
colorbar. Matches the `_v2` E3SMS layouts (log-pressure axis, Gaussian
smoothing, centred 0-contour highlighted).

```python
from apinter.plotting import plot_walker_section, plot_hadley_section, plot_velpot_panel

levels = np.arange(-2, 2.1, 0.2)

fig, ax = plt.subplots(figsize=(12, 4))
plot_walker_section(ax, psi_walker, lon, plev_hpa, levels=levels,
                    lon_plot_bounds=(100, 360),
                    left_title='(a)', right_title='ERA5',
                    show_xlabel=True, show_ylabel=True)

fig, ax = plt.subplots(figsize=(6, 4))
plot_hadley_section(ax, psi_hadley_atlantic, lat, plev_hpa, levels=levels,
                    lat_plot_bounds=(-30, 30),
                    left_title='(b)', right_title='Atlantic')

# Velocity potential + divergent-wind overlay at 200 hPa
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=180)})
cf = plot_velpot_panel(ax, chi * 1e-6, u_div, v_div, lat, lon,
                       levels=np.arange(-10, 10.1, 1),
                       quiver_step=8, min_wind=0.1)
```

### Omega lead-lag regression profiles (Paper_1)

```python
from apinter.plotting import plot_omega_lead_lag_profile, create_omega_regression_plots

# Single model, single index
fig, axs = plot_omega_lead_lag_profile(
    model_results, model_name='HadISST',
    index_name='tamv', var='regression',
    vmin=-0.003, vmax=0.003,
    savepath='figures/HadISST_tamv_omega_lead_lag.png',
)

# All indices in one call (writes one PNG per index)
create_omega_regression_plots(
    model_results, model_name='HadISST',
    output_dir='figures/omega_lead_lag',
    vmin=-0.003, vmax=0.003,
)
```

The vertical separators at 30°E / 110°E / 180°E / 280°E and the
*Indian / WP / CEP / Atlantic* region labels are drawn automatically — they
match the Paper_1 omega-regression figure convention.

---

## Significance

```python
from apinter.stats import (
    autocorrelation_function,          # scalar ACF of a 1D series
    effective_degrees_of_freedom,      # scalar Pyper-Peterman Ne for two series
    autocorrelation_numpy_vectorized,  # (time, space) ACF — used internally
    calculate_neff_vectorized,         # Bretherton integral Ne — used internally
)
```

Use the scalar versions for point-wise or single-series diagnostics. The
vectorized versions power `regression_lags`'s significance test — you
rarely call them directly.

---

## Putting it together: a minimal Paper_1 workflow

```python
from apinter.io import load_obs_sst
from apinter.indices import calculate_multiple_indices, gridded_anomalies
from apinter.stats import regression_lags

sst = load_obs_sst('hadisst', sim_time=slice('1891', '2014'))

indices = calculate_multiple_indices(sst, {
    'tamv': ((280, 340), (0, 30)),
    'tpdv': ((180, 280), (-20, 20)),
})

lpf_ssta = gridded_anomalies(sst, cutoff_period=132, normalize=True)

# Partial lead-lag regression at -15yr, -10yr, -5yr, concurrent
reg = regression_lags(
    field=lpf_ssta.sel(time=slice("1958", "2014")),
    target_index=indices['tamv'].sel(time=slice("1958", "2014")),
    confounder=indices['tpdv'].sel(time=slice("1958", "2014")),
    lags=[-180, -120, -60, 0],
    compute_significance=True,
)

reg['target_beta'].sel(lag=-120).plot()    # TAMV partial at 10-yr lead
```
