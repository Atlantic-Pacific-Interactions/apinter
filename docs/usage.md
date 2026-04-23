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
