"""Linear trend computations (global, spatial, zonal, running, seasonal)."""
import numpy as np
import xarray as xr


def global_mean_trend(ds: xr.DataArray, trend_period: int = 120) -> xr.DataArray:
    """
    Area-weighted global-mean linear trend scaled by trend_period.

    trend_period: 12 = per year, 120 = per decade.
    """
    weights = np.cos(np.deg2rad(ds.lat))
    ds_weighted = ds.weighted(weights)
    global_mean = ds_weighted.mean(dim=["lat", "lon"])

    global_mean = global_mean.copy()
    global_mean['time'] = np.arange(global_mean.time.size)

    p = global_mean.polyfit(dim='time', deg=1)
    return p.polyfit_coefficients.sel(degree=1) * trend_period


def spatial_trend(ds: xr.DataArray, trend_period: int = 120) -> xr.DataArray:
    """Per-grid-cell linear trend scaled by trend_period."""
    ds = ds.copy()
    ds['time'] = np.arange(ds.time.size)

    p = ds.polyfit(dim='time', deg=1)
    return p.polyfit_coefficients.sel(degree=1) * trend_period


def zonal_mean_trend(ds: xr.DataArray, trend_period: int = 120) -> xr.DataArray:
    """Zonal-mean linear trend per latitude band."""
    zonal_mean = ds.mean(dim='lon')
    zonal_mean = zonal_mean.copy()
    zonal_mean['time'] = np.arange(zonal_mean.time.size)

    p = zonal_mean.polyfit(dim='time', deg=1)
    return p.polyfit_coefficients.sel(degree=1) * trend_period


def running_trend(ds: xr.DataArray, window: int = 360,
                  trend_period: int = 120) -> xr.DataArray:
    """Running-window linear trend per grid cell (window in time steps)."""
    def single_trend(y):
        t = np.arange(y.size)
        coeff = np.polyfit(t, y, 1)[0]
        return coeff * trend_period

    rolling = ds.rolling(time=window, center=True).construct('window')
    return xr.apply_ufunc(
        single_trend, rolling,
        input_core_dims=[['window']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
    )


def assign_season(ds) -> xr.DataArray:
    """Attach a 'season' coordinate (DJF/MAM/JJA/SON) to the time dimension."""
    month = ds['time'].dt.month
    return xr.DataArray(
        np.select(
            [month.isin([12, 1, 2]),
             month.isin([3, 4, 5]),
             month.isin([6, 7, 8]),
             month.isin([9, 10, 11])],
            ['DJF', 'MAM', 'JJA', 'SON']),
        coords=ds.time.coords,
        name='season',
    )


def seasonal_trend(ds: xr.DataArray, season: str = 'DJF',
                   trend_period: int = 120) -> xr.DataArray:
    """Linear trend for a specified season ('DJF' | 'MAM' | 'JJA' | 'SON')."""
    s = assign_season(ds)
    ds_season = ds.sel(time=s == season)

    ds_season = ds_season.copy()
    ds_season['time'] = np.arange(ds_season.time.size)

    p = ds_season.polyfit(dim='time', deg=1)
    return p.polyfit_coefficients.sel(degree=1) * trend_period


def all_seasonal_trends(ds: xr.DataArray, trend_period: int = 120) -> xr.Dataset:
    """Dataset of linear trends for DJF/MAM/JJA/SON."""
    return xr.Dataset({
        season: seasonal_trend(ds, season=season, trend_period=trend_period)
        for season in ['DJF', 'MAM', 'JJA', 'SON']
    })
