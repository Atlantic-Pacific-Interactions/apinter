"""Autocorrelation and effective degrees of freedom (Pyper & Peterman 1998)."""
import numpy as np
import xarray as xr


def autocorrelation_function(data: xr.DataArray, max_lag: int = None) -> np.ndarray:
    """
    Autocorrelation function for 1D time-series data.

    Returns normalized autocorrelation from lag 0 to max_lag.
    """
    values = data.values
    values = values[~np.isnan(values)]

    if len(values) < 2:
        if max_lag is None:
            return np.array([np.nan])
        return np.full(max_lag + 1, np.nan)

    n = len(values)
    if max_lag is None:
        max_lag = n - 1

    values_centered = values - np.mean(values)
    autocorr = np.correlate(values_centered, values_centered, mode='full')
    autocorr = autocorr[n - 1:]

    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    else:
        autocorr = np.full(len(autocorr), np.nan)

    return autocorr[:max_lag + 1]


def effective_degrees_of_freedom(data1: xr.DataArray, data2: xr.DataArray) -> float:
    """
    Effective degrees of freedom (Pyper & Peterman 1998) for two time series.

    1/Ne ≈ 1/N + (2/N) Σ_{j=1}^{N-1} (N-j)/N * rho_xx(j) * rho_yy(j)
    """
    values1 = data1.values
    values2 = data2.values

    valid_mask = ~(np.isnan(values1) | np.isnan(values2))
    values1 = values1[valid_mask]
    values2 = values2[valid_mask]

    n = len(values1)
    if n < 3:
        return np.nan

    data1_clean = xr.DataArray(values1, dims=['time'])
    data2_clean = xr.DataArray(values2, dims=['time'])

    rho_xx = autocorrelation_function(data1_clean, max_lag=n - 1)
    rho_yy = autocorrelation_function(data2_clean, max_lag=n - 1)

    if np.any(np.isnan(rho_xx)) or np.any(np.isnan(rho_yy)):
        return np.nan

    sum_term = 0.0
    for j in range(1, n):
        sum_term += (n - j) / n * rho_xx[j] * rho_yy[j]

    one_over_ne = 1.0 / n + 2.0 / n * sum_term

    if one_over_ne > 0:
        ne = 1.0 / one_over_ne
        ne = np.maximum(1.0, np.minimum(ne, float(n)))
    else:
        ne = np.nan

    return ne
