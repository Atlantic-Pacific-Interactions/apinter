"""Autocorrelation and effective degrees of freedom.

Two families:
  - Scalar (Pyper & Peterman 1998): autocorrelation_function + effective_degrees_of_freedom.
    Used for significance of point-wise or time-series correlations.
  - Vectorized (Bretherton et al. 1999 integral time scale):
    autocorrelation_numpy_vectorized + calculate_neff_vectorized. Used by
    regression_lags for spatial-map significance.
"""
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


def autocorrelation_numpy_vectorized(data: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Vectorized autocorrelation function (ACF) for (time, space) arrays.

    Input: np.ndarray of shape (time,) or (time, space).
    Output: np.ndarray of shape (max_lag+1, space). ACF at lag 0 is 1.0;
    NaN where variance is zero.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    data_mean = np.mean(data, axis=0)
    data_centered = data - data_mean
    var = np.sum(data_centered ** 2, axis=0)

    var = np.where(var == 0, np.nan, var)

    acf = np.zeros((max_lag + 1, data.shape[1]))
    acf[0, :] = 1.0

    for lag in range(1, max_lag + 1):
        cov = np.sum(data_centered[:-lag] * data_centered[lag:], axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            acf[lag, :] = cov / var
    return acf


def calculate_neff_vectorized(x_series: np.ndarray,
                              y_field: np.ndarray,
                              max_lag_cap: int = 60) -> np.ndarray:
    """
    Vectorized effective sample size (Bretherton et al. 1999 integral time scale).

    1/Ne = 1/N + (2/N) Σ_{j=1..L} (N-j)/N * rho_xx(j) * rho_yy(j)

    Inputs
    ------
    x_series : (time,) 1D array (predictor).
    y_field  : (time, space) 2D array (response, flattened spatial).
    max_lag_cap : int, cap on summation range for memory efficiency (default 60).

    Returns
    -------
    Ne : (space,) array, clipped to [2, N]. NaN where variance is zero.
    """
    n = len(x_series)
    if n < 3:
        return np.full(y_field.shape[1], np.nan)

    x_in = x_series.reshape(-1, 1)
    max_lag = min(n - 2, max_lag_cap)

    rho_xx = autocorrelation_numpy_vectorized(x_in, max_lag)
    rho_yy = autocorrelation_numpy_vectorized(y_field, max_lag)

    sum_term = np.zeros(y_field.shape[1])
    for j in range(1, max_lag + 1):
        sum_term += (n - j) / n * rho_xx[j, 0] * rho_yy[j, :]

    inv_ne = (1.0 / n) + (2.0 / n) * sum_term
    with np.errstate(divide='ignore', invalid='ignore'):
        ne = np.clip(1.0 / inv_ne, 2.0, float(n))
    return ne
