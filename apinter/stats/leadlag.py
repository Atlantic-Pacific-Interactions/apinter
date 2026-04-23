"""Lead-lag regression: full-field and partial (two-predictor) variants."""
import logging
from typing import Any, Dict

import numpy as np
import xarray as xr
from scipy import stats

from .significance import effective_degrees_of_freedom

logger = logging.getLogger(__name__)


def calculate_lead_lag_regression(index_data: xr.DataArray,
                                  field_data: xr.DataArray,
                                  max_lag_years: int = 15,
                                  lag_step: int = 60,
                                  alpha: float = 0.05) -> Dict[int, Dict[str, Any]]:
    """
    Lead-lag regression of `field_data` on `index_data` at month-resolution lags.

    For each lag, computes regression coefficient, correlation, per-point
    effective DOF (Pyper-Peterman), and a significance mask at level alpha.

    Parameters
    ----------
    index_data : xr.DataArray with time dim.
    field_data : xr.DataArray with time dim (plus any spatial dims).
    max_lag_years : int
    lag_step : int (months)
    alpha : float (two-tailed)

    Returns
    -------
    dict keyed by lag-in-months, each holding
      {regression, correlation, significant, ne, r_critical, n_effective}.
    """
    lag_range = range(-max_lag_years * 12, (max_lag_years + 1) * 12, lag_step)
    lag_coeffs: Dict[int, Dict[str, Any]] = {}

    print(f"  Calculating {len(list(lag_range))} lags from {-max_lag_years} to {+max_lag_years} years")

    for i, lag in enumerate(lag_range):
        print(f"    Processing lag {lag:3d} months ({i + 1}/{len(list(lag_range))})")

        field_shifted = field_data.shift(time=lag)

        reg_coeff = xr.cov(index_data, field_shifted, dim="time") / index_data.var(dim="time")
        cor_coeff = xr.corr(index_data, field_shifted, dim="time")

        n_effective = len(index_data) - abs(lag // 12)

        print(f"      Calculating effective degrees of freedom...")
        ne_map = xr.apply_ufunc(
            lambda field_point: effective_degrees_of_freedom(
                index_data,
                xr.DataArray(field_point, dims=['time'])
            ),
            field_shifted,
            input_core_dims=[['time']],
            vectorize=True,
        )

        ne_valid = ne_map.fillna(n_effective - 2)
        ne_for_test = xr.where(ne_valid > 2, ne_valid, 3)

        t_crit = stats.t.ppf(1 - alpha / 2, ne_for_test - 2)
        r_crit = t_crit / np.sqrt(ne_for_test - 2 + t_crit ** 2)

        sig_mask = (np.abs(cor_coeff) > r_crit) & (~np.isnan(cor_coeff))

        lag_coeffs[lag] = {
            'regression': reg_coeff,
            'correlation': cor_coeff,
            'significant': sig_mask,
            'ne': ne_map,
            'r_critical': r_crit,
            'n_effective': n_effective,
        }

        n_sig = sig_mask.sum().values
        n_total = (~np.isnan(cor_coeff)).sum().values
        if n_total > 0:
            print(f"      Significant points: {n_sig}/{n_total} ({100 * n_sig / n_total:.1f}%)")
            print(f"      Ne stats: mean={ne_map.mean().values:.1f}, median={ne_map.median().values:.1f}")
            print(f"      Regression coeff range: [{reg_coeff.min().values:.4f}, {reg_coeff.max().values:.4f}]")
        else:
            print(f"      No valid data points")

    return lag_coeffs


def calculate_partial_lead_lag_regression(ts1: xr.DataArray,
                                          ts2: xr.DataArray,
                                          field: xr.DataArray,
                                          max_lag_years: int = 15) -> Dict[str, Any]:
    """
    Partial lead-lag regression of `field` on two predictor time series (ts1, ts2).

    Auto-detects spatial dims — works for 2D (lat, lon) and 3D (level, lat, lon) fields.

    Returns dict with lags (months), tamv_beta (ts1 coeff), tpdv_beta (ts2 coeff),
    and per-dim coordinate arrays.
    """
    ts1, ts2, field = xr.align(ts1, ts2, field, join='inner')

    start_date = str(field.time.min().values)
    end_date = str(field.time.max().values)
    logging.info(f"Alignment complete. Processing time range: {start_date} to {end_date}")
    logging.info(f"Total time steps: {len(field.time)}")

    spatial_dims = [d for d in field.dims if d != 'time']
    spatial_shape = [field.sizes[d] for d in spatial_dims]

    if len(field.time) <= max_lag_years * 12:
        raise ValueError(f"Time series is too short for a {max_lag_years} year lag.")

    lags = np.arange(-max_lag_years, max_lag_years + 1) * 12

    res_beta1 = np.zeros((len(lags), *spatial_shape))
    res_beta2 = np.zeros((len(lags), *spatial_shape))

    for i, lag in enumerate(lags):
        if lag < 0:
            y = field.values[-lag:]
            x1, x2 = ts1.values[:lag], ts2.values[:lag]
        elif lag > 0:
            y = field.values[:-lag]
            x1, x2 = ts1.values[lag:], ts2.values[lag:]
        else:
            y, x1, x2 = field.values, ts1.values, ts2.values

        X = np.column_stack([x1, x2, np.ones(len(x1))])
        y_flat = y.reshape(y.shape[0], -1)
        beta = np.linalg.lstsq(X, y_flat, rcond=None)[0]

        res_beta1[i] = beta[0].reshape(spatial_shape)
        res_beta2[i] = beta[1].reshape(spatial_shape)

    return {
        'lags': lags,
        'tamv_beta': res_beta1,
        'tpdv_beta': res_beta2,
        'coords': {d: field[d].values for d in spatial_dims},
    }
