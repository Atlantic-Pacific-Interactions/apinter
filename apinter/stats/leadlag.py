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
                                          max_lag_years: int = 15,
                                          compute_significance: bool = False) -> Dict[str, Any]:
    """
    Partial lead-lag regression of `field` on two predictor time series (ts1, ts2).

    Legacy-compatible shim: delegates to apinter.stats.regression.regression_lags
    and returns a dict in the old format (lags, tamv_beta, tpdv_beta, coords).
    Passing compute_significance=True also adds tamv_pval, tpdv_pval.

    Prefer regression_lags(field, ts1, confounder=ts2, lags=...) directly for new code.
    """
    from .regression import regression_lags

    if len(field.time) <= max_lag_years * 12:
        raise ValueError(f"Time series is too short for a {max_lag_years} year lag.")

    lags = np.arange(-max_lag_years, max_lag_years + 1) * 12
    ds = regression_lags(field, ts1, lags=list(lags), confounder=ts2,
                         compute_significance=compute_significance)
    if ds is None:
        raise RuntimeError("regression_lags returned None (overlap too short).")

    spatial_dims = [d for d in field.dims if d != 'time']

    out: Dict[str, Any] = {
        'lags': lags,
        'tamv_beta': ds['target_beta'].values,
        'tpdv_beta': ds['confounder_beta'].values,
        'coords': {d: field[d].values for d in spatial_dims},
    }
    if compute_significance:
        out['tamv_pval'] = ds['target_pval'].values
        out['tpdv_pval'] = ds['confounder_pval'].values
    return out
