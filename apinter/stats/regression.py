"""Vectorized grid-point regression and lead-lag correlation between time series.

Canonical entry point: `regression_lags` — handles simple or partial regression at
one or many lags with effective-DoF significance (Bretherton integral time scale).
"""
import logging
from typing import Dict, Iterable, Optional

import numpy as np
import xarray as xr
from scipy import stats

from .significance import calculate_neff_vectorized

logger = logging.getLogger(__name__)


def regression_lags(
    field: xr.DataArray,
    target_index: xr.DataArray,
    lags: Iterable[int] = (0,),
    confounder: Optional[xr.DataArray] = None,
    compute_significance: bool = True,
    min_samples: int = 10,
    min_overlap_months: int = 24,
) -> Optional[xr.Dataset]:
    """
    Lead-lag (optionally partial) regression of `field` on `target_index`.

    For each lag in `lags` (months; notebook 34 convention):
        lag < 0: target_index leads field by |lag| months
        lag > 0: field leads target_index by lag months
        lag = 0: concurrent

    If `confounder` is None:
        field(t+lag) = beta * target(t) + intercept + ε

    If `confounder` is given (same time base as target_index):
        field(t+lag) = target_beta * target(t) + confounder_beta * confounder(t)
                       + intercept + ε

    Fit is spatial-vectorized via np.linalg.lstsq. p-values use Bretherton-style
    effective sample size with the integral time scale, then a two-sided t-test.

    Returns
    -------
    xr.Dataset with dims (lag, *field_spatial_dims). Variables:
        no confounder:  beta, p_value (if compute_significance)
        with confounder: target_beta, confounder_beta,
                         target_pval, confounder_pval (if compute_significance)
    Returns None if overlap shorter than min_overlap_months.
    """
    lags = list(lags)

    if confounder is None:
        ts1, field = xr.align(target_index, field, join='inner')
        ts2 = None
    else:
        ts1, ts2, field = xr.align(target_index, confounder, field, join='inner')

    if len(field.time) < min_overlap_months:
        logger.warning(
            f"Overlap ({len(field.time)} months) < min_overlap_months={min_overlap_months}. Skipping."
        )
        return None

    spatial_dims = tuple(d for d in field.dims if d != 'time')
    spatial_shape = tuple(field.sizes[d] for d in spatial_dims)
    spatial_flat = int(np.prod(spatial_shape)) if spatial_shape else 1

    y_vals = field.values.reshape(field.shape[0], spatial_flat)
    x1_vals = ts1.values
    x2_vals = ts2.values if ts2 is not None else None

    nlags = len(lags)
    beta_target = np.full((nlags, spatial_flat), np.nan, dtype=np.float64)
    pval_target = np.full((nlags, spatial_flat), np.nan, dtype=np.float64) if compute_significance else None
    if confounder is not None:
        beta_confounder = np.full((nlags, spatial_flat), np.nan, dtype=np.float64)
        pval_confounder = (
            np.full((nlags, spatial_flat), np.nan, dtype=np.float64)
            if compute_significance else None
        )

    for i, lag in enumerate(lags):
        if lag < 0:
            shift = abs(lag)
            y_run = y_vals[shift:, :]
            x1_run = x1_vals[:-shift]
            x2_run = x2_vals[:-shift] if x2_vals is not None else None
        elif lag > 0:
            shift = lag
            y_run = y_vals[:-shift, :]
            x1_run = x1_vals[shift:]
            x2_run = x2_vals[shift:] if x2_vals is not None else None
        else:
            y_run, x1_run = y_vals, x1_vals
            x2_run = x2_vals

        if x2_run is not None:
            valid = ~(np.isnan(x1_run) | np.isnan(x2_run))
        else:
            valid = ~np.isnan(x1_run)

        if np.sum(valid) < min_samples:
            continue

        x1_final = x1_run[valid]
        y_final = y_run[valid, :]
        n_samples = len(x1_final)

        if x2_run is not None:
            x2_final = x2_run[valid]
            X = np.column_stack([x1_final, x2_final, np.ones(n_samples)])
            n_predictors = 2
        else:
            X = np.column_stack([x1_final, np.ones(n_samples)])
            n_predictors = 1

        beta, residuals, rank, _s = np.linalg.lstsq(X, y_final, rcond=None)
        b1 = beta[0, :]
        beta_target[i, :] = b1
        if confounder is not None:
            b2 = beta[1, :]
            beta_confounder[i, :] = b2

        if not compute_significance:
            continue

        # Sum of squared residuals (spatial-vectorized)
        if residuals.size > 0 and residuals.shape == (y_final.shape[1],):
            ssr = residuals
        else:
            y_pred = X @ beta
            ssr = np.sum((y_final - y_pred) ** 2, axis=0)

        dof_nominal = n_samples - (n_predictors + 1)
        if dof_nominal <= 0:
            continue
        mse = ssr / dof_nominal

        try:
            xtx_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            continue
        xtx_diag = np.diag(xtx_inv)

        # Target predictor p-value
        se_nom_1 = np.sqrt(mse * xtx_diag[0])
        neff_1 = calculate_neff_vectorized(x1_final, y_final)
        with np.errstate(divide='ignore', invalid='ignore'):
            se_eff_1 = se_nom_1 * np.sqrt(n_samples / neff_1)
            t_stat_1 = b1 / se_eff_1
        dof_eff_1 = np.maximum(neff_1 - (n_predictors + 1), 1)
        p1 = 2 * (1 - stats.t.cdf(np.abs(t_stat_1), df=dof_eff_1))
        pval_target[i, :] = p1

        if confounder is not None:
            se_nom_2 = np.sqrt(mse * xtx_diag[1])
            neff_2 = calculate_neff_vectorized(x2_final, y_final)
            with np.errstate(divide='ignore', invalid='ignore'):
                se_eff_2 = se_nom_2 * np.sqrt(n_samples / neff_2)
                t_stat_2 = b2 / se_eff_2
            dof_eff_2 = np.maximum(neff_2 - (n_predictors + 1), 1)
            p2 = 2 * (1 - stats.t.cdf(np.abs(t_stat_2), df=dof_eff_2))
            pval_confounder[i, :] = p2

    out_dims = ('lag',) + spatial_dims
    out_shape = (nlags,) + spatial_shape
    coords = {'lag': list(lags), **{d: field[d] for d in spatial_dims}}

    if confounder is None:
        ds_vars = {'beta': (out_dims, beta_target.reshape(out_shape))}
        if compute_significance:
            ds_vars['p_value'] = (out_dims, pval_target.reshape(out_shape))
    else:
        ds_vars = {
            'target_beta': (out_dims, beta_target.reshape(out_shape)),
            'confounder_beta': (out_dims, beta_confounder.reshape(out_shape)),
        }
        if compute_significance:
            ds_vars['target_pval'] = (out_dims, pval_target.reshape(out_shape))
            ds_vars['confounder_pval'] = (out_dims, pval_confounder.reshape(out_shape))

    return xr.Dataset(ds_vars, coords=coords)


def correlation_lags(ts1: xr.DataArray,
                     ts2: xr.DataArray,
                     max_lag: int = 12) -> xr.Dataset:
    """
    Lead-lag Pearson correlation between two 1D time series.

    For each lag k in [-max_lag, +max_lag] (time steps), computes
    corr(ts1, ts2 shifted by k).
      lag < 0: ts2 leads ts1 by |lag| steps
      lag > 0: ts1 leads ts2 by lag steps
      lag = 0: concurrent

    The caller is responsible for slicing ts1 and ts2 to the desired time
    window before calling. The function does not apply any internal
    time-range selection.

    Parameters
    ----------
    ts1, ts2 : xr.DataArray
        1D time series sharing the 'time' dimension. Must already be
        aligned; otherwise an inner join is applied.
    max_lag : int
        Maximum absolute lag in time-step units (months for monthly data).

    Returns
    -------
    xr.Dataset with coord `lag` (length 2*max_lag+1) and variables `r`,
    `p_value` (two-sided scipy.stats.linregress).
    """
    ts1, ts2 = xr.align(ts1, ts2, join='inner')
    n = ts1.sizes['time']
    if n < 2 * max_lag + 3:
        raise ValueError(
            f"Overlap length {n} too short for max_lag={max_lag}."
        )

    x = ts1.isel(time=slice(max_lag, n - max_lag)).values
    y = ts2.values

    lags = np.arange(-max_lag, max_lag + 1)
    r_vals = np.empty(lags.size, dtype=float)
    p_vals = np.empty(lags.size, dtype=float)

    for i, lag in enumerate(lags):
        start = max_lag + lag
        y_shifted = y[start:start + x.size]
        _, _, r, p, _ = stats.linregress(x, y_shifted)
        r_vals[i] = r
        p_vals[i] = p

    return xr.Dataset(
        {'r': (['lag'], r_vals), 'p_value': (['lag'], p_vals)},
        coords={'lag': lags},
    )


def mmm_correlation_lags(per_model_results: Dict[str, xr.Dataset],
                         exclude: Optional[Iterable[str]] = None
                         ) -> xr.Dataset:
    """
    Multi-model-mean of per-model correlation_lags Datasets.

    Parameters
    ----------
    per_model_results : {model_name: correlation_lags Dataset}
        Each value is a Dataset from correlation_lags (dims: lag; vars: r, p_value).
    exclude : iterable of model names, optional
        Models to skip (e.g., observational datasets). Default: include all.

    Returns
    -------
    xr.Dataset with `lag` coord and variables:
        r_mean, r_std, p_mean, n_models
    """
    exclude = set(exclude or [])
    included = [name for name in per_model_results if name not in exclude]
    if not included:
        raise ValueError("No models left after applying exclude list.")

    stacked_r = xr.concat(
        [per_model_results[name]['r'] for name in included],
        dim=xr.DataArray(included, dims='model', name='model'),
    )
    stacked_p = xr.concat(
        [per_model_results[name]['p_value'] for name in included],
        dim=xr.DataArray(included, dims='model', name='model'),
    )

    return xr.Dataset({
        'r_mean': stacked_r.mean('model'),
        'r_std': stacked_r.std('model'),
        'p_mean': stacked_p.mean('model'),
        'n_models': xr.DataArray(len(included)),
    })
