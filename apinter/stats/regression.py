"""Vectorized grid-point regression and lead-lag correlation between time series.

Canonical entry point: `regression_lags` — handles simple or partial regression at
one or many lags with effective-DoF significance (Bretherton integral time scale).
"""
import logging
from typing import Dict, Iterable, Optional, Tuple

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


def _linear_regression(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Linear regression between 1D x and y arrays (module-level for picklability).

    Returns (slope, intercept, r_value, p_value, std_err), NaN if < 3 valid points.
    """
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if valid_mask.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
        return slope, intercept, r_value, p_value, std_err
    except Exception as e:
        logger.warning(f"Regression failed: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan


def calculate_regression_vectorize(index: xr.DataArray,
                                   gridded_data: xr.DataArray,
                                   coord_name: str = 'time',
                                   align_time: bool = True) -> xr.Dataset:
    """
    Vectorized grid-point linear regression between a 1D index and gridded data.

    Returns a Dataset with: slope, intercept, r_value, p_value, std_err.
    r_value is identical to scipy.stats.pearsonr.
    """
    logger.info("Starting vectorized regression calculation")
    logger.info(f"Index shape: {index.shape}, gridded shape: {gridded_data.shape}")

    if align_time:
        index[coord_name] = gridded_data[coord_name]
        index, gridded_data = xr.align(index, gridded_data, join='inner', copy=False)
        logger.info(f"Aligned index shape: {index.shape}, gridded: {gridded_data.shape}")

    index_nan_count = np.isnan(index.values).sum()
    gridded_nan_count = np.isnan(gridded_data.values).sum()
    if index_nan_count > 0:
        logger.warning(f"Index contains {index_nan_count} NaN values")
    if gridded_nan_count > 0:
        logger.warning(f"Gridded data contains {gridded_nan_count} NaN values")

    regression_results = xr.apply_ufunc(
        _linear_regression,
        index,
        gridded_data,
        input_core_dims=[[coord_name], [coord_name]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float, float, float, float],
        dask_gufunc_kwargs={'allow_rechunk': True},
    )

    regression_ds = xr.Dataset({
        'slope': regression_results[0],
        'intercept': regression_results[1],
        'r_value': regression_results[2],
        'p_value': regression_results[3],
        'std_err': regression_results[4],
    })

    regression_ds.attrs.update({
        'description': 'Linear regression statistics between index and gridded data',
        'index_name': index.name or 'index',
        'gridded_data_name': gridded_data.name or 'gridded_data',
        'coordinate_name': coord_name,
        'method': 'scipy.stats.linregress',
    })

    regression_ds['slope'].attrs.update({'long_name': 'Regression slope'})
    regression_ds['intercept'].attrs.update({'long_name': 'Regression intercept'})
    regression_ds['r_value'].attrs.update({'long_name': 'Pearson correlation coefficient'})
    regression_ds['p_value'].attrs.update({'long_name': 'Two-tailed p-value'})
    regression_ds['std_err'].attrs.update({'long_name': 'Standard error of slope'})

    return regression_ds.compute().copy()


def calculate_correlation(model_name: str,
                          ds1: Dict, ds2: Dict,
                          shift_time: int = 12,
                          data_type: str = 'hpf_data') -> Tuple[Dict, Dict]:
    """
    Lead-lag correlation between two model time series at monthly lags
    from -shift_time to +shift_time. Series are selected over 1955-2009.

    Parameters
    ----------
    model_name : str
    ds1, ds2 : nested dicts {data_type: {model_name: DataArray}}
        ds1 is unshifted (e.g. NTA/AMO); ds2 is shifted (e.g. CTI).
    shift_time : int
        Max lag in months.
    data_type : str
        Key into ds1/ds2 (e.g. 'hpf_data', 'lpf_data').

    Returns
    -------
    (r_values, p_values) : dict keyed by lag in months.
    """
    r_values: Dict = {}
    p_values: Dict = {}

    time_slice = slice('1955', '2009')
    nta_data_ds = ds1[data_type][model_name].sel(time=time_slice)
    cti_data_ds = ds2[data_type][model_name].sel(time=time_slice)
    nta_data_ds = nta_data_ds.isel(time=slice(shift_time, -shift_time))

    for i in range(0, 2 * shift_time + 1):
        if i < 2 * shift_time:
            cti_shifted = cti_data_ds.isel(time=slice(i, -2 * shift_time + i))
        else:
            cti_shifted = cti_data_ds.isel(time=slice(i, None))

        _, _, r_value, p_value, _ = stats.linregress(nta_data_ds.values,
                                                     cti_shifted.values)
        lag = i - shift_time
        r_values[lag] = r_value
        p_values[lag] = p_value

    return r_values, p_values


def calculate_multi_model_mean_correlation(models,
                                           results: Dict,
                                           shift_time: int = 12,
                                           data_type: str = 'hpf_data'):
    """
    Multi-model mean of lead-lag correlation, excluding observational/E3SM runs.

    Excludes models in {'HadISST', 'ERSST', 'E3SM-MMF', 'E3SMv2'}.

    Returns (r_mean, p_mean, std_err) each keyed by lag.
    """
    r_values: Dict = {i: [] for i in range(-shift_time, shift_time + 1)}
    p_values: Dict = {i: [] for i in range(-shift_time, shift_time + 1)}

    exclude = {'HadISST', 'ERSST', 'E3SM-MMF', 'E3SMv2'}
    for model in models:
        if model in exclude:
            continue
        for i in range(-shift_time, shift_time + 1):
            r_values[i].append(results[model][0][i])
            p_values[i].append(results[model][1][i])

    r_mean = {i: np.mean(r_values[i]) for i in range(-shift_time, shift_time + 1)}
    p_mean = {i: np.mean(p_values[i]) for i in range(-shift_time, shift_time + 1)}
    std_err = {i: np.std(r_values[i]) for i in range(-shift_time, shift_time + 1)}

    return r_mean, p_mean, std_err
