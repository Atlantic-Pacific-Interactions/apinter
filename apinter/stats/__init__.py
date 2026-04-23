"""Statistics: trends, regression, correlation, significance.

Canonical regression entry point: `regression_lags` — simple or partial
(with a confounder), concurrent or multi-lag, with Bretherton-effective-DoF
significance. Use it for all field-on-index regression.

For 1D-on-1D lead-lag correlation between two time series, use
`calculate_correlation` / `calculate_multi_model_mean_correlation`.
"""
from .trends import (
    global_mean_trend,
    spatial_trend,
    zonal_mean_trend,
    running_trend,
    assign_season,
    seasonal_trend,
    all_seasonal_trends,
)
from .regression import (
    regression_lags,
    calculate_regression_vectorize,
    calculate_correlation,
    calculate_multi_model_mean_correlation,
)
from .significance import (
    autocorrelation_function,
    effective_degrees_of_freedom,
    autocorrelation_numpy_vectorized,
    calculate_neff_vectorized,
)

__all__ = [
    # trends
    "global_mean_trend",
    "spatial_trend",
    "zonal_mean_trend",
    "running_trend",
    "assign_season",
    "seasonal_trend",
    "all_seasonal_trends",
    # regression + correlation (all field/time-series regression lives here)
    "regression_lags",
    "calculate_regression_vectorize",
    "calculate_correlation",
    "calculate_multi_model_mean_correlation",
    # significance
    "autocorrelation_function",
    "effective_degrees_of_freedom",
    "autocorrelation_numpy_vectorized",
    "calculate_neff_vectorized",
]
