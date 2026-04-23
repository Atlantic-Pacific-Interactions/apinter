"""Statistics: trends, regression, lead-lag, significance testing.

Canonical regression entry point: `regression_lags` — simple or partial,
one or many lags, with Bretherton-effective-DoF significance.
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
from .leadlag import (
    calculate_lead_lag_regression,
    calculate_partial_lead_lag_regression,
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
    # regression (canonical first)
    "regression_lags",
    "calculate_regression_vectorize",
    "calculate_correlation",
    "calculate_multi_model_mean_correlation",
    # leadlag (legacy shims)
    "calculate_lead_lag_regression",
    "calculate_partial_lead_lag_regression",
    # significance
    "autocorrelation_function",
    "effective_degrees_of_freedom",
    "autocorrelation_numpy_vectorized",
    "calculate_neff_vectorized",
]
