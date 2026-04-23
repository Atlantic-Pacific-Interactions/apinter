"""Statistics: trends, regression, lead-lag, significance testing."""
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
)

__all__ = [
    "global_mean_trend",
    "spatial_trend",
    "zonal_mean_trend",
    "running_trend",
    "assign_season",
    "seasonal_trend",
    "all_seasonal_trends",
    "calculate_regression_vectorize",
    "calculate_correlation",
    "calculate_multi_model_mean_correlation",
    "calculate_lead_lag_regression",
    "calculate_partial_lead_lag_regression",
    "autocorrelation_function",
    "effective_degrees_of_freedom",
]
