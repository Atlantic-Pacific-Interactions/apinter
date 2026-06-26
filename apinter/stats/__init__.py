"""Statistics: trends, regression, correlation, significance.

Regression and correlation entry points:
  - regression_lags(field, target_index, lags, confounder=None, ...):
      field-on-index regression (simple or partial), concurrent or multi-lag.
      target_index is a single 1D predictor shared by every pixel.
  - pointwise_regression(x_field, y_field, ...):
      grid-cell-local regression — BOTH operands vary spatially; each pixel
      uses only its own (x(t), y(t)) pair (e.g. local SST' vs local precip').
  - correlation_lags(ts1, ts2, max_lag=12):
      1D-on-1D lead-lag Pearson correlation (no internal time selection).
  - mmm_correlation_lags(per_model_results, exclude=None):
      multi-model mean of per-model correlation_lags results.
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
    pointwise_regression,
    correlation_lags,
    mmm_correlation_lags,
)
from .significance import (
    autocorrelation_function,
    effective_degrees_of_freedom,
    autocorrelation_numpy_vectorized,
    calculate_neff_vectorized,
    calculate_neff_pointwise,
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
    # regression + correlation
    "regression_lags",
    "pointwise_regression",
    "correlation_lags",
    "mmm_correlation_lags",
    # significance
    "autocorrelation_function",
    "effective_degrees_of_freedom",
    "autocorrelation_numpy_vectorized",
    "calculate_neff_vectorized",
    "calculate_neff_pointwise",
]
