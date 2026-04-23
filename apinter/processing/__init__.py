"""Data transformations: anomalies, filters, region extraction."""
from .anomalies import (
    detrend_dim,
    compute_anomalies,
    calculate_anomalies_and_filter,
)
from .filters import lanczos_lowpass
from .regions import wgt_areaave, extract_region

__all__ = [
    "detrend_dim",
    "compute_anomalies",
    "calculate_anomalies_and_filter",
    "lanczos_lowpass",
    "wgt_areaave",
    "extract_region",
]
