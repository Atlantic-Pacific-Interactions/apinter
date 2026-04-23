"""Data transformations: anomalies, filters, region extraction.

Core helpers:
  - detrend_dim, compute_anomaly  (anomalies.py)
  - lanczos_lowpass, apply_lowpass  (filters.py)
  - wgt_areaave  (regions.py)
"""
from .anomalies import detrend_dim, compute_anomaly
from .filters import lanczos_lowpass, apply_lowpass
from .regions import wgt_areaave

__all__ = [
    "detrend_dim",
    "compute_anomaly",
    "lanczos_lowpass",
    "apply_lowpass",
    "wgt_areaave",
]
