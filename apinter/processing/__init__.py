"""Data transformations: anomalies, filters, region extraction, regridding.

Core helpers:
  - detrend_dim, compute_anomaly  (anomalies.py)
  - lanczos_lowpass, apply_lowpass  (filters.py)
  - wgt_areaave  (regions.py)
  - regrid_to_1deg, regrid_dict_to_1deg  (regrid.py — xesmf-based)
"""
from .anomalies import detrend_dim, compute_anomaly
from .filters import lanczos_lowpass, apply_lowpass
from .regions import wgt_areaave
from .regrid import regrid_to_1deg, regrid_dict_to_1deg

__all__ = [
    "detrend_dim",
    "compute_anomaly",
    "lanczos_lowpass",
    "apply_lowpass",
    "wgt_areaave",
    "regrid_to_1deg",
    "regrid_dict_to_1deg",
]
