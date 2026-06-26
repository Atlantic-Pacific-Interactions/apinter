"""Data transformations: anomalies, filters, region extraction, regridding.

Core helpers:
  - detrend_dim, compute_anomaly, standardize_time_to_month_start  (anomalies.py)
  - lanczos_lowpass, apply_lowpass  (filters.py)
  - wgt_areaave  (regions.py)
  - regrid_to_1deg, regrid_dict_to_1deg  (regrid.py — xesmf-based)
  - load_ocean_mask, apply_ocean_mask  (masks.py)
"""
from .anomalies import detrend_dim, compute_anomaly, standardize_time_to_month_start
from .filters import lanczos_lowpass, apply_lowpass
from .regions import wgt_areaave
from .regrid import regrid_to_1deg, regrid_dict_to_1deg
from .masks import load_ocean_mask, apply_ocean_mask

__all__ = [
    "detrend_dim",
    "compute_anomaly",
    "standardize_time_to_month_start",
    "lanczos_lowpass",
    "apply_lowpass",
    "wgt_areaave",
    "regrid_to_1deg",
    "regrid_dict_to_1deg",
    "load_ocean_mask",
    "apply_ocean_mask",
]
