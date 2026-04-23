"""Climate index calculation (canonical Paper_1 pipeline).

Pipeline: monthly anomaly -> optional detrend -> area-weighted regional mean ->
low-pass filter (Lanczos by default) -> optional standardization by std.
"""
import logging
from typing import Dict, Optional, Tuple

import xarray as xr

from .processing.anomalies import compute_anomaly
from .processing.filters import apply_lowpass
from .processing.regions import wgt_areaave

logger = logging.getLogger(__name__)


def calculate_index(sst_data: xr.DataArray,
                    lon_bounds: Tuple[float, float],
                    lat_bounds: Tuple[float, float],
                    detrend: bool = True,
                    cutoff_period: float = 132,
                    method: str = 'lanczos',
                    normalize: bool = True) -> xr.DataArray:
    """
    Climate index from SST (canonical Paper_1 pipeline).

    anomaly -> optional detrend -> area-weighted regional mean ->
    low-pass filter -> optional standardize.

    Parameters
    ----------
    sst_data : xr.DataArray with dims (time, lat, lon).
    lon_bounds, lat_bounds : (min, max).
    detrend : Linear detrend after anomaly (default True).
    cutoff_period : Filter time scale in months (default 132 = 11 years).
    method : {'lanczos', 'running_mean', None}.
    normalize : Divide by std after filtering (default True).
    """
    logger.info(
        f"Calculating index: lon {lon_bounds}, lat {lat_bounds}, "
        f"method={method}, cutoff={cutoff_period} months"
    )
    anom = compute_anomaly(sst_data, detrend=detrend)
    regional = wgt_areaave(anom, lat_bounds[0], lat_bounds[1],
                           lon_bounds[0], lon_bounds[1])
    filtered = apply_lowpass(regional, cutoff_period, method)
    if normalize:
        filtered = filtered / filtered.std('time')
    return filtered


def calculate_multiple_indices(
    sst_data: xr.DataArray,
    regions: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    detrend: bool = True,
    cutoff_period: float = 132,
    method: str = 'lanczos',
    normalize: bool = True,
) -> Dict[str, xr.DataArray]:
    """
    Multiple regional indices from one SST field.

    Computes the gridded anomaly once, then derives each regional index.

    regions : {name: ((lon_min, lon_max), (lat_min, lat_max))}.
    """
    logger.info(f"Calculating {len(regions)} indices (method={method})")
    anom = compute_anomaly(sst_data, detrend=detrend)

    out: Dict[str, xr.DataArray] = {}
    for name, (lon_bounds, lat_bounds) in regions.items():
        regional = wgt_areaave(anom, lat_bounds[0], lat_bounds[1],
                               lon_bounds[0], lon_bounds[1])
        filtered = apply_lowpass(regional, cutoff_period, method)
        if normalize:
            filtered = filtered / filtered.std('time')
        out[name] = filtered
    return out


def gridded_anomalies(
    data: xr.DataArray,
    lon_bounds: Optional[Tuple[float, float]] = None,
    lat_bounds: Optional[Tuple[float, float]] = None,
    detrend: bool = True,
    cutoff_period: Optional[float] = 132,
    method: str = 'lanczos',
    normalize: bool = True,
) -> xr.DataArray:
    """
    Gridded low-pass-filtered anomaly (keeps spatial dims).

    Pipeline: optional region crop -> monthly anomaly -> optional detrend ->
    optional low-pass filter -> optional standardize.

    Pass cutoff_period=None or method=None to skip the filter step.
    """
    is_regional = lon_bounds is not None and lat_bounds is not None
    if is_regional:
        mask_lon = (data.lon >= lon_bounds[0]) & (data.lon <= lon_bounds[1])
        mask_lat = (data.lat >= lat_bounds[0]) & (data.lat <= lat_bounds[1])
        data = data.where(mask_lon & mask_lat, drop=True)

    anom = compute_anomaly(data, detrend=detrend)

    if cutoff_period is None or method is None or method == 'none':
        filtered = anom
    else:
        filtered = apply_lowpass(anom, cutoff_period, method)

    if normalize:
        filtered = filtered / filtered.std('time')
    return filtered
