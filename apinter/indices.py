"""Climate index calculation (canonical Paper_1 pipeline with Lanczos low-pass).

Pipeline: monthly anomaly -> optional detrend -> area-weighted regional mean ->
low-pass filter (Lanczos by default) -> optional standardization by std.
"""
import logging
from typing import Dict, Optional, Tuple

import xarray as xr

from .processing.anomalies import detrend_dim
from .processing.filters import lanczos_lowpass
from .processing.regions import wgt_areaave

logger = logging.getLogger(__name__)


def compute_anomaly_field(data: xr.DataArray, detrend: bool = True) -> xr.DataArray:
    """
    Gridded monthly anomaly with optional linear detrend.

    Subtracts monthly climatology; if detrend=True, applies linear detrend on time.
    Shape is preserved (no spatial averaging).
    """
    clim = data.groupby('time.month').mean('time')
    anom = data.groupby('time.month') - clim
    if detrend:
        anom = detrend_dim(anom, 'time')
    return anom


def _apply_filter(data: xr.DataArray, cutoff_period: float, method: str) -> xr.DataArray:
    """Dispatch low-pass filter by method."""
    if method == 'lanczos':
        return lanczos_lowpass(data, cutoff_period)
    if method == 'running_mean':
        window = int(round(cutoff_period))
        return data.rolling(time=window, center=True, min_periods=1).mean()
    if method is None or method == 'none':
        return data
    raise ValueError(
        f"Unknown method {method!r}. Expected 'lanczos', 'running_mean', or None."
    )


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
    anom = compute_anomaly_field(sst_data, detrend=detrend)
    regional = wgt_areaave(anom, lat_bounds[0], lat_bounds[1],
                           lon_bounds[0], lon_bounds[1])
    filtered = _apply_filter(regional, cutoff_period, method)
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
    anom = compute_anomaly_field(sst_data, detrend=detrend)

    out: Dict[str, xr.DataArray] = {}
    for name, (lon_bounds, lat_bounds) in regions.items():
        regional = wgt_areaave(anom, lat_bounds[0], lat_bounds[1],
                               lon_bounds[0], lon_bounds[1])
        filtered = _apply_filter(regional, cutoff_period, method)
        if normalize:
            filtered = filtered / filtered.std('time')
        out[name] = filtered
    return out


def calculate_gridded_anomalies(
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

    optional region crop -> monthly anomaly -> optional detrend ->
    optional low-pass filter -> optional standardize.

    Pass cutoff_period=None or method=None to skip the filter step.
    """
    is_regional = lon_bounds is not None and lat_bounds is not None
    if is_regional:
        lat_name = 'lat' if 'lat' in data.dims else 'latitude'
        lon_name = 'lon' if 'lon' in data.dims else 'longitude'
        mask_lon = (data[lon_name] >= lon_bounds[0]) & (data[lon_name] <= lon_bounds[1])
        mask_lat = (data[lat_name] >= lat_bounds[0]) & (data[lat_name] <= lat_bounds[1])
        data = data.where(mask_lon & mask_lat, drop=True)

    anom = compute_anomaly_field(data, detrend=detrend)

    if cutoff_period is None or method is None or method == 'none':
        filtered = anom
    else:
        filtered = _apply_filter(anom, cutoff_period, method)

    if normalize:
        filtered = filtered / filtered.std('time')
    return filtered
