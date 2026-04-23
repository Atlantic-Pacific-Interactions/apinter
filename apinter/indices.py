"""Climate index calculation (TAMV, TPDV, AMO, ENSO, etc.)."""
import logging
from typing import Dict, Optional, Tuple

import xarray as xr

from .processing.anomalies import compute_anomalies, detrend_dim
from .processing.regions import extract_region

logger = logging.getLogger(__name__)


def calculate_index(sst_data: xr.DataArray,
                    lon_bounds: Tuple[float, float],
                    lat_bounds: Tuple[float, float],
                    detrend: bool = True,
                    rolling_time: Optional[int] = None
                    ) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute a climate index from SST for one lat/lon box.

    Pipeline: extract region (area-weighted mean) -> optional detrend ->
    monthly-climatology anomalies -> optional rolling mean -> normalize.

    Returns (index_anomaly, index_normalized).
    """
    logger.info(f"Calculating index for region: lon {lon_bounds}, lat {lat_bounds}")
    logger.info(f"Detrend: {detrend}, Rolling time: {rolling_time}")

    sst_region = extract_region(sst_data, lon_bounds, lat_bounds)

    if detrend:
        logger.info("Applying detrending")
        sst_region = detrend_dim(sst_region, dim='time')

    logger.info("Computing anomalies")
    return compute_anomalies(sst_region, rolling_time)


def calculate_multiple_indices(sst_data: xr.DataArray,
                               regions: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
                               detrend: bool = True,
                               rolling_time: Optional[int] = None
                               ) -> Tuple[Dict[str, xr.DataArray], Dict[str, xr.DataArray]]:
    """
    Compute multiple regional indices from one SST field.

    regions: {name: ((lon_min, lon_max), (lat_min, lat_max))}.
    Returns (anomalies, normalized_anomalies) as dicts keyed by region name.
    """
    logger.info(f"Calculating indices for {len(regions)} regions")

    anomalies: Dict[str, xr.DataArray] = {}
    normalized: Dict[str, xr.DataArray] = {}

    for name, (lon_bounds, lat_bounds) in regions.items():
        logger.info(f"Processing region: {name}")
        sst_region = extract_region(sst_data, lon_bounds, lat_bounds)
        if detrend:
            sst_region = detrend_dim(sst_region, dim='time')
        anomalies[name], normalized[name] = compute_anomalies(sst_region, rolling_time)

    return anomalies, normalized


def calculate_gridded_anomalies(data: xr.DataArray,
                                lon_bounds: Optional[Tuple[float, float]] = None,
                                lat_bounds: Optional[Tuple[float, float]] = None,
                                detrend: bool = True,
                                rolling_time: Optional[int] = None
                                ) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Gridded anomalies (keeps spatial dims) with optional regional subset + detrend.

    Returns (anomaly, normalized).
    """
    is_regional = lon_bounds is not None and lat_bounds is not None
    region_info = f"lon {lon_bounds}, lat {lat_bounds}" if is_regional else "global"

    logger.info(f"Calculating gridded anomalies for {region_info}")
    logger.info(f"Input data shape: {data.shape}")

    if is_regional:
        lat_name = 'lat' if 'lat' in data.dims else 'latitude'
        lon_name = 'lon' if 'lon' in data.dims else 'longitude'
        mask_lon = (data[lon_name] >= lon_bounds[0]) & (data[lon_name] <= lon_bounds[1])
        mask_lat = (data[lat_name] >= lat_bounds[0]) & (data[lat_name] <= lat_bounds[1])
        data = data.where(mask_lon & mask_lat, drop=True)

    if detrend:
        data = detrend_dim(data, dim='time')

    return compute_anomalies(data, rolling_time)
