"""Region extraction and area-weighted averaging."""
from typing import Tuple

import numpy as np
import xarray as xr


def wgt_areaave(indat: xr.DataArray,
                latS: float, latN: float,
                lonW: float, lonE: float) -> xr.DataArray:
    """Cosine-latitude weighted area average over a lat/lon box."""
    lat = indat.lat
    lon = indat.lon

    iplat = lat.where((lat >= latS) & (lat <= latN), drop=True)
    iplon = lon.where((lon >= lonW) & (lon <= lonE), drop=True)

    wgt = np.cos(np.deg2rad(lat))
    return indat.sel(lat=iplat, lon=iplon).weighted(wgt).mean(("lon", "lat"), skipna=True)


def extract_region(data: xr.DataArray,
                   lon_bounds: Tuple[float, float],
                   lat_bounds: Tuple[float, float]) -> xr.DataArray:
    """
    Extract a lat/lon region and compute its cosine-weighted areal mean.

    Accepts either 'lat'/'lon' or 'latitude'/'longitude' coordinate names.
    """
    lat_name = 'lat' if 'lat' in data.dims else 'latitude'
    lon_name = 'lon' if 'lon' in data.dims else 'longitude'

    mask_lon = (data[lon_name] >= lon_bounds[0]) & (data[lon_name] <= lon_bounds[1])
    mask_lat = (data[lat_name] >= lat_bounds[0]) & (data[lat_name] <= lat_bounds[1])

    region = data.where(mask_lon & mask_lat, drop=True)
    return region.weighted(np.cos(np.deg2rad(region[lat_name]))).mean((lon_name, lat_name))
