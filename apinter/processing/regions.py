"""Region extraction and area-weighted averaging."""
import numpy as np
import xarray as xr


def wgt_areaave(indat: xr.DataArray,
                latS: float, latN: float,
                lonW: float, lonE: float) -> xr.DataArray:
    """Cosine-latitude weighted area average over a lat/lon box.

    Requires 'lat' and 'lon' coords (package-wide convention).
    """
    lat = indat.lat
    lon = indat.lon

    iplat = lat.where((lat >= latS) & (lat <= latN), drop=True)
    iplon = lon.where((lon >= lonW) & (lon <= lonE), drop=True)

    wgt = np.cos(np.deg2rad(lat))
    return indat.sel(lat=iplat, lon=iplon).weighted(wgt).mean(("lon", "lat"), skipna=True)
