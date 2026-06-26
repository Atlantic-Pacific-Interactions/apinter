"""Land/ocean masking."""
import xarray as xr

LSMASK_PATH = '/pscratch/sd/y/yanxia/DATA/LandMask/lsmask.nc'


def load_ocean_mask(path: str = LSMASK_PATH, sort_lat: bool = True) -> xr.DataArray:
    """Load the NOAA land/sea mask (1=ocean, 0=land) at 1-deg resolution."""
    ds = xr.open_dataset(path)
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    mask = ds['mask']
    if 'time' in mask.dims:
        mask = mask.isel(time=0).drop_vars('time', errors='ignore')
    if sort_lat:
        mask = mask.sortby('lat')
    return mask


def apply_ocean_mask(field: xr.DataArray, mask: xr.DataArray = None,
                      path: str = LSMASK_PATH) -> xr.DataArray:
    """Mask land points in `field` (NaN over land) via nearest-neighbor
    interpolation of the ocean mask onto `field`'s lat/lon grid."""
    if mask is None:
        mask = load_ocean_mask(path=path)
    mask_interp = mask.interp(lat=field.lat, lon=field.lon, method='nearest')
    return field.where(mask_interp == 1)
