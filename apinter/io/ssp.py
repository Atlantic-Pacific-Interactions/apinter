"""SSP-scenario data: historical (1850-2014) + SSP (2015-2100) concatenation.

Generalizes the three per-variable functions in the legacy
`SSP/src/ssp_data_loading.py` (sst / omega / precip) into a single
`load_and_concat(var, ssp, model, ...)`.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import xarray as xr

from apinter.config import CMIP6_DIR, LSMSK_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-variable spec for SSP
# ---------------------------------------------------------------------------

# Historical and SSP branches can use different variable names and directory
# layouts (e.g. SST: historical `ts` in Kelvin with land; SSP `tos` in degC,
# ocean-only). Each entry declares both.
SSP_VARS: Dict[str, Dict] = {
    'sst': {
        'hist_var': 'ts', 'hist_subdir': '1850-2015-atmos', 'hist_file': 'ts.zarr',
        'ssp_var':  'tos', 'ssp_subdir': 'ocean',            'ssp_file':  'tos_1deg_2015-2100.zarr',
        'hist_to_celsius': True, 'hist_sst_filter': True,
        'meridional_mean': None,
    },
    'wap': {
        'hist_var': 'wap', 'hist_subdir': '1850-2015-atmos', 'hist_file': 'wap.zarr',
        'ssp_var':  'wap', 'ssp_subdir': 'atmos',            'ssp_file':  'wap_1deg_2015-2100.zarr',
        'meridional_mean': (-5, 5),   # legacy: 5S-5N mean for omega
    },
    'pr':  {
        'hist_var': 'pr',  'hist_subdir': '1850-2015-atmos', 'hist_file': 'pr.zarr',
        'ssp_var':  'pr',  'ssp_subdir': 'atmos',            'ssp_file':  'pr_1deg_2015-2100.zarr',
    },
    'prc': {
        'hist_var': 'prc', 'hist_subdir': '1850-2015-atmos', 'hist_file': 'prc.zarr',
        'ssp_var':  'prc', 'ssp_subdir': 'atmos',            'ssp_file':  'prc_1deg_2015-2100.zarr',
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_land_mask_cache = None


def _get_land_mask() -> xr.DataArray:
    global _land_mask_cache
    if _land_mask_cache is not None:
        return _land_mask_cache
    ds = xr.open_dataset(LSMSK_PATH)
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon'})
    mask = ds['mask']
    if 'time' in mask.dims:
        mask = mask.isel(time=0)
    _land_mask_cache = mask
    return mask


def _rename_coords(da: xr.DataArray) -> xr.DataArray:
    if 'latitude' in da.dims:
        da = da.rename({'latitude': 'lat'})
    if 'longitude' in da.dims:
        da = da.rename({'longitude': 'lon'})
    if 'plev' in da.dims:
        level_values = (da.plev / 100).round().astype(int)
        da = da.assign_coords(level=(['plev'], level_values.values))
        da = da.swap_dims({'plev': 'level'})
        da = da.drop_vars('plev')
    return da


def _open_branch(path: Path, var: str) -> xr.DataArray:
    ds = xr.open_zarr(str(path))
    da = ds[var].squeeze(drop=True)
    da = _rename_coords(da)
    return da


def _align_grid(ref: xr.DataArray, other: xr.DataArray) -> xr.DataArray:
    """Interpolate `other` onto `ref` lat/lon grid if they differ."""
    if not np.array_equal(ref.lat.values, other.lat.values) or \
       not np.array_equal(ref.lon.values, other.lon.values):
        other = other.interp(lat=ref.lat, lon=ref.lon, method='linear')
    return other


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def get_ssp_models(ssp: str, var: str = 'sst',
                   base_path: Union[str, Path] = CMIP6_DIR) -> List[str]:
    """Sorted list of models with both historical and SSP data for `var`."""
    if var not in SSP_VARS:
        raise ValueError(f"Unknown var {var!r}. Choices: {sorted(SSP_VARS)}")

    spec = SSP_VARS[var]
    base_path = Path(base_path)
    skip = {'logs', 'temp_files', 'tmp_regrid', 'tmp_regrid_ssp', 'historical'}

    models = []
    for d in sorted(base_path.iterdir()):
        if not d.is_dir() or d.name in skip or d.name.startswith('.'):
            continue
        hist_ok = (d / spec['hist_subdir'] / spec['hist_file']).exists()
        ssp_ok = (d / spec['ssp_subdir'] / ssp / spec['ssp_file']).exists()
        if hist_ok and ssp_ok:
            models.append(d.name)
    return sorted(models)


# ---------------------------------------------------------------------------
# Generic concat loader
# ---------------------------------------------------------------------------

def load_and_concat(var: str,
                    ssp: str,
                    model: str,
                    full_time: slice = slice('1850-01-01', '2100-12-31'),
                    base_path: Union[str, Path] = CMIP6_DIR) -> xr.DataArray:
    """Load historical + SSP branches for one variable/model and concatenate.

    Parameters
    ----------
    var : str
        Variable key into ``SSP_VARS`` (e.g. 'sst', 'wap', 'pr', 'prc').
    ssp : str
        SSP scenario (e.g. 'ssp126', 'ssp245', 'ssp370', 'ssp585').
    model : str
        CMIP6 model name.
    full_time : slice
        Time range of the concatenated output.
    base_path : Path-like
        CMIP6 root (default: apinter.config.CMIP6_DIR).

    Returns
    -------
    xr.DataArray
        Continuous series covering historical through SSP.
    """
    if var not in SSP_VARS:
        raise ValueError(f"Unknown var {var!r}. Choices: {sorted(SSP_VARS)}")
    spec = SSP_VARS[var]
    base_path = Path(base_path)

    hist_path = base_path / model / spec['hist_subdir'] / spec['hist_file']
    ssp_path  = base_path / model / spec['ssp_subdir']  / ssp / spec['ssp_file']

    hist = _open_branch(hist_path, spec['hist_var']).sel(
        time=slice('1850-01-01', '2014-12-31')
    )
    ssp_ = _open_branch(ssp_path, spec['ssp_var']).sel(
        time=slice('2015-01-01', '2100-12-31')
    )

    # SST-specific K->C + outlier filter on historical branch
    # (E3SMS/mean_state convention — see apinter.io.cmip6 CMIP6_VARS for rationale)
    if spec.get('hist_to_celsius'):
        hist = hist - 273.15
    if spec.get('hist_sst_filter'):
        hist = hist.where((hist > -10) & (hist < 40))
    elif spec.get('hist_ocean_mask'):
        mask = _get_land_mask()
        if not np.array_equal(mask.lat.values, hist.lat.values):
            mask = mask.interp(lat=hist.lat, lon=hist.lon, method='nearest')
        hist = hist.where(mask == 1)

    hist = hist.load()
    ssp_ = ssp_.load()

    # Optional meridional mean (e.g. 5S-5N for omega)
    if spec.get('meridional_mean') is not None:
        lo, hi = spec['meridional_mean']
        hist = hist.sel(lat=slice(lo, hi)).mean('lat')
        ssp_ = ssp_.sel(lat=slice(lo, hi)).mean('lat')

    # Align vertical levels if 3D (e.g. wap)
    if 'level' in hist.dims and 'level' in ssp_.dims:
        if not np.array_equal(hist.level.values, ssp_.level.values):
            common = np.intersect1d(hist.level.values, ssp_.level.values)
            hist = hist.sel(level=common)
            ssp_ = ssp_.sel(level=common)

    # Align lat/lon if needed
    if 'lat' in hist.dims and 'lat' in ssp_.dims:
        ssp_ = _align_grid(hist, ssp_)
    elif 'lon' in hist.dims and 'lon' in ssp_.dims and not np.array_equal(hist.lon.values, ssp_.lon.values):
        ssp_ = ssp_.interp(lon=hist.lon, method='linear')

    hist['time'] = hist.time.values.astype('datetime64[ns]')
    ssp_['time'] = ssp_.time.values.astype('datetime64[ns]')

    out_name = var
    hist.name = out_name
    ssp_.name = out_name

    combined = xr.concat([hist, ssp_], dim='time').sel(time=full_time)
    logger.info(
        f"  {model}/{ssp}: {var} shape {combined.shape}, "
        f"{str(combined.time.values[0])[:7]} to {str(combined.time.values[-1])[:7]}"
    )
    return combined
