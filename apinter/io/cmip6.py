"""Generic CMIP6 multi-model loader.

Replaces the 9 per-variable `load_cmip6_<var>` functions in the legacy
`Paper_2/src/load_cmip6.py` with one dispatcher plus a spec dict.

Usage
-----
    from apinter.io import load_cmip6

    sst = load_cmip6('ts')                             # Kelvin->Celsius + ocean mask
    omega = load_cmip6('wap', sim_time=slice('1980', '2014'))
    ua_850 = load_cmip6('ua', level=850)               # pressure-level slice

Each call returns ``dict[model_name, xr.DataArray]`` with `lat`, `lon`
(and `level`, if 3D) coords. Legacy `load_cmip6_*` names remain as thin
compatibility wrappers.
"""
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import xarray as xr

from apinter.config import CMIP6_DIR, LSMSK_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-variable spec
# ---------------------------------------------------------------------------

# Each entry describes where the data live and how to post-process.
#   subdir     : model-relative directory containing the file
#   ext        : 'zarr' or 'nc'
#   to_celsius : subtract 273.15 from the values (atmospheric temperature)
#   sst_filter : keep only (-10 < value < 40) — E3SMS/mean_state convention
#                for `ts`: removes sea-ice surfaces and regridding artifacts
#                (e.g. spurious 0-K cells) that survive land masking.
#                Complements ocean_mask below, does not replace it.
#   ocean_mask : apply the explicit land-sea mask. Required for `ts` (CLAUDE.md
#                hard rule: land contaminates SST) and for natively ocean-only
#                variables like `zos`.
CMIP6_VARS: Dict[str, Dict] = {
    'ts':     {'subdir': '1850-2015-atmos', 'ext': 'zarr',
               'to_celsius': True, 'sst_filter': True, 'ocean_mask': True},
    'wap':    {'subdir': '1850-2015-atmos', 'ext': 'zarr'},
    'zg':     {'subdir': '1850-2015-atmos', 'ext': 'zarr'},
    'psl':    {'subdir': '1850-2015-atmos', 'ext': 'zarr'},
    'pr':     {'subdir': '1850-2015-atmos', 'ext': 'zarr'},
    'zos':    {'subdir': '1850-2015-atmos', 'ext': 'zarr',
               'ocean_mask': True},
    'ua':     {'subdir': '1850-2015-atmos', 'ext': 'nc'},
    'va':     {'subdir': '1850-2015-atmos', 'ext': 'nc'},
    'ta':     {'subdir': '1850-2015-atmos', 'ext': 'nc'},
    'thetao': {'subdir': 'ocean',           'ext': 'zarr'},
    'tauu':   {'subdir': 'atmos',           'ext': 'zarr'},
    'hus':    {'subdir': '1850-2015-atmos', 'ext': 'zarr'},
    'huss':   {'subdir': '1850-2015-atmos', 'ext': 'zarr'},
    'sfcWind': {'subdir': '1850-2015-atmos', 'ext': 'zarr'},
    'hfls':   {'subdir': '1850-2015-atmos', 'ext': 'zarr'},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_land_mask_cache = None


def _get_land_mask() -> xr.DataArray:
    """Load and cache the land-sea mask (ocean=1, land=0), as lat/lon."""
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
    """Standardize coord names to short form: latitude/longitude -> lat/lon.
    Also converts plev (Pa) -> level (hPa) as the pressure coord."""
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


def get_cmip6_models(base_path: Union[str, Path] = CMIP6_DIR) -> List[str]:
    """Return sorted list of model directories under `base_path`."""
    base_path = Path(base_path)
    if not base_path.exists():
        return []
    skip = {'logs', 'temp_files', 'tmp_regrid', 'tmp_regrid_ssp', 'historical'}
    return sorted(
        p.name for p in base_path.iterdir()
        if p.is_dir() and p.name not in skip and not p.name.startswith('.')
    )

# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------

def load_cmip6(var: str,
               sim_time: slice = slice('1980-01-01', '2014-12-31'),
               models: Optional[Iterable[str]] = None,
               level: Optional[int] = None,
               base_path: Union[str, Path] = CMIP6_DIR,
               ) -> Dict[str, xr.DataArray]:
    """Load a single variable across CMIP6 models.

    Parameters
    ----------
    var : str
        Variable name; must be a key of ``CMIP6_VARS``.
    sim_time : slice
        Time range passed to ``.sel(time=...)`` (default 1980-2014).
    models : iterable of str, optional
        Subset of models to load. Default: all models under ``base_path``.
    level : int, optional
        For 3D atmospheric variables (``ua``, ``va``, ``wap``, ``zg``), select
        a single pressure level in hPa after renaming plev->level.
    base_path : Path-like, optional
        CMIP6 root directory (default: ``apinter.config.CMIP6_DIR``).

    Returns
    -------
    dict[model_name, xr.DataArray]
        Models with missing files are silently skipped.
    """
    if var not in CMIP6_VARS:
        raise ValueError(
            f"Unknown variable {var!r}. Choices: {sorted(CMIP6_VARS)}"
        )

    spec = CMIP6_VARS[var]
    base_path = Path(base_path)
    model_list = list(models) if models is not None else get_cmip6_models(base_path)

    need_mask = spec.get('ocean_mask', False)
    mask = _get_land_mask() if need_mask else None

    out: Dict[str, xr.DataArray] = {}
    for model in model_list:
        path = base_path / model / spec['subdir'] / f"{var}.{spec['ext']}"
        if not path.exists():
            logger.warning(f"{var}: {path} not found for model {model}")
            continue

        try:
            if spec['ext'] == 'zarr':
                ds = xr.open_zarr(str(path))
            else:
                ds = xr.open_dataset(path)

            da = ds[var].squeeze(drop=True).sel(time=sim_time)
            da = _rename_coords(da)
            # Some models (e.g. GISS) use extreme fill values (~1e27) instead
            # of NaN; mask before any arithmetic so they can't survive as
            # finite-but-wrong values downstream.
            da = da.where(np.abs(da) < 1e10)

            if level is not None and 'level' in da.dims:
                da = da.sel(level=level)

            if spec.get('to_celsius'):
                da = da - 273.15
            if spec.get('sst_filter'):
                # E3SMS/mean_state convention: removes regridding artifacts
                # (0-K cells) and sea-ice surface temps that a land mask
                # alone wouldn't catch. Does NOT replace the land mask below
                # (CLAUDE.md hard rule: land contaminates SST — many land
                # cells fall inside -10..40 C too).
                da = da.where((da > -10) & (da < 40))
            if need_mask:
                da = da.where(mask == 1)

            out[model] = da
            logger.info(f"Loaded {var} for {model} (shape={da.shape})")
        except Exception as e:
            logger.exception(f"Error loading {var} for {model}: {e}")
            continue

    return out


# ---------------------------------------------------------------------------
# Backward-compatible wrappers
# ---------------------------------------------------------------------------

def load_cmip6_sst(**kw):    return load_cmip6('ts',     **kw)
def load_cmip6_omega(**kw):  return load_cmip6('wap',    **kw)
def load_cmip6_zg(**kw):     return load_cmip6('zg',     **kw)
def load_cmip6_psl(**kw):    return load_cmip6('psl',    **kw)
def load_cmip6_pr(**kw):     return load_cmip6('pr',     **kw)
def load_cmip6_zos(**kw):    return load_cmip6('zos',    **kw)
def load_cmip6_thetao(**kw): return load_cmip6('thetao', **kw)
def load_cmip6_tauu(**kw):   return load_cmip6('tauu',   **kw)


def load_cmip6_wind(target_level: int = 850, **kw) -> Dict[str, Dict[str, xr.DataArray]]:
    """Legacy-compatible wind loader returning ``{model: {'ua': da, 'va': da}}``.

    Prefer ``load_cmip6('ua', level=...)`` / ``load_cmip6('va', level=...)`` for new code.
    """
    ua = load_cmip6('ua', level=target_level, **kw)
    va = load_cmip6('va', level=target_level, **kw)
    models = set(ua) & set(va)
    return {m: {'ua': ua[m], 'va': va[m]} for m in sorted(models)}
