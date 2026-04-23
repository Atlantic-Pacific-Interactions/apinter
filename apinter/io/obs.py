"""Observational SST loaders (HadISST, ERSST, COBE).

ERA5 reanalysis lives in apinter.io.era5. Ocean reanalysis (ORAS5) lives
in apinter.io.oras5.
"""
import logging
from typing import Dict

import numpy as np
import xarray as xr

from apinter.config import COBE_PATH, ERSST_PATH, HADISST_PATH

logger = logging.getLogger(__name__)


OBS_SST_SOURCES: Dict[str, Dict] = {
    'hadisst': {'path': HADISST_PATH, 'var': 'sst', 'missing': -1000.0},
    'ersst':   {'path': ERSST_PATH,   'var': 'sst', 'missing': -1000.0},
    'cobesst': {'path': COBE_PATH,    'var': 'sst', 'missing': None},
}


def load_obs_sst(source: str = 'hadisst',
                 sim_time: slice = slice('1850', '2023')) -> xr.DataArray:
    """Load an observational SST dataset.

    Parameters
    ----------
    source : {'hadisst', 'ersst', 'cobesst'}
    sim_time : slice
        Time range passed to ``.sel(time=...)``.

    Returns
    -------
    xr.DataArray with dims (time, lat, lon). Missing values set to NaN.
    """
    if source not in OBS_SST_SOURCES:
        raise ValueError(
            f"Unknown source {source!r}. Choices: {sorted(OBS_SST_SOURCES)}"
        )
    spec = OBS_SST_SOURCES[source]

    ds = xr.open_dataset(spec['path'])
    sst = ds[spec['var']].sel(time=sim_time).squeeze()

    if 'latitude' in sst.dims:
        sst = sst.rename({'latitude': 'lat', 'longitude': 'lon'})

    if spec['missing'] is not None:
        sst = sst.where(sst != spec['missing'], np.nan)

    logger.info(f"Loaded {source} SST: shape={sst.shape}")
    return sst
