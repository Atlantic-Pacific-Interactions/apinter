"""Detrending and anomaly utilities."""
import logging
from typing import Optional, Tuple

import xarray as xr

from .filters import lanczos_lowpass

logger = logging.getLogger(__name__)


def detrend_dim(da: xr.DataArray, dim: str = 'time', deg: int = 1) -> xr.DataArray:
    """Detrend data along a single dimension via polynomial fit."""
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit


def compute_anomalies(data: xr.DataArray,
                      rolling_time: Optional[int] = None
                      ) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute anomalies and normalized anomalies from climatological monthly means.

    Climatology is computed over complete calendar years only.

    Returns
    -------
    (anomaly, normalized_anomaly)
    """
    idx_complete_years = data.time.shape[0] // 12
    clim = data[:idx_complete_years * 12].groupby('time.month').mean('time')
    ano = data.groupby('time.month') - clim

    if rolling_time:
        ano = ano.rolling(time=rolling_time, center=True, min_periods=1).mean()

    ano_norm = ano / ano.std()

    ano.attrs['clim_start'] = data.time[0].values
    ano.attrs['clim_end'] = data.time[:idx_complete_years * 12][-1].values
    ano_norm.attrs['clim_start'] = data.time[0].values
    ano_norm.attrs['clim_end'] = data.time[:idx_complete_years * 12][-1].values

    return ano, ano_norm


def calculate_anomalies_and_filter(data: xr.DataArray,
                                   variable_name: str = "data",
                                   cutoff_period: float = 132,
                                   method: str = 'lanczos') -> xr.DataArray:
    """
    Compute detrended, low-pass-filtered monthly anomalies.

    Pipeline: monthly climatology -> anomalies -> linear detrend -> filter.

    Parameters
    ----------
    data : xr.DataArray
        Input with a 'time' dimension.
    variable_name : str
        Label for log messages.
    cutoff_period : float
        Time scale in months. For 'lanczos' this is the low-pass cutoff;
        for 'running_mean' this is the window length. Default 132 (11 years).
    method : {'lanczos', 'running_mean'}
        Filter method.
    """
    logger.info(f"  Calculating climatology and anomalies for {variable_name}...")
    data_clm = data.groupby('time.month').mean(dim='time')
    data_anm = data.groupby('time.month') - data_clm

    logger.info(f"  Detrending {variable_name}...")
    data_danm = detrend_dim(data_anm, 'time')

    logger.info(f"  Applying {method} filter (cutoff={cutoff_period} months) to {variable_name}...")
    if method == 'lanczos':
        return lanczos_lowpass(data_danm, cutoff_period)
    if method == 'running_mean':
        window = int(round(cutoff_period))
        return data_danm.rolling(time=window, center=True, min_periods=1).mean()
    raise ValueError(f"Unknown method {method!r}. Expected 'lanczos' or 'running_mean'.")
