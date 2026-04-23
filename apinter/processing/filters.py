"""Temporal filters."""
import numpy as np
import xarray as xr
from scipy import ndimage


def lanczos_lowpass(data: xr.DataArray, cutoff_period: float, window: int = None,
                    time_dim: str = 'time') -> xr.DataArray:
    """
    Apply Lanczos low-pass filter to time series data.

    Parameters
    ----------
    data : xr.DataArray
        Time series data. Filtering is applied along the time dimension.
    cutoff_period : float
        Cutoff period in same units as data spacing
        (e.g., 132 for 11-year cutoff with monthly data).
    window : int, optional
        Filter window length. If None, auto-set to 1.5x cutoff_period (odd).
    time_dim : str, optional
        Name of time dimension (default: 'time').

    Returns
    -------
    xr.DataArray
        Low-pass filtered time series.
    """
    time_axis = data.get_axis_num(time_dim)
    input_data = data.values
    coords = data.coords
    dims = data.dims
    attrs = data.attrs
    name = data.name

    if window is None:
        window = int(1.5 * cutoff_period)
        if window % 2 == 0:
            window += 1

    cutoff_freq = 1.0 / cutoff_period
    order = ((window - 1) // 2) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2

    w[n] = 2 * cutoff_freq
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff_freq * k) / (np.pi * k)
    w[n - 1:0:-1] = firstfactor * sigma
    w[n + 1:-1] = firstfactor * sigma

    weights = w[1:-1]
    filtered_values = ndimage.convolve1d(input_data, weights, axis=time_axis,
                                          mode='constant', cval=0.0)

    return xr.DataArray(filtered_values, coords=coords, dims=dims,
                        attrs=attrs, name=name)


def apply_lowpass(data: xr.DataArray,
                  cutoff_period: float,
                  method: str = 'lanczos',
                  time_dim: str = 'time') -> xr.DataArray:
    """
    Dispatch low-pass filter by method.

    Parameters
    ----------
    data : xr.DataArray with a time dim.
    cutoff_period : float
        Time scale in the time-dim spacing units (e.g., months).
    method : {'lanczos', 'running_mean', None, 'none'}
        'lanczos': Lanczos low-pass with `cutoff_period` as cutoff.
        'running_mean': centered rolling mean with window=int(round(cutoff_period)).
        None / 'none': pass through unchanged.
    time_dim : str

    Returns
    -------
    xr.DataArray
        Filtered series (same shape as input).
    """
    if method == 'lanczos':
        return lanczos_lowpass(data, cutoff_period, time_dim=time_dim)
    if method == 'running_mean':
        window = int(round(cutoff_period))
        return data.rolling({time_dim: window}, center=True, min_periods=1).mean()
    if method is None or method == 'none':
        return data
    raise ValueError(
        f"Unknown method {method!r}. Expected 'lanczos', 'running_mean', or None."
    )
