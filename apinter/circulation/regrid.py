"""Interpolate (pressure, longitude) or (pressure, latitude) fields to a common grid.

Two variants are kept because the Walker and Hadley scripts need slightly
different behavior at longitude boundaries:

  - ``interp_to_common_lon`` wraps the periodic longitude coordinate
    before interpolation (for Walker and velocity-potential overlays).
  - ``interp_to_common_lat`` does not wrap (latitude is not periodic).

Ported from ``E3SMS/scripts/circulation/plot_walker_streamfunction_v2.py``
and ``plot_hadley_cell_v2.py``.
"""
import numpy as np
from scipy.interpolate import interp1d

from apinter.config import COMMON_LON, COMMON_PLEV


def interp_to_common_lon(data: np.ndarray,
                         lon: np.ndarray,
                         plev_hpa: np.ndarray,
                         target_lon: np.ndarray = COMMON_LON,
                         target_plev: np.ndarray = COMMON_PLEV) -> np.ndarray:
    """Interpolate (plev, lon) to (target_plev, target_lon) with cyclic-lon wrap.

    Parameters
    ----------
    data : (nplev, nlon) np.ndarray
    lon : (nlon,) np.ndarray — degrees east, monotone.
    plev_hpa : (nplev,) np.ndarray — hPa.
    target_lon : (nlon_out,) np.ndarray
    target_plev : (nplev_out,) np.ndarray
    """
    nplev_out = len(target_plev)
    nlon = len(lon)

    # Interp in pressure first, per column
    tmp = np.full((nplev_out, nlon), np.nan)
    for j in range(nlon):
        col = data[:, j]
        mask = np.isfinite(col)
        if mask.sum() < 2:
            continue
        f = interp1d(plev_hpa[mask], col[mask],
                     bounds_error=False, fill_value=np.nan)
        tmp[:, j] = f(target_plev)

    # Pad for cyclic longitude wrap
    lon_ext = np.concatenate([lon[-1:] - 360, lon, lon[:1] + 360])
    nlon_out = len(target_lon)
    out = np.full((nplev_out, nlon_out), np.nan)
    for k in range(nplev_out):
        row = tmp[k, :]
        row_ext = np.concatenate([row[-1:], row, row[:1]])
        mask = np.isfinite(row_ext)
        if mask.sum() < 2:
            continue
        f = interp1d(lon_ext[mask], row_ext[mask],
                     bounds_error=False, fill_value=np.nan)
        out[k, :] = f(target_lon)
    return out


def interp_to_common_lat(psi: np.ndarray,
                         lat: np.ndarray,
                         plev_hpa: np.ndarray,
                         target_lat: np.ndarray,
                         target_plev: np.ndarray = COMMON_PLEV) -> np.ndarray:
    """Interpolate ψ(plev, lat) to (target_plev, target_lat). No wrapping.

    Parameters
    ----------
    psi : (nplev, nlat) np.ndarray
    lat : (nlat,) np.ndarray — degrees north, monotone.
    plev_hpa : (nplev,) np.ndarray — hPa.
    target_lat : (nlat_out,) np.ndarray
    target_plev : (nplev_out,) np.ndarray
    """
    nplev_out = len(target_plev)
    nlat = len(lat)

    psi_p = np.full((nplev_out, nlat), np.nan)
    for j in range(nlat):
        col = psi[:, j]
        mask = np.isfinite(col)
        if mask.sum() < 2:
            continue
        f = interp1d(plev_hpa[mask], col[mask],
                     bounds_error=False, fill_value=np.nan)
        psi_p[:, j] = f(target_plev)

    nlat_out = len(target_lat)
    psi_out = np.full((nplev_out, nlat_out), np.nan)
    for k in range(nplev_out):
        row = psi_p[k, :]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        f = interp1d(lat[mask], row[mask],
                     bounds_error=False, fill_value=np.nan)
        psi_out[k, :] = f(target_lat)
    return psi_out
