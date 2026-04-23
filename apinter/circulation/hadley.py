"""Hadley-circulation meridional mass streamfunction.

Two constructions:

  - Regional sector (Atlantic, Pacific, or user-chosen longitude box):
        ψ(φ, p) = (Δλ a cosφ / g) ∫_0^p v_χ dp'
    where v_χ is the divergent meridional wind averaged over the sector.
    Output in units of 10^10 kg/s.

  - Zonal-mean (global) Hadley:
        ψ(φ, p) = (2π a cosφ / g) ∫_0^p [v] dp'
    Output in units of 10^10 kg/s.

Ported from ``E3SMS/scripts/circulation/plot_hadley_cell_v2.py`` (regional)
and ``plot_hadley_cell_global.py`` (global).
"""
import numpy as np

from apinter.config import EARTH_RADIUS as _A, GRAVITY as _G


def calc_streamfunction(vd_sector: np.ndarray,
                        lat: np.ndarray,
                        plev_pa: np.ndarray,
                        lon_w: float,
                        lon_e: float) -> np.ndarray:
    """Regional Hadley cell stream function from sector-mean divergent v.

    Parameters
    ----------
    vd_sector : (nplev, nlat) np.ndarray
        Divergent v, longitude-averaged over the sector [lon_w, lon_e], m/s.
    lat : (nlat,) np.ndarray
        Latitudes in degrees.
    plev_pa : (nplev,) np.ndarray
        Pressure levels in Pa (not hPa — matches notebook v2 convention).
    lon_w, lon_e : float
        Sector bounds in degrees. Δλ = lon_e - lon_w (in radians).

    Returns
    -------
    psi : (nplev, nlat) np.ndarray in units of 10^10 kg/s, aligned with
    the input ``plev_pa`` ordering.
    """
    coslat = np.cos(np.deg2rad(lat))
    delta_lambda = np.deg2rad(lon_e - lon_w)

    sort_idx = np.argsort(plev_pa)
    plev_sorted = plev_pa[sort_idx]
    v_sorted = vd_sector[sort_idx, :]
    v_sorted = np.where(np.isnan(v_sorted), 0.0, v_sorted)

    nplev = len(plev_sorted)
    psi = np.zeros_like(v_sorted)
    for k in range(1, nplev):
        dp = plev_sorted[k] - plev_sorted[k - 1]
        psi[k, :] = psi[k - 1, :] + v_sorted[k, :] * dp

    psi = (delta_lambda * _A / _G) * coslat[None, :] * psi
    psi = psi / 1e10

    psi_out = np.empty_like(psi)
    psi_out[sort_idx, :] = psi
    return psi_out


def calc_streamfunction_global(v_mean: np.ndarray,
                               lat: np.ndarray,
                               plev_pa: np.ndarray) -> np.ndarray:
    """Global (zonal-mean) Hadley cell stream function.

    Parameters
    ----------
    v_mean : (nplev, nlat) np.ndarray
        Global zonal-mean v (after time mean), m/s.
    lat : (nlat,) np.ndarray
        Latitudes in degrees.
    plev_pa : (nplev,) np.ndarray
        Pressure levels in Pa.

    Returns
    -------
    psi : (nplev, nlat) np.ndarray in units of 10^10 kg/s.
    """
    coslat = np.cos(np.deg2rad(lat))

    sort_idx = np.argsort(plev_pa)
    plev_sorted = plev_pa[sort_idx]
    v_sorted = v_mean[sort_idx, :]

    nplev = len(plev_sorted)
    psi = np.zeros_like(v_sorted)
    for k in range(1, nplev):
        dp = plev_sorted[k] - plev_sorted[k - 1]
        psi[k, :] = psi[k - 1, :] + v_sorted[k, :] * dp

    psi = (2 * np.pi * _A / _G) * coslat[None, :] * psi
    psi = psi / 1e10

    psi_out = np.empty_like(psi)
    psi_out[sort_idx, :] = psi
    return psi_out
