"""Walker-circulation mass streamfunction.

Two canonical constructions:

  - From equatorial divergent U (Yu & Zwiers 2010 / Ma & Zhou 2016 convention):
        ψ(λ, p) = (2π a cosφ₀ / g) ∫_0^p u_D(λ, p') dp'
    Output in units of 10^11 kg/s.

  - From ω integrated zonally (legacy Paper_1 / E3SMS alternative):
        ψ(λ, p) = (a² cos²φ₀ Δφ / g) ∫_0^λ ω(λ', p) dλ'
    Output in units of 10^10 kg/s.

Ported from ``E3SMS/scripts/circulation/plot_walker_streamfunction_v2.py``
and ``plot_walker_circulation.py``.
"""
from typing import Tuple

import numpy as np

from apinter.config import EARTH_RADIUS as _A, GRAVITY as _G


def calc_walker_sf(ud_eq: np.ndarray,
                   plev_hpa: np.ndarray,
                   phi0_deg: float = 0.0) -> np.ndarray:
    """Walker stream function from equatorial divergent u.

    ψ(λ, p) = (2π a cosφ₀ / g) ∫_0^p u_D dp'

    Parameters
    ----------
    ud_eq : (nplev, nlon) np.ndarray
        Divergent u averaged over an equatorial band, in m/s.
    plev_hpa : (nplev,) np.ndarray
        Pressure levels in hPa. Not required to be sorted.
    phi0_deg : float
        Band-center latitude in degrees (default 0 = equator).

    Returns
    -------
    psi : (nplev, nlon) np.ndarray
        Stream function in units of 10^11 kg/s, aligned with the input
        ``plev_hpa`` ordering.
    """
    cosphi = np.cos(np.deg2rad(phi0_deg))

    sort_idx = np.argsort(plev_hpa)
    plev_sorted = plev_hpa[sort_idx] * 100  # hPa -> Pa
    ud_sorted = ud_eq[sort_idx, :]

    # Fill NaN with 0 for integration (e.g. levels below the surface).
    ud_sorted = np.where(np.isnan(ud_sorted), 0.0, ud_sorted)

    nplev = len(plev_sorted)
    psi = np.zeros_like(ud_sorted)
    for k in range(1, nplev):
        dp = plev_sorted[k] - plev_sorted[k - 1]
        psi[k, :] = psi[k - 1, :] + ud_sorted[k, :] * dp

    psi = (2 * np.pi * _A * cosphi / _G) * psi / 1e11

    psi_out = np.empty_like(psi)
    psi_out[sort_idx, :] = psi
    return psi_out


def omega_to_streamfunction(omega: np.ndarray,
                            lon: np.ndarray,
                            plev_hpa: np.ndarray,
                            phi0_deg: float = 0.0,
                            band_half_width_deg: float = 5.0) -> np.ndarray:
    """Walker stream function from ω integrated zonally.

    ψ(λ, p) = (a² cos²φ₀ Δφ / g) ∫_0^λ ω(λ', p) dλ'

    Parameters
    ----------
    omega : (nplev, nlon) np.ndarray
        Equatorial-band-mean vertical velocity in Pa/s.
    lon : (nlon,) np.ndarray
        Longitudes in degrees.
    plev_hpa : (nplev,) np.ndarray
        Pressure levels in hPa (not used directly; returned untouched).
    phi0_deg : float
        Band-center latitude in degrees.
    band_half_width_deg : float
        Half-width of the equatorial averaging band in degrees. ``Δφ`` is
        twice this, in radians.

    Returns
    -------
    psi : (nplev, nlon) np.ndarray in units of 10^10 kg/s.
    """
    cosphi = np.cos(np.deg2rad(phi0_deg))
    dphi = 2.0 * np.deg2rad(band_half_width_deg)

    nplev, nlon = omega.shape
    psi = np.zeros_like(omega, dtype=float)
    for j in range(1, nlon):
        dl = np.deg2rad(lon[j] - lon[j - 1])
        # Trapezoidal step in longitude
        omega_avg = 0.5 * (omega[:, j - 1] + omega[:, j])
        psi[:, j] = psi[:, j - 1] + omega_avg * dl

    scale = _A ** 2 * cosphi ** 2 * dphi / _G
    return psi * scale / 1e10
