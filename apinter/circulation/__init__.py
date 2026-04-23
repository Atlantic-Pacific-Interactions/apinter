"""Atmospheric circulation: Walker and Hadley mass stream function.

Scientific primitives, no plotting. Top-level entry points:

    # Helmholtz decomposition of 2D wind
    get_divergent_u, get_divergent_v, compute_velpot

    # Walker stream function
    calc_walker_sf              # from divergent U, output 10^11 kg/s
    omega_to_streamfunction     # from ω integrated zonally, output 10^10 kg/s

    # Hadley stream function
    calc_streamfunction         # regional, output 10^10 kg/s
    calc_streamfunction_global  # zonal-mean, output 10^10 kg/s

    # Common-grid regridding for MMM overlays
    interp_to_common_lon, interp_to_common_lat

    # Li et al. (2006) ψ/φ minimization solver (Paper_1)
    psi_phi  (submodule — use ``from apinter.circulation.psi_phi import uv2psiphi``)
"""
from .helmholtz import get_divergent_u, get_divergent_v, compute_velpot
from .walker import calc_walker_sf, omega_to_streamfunction
from .hadley import calc_streamfunction, calc_streamfunction_global
from .regrid import interp_to_common_lon, interp_to_common_lat
from . import psi_phi  # expose submodule

__all__ = [
    # helmholtz
    "get_divergent_u",
    "get_divergent_v",
    "compute_velpot",
    # walker
    "calc_walker_sf",
    "omega_to_streamfunction",
    # hadley
    "calc_streamfunction",
    "calc_streamfunction_global",
    # regrid
    "interp_to_common_lon",
    "interp_to_common_lat",
    # submodule
    "psi_phi",
]
