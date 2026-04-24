"""Surface heat flux term for the mixed-layer heat budget.

sfcflx = (Qnet - Qpen) / (ρ cp H),
where Qpen is the shortwave that penetrates below the ML via the
Paulson & Simpson (1977) Type I water two-band formula:
    Qpen = Qsw * (SW_R * exp(-H/SW_D1) + (1 - SW_R) * exp(-H/SW_D2))

Port of Matlab heat_budget_rd_lme.m lines 98-105.
"""
import numpy as np
import xarray as xr

from .constants import RHO, CP, SW_R, SW_D1, SW_D2, MLD_MIN


def surface_heat_flux(qnet, qsw, mld):
    """Surface-flux contribution to the ML temperature tendency (°C/s).

    Parameters
    ----------
    qnet : xr.DataArray (time, lat, lon)
        Net surface heat flux (W/m²), positive into ocean.
    qsw : xr.DataArray (time, lat, lon)
        Net shortwave at the surface (W/m²).
    mld : xr.DataArray (time, lat, lon)
        Mixed layer depth (m).
    """
    mld_invalid = (mld <= 0) | np.isnan(mld)
    mld_safe = mld.clip(min=MLD_MIN)

    qpen = qsw * (SW_R * np.exp(-mld_safe / SW_D1)
                  + (1 - SW_R) * np.exp(-mld_safe / SW_D2))

    sfcflx = (qnet - qpen) / (RHO * CP * mld_safe)
    return xr.where(mld_invalid, np.nan, sfcflx)
