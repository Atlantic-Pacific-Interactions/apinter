"""Physical constants for the mixed-layer heat budget.

Values match the Matlab reference code (heat_budget_rd_lme.m) used by
Graham et al. (2014) and Stevenson et al. (2017).
"""

# Seawater reference properties (Graham 2014 / Stevenson 2017)
RHO = 1025.0    # Reference density of seawater [kg/m³]
CP = 3993.0     # Specific heat of seawater [J/(kg·K)]

# Earth geometry (equatorial radius, matches Matlab reference)
RE = 6378.0e3   # [m]

# Shortwave penetration — Paulson & Simpson (1977), Type I water:
#   qpen = qsw * (SW_R * exp(-H/SW_D1) + (1-SW_R) * exp(-H/SW_D2))
SW_R = 0.58     # fraction for the visible band
SW_D1 = 0.35    # attenuation depth for visible [m]
SW_D2 = 23.0    # attenuation depth for UV/blue [m]

# Project baseline monthly climatology period
CLIM_START = 1981
CLIM_END = 2010

# Numerical guard against division-by-zero in per-MLD normalisations
MLD_MIN = 1.0   # [m]
