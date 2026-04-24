"""Mixed-layer heat budget with Reynolds decomposition.

Two backends:
  - Regular lat/lon grid (approximate spherical operators): mld, advection,
    entrainment, surface_flux, ml_tendency.
  - Native NEMO ORCA C-grid (exact metrics from mesh_mask.nc): nemo_grid,
    nemo_mld, nemo_advection, nemo_entrainment. Recommended for ORAS5.

References
----------
Graham, F.S. et al. (2014). Effectiveness of the Bjerknes stability index in
    representing ocean dynamics. Climate Dynamics, 43, 2399-2414.
    — Reynolds decomposition of horizontal / vertical advection.
Stevenson, S. et al. (2017). Spurious precipitation variability in the Last
    Millennium Ensemble. Climate Dynamics.
    — mld utilities: depth-weighted MLD averaging and sub-MLD sampling.
Nnamchi, H.C. et al. (2021). Atlantic-Pacific link to the tropical climate.
    Nature Communications.
    — ml_tendency: partial-cell interpolation for ⟨T⟩ at MLD base.
Paulson, C.A. & Simpson, J.J. (1977). Irradiance measurements in the upper
    ocean. J. Phys. Oceanogr., 7, 952-956.
    — surface_flux: Type I water two-band shortwave penetration.
Madec, G. & the NEMO System Team (2019). NEMO ocean engine. Scientific Notes
    of Climate Modelling Center 27, Institut Pierre-Simon Laplace.
    — nemo_grid: Arakawa C-grid discretisation, metric tensors, mesh_mask.nc.
"""
from .constants import (
    RHO, CP, RE,
    SW_R, SW_D1, SW_D2,
    MLD_MIN,
    CLIM_START, CLIM_END,
)
from .tendency import compute_tendency, compute_anomaly_tendency
from .mld import mldavg_varytime, submld_varytime, botmld_varytime
from .surface_flux import surface_heat_flux
from .advection import advection_ml_rd
from .entrainment import compute_w_from_continuity, vertadv_ml_rd
from .ml_tendency import ml_mean_temperature, ml_tendency
from .nemo_grid import NemoGrid
from .nemo_mld import (
    mldavg_varytime as nemo_mldavg,
    submld_varytime as nemo_submld,
)

__all__ = [
    'RHO', 'CP', 'RE',
    'SW_R', 'SW_D1', 'SW_D2',
    'MLD_MIN',
    'CLIM_START', 'CLIM_END',
    'compute_tendency', 'compute_anomaly_tendency',
    'mldavg_varytime', 'submld_varytime', 'botmld_varytime',
    'surface_heat_flux',
    'advection_ml_rd',
    'compute_w_from_continuity', 'vertadv_ml_rd',
    'ml_mean_temperature', 'ml_tendency',
    'NemoGrid',
    'nemo_mldavg', 'nemo_submld',
]
