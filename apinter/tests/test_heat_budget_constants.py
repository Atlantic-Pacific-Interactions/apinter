"""Sanity checks on heat-budget physical constants."""
import apinter.config as cfg
from apinter.heat_budget import (
    RHO, CP, RE, SW_R, SW_D1, SW_D2, MLD_MIN,
    CLIM_START, CLIM_END,
)


def test_physical_constants_in_expected_ranges():
    # Seawater reference density and heat capacity (Graham 2014 / Matlab ref)
    assert 1020.0 <= RHO <= 1030.0, f"ρ = {RHO} not a seawater density"
    assert 3900.0 <= CP <= 4100.0, f"cp = {CP} not a seawater heat capacity"
    # Earth radius in metres
    assert 6.0e6 <= RE <= 6.5e6, f"RE = {RE} not a plausible Earth radius"
    # Paulson & Simpson (1977) Type I water coefficients
    assert 0.0 < SW_R < 1.0
    assert SW_D1 < SW_D2
    assert SW_D1 > 0.0 and SW_D2 > 0.0
    # MLD safety guard must be a small positive depth
    assert 0.0 < MLD_MIN < 5.0


def test_climatology_period_default():
    # Project convention: 1981-2010 monthly climatology
    assert CLIM_START == 1981
    assert CLIM_END == 2010
    assert CLIM_START < CLIM_END


def test_config_has_oras5_mesh_path():
    # Spec: apinter.config must expose the mesh_mask path for NemoGrid().
    from apinter.config import ORAS5_MESH_PATH, ORAS5_DIR
    assert ORAS5_MESH_PATH == ORAS5_DIR / "mesh_mask.nc"
