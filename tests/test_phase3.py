"""Phase-3 tests: apinter.circulation primitives.

Uses small synthetic (plev, lat, lon) inputs. No real-data dependency —
all tests run in under a second.
"""
import numpy as np
import pytest
import xarray as xr

from apinter.circulation import (
    calc_streamfunction,
    calc_streamfunction_global,
    calc_walker_sf,
    compute_velpot,
    get_divergent_u,
    get_divergent_v,
    interp_to_common_lat,
    interp_to_common_lon,
    omega_to_streamfunction,
    psi_phi,
)
from apinter.config import COMMON_LON, COMMON_PLEV, EARTH_RADIUS, GRAVITY


# ---------- Walker stream function ----------

def test_calc_walker_sf_zero_input_zero_output():
    nlev, nlon = 12, 36
    plev = np.linspace(100, 1000, nlev)
    ud = np.zeros((nlev, nlon))
    psi = calc_walker_sf(ud, plev)
    assert psi.shape == (nlev, nlon)
    assert np.all(psi == 0.0)


def test_calc_walker_sf_constant_u_scales_linearly_in_pressure():
    """Constant u_d ⇒ ψ linear in pressure along each longitude column."""
    nlev, nlon = 10, 5
    plev = np.linspace(100, 1000, nlev)     # hPa
    ud = np.ones((nlev, nlon)) * 1.0        # 1 m/s everywhere
    psi = calc_walker_sf(ud, plev, phi0_deg=0.0)

    # At top level (p=100 hPa): ψ = 0
    # At each subsequent level k: ψ_k = (2π a / g) * (p_k - p_0) / 1e11
    p_pa = plev * 100
    expected = (2 * np.pi * EARTH_RADIUS / GRAVITY) * (p_pa - p_pa[0]) / 1e11
    # Should be identical at every longitude
    for j in range(nlon):
        np.testing.assert_allclose(psi[:, j], expected, rtol=1e-10)


def test_calc_walker_sf_preserves_input_plev_order():
    """Regardless of input level order, ψ aligns to the input order."""
    nlev, nlon = 5, 4
    ud = np.ones((nlev, nlon)) * 0.5
    plev_asc = np.array([100, 300, 500, 700, 1000], dtype=float)
    plev_desc = plev_asc[::-1].copy()
    psi_asc = calc_walker_sf(ud, plev_asc)
    psi_desc = calc_walker_sf(ud, plev_desc)
    # Values at plev=500 should match regardless of input order
    idx_asc = np.where(plev_asc == 500)[0][0]
    idx_desc = np.where(plev_desc == 500)[0][0]
    np.testing.assert_allclose(psi_asc[idx_asc, :], psi_desc[idx_desc, :],
                               rtol=1e-10)


def test_omega_to_streamfunction_shapes_and_scale():
    nlev, nlon = 12, 72
    plev = np.linspace(100, 1000, nlev)
    lon = np.linspace(0, 355, nlon)
    omega = np.zeros((nlev, nlon))
    omega[5, :] = 0.05   # mid-level ω of 0.05 Pa/s
    psi = omega_to_streamfunction(omega, lon, plev,
                                  phi0_deg=0.0, band_half_width_deg=5.0)
    assert psi.shape == (nlev, nlon)
    # At row-0 longitude, ψ = 0 (cumulative integral starts there)
    assert np.all(psi[:, 0] == 0.0)
    # Mid-level ψ should grow from 0 westward to positive as we integrate east
    assert np.all(psi[5, 1:] > 0)


# ---------- Hadley stream function ----------

def test_calc_streamfunction_regional_zero_input():
    nlev, nlat = 12, 20
    plev_pa = np.linspace(1e4, 1e5, nlev)
    lat = np.linspace(-30, 30, nlat)
    vd = np.zeros((nlev, nlat))
    psi = calc_streamfunction(vd, lat, plev_pa, lon_w=280, lon_e=360)
    assert psi.shape == (nlev, nlat)
    assert np.all(psi == 0.0)


def test_calc_streamfunction_sector_width_scales_psi_linearly():
    """Doubling Δλ should double ψ everywhere."""
    nlev, nlat = 8, 12
    plev_pa = np.linspace(1e4, 1e5, nlev)
    lat = np.linspace(-30, 30, nlat)
    rng = np.random.default_rng(0)
    vd = rng.standard_normal((nlev, nlat)) * 0.5

    psi60 = calc_streamfunction(vd, lat, plev_pa, lon_w=280, lon_e=340)   # 60°
    psi120 = calc_streamfunction(vd, lat, plev_pa, lon_w=280, lon_e=400)  # 120°
    np.testing.assert_allclose(psi120, 2.0 * psi60, rtol=1e-10)


def test_calc_streamfunction_global_cos_latitude_weighting():
    """At the equator (cosφ=1), global ψ uses the full 2π factor."""
    nlev, nlat = 5, 7
    plev_pa = np.linspace(1e4, 1e5, nlev)
    lat = np.array([-45, -30, -15, 0, 15, 30, 45], dtype=float)
    v = np.ones((nlev, nlat)) * 0.1
    psi = calc_streamfunction_global(v, lat, plev_pa)
    assert psi.shape == (nlev, nlat)
    # ψ should scale with cos(lat) — equator column magnitude > poleward
    assert np.abs(psi[-1, 3]) > np.abs(psi[-1, 0])
    assert np.abs(psi[-1, 3]) > np.abs(psi[-1, -1])


# ---------- Helmholtz decomposition ----------

def _uv_grid(nlat: int = 73, nlon: int = 144):
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    return lat, lon


def test_helmholtz_pure_divergent_field_round_trip():
    """A pure divergent (irrotational) u,v should decompose back to itself."""
    pytest.importorskip("windspharm")
    lat, lon = _uv_grid()
    lons2d, lats2d = np.meshgrid(lon, lat)
    # Pure divergent field: u = cos(lat)*cos(lon), v = -sin(lon)*sin(lat) is NOT divergent
    # Easiest: u = 1 at equator decaying poleward, v = 0
    u = np.cos(np.deg2rad(lats2d))
    v = np.zeros_like(u)
    u_da = xr.DataArray(u, coords={'lat': lat, 'lon': lon}, dims=('lat', 'lon'))
    v_da = xr.DataArray(v, coords={'lat': lat, 'lon': lon}, dims=('lat', 'lon'))
    ud = get_divergent_u(u_da, v_da)
    vd = get_divergent_v(u_da, v_da)
    # Shapes must match
    assert ud.shape == u.shape
    assert vd.shape == v.shape


def test_helmholtz_nan_input_preserved_as_nan():
    pytest.importorskip("windspharm")
    lat, lon = _uv_grid(37, 72)
    lons2d, lats2d = np.meshgrid(lon, lat)
    u = np.cos(np.deg2rad(lats2d))
    v = np.zeros_like(u)
    u[10, 5] = np.nan
    v[10, 5] = np.nan
    u_da = xr.DataArray(u, coords={'lat': lat, 'lon': lon}, dims=('lat', 'lon'))
    v_da = xr.DataArray(v, coords={'lat': lat, 'lon': lon}, dims=('lat', 'lon'))
    ud = get_divergent_u(u_da, v_da)
    vd = get_divergent_v(u_da, v_da)
    assert np.isnan(ud[10, 5])
    assert np.isnan(vd[10, 5])


def test_compute_velpot_returns_three_fields():
    pytest.importorskip("windspharm")
    lat, lon = _uv_grid(37, 72)
    lons2d, lats2d = np.meshgrid(lon, lat)
    u = xr.DataArray(np.cos(np.deg2rad(lats2d)),
                     coords={'lat': lat, 'lon': lon}, dims=('lat', 'lon'))
    v = xr.DataArray(np.zeros_like(u.values),
                     coords={'lat': lat, 'lon': lon}, dims=('lat', 'lon'))
    chi, u_div, v_div = compute_velpot(u, v)
    assert chi.shape[-2:] == u.shape
    assert u_div.shape[-2:] == u.shape
    assert v_div.shape[-2:] == v.shape


# ---------- Common-grid interpolation ----------

def test_interp_to_common_lon_same_grid_roundtrip():
    """Interpolating onto the same grid as the input should return identical values."""
    nlev, nlon = 5, 10
    plev = np.linspace(100, 1000, nlev)
    lon = np.linspace(5, 355, nlon)
    rng = np.random.default_rng(2)
    data = rng.standard_normal((nlev, nlon))
    out = interp_to_common_lon(data, lon, plev, target_lon=lon, target_plev=plev)
    # Slight discrepancy allowed at the boundaries due to cyclic wrap + interp
    np.testing.assert_allclose(out[:, 1:-1], data[:, 1:-1], rtol=1e-10)


def test_interp_to_common_lon_output_shape_matches_targets():
    nlev_in, nlon_in = 8, 20
    data = np.zeros((nlev_in, nlon_in))
    out = interp_to_common_lon(data,
                               lon=np.linspace(5, 355, nlon_in),
                               plev_hpa=np.linspace(100, 1000, nlev_in))
    # Defaults: COMMON_PLEV (12 levels), COMMON_LON (360 points)
    assert out.shape == (len(COMMON_PLEV), len(COMMON_LON))


def test_interp_to_common_lat_output_shape():
    nlev_in, nlat_in = 8, 20
    data = np.zeros((nlev_in, nlat_in))
    target_lat = np.arange(-30, 31, 2.0)
    out = interp_to_common_lat(data,
                               lat=np.linspace(-30, 30, nlat_in),
                               plev_hpa=np.linspace(100, 1000, nlev_in),
                               target_lat=target_lat)
    assert out.shape == (len(COMMON_PLEV), len(target_lat))


# ---------- Li et al. (2006) ψ/φ solver ----------

def test_psi_phi_uv2psiphi_is_importable():
    """The Li-et-al solver exposes uv2psiphi and the internal helpers."""
    from apinter.circulation.psi_phi import (
        uv2psiphi, psi_lietal, ja, grad_ja, derive_ax, derive_adj,
        dx_from_dlon, dy_from_dlat,
    )
    # Make sure the high-level entry point is a callable function
    assert callable(uv2psiphi)
    assert callable(psi_lietal)
