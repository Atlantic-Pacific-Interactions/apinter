"""Phase-4 tests: apinter.plotting smoke tests.

We drive each plotting function with a small synthetic DataArray/Dataset
and check it produces a Figure without error (Matplotlib ``Agg`` backend).
No reference-image diffing — that's too brittle for unit tests.
"""
import matplotlib
matplotlib.use("Agg")  # must be set before pyplot import
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest
import xarray as xr


# ---------- Fixtures ----------

@pytest.fixture
def index_series():
    n = 120
    time = pd.date_range("1990-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(n).cumsum() * 0.05
    return xr.DataArray(vals, coords={"time": time}, dims=["time"], name="index")


@pytest.fixture
def small_regression_ds():
    lat = np.arange(-30, 31, 5.0)
    lon = np.arange(0, 360, 10.0)
    rng = np.random.default_rng(1)
    beta = rng.standard_normal((lat.size, lon.size)) * 0.05
    p = rng.uniform(0, 1, size=(lat.size, lon.size))
    return xr.Dataset(
        {
            "beta": (("lat", "lon"), beta),
            "p_value": (("lat", "lon"), p),
        },
        coords={"lat": lat, "lon": lon},
    )


@pytest.fixture
def small_trend_da():
    lat = np.arange(-30, 31, 5.0)
    lon = np.arange(0, 360, 10.0)
    rng = np.random.default_rng(2)
    data = rng.standard_normal((lat.size, lon.size)) * 0.3
    return xr.DataArray(
        data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"),
        name="trend",
    )


@pytest.fixture
def walker_psi():
    plev = np.array([100, 200, 300, 500, 700, 850, 1000], dtype=float)
    lon = np.arange(100, 361, 5.0)
    rng = np.random.default_rng(3)
    psi = rng.standard_normal((plev.size, lon.size)) * 0.5
    return psi, lon, plev


@pytest.fixture
def hadley_psi():
    plev = np.array([100, 200, 300, 500, 700, 850, 1000], dtype=float)
    lat = np.arange(-30, 31, 2.0)
    rng = np.random.default_rng(4)
    psi = rng.standard_normal((plev.size, lat.size))
    return psi, lat, plev


@pytest.fixture
def velpot_fields():
    lat = np.arange(-30, 31, 5.0)
    lon = np.arange(0, 360, 10.0)
    rng = np.random.default_rng(5)
    chi = rng.standard_normal((lat.size, lon.size))
    u_div = rng.standard_normal((lat.size, lon.size)) * 0.3
    v_div = rng.standard_normal((lat.size, lon.size)) * 0.3
    return chi, u_div, v_div, lat, lon


# ---------- index_ts.py ----------

def test_plot_index_ts_smoke(index_series):
    from apinter.plotting import plot_index_ts
    fig, ax = plt.subplots()
    plot_index_ts(index_series, ax, left_title='OBS',
                  center_title='TAMV', right_title='11yr LPF')
    assert ax.get_title(loc='left') == 'OBS'
    plt.close(fig)


def test_plot_index_grid_smoke(index_series):
    from apinter.plotting import plot_index_grid
    indices = {'HadISST': index_series, 'ERSST': index_series,
               'COBE': index_series, 'MMM': index_series}
    fig = plot_index_grid(indices, ncols=2)
    assert len(fig.axes) >= 4
    plt.close(fig)


# ---------- spatial_maps.py ----------

def test_plot_regression_map_smoke(small_regression_ds):
    import cartopy.crs as ccrs
    from apinter.plotting import plot_regression_map
    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.Robinson(central_longitude=180)}
    )
    im = plot_regression_map(ax, small_regression_ds,
                             vmin=-0.1, vmax=0.1,
                             add_stippling=True, stipple_thinning=2)
    assert im is not None
    plt.close(fig)


def test_plot_multiple_regression_maps_smoke(small_regression_ds):
    from apinter.plotting import plot_multiple_regression_maps
    fig = plot_multiple_regression_maps(
        [small_regression_ds, small_regression_ds],
        variable='beta', nrows=1, ncols=2, figsize=(10, 4),
        titles=['A', 'B'], add_stippling=False,
    )
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_trend_map_smoke(small_trend_da):
    from apinter.plotting import plot_trend_map
    fig = plot_trend_map(small_trend_da, vmin=-1, vmax=1,
                         title='synthetic', colorbar_label='K/decade',
                         figsize=(8, 4))
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_cmip6_trends_smoke(small_trend_da):
    from apinter.plotting import plot_cmip6_trends
    trends = {f'Model_{i}': small_trend_da for i in range(3)}
    fig = plot_cmip6_trends(trends, vmin=-1, vmax=1, ncols=3,
                            figsize_per_panel=(3, 2.5), title='MMM trends')
    assert len(fig.axes) >= 3
    plt.close(fig)


# ---------- circulation.py ----------

def test_plot_walker_section_smoke(walker_psi):
    from apinter.plotting import plot_walker_section
    psi, lon, plev = walker_psi
    fig, ax = plt.subplots()
    levels = np.linspace(-1, 1, 11)
    cf = plot_walker_section(ax, psi, lon, plev, levels=levels,
                             lon_plot_bounds=(100, 360))
    assert cf is not None
    plt.close(fig)


def test_plot_hadley_section_smoke(hadley_psi):
    from apinter.plotting import plot_hadley_section
    psi, lat, plev = hadley_psi
    fig, ax = plt.subplots()
    levels = np.linspace(-2, 2, 11)
    cf = plot_hadley_section(ax, psi, lat, plev, levels=levels,
                             lat_plot_bounds=(-30, 30))
    assert cf is not None
    plt.close(fig)


def test_plot_velpot_panel_smoke(velpot_fields):
    from apinter.plotting import plot_velpot_panel
    chi, u, v, lat, lon = velpot_fields
    fig, ax = plt.subplots()
    levels = np.linspace(-3, 3, 11)
    cf = plot_velpot_panel(ax, chi, u, v, lat, lon, levels=levels)
    assert cf is not None
    plt.close(fig)


# ---------- omega_regression.py ----------

def test_plot_omega_lead_lag_profile_smoke():
    """Synthetic lead-lag omega dataset with two lags."""
    from apinter.plotting import plot_omega_lead_lag_profile
    lev = np.array([1000, 850, 700, 500, 300, 200, 100], dtype=float)
    lon = np.arange(0, 360, 5.0)
    rng = np.random.default_rng(7)

    def _field():
        return xr.DataArray(
            rng.standard_normal((lev.size, lon.size)) * 0.002,
            coords={"level": lev, "lon": lon}, dims=("level", "lon"),
        )

    data = {'tpdv': {0: {'regression': _field()},
                     -60: {'regression': _field()}}}
    fig, axs = plot_omega_lead_lag_profile(
        data, model_name='CESM2', index_name='tpdv',
        figsize=(14, 5), savepath=None,
    )
    assert fig is not None and axs is not None
    plt.close(fig)
