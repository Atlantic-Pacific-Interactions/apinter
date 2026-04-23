"""Shared pytest fixtures and legacy-sys.path setup for the apinter test suite.

Keeping the legacy `Paper_1/src` and `./src` directories on `sys.path` lets the
parity tests import the old implementations directly for bit-level comparison.
Will go away in phase 5 when the legacy folders are removed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Paper_1" / "src"))
sys.path.insert(0, str(REPO / "src"))


@pytest.fixture
def ts_1d():
    """Monthly time series 1980-01 .. 2014-12 (420 months) with trend+seasonal+noise."""
    n = 420
    time = pd.date_range("1980-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(42)
    t = np.arange(n)
    values = (
        0.01 * t
        + 2.0 * np.sin(2 * np.pi * t / 12)
        + rng.standard_normal(n) * 0.5
    )
    return xr.DataArray(values, coords={"time": time}, dims=["time"], name="x")


@pytest.fixture
def field_2d():
    """Monthly (time, lat, lon) field on a 5x6 grid."""
    n = 120
    time = pd.date_range("1980-01-01", periods=n, freq="MS")
    lat = np.array([-30., -15., 0., 15., 30.])
    lon = np.array([0., 60., 120., 180., 240., 300.])
    rng = np.random.default_rng(0)
    t = np.arange(n)
    data = (
        0.005 * t[:, None, None]
        + 1.0 * np.sin(2 * np.pi * t[:, None, None] / 12)
        + rng.standard_normal((n, 5, 6)) * 0.3
    )
    return xr.DataArray(
        data,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"], name="x",
    )
