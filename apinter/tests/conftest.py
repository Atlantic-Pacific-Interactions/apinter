"""Shared pytest fixtures for the apinter test suite.

Tests with ``_matches_legacy`` in their names compare apinter functions
against the original Paper_1/src and ./src implementations for a bit-level
parity check. Those legacy folders only exist inside the
``Midlat-Atlantic-Pacific-Interactions`` mono-repo — when apinter is
checked out or pip-installed on its own, those tests auto-skip.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Search a handful of likely locations for the legacy mono-repo.
_this = Path(__file__).resolve()
_candidates = [
    Path(os.environ.get("APINTER_LEGACY_ROOT", "")) if os.environ.get("APINTER_LEGACY_ROOT") else None,
    # Sibling to a standalone apinter clone at /pscratch/sd/y/yanxia/apinter
    _this.parent.parent.parent.parent / "Midlat-Atlantic-Pacific-Interactions",
    # Inside the mono-repo (apinter/ lives at the mono-repo root)
    _this.parent.parent.parent,
    # Default Perlmutter path
    Path("/pscratch/sd/y/yanxia/Midlat-Atlantic-Pacific-Interactions"),
]
_candidates = [c for c in _candidates if c is not None]

LEGACY_PATHS: list[Path] = []
for root in _candidates:
    p1 = root / "Paper_1" / "src"
    p2 = root / "src"
    if p1.is_dir() and p2.is_dir():
        LEGACY_PATHS = [p1, p2]
        break

HAS_LEGACY = bool(LEGACY_PATHS)
for p in LEGACY_PATHS:
    sys.path.insert(0, str(p))


def pytest_collection_modifyitems(config, items):
    """Auto-skip *_matches_legacy tests when the legacy source tree is absent."""
    if HAS_LEGACY:
        return
    skip = pytest.mark.skip(
        reason="legacy Paper_1/src and ./src not present "
               "(set APINTER_LEGACY_ROOT to re-enable parity tests)"
    )
    for item in items:
        if "matches_legacy" in item.name:
            item.add_marker(skip)


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
