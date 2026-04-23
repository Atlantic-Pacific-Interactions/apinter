"""Large Ensemble loaders for NERSC community mirrors.

Covers two ensembles:

  - **CESM1-LE** (``NERSC_CESM1_LENS_DIR`` = /global/cfs/cdirs/m2637/LENS)
    40-member CESM1-CAM5 historical + RCP8.5 ensemble, 1920–2100 monthly.
    Variables available: ``tas``, ``ts`` only.

  - **LOCA2** (``NERSC_LOCA2_DIR`` = /global/cfs/cdirs/m3522/datalake/LOCA2)
    Statistically downscaled (~6 km / 0.0625°) CMIP6 ensemble.
    Variables: ``pr``, ``tasmax``, ``tasmin``, ``DTR``.
    Experiments: ``historical`` (1950-2014), ``ssp370`` (2015-2100).
    25+ CMIP6 models incl. ``CESM2-LENS`` (10 members).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import xarray as xr

from apinter.config import NERSC_CESM1_LENS_DIR, NERSC_LOCA2_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CESM1-LE
# ---------------------------------------------------------------------------

_CESM1_FILE_RE = re.compile(
    r"^(?P<var>[a-z]+)_Amon_CESM1-CAM5_historical_rcp85_"
    r"(?P<member>r\d+i\d+p\d+)_\d{6}-\d{6}\.nc$"
)

CESM1_LENS_VARS = ('tas', 'ts')


def list_cesm1_lens_members(variable: str = 'ts',
                            base_path: Union[str, Path] = NERSC_CESM1_LENS_DIR,
                            ) -> List[str]:
    """Return sorted member ids (e.g. ['r1i1p1', 'r2i1p1', ...]) present for
    the given variable."""
    var_dir = Path(base_path) / variable
    if not var_dir.is_dir():
        return []
    members = []
    for f in var_dir.iterdir():
        m = _CESM1_FILE_RE.match(f.name)
        if m and m.group('var') == variable:
            members.append(m.group('member'))
    # CESM member names include r1..r40; sort numerically by the leading r\d+
    def _key(mid: str) -> int:
        m = re.match(r"r(\d+)", mid)
        return int(m.group(1)) if m else 0
    return sorted(set(members), key=_key)


def load_cesm1_lens(variable: str = 'ts',
                    member_ids: Optional[Iterable[str]] = None,
                    sim_time: Optional[slice] = None,
                    base_path: Union[str, Path] = NERSC_CESM1_LENS_DIR,
                    ) -> Dict[str, xr.DataArray]:
    """Load CESM1-CAM5 LENS for a single variable across ensemble members.

    Parameters
    ----------
    variable : {'tas', 'ts'}
        Only these are on the NERSC mirror.
    member_ids : iterable, optional
        Default: every member present for the variable.
    sim_time : slice, optional
    base_path : Path-like

    Returns
    -------
    dict[member_id, xr.DataArray]
    """
    if variable not in CESM1_LENS_VARS:
        raise ValueError(
            f"Unknown CESM1-LE variable {variable!r}. "
            f"Only {CESM1_LENS_VARS} are on the NERSC mirror."
        )
    base_path = Path(base_path)
    var_dir = base_path / variable

    if member_ids is None:
        member_ids = list_cesm1_lens_members(variable, base_path)

    out: Dict[str, xr.DataArray] = {}
    for mid in member_ids:
        # Some members (r1i1p1) span 1850-2100 while r2...r40 span 1920-2100,
        # so glob for the file rather than hardcoding the date range.
        candidates = sorted(var_dir.glob(
            f"{variable}_Amon_CESM1-CAM5_historical_rcp85_{mid}_*.nc"
        ))
        if not candidates:
            logger.warning(f"CESM1-LE {mid}: no {variable} file found")
            continue
        # Pick the widest time span (filename date pattern is YYYYMM-YYYYMM
        # — width ≈ end-year – start-year is well ordered by filename sort
        # in practice; just take the first.)
        fp = candidates[0]
        try:
            ds = xr.open_dataset(fp)
            da = ds[variable]
            if sim_time is not None and 'time' in da.dims:
                da = da.sel(time=sim_time)
            out[mid] = da
            logger.info(f"Loaded CESM1-LE {mid}/{variable}: shape={da.shape}")
        except Exception as e:
            logger.exception(f"CESM1-LE {mid}: {e}")
    return out


# ---------------------------------------------------------------------------
# LOCA2
# ---------------------------------------------------------------------------

LOCA2_VARS = ('pr', 'tasmax', 'tasmin', 'DTR')
LOCA2_EXPERIMENTS = ('historical', 'ssp370')


def list_loca2_models(base_path: Union[str, Path] = NERSC_LOCA2_DIR
                      ) -> List[str]:
    """Sorted list of downscaled CMIP6 model names under LOCA2 (excludes
    ``monthly``, ``scripts``, ``training_data`` helper subdirs)."""
    skip = {'monthly', 'scripts', 'training_data'}
    base_path = Path(base_path)
    if not base_path.is_dir():
        return []
    return sorted(
        p.name for p in base_path.iterdir()
        if p.is_dir() and p.name not in skip
    )


def list_loca2_members(model: str,
                       resolution: str = '0p0625deg',
                       base_path: Union[str, Path] = NERSC_LOCA2_DIR,
                       ) -> List[str]:
    """Sorted ensemble members for a given LOCA2 model + resolution."""
    model_dir = Path(base_path) / model / resolution
    if not model_dir.is_dir():
        return []
    def _key(mid: str) -> int:
        m = re.match(r"r(\d+)", mid)
        return int(m.group(1)) if m else 0
    return sorted(
        (p.name for p in model_dir.iterdir() if p.is_dir()),
        key=_key,
    )


def load_loca2(model: str = 'CESM2-LENS',
               variable: str = 'pr',
               experiment: str = 'historical',
               member_ids: Optional[Iterable[str]] = None,
               resolution: str = '0p0625deg',
               sim_time: Optional[slice] = None,
               base_path: Union[str, Path] = NERSC_LOCA2_DIR,
               ) -> Dict[str, xr.DataArray]:
    """Load LOCA2 statistically-downscaled data for one model/experiment/var.

    Parameters
    ----------
    model : str
        e.g. 'CESM2-LENS', 'ACCESS-CM2', 'MPI-ESM1-2-HR'. See
        :func:`list_loca2_models` for all available.
    variable : {'pr', 'tasmax', 'tasmin', 'DTR'}
    experiment : {'historical', 'ssp370'}
    member_ids : iterable, optional
        Default: every member present.
    resolution : str
        '0p0625deg' (~6 km) is the standard.
    sim_time : slice, optional
    base_path : Path-like

    Returns
    -------
    dict[member_id, xr.DataArray]
    """
    if variable not in LOCA2_VARS:
        raise ValueError(
            f"Unknown LOCA2 variable {variable!r}. Choices: {LOCA2_VARS}"
        )
    if experiment not in LOCA2_EXPERIMENTS:
        raise ValueError(
            f"Unknown LOCA2 experiment {experiment!r}. Choices: {LOCA2_EXPERIMENTS}"
        )
    base_path = Path(base_path)
    model_dir = base_path / model / resolution

    if member_ids is None:
        member_ids = list_loca2_members(model, resolution, base_path)

    out: Dict[str, xr.DataArray] = {}
    for mid in member_ids:
        var_dir = model_dir / mid / experiment / variable
        if not var_dir.is_dir():
            logger.warning(f"LOCA2 {model}/{mid}/{experiment}/{variable}: dir missing")
            continue
        # LOCA2 often keeps multiple version suffixes (e.g. ``v20220519`` and
        # ``v20240915``) for the same member/experiment/variable. Take only
        # the newest — lexicographic max of the ``LOCA_*_vYYYYMMDD.nc`` tail.
        nc_files = sorted(var_dir.glob('*.nc'))
        if not nc_files:
            logger.warning(f"LOCA2 {model}/{mid}/{experiment}/{variable}: no .nc files")
            continue
        fp = nc_files[-1]
        try:
            ds = xr.open_dataset(fp)
            da = ds[variable] if variable in ds.data_vars else ds[list(ds.data_vars)[0]]
            if sim_time is not None and 'time' in da.dims:
                da = da.sel(time=sim_time)
            out[mid] = da
            logger.info(
                f"Loaded LOCA2 {model}/{mid}/{experiment}/{variable} "
                f"({fp.name}): shape={da.shape}"
            )
        except Exception as e:
            logger.exception(f"LOCA2 {model}/{mid}: {e}")
    return out
