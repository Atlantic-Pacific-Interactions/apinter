"""CMIP6 loader for the canonical ESGF mirror at NERSC.

The mirror at ``/global/cfs/cdirs/m3522/cmip6/CMIP6`` follows the full ESGF
Data Reference Syntax (DRS):

    {base}/{activity_id}/{institution_id}/{source_id}/{experiment_id}/
          {member_id}/{table_id}/{variable_id}/{grid_label}/{version}/*.nc

e.g.
    CMIP/NCAR/CESM2/historical/r1i1p1f1/Amon/ts/gn/v20190308/
        ts_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc

This is native-grid data: each model keeps its own resolution, rotated pole,
or ocean-grid (``gn``), etc. If you need a multi-model mean you must regrid
each DataArray to a common grid yourself (``apinter.circulation.regrid`` or
``xesmf``).

Contrast with :func:`apinter.io.cmip6.load_cmip6`, which serves the user's
1°-regridded zarr mirror under ``/pscratch/sd/y/yanxia/CMIP6``.
"""
from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import xarray as xr

from apinter.config import NERSC_CMIP6_DIR

logger = logging.getLogger(__name__)


# Common experiment_id -> activity_id mapping. Extend as needed; users can
# always pass ``activity_id=`` explicitly to bypass the lookup.
_EXPERIMENT_ACTIVITY = {
    # CMIP
    'historical': 'CMIP', 'piControl': 'CMIP', 'amip': 'CMIP',
    '1pctCO2': 'CMIP', 'abrupt-4xCO2': 'CMIP',
    # ScenarioMIP
    'ssp119': 'ScenarioMIP', 'ssp126': 'ScenarioMIP',
    'ssp245': 'ScenarioMIP', 'ssp370': 'ScenarioMIP',
    'ssp460': 'ScenarioMIP', 'ssp585': 'ScenarioMIP',
    'ssp534-over': 'ScenarioMIP',
    # DAMIP
    'hist-nat': 'DAMIP', 'hist-GHG': 'DAMIP', 'hist-aer': 'DAMIP',
    # HighResMIP
    'hist-1950': 'HighResMIP', 'highres-future': 'HighResMIP',
}


# Default ``sim_time`` window per experiment — the canonical period each
# branch was designed to cover. ``None`` (or unmapped experiments) means
# "load everything the files contain". Pass ``sim_time=`` to override —
# e.g. ``sim_time=slice('2015-01-01', '2300-12-31')`` for the SSP1-2.6 /
# SSP5-8.5 long extension.
_DEFAULT_SIM_TIME: Dict[str, slice] = {
    'historical': slice('1850-01-01', '2014-12-31'),
    'ssp119':     slice('2015-01-01', '2100-12-31'),
    'ssp126':     slice('2015-01-01', '2100-12-31'),
    'ssp245':     slice('2015-01-01', '2100-12-31'),
    'ssp370':     slice('2015-01-01', '2100-12-31'),
    'ssp460':     slice('2015-01-01', '2100-12-31'),
    'ssp585':     slice('2015-01-01', '2100-12-31'),
    'ssp534-over': slice('2015-01-01', '2100-12-31'),
}


# CMOR filename time-range token: ``..._<start>-<end>[-clim].nc`` where each
# of <start>/<end> is YYYY[MM[DD[hhmm[ss]]]] (4/6/8/12/14 digits, equal width).
_FNAME_DATE_RE = re.compile(r'_(\d{4,14})-(\d{4,14})(?:-clim)?\.nc$')


def _resolve_activity(experiment_id: str,
                      activity_id: Optional[str]) -> str:
    if activity_id is not None:
        return activity_id
    if experiment_id in _EXPERIMENT_ACTIVITY:
        return _EXPERIMENT_ACTIVITY[experiment_id]
    raise ValueError(
        f"Unknown experiment_id {experiment_id!r}; pass activity_id= explicitly."
    )


def _pick_version(var_dir: Path, version: str) -> Optional[Path]:
    """Pick a version subdirectory. ``'latest'`` sorts lexicographically and
    takes the last (works for ``vYYYYMMDD`` naming)."""
    versions = sorted(p for p in var_dir.iterdir() if p.is_dir())
    if not versions:
        return None
    if version == 'latest':
        return versions[-1]
    for v in versions:
        if v.name == version:
            return v
    return None


def _normalize_cmor_date(s: str, *, end: bool) -> str:
    """Pad a CMOR date token (YYYY..YYYYMMDDhhmmss) to a 14-char string for
    lexicographic comparison. ``end=True`` pads to the *latest* instant
    inside the resolution (e.g. month → '31235959'); ``end=False`` pads to
    the earliest. Day is padded to '31' regardless of month length — this
    overestimates by at most a couple of days, which only causes us to keep
    a borderline file and let xarray's ``.sel(time=...)`` filter it precisely."""
    if len(s) == 4:    return s + ('1231235959' if end else '0101000000')
    if len(s) == 6:    return s + ('31235959'   if end else '01000000')
    if len(s) == 8:    return s + ('235959'     if end else '000000')
    if len(s) == 10:   return s + ('5959'       if end else '0000')
    if len(s) == 12:   return s + ('59'         if end else '00')
    if len(s) == 14:   return s
    raise ValueError(f"Unexpected CMOR date token {s!r}")


def _parse_filename_date_range(path: Path) -> Optional[Tuple[str, str]]:
    """Extract (start, end) as 14-char zero-padded numeric strings from a
    CMOR-style filename. Returns ``None`` if the filename has no time token
    (fx/Ofx fixed fields), or if start/end have mismatched widths."""
    m = _FNAME_DATE_RE.search(path.name)
    if m is None:
        return None
    a, b = m.group(1), m.group(2)
    if len(a) != len(b):
        return None
    try:
        return _normalize_cmor_date(a, end=False), _normalize_cmor_date(b, end=True)
    except ValueError:
        return None


def _slice_to_padded_strs(sim_time: slice) -> Optional[Tuple[str, str]]:
    """Render slice bounds as 14-char zero-padded numeric strings for
    lexicographic comparison against filename date tokens. Returns ``None``
    if the slice doesn't have parseable bounds."""
    def _pad(b, end: bool) -> str:
        if b is None:
            return '99999999999999' if end else '00000000000000'
        digits = ''.join(ch for ch in str(b) if ch.isdigit())
        if not digits:
            raise ValueError(f"slice bound {b!r} has no digits")
        return _normalize_cmor_date(digits, end=end)
    try:
        return _pad(sim_time.start, end=False), _pad(sim_time.stop, end=True)
    except ValueError:
        return None


def _filter_paths_by_sim_time(paths: List[Path],
                              sim_time: Optional[slice]) -> List[Path]:
    """Drop NetCDF files whose filename-encoded time range doesn't overlap
    ``sim_time``. Files with no time token (fx/Ofx) and files whose token we
    can't parse are kept unconditionally — safer to over-include than skip
    real data."""
    if sim_time is None:
        return paths
    bounds = _slice_to_padded_strs(sim_time)
    if bounds is None:
        return paths
    sim_start, sim_end = bounds
    keep: List[Path] = []
    for p in paths:
        rng = _parse_filename_date_range(p)
        if rng is None:
            keep.append(p)
            continue
        f_start, f_end = rng
        if f_end >= sim_start and f_start <= sim_end:
            keep.append(p)
    return keep


def _to_datetime64_if_possible(da: xr.DataArray) -> xr.DataArray:
    """Convert a cftime time index back to ``datetime64[ns]`` for downstream
    compatibility, when the calendar is Gregorian-like and dates fit in the
    [1677, 2262] range. Leaves the array untouched if conversion isn't safe
    (non-standard calendar, dates out of range, missing time coord)."""
    if 'time' not in da.coords:
        return da
    try:
        from xarray.coding.cftimeindex import CFTimeIndex  # type: ignore
    except ImportError:
        return da
    idx = da.indexes.get('time')
    if not isinstance(idx, CFTimeIndex):
        return da
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            new_idx = idx.to_datetimeindex()
    except Exception:
        return da
    return da.assign_coords(time=new_idx)


def _open_files(paths: List[Path], variable_id: str,
                sim_time: Optional[slice]) -> xr.DataArray:
    """Open one or more NetCDF files, concat on time if needed, slice time.

    Files are pre-filtered against ``sim_time`` using their CMOR filename
    date token before opening — this avoids the SSP long-extension trap
    where the 2101-2300 file overflows ``datetime64[ns]`` and forces a
    cftime fallback that ``combine_by_coords`` then refuses to merge with
    the 2015-2100 (datetime64) file. ``use_cftime=True`` is set as a safety
    net for the case where the user *does* request the long extension."""
    paths = _filter_paths_by_sim_time(paths, sim_time)
    if not paths:
        raise FileNotFoundError(
            f"No files overlap sim_time={sim_time!r}"
        )
    if len(paths) == 1:
        ds = xr.open_dataset(paths[0], use_cftime=True)
    else:
        ds = xr.open_mfdataset(
            [str(p) for p in paths],
            combine='by_coords',
            parallel=False,
            use_cftime=True,
        )
    # Select the requested variable explicitly — some CMIP6 NetCDFs list
    # ``time_bnds`` (a datetime64 array) as a data_var, so don't just grab
    # ``list(ds.data_vars)[0]``.
    if variable_id not in ds.data_vars:
        raise KeyError(
            f"Variable {variable_id!r} not in file — available: {list(ds.data_vars)}"
        )
    da = ds[variable_id]
    if sim_time is not None and 'time' in da.dims:
        da = da.sel(time=sim_time)
    return _to_datetime64_if_possible(da)


def list_nersc_cmip6_models(experiment_id: str,
                            variable_id: str,
                            table_id: str = 'Amon',
                            member_id: str = 'r1i1p1f1',
                            grid_label: str = 'gn',
                            activity_id: Optional[str] = None,
                            base_path: Union[str, Path] = NERSC_CMIP6_DIR,
                            ) -> List[str]:
    """Return sorted model source_ids that have the requested
    (experiment, variable, table, member, grid_label) combo present."""
    base_path = Path(base_path)
    activity = _resolve_activity(experiment_id, activity_id)
    activity_dir = base_path / activity
    if not activity_dir.is_dir():
        return []

    found: List[str] = []
    try:
        institutes = sorted(activity_dir.iterdir())
    except PermissionError:
        return []
    for inst_dir in institutes:
        try:
            if not inst_dir.is_dir():
                continue
            for model_dir in sorted(inst_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                var_dir = (model_dir / experiment_id / member_id / table_id
                           / variable_id / grid_label)
                if _safe_is_dir(var_dir):
                    found.append(model_dir.name)
        except PermissionError:
            # Some institute subtrees are restricted (e.g. E3SM ocean dirs);
            # skip and keep going.
            continue
    return sorted(found)


def _safe_is_dir(p: Path) -> bool:
    """Path.is_dir() that returns False on PermissionError instead of raising."""
    try:
        return p.is_dir()
    except PermissionError:
        return False


def load_nersc_cmip6(variable_id: str,
                    experiment_id: str,
                    source_ids: Optional[Iterable[str]] = None,
                    activity_id: Optional[str] = None,
                    table_id: str = 'Amon',
                    member_id: str = 'r1i1p1f1',
                    grid_label: str = 'gn',
                    version: str = 'latest',
                    sim_time: Optional[slice] = None,
                    base_path: Union[str, Path] = NERSC_CMIP6_DIR,
                    ) -> Dict[str, xr.DataArray]:
    """Load one variable across models from the NERSC ESGF CMIP6 mirror.

    Parameters
    ----------
    variable_id : str        e.g. 'ts', 'tas', 'pr', 'wap', 'ua'
    experiment_id : str      e.g. 'historical', 'ssp585'
    source_ids : iterable, optional
        Models to load. Default: every model that has the requested combo.
    activity_id : str, optional
        Override the default activity mapping ('CMIP' for historical,
        'ScenarioMIP' for SSPs, etc.).
    table_id : str           default 'Amon'.
    member_id : str          default 'r1i1p1f1'. Pass a different member
                             explicitly, or use :func:`load_nersc_cmip6_ensemble`
                             for multiple members of one model.
    grid_label : str         'gn' (native), 'gr' (regridded), etc.
    version : str            'latest' picks the newest ``vYYYYMMDD``; or
                             pass a specific version string.
    sim_time : slice, optional
        Time window to load. Defaults to the canonical period for the
        experiment: 1850-2014 for ``historical``, 2015-2100 for any SSP.
        Pass an explicit slice to override — e.g.
        ``sim_time=slice('2015-01-01', '2300-12-31')`` for the SSP1-2.6 /
        SSP5-8.5 long extension, or ``sim_time=None`` (after manually
        passing it through) for "load whatever is on disk".
    base_path : Path-like

    Returns
    -------
    dict[source_id, xr.DataArray]
        Keys are model source_ids. Models without the requested combo are
        silently skipped. DataArrays are on each model's native grid.
    """
    base_path = Path(base_path)
    activity = _resolve_activity(experiment_id, activity_id)
    if sim_time is None:
        sim_time = _DEFAULT_SIM_TIME.get(experiment_id)

    if source_ids is None:
        source_ids = list_nersc_cmip6_models(
            experiment_id=experiment_id, variable_id=variable_id,
            table_id=table_id, member_id=member_id,
            grid_label=grid_label, activity_id=activity,
            base_path=base_path,
        )

    out: Dict[str, xr.DataArray] = {}
    for source_id in source_ids:
        var_dir = _find_var_dir(base_path, activity, source_id,
                                experiment_id, member_id, table_id,
                                variable_id, grid_label)
        if var_dir is None:
            logger.warning(f"{source_id}: {variable_id}/{experiment_id}/{member_id}/"
                           f"{table_id}/{grid_label} not found")
            continue
        version_dir = _pick_version(var_dir, version)
        if version_dir is None:
            logger.warning(f"{source_id}: no version under {var_dir}")
            continue
        nc_files = sorted(version_dir.glob('*.nc'))
        if not nc_files:
            logger.warning(f"{source_id}: no .nc files in {version_dir}")
            continue
        try:
            out[source_id] = _open_files(nc_files, variable_id, sim_time)
            logger.info(f"Loaded {source_id}/{variable_id}/{experiment_id}/"
                        f"{member_id}: shape={out[source_id].shape}")
        except Exception as e:  # individual model load failure shouldn't abort
            logger.exception(f"Failed to load {source_id}: {e}")

    return out


def load_nersc_cmip6_ensemble(variable_id: str,
                              experiment_id: str,
                              source_id: str,
                              member_ids: Optional[Iterable[str]] = None,
                              activity_id: Optional[str] = None,
                              table_id: str = 'Amon',
                              grid_label: str = 'gn',
                              version: str = 'latest',
                              sim_time: Optional[slice] = None,
                              base_path: Union[str, Path] = NERSC_CMIP6_DIR,
                              ) -> Dict[str, xr.DataArray]:
    """Load an ensemble of members from a single CMIP6 model.

    Parameters
    ----------
    variable_id, experiment_id, source_id : str
    member_ids : iterable, optional
        Members to load. Default: every member that has the combo.
    Remaining params: see :func:`load_nersc_cmip6`.

    Returns
    -------
    dict[member_id, xr.DataArray]
    """
    base_path = Path(base_path)
    activity = _resolve_activity(experiment_id, activity_id)
    if sim_time is None:
        sim_time = _DEFAULT_SIM_TIME.get(experiment_id)

    # Find member_id directories under the model
    model_root = _find_model_dir(base_path, activity, source_id)
    if model_root is None:
        return {}
    exp_dir = model_root / experiment_id
    if not exp_dir.is_dir():
        return {}

    if member_ids is None:
        member_ids = sorted(p.name for p in exp_dir.iterdir() if p.is_dir())

    out: Dict[str, xr.DataArray] = {}
    for mid in member_ids:
        var_dir = (exp_dir / mid / table_id / variable_id / grid_label)
        if not var_dir.is_dir():
            continue
        version_dir = _pick_version(var_dir, version)
        if version_dir is None:
            continue
        nc_files = sorted(version_dir.glob('*.nc'))
        if not nc_files:
            continue
        try:
            out[mid] = _open_files(nc_files, variable_id, sim_time)
        except Exception as e:
            logger.exception(f"{source_id}/{mid}: {e}")
    return out


# ---------------------------------------------------------------------------
# internal dir-walking helpers
# ---------------------------------------------------------------------------

def _find_var_dir(base: Path, activity: str, source_id: str,
                  experiment_id: str, member_id: str, table_id: str,
                  variable_id: str, grid_label: str) -> Optional[Path]:
    """Return the var/grid_label directory or None."""
    model_dir = _find_model_dir(base, activity, source_id)
    if model_dir is None:
        return None
    var_dir = (model_dir / experiment_id / member_id / table_id
               / variable_id / grid_label)
    return var_dir if _safe_is_dir(var_dir) else None


def _find_model_dir(base: Path, activity: str,
                    source_id: str) -> Optional[Path]:
    """Walk {base}/{activity}/*/<source_id>/ to find the model's root dir."""
    activity_dir = base / activity
    if not _safe_is_dir(activity_dir):
        return None
    try:
        institutes = list(activity_dir.iterdir())
    except PermissionError:
        return None
    for inst_dir in institutes:
        if not _safe_is_dir(inst_dir):
            continue
        candidate = inst_dir / source_id
        if _safe_is_dir(candidate):
            return candidate
    return None
