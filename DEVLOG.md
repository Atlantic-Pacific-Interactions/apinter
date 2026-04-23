# apinter — `dev` branch log

This file records what's added on the `dev` branch on top of `main`, in
reverse chronological order. The `main` branch stays the stable
`pip install`-able version; experimental loaders and processing helpers
live here until they're proven and ready to promote.

To install the `dev` branch:

```bash
pip install --upgrade git+https://github.com/Atlantic-Pacific-Interactions/apinter.git@dev
```

To run the regular fast suite (≈80 s, 75 + dev tests):

```bash
pytest --pyargs apinter.tests
```

To also run the slow opt-in all-models survey (≈4 min):

```bash
pytest --pyargs apinter.tests -m slow
```

---

## 2026-04-23 — NERSC central-mirror access + xesmf regrid + ocean coverage

### Added

- **`apinter.io.load_nersc_cmip6`** (and `load_nersc_cmip6_ensemble`,
  `list_nersc_cmip6_models`) — ESGF DRS loader for the canonical NERSC
  CMIP6 mirror at `/global/cfs/cdirs/m3522/cmip6/CMIP6/`. Returns
  **native-grid** data; auto-infers `activity_id` for CMIP /
  ScenarioMIP / DAMIP / HighResMIP experiments.
  Dir walks tolerate `PermissionError` from restricted institute subtrees
  (e.g. `E3SM-Project/E3SM-1-1/...Omon/thetao/`).
- **`apinter.io.load_cesm1_lens`**, **`apinter.io.load_loca2`** plus
  list helpers — Large Ensemble loaders for `m2637/LENS` (CESM1-CAM5
  40-member ts/tas) and `m3522/datalake/LOCA2` (downscaled 6 km CMIP6
  ensemble incl. 10-member CESM2-LENS). Latest LOCA2 version suffix
  picked automatically when multiple coexist.
- **`apinter.io.OBS_SST_SOURCES`** — three new keys (`hadisst_nersc`,
  `cobe2_nersc`, `oisst_v2_nersc`) pointing at
  `/global/cfs/cdirs/m3522/datalake/{HadISST,COBE2,OISST_V2}/`.
- **`apinter.processing.regrid_to_1deg`** and **`regrid_dict_to_1deg`** —
  xesmf-based bilinear regrid to the canonical 1° lat/lon grid
  (`COMMON_LAT × COMMON_LON`). Auto-detects `lat`/`latitude`/`nav_lat`/
  `rlat` (and lon variants); handles **rectilinear and curvilinear**
  inputs (e.g. CESM2 ocean `lat(nlat, nlon)`) without code changes;
  handles 3D `(time, lat, lon)` and 4D `(time, lev, lat, lon)`
  uniformly — xesmf regrids the lat/lon plane only.
- **`apinter.config`** — adds `NERSC_CMIP6_DIR`, `NERSC_CESM1_LENS_DIR`,
  `NERSC_LOCA2_DIR`, `NERSC_DATALAKE_DIR`.
- **`pyproject.toml`** — adds `regrid = ["xesmf>=0.8"]` optional-extra,
  registers the `slow` pytest marker, and sets
  `addopts = "-m 'not slow'"` so the regular suite stays fast.

### Tests added (`apinter/tests/`)

| File | Tests | Notes |
|---|---:|---|
| `test_nersc_io.py` | 15 | smoke loads of every NERSC-mirror layout; auto-skip if path absent |
| `test_regrid.py` (regular) | 7 | synthetic + 3D ts + 4D wap + curvilinear tos (3D) + curvilinear thetao (4D) + dict-of-models |
| `test_regrid.py` (slow) | 3 | parametrized survey over all models for `ts`/Amon, `wap`/Amon, `tos`/Omon |
| `test_pipeline.py` | 1 | **end-to-end**: `load_nersc_cmip6 → regrid_to_1deg → K→C+filter → calculate_index → gridded_anomalies → regression_lags`. Proves the apinter downstream pipeline is unchanged whether data comes from NERSC or the user's pre-regridded zarr mirror. |

### Verification highlights

- `regrid_to_1deg(CESM2 historical ts)` matches the user's CDO-regridded
  zarr mirror at `/pscratch/sd/y/yanxia/CMIP6/CESM2/` to **4 decimals on
  every ocean cell** (e.g. 27.9981 °C and 25.4308 °C at two equatorial
  Pacific points). xesmf bilinear ≡ CDO `remapbil` for this case.
- All-models survey (run with `pytest -m slow`):
  - `ts`  : 36/38 succeed (`r1i1p1f1` historical)
  - `wap` : 35/37 succeed
  - `tos` : runs after the `table_id='Omon'` fix below

### Known limitations / accepted failures

- **ICON-ESM-LR** — icosahedral / unstructured triangular grid;
  `regrid_to_1deg` requires a rectilinear or curvilinear lat/lon
  representation. Listed in `EXPECTED_FAILURES` in the slow survey.
- **IITM-ESM** — empty version directory on the NERSC mirror. Same.
- A few institute subtrees (e.g. `E3SM-Project/E3SM-1-1/.../Omon/thetao/`)
  are permission-restricted; the dir walk now skips them silently
  rather than crashing.

### Bug fixes within dev

- `_open_files` in `nersc_cmip6.py` now selects the requested
  `variable_id` explicitly. Previously it grabbed
  `list(ds.data_vars)[0]`, which returned `time_bnds` for some files.
- `load_cesm1_lens` no longer hardcodes the `192001-210012.nc` time
  range; it globs `r{NN}i{NN}p{NN}_*.nc` to pick up the wider
  `185001-210012.nc` file that exists for `r1i1p1`.
- `load_loca2` picks the newest version suffix when multiple
  `LOCA_*_vYYYYMMDD.nc` files coexist for one member/var/experiment.
- The slow-survey test passes `table_id` to the load call (was
  defaulting to `Amon` for all variables, breaking ocean tests).

### Workflow sketch — regridded NERSC data feeds the standard pipeline

```python
from apinter.io import load_nersc_cmip6
from apinter.processing import regrid_to_1deg
from apinter.indices import calculate_index, gridded_anomalies
from apinter.stats import regression_lags

# Load + regrid + match the zarr mirror's K->C + outlier filter
sst_native = load_nersc_cmip6(
    variable_id='ts', experiment_id='historical',
    source_ids=['CESM2'],
    sim_time=slice('1980-01-01', '2014-12-31'),
)['CESM2']
sst_1deg = regrid_to_1deg(sst_native)
sst = (sst_1deg - 273.15).where(lambda x: (x > -10) & (x < 40))

# Drop into the existing apinter functions unchanged
tamv = calculate_index(sst, lon_bounds=(280, 340), lat_bounds=(0, 30))
tpdv = calculate_index(sst, lon_bounds=(180, 280), lat_bounds=(-20, 20))
ssta = gridded_anomalies(sst, cutoff_period=132, normalize=True)
reg  = regression_lags(field=ssta, target_index=tamv, confounder=tpdv,
                       lags=[-180, -120, -60, 0], compute_significance=True)
```

### Next on dev

- Optional convenience helper `apply_cmip6_sst_normalization(da)` that
  wraps the `K-273.15 + (-10 < sst < 40)` step (matches the inline
  branch in `apinter.io.cmip6.load_cmip6` for `ts`).
- Surface ocean variables (`tos`, `sos`) as named compat wrappers
  similar to the `load_cmip6_*` family.
- Once the dev API stabilizes and a couple of paper notebooks have used
  it, promote the relevant pieces to `main` and tag a release.
