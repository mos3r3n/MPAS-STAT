"""IODA obs-space file naming and HDF5 reading for the statistics stage.
"""
import glob
import os
import sys
from datetime import datetime
from typing import Dict, Optional, Union

import h5py
import numpy as np

from .sensor_config import get_obs_config, source_obs_name


def _resolve_h5_path(fname: Optional[str]) -> Optional[str]:
    """Resolve an obsout_*.h5 path, falling back to a wildcard match.

    The fallback is deliberately conservative: a missing background file
    must never resolve to the separate `..._ana.h5` ensemble-analysis file,
    which would silently use OMA where OMB was requested.
    """
    if fname is None:
        return None
    if os.path.exists(fname):
        return fname
    stem = fname[:-3] if fname.endswith(".h5") else fname
    matches = sorted(glob.glob(stem + "*.h5"))
    if fname.endswith("_ana.h5"):
        matches = [m for m in matches if m.endswith("_ana.h5")]
    elif os.path.basename(fname).startswith("obsout_da_"):
        matches = [m for m in matches if not m.endswith("_ana.h5")]
    return matches[-1] if matches else None


def build_filename(base_dir: str, exp_name: str, exp_type: str, exp_member: int,
                  cycle_time: datetime, diag_obs: str) -> Dict[str, str]:
    """Construct the OMB/OMA (background/analysis, cycling DA) filenames.
       Falls back to obsout_da/hofx_{diag_obs}*.h5 if exact file doesn't exist.

    OMA file layout depends on exp_member (n_member):
      - Deterministic DA (n_member == 1, e.g. 3DEnVar/hybrid with a single
        outer loop's worth of output): background AND analysis innovations
        live together in the SAME obsout_da_<obs>.h5 file, as separate
        'ombg'/'oman' (or 'hofx0'/'hofx<outerloop>') groups -- see
        load_observation_data(). filedict["ana"] == filedict["bak"] here;
        which group gets read is controlled entirely by the `outerloop`
        argument passed to load_observation_data() at each call site.
      - Ensemble DA (n_member > 1, e.g. LETKF/GETKF): the OMA innovations
        are written to a SEPARATE file, obsout_da_<obs>_ana.h5, alongside
        the background file's obsout_da_<obs>.h5 -- so filedict["ana"]
        points at that "..._ana.h5" file instead.
    """
    date_str = cycle_time.strftime("%Y%m%d%H")
    filedict = {"bak": None, "ana": None}
    resolve = _resolve_h5_path

    # --------------------------------------------------
    # Some obs types share another sensor's IODA file (e.g. profiler reads
    # the sondes obsout file and is distinguished by ObsType -- see
    # mpas_stat.obs.sensor_config.ObservationConfig.file_alias).
    # --------------------------------------------------
    obs_name = source_obs_name(diag_obs)

    if exp_type == 'da':
        # ---- background (OMB / hofx0)
        bak = (
            f"{base_dir}/{exp_name}/CyclingDA/{date_str}/dbOut/"
            f"obsout_da_{obs_name}.h5"
        )
        filedict["bak"] = resolve(bak)

        # ---- analysis (OMA)
        if exp_member > 1:
            # Ensemble DA (LETKF/GETKF): OMA lives in a separate "_ana.h5"
            # file alongside the background file, not as a group inside it.
            ana = (
                f"{base_dir}/{exp_name}/CyclingDA/{date_str}/dbOut/"
                f"obsout_da_{obs_name}_ana.h5"
            )
            filedict["ana"] = resolve(ana)
        else:
            # Deterministic (single-member) DA: background and analysis
            # innovations are both groups inside the SAME file.
            filedict["ana"] = filedict["bak"]
    # ==================================================
    # HOFX ONLY
    # ==================================================
    else:

        bak = (
            f"{base_dir}/{exp_name}/{date_str}/dbOut/"
            f"obsout_hofx_{obs_name}.h5"
        )
        filedict["bak"] = resolve(bak)

        # no analysis for hofx-only experiments
        filedict["ana"] = None

    return filedict


def build_hofx_forecast_filename(hofx_dir: str, exp_name: str, init_time: datetime,
                                lead_hour: int, diag_obs: str) -> Optional[str]:
    """Construct the OMF (obs-minus-forecast, extended-forecast skill)
    HofX filename for one (experiment, init cycle, forecast lead hour):
        <hofx_dir>/<exp_name>/HofX/<init_cycle>/f<HHH>/dbOut/obsout_hofx_<obs>.h5
    Falls back to a wildcard match if the exact filename doesn't exist, same
    as build_filename()."""
    obs_name = source_obs_name(diag_obs)
    init_str = init_time.strftime("%Y%m%d%H")
    fname = (
        f"{hofx_dir}/{exp_name}/HofX/{init_str}/f{int(lead_hour):03d}/dbOut/"
        f"obsout_hofx_{obs_name}.h5"
    )
    return _resolve_h5_path(fname)


def _hofx_fill_mask(hofx_arr: np.ndarray, fill_threshold: float = 1.0e30) -> np.ndarray:
    """Boolean mask, True where `hofx_arr` is a MISSING/invalid HofX value:
    NaN, +/-Inf, or a large numeric fill sentinel -- IODA HofX output uses
    the default netCDF fill value (~9.9692e+36 for float32/float64) for
    observations a forecast could not be evaluated at (e.g. an observation
    outside a REGIONAL MPAS-JEDI domain's extent, or a failed/undefined
    forecast column). No physically real value of any variable this
    package handles comes remotely close to `fill_threshold`, so this is a
    safe, convention-agnostic missing-value test regardless of whether a
    given file actually used NaN or a fill sentinel."""
    arr = np.asarray(hofx_arr, dtype=np.float64)
    return ~np.isfinite(arr) | (np.abs(arr) >= fill_threshold)


def _apply_hofx_qc(qc: Optional[np.ndarray], hofx_arr: np.ndarray,
                   remap_background_check: bool = True) -> np.ndarray:
    """Finalize the QC mask for a HofX-DERIVED innovation (the branch taken
    whenever `hofx_used is not None` in load_observation_data() -- i.e. the
    'hofx'/'hofx0'/'hofx0_mean'/'hofx<outerloop>' fallback, never the
    precomputed 'ombg'/'oman' path, which is the DA system's own already-QC'd
    output and is left untouched). Two things happen here:

      1. QC flag remap: for a HofX-derived innovation, both 0 (nominal
         pass) AND 19 count as passing, not just 0 -- unlike the plain
         qc == 0 convention every OTHER path in this module uses. Rather
         than teach every downstream consumer (calculate_stats(),
         compute_stats_2d_latlon(), compute_stats_2d_lat_vert(), all of
         which just check `qc == 0`) about this hofx-only exception, 19 is
         remapped to 0 right here, once, so the rest of the pipeline keeps
         a single uniform "0 == pass" convention regardless of where a qc
         array came from.
      2. HofX-missing exclusion (see _hofx_fill_mask): excluded (set to
         NaN) whether or not a prior QC value already excluded them, and
         QC is properly initialized (to all-valid elsewhere) even when
         `qc` came in as None -- e.g. a pure HofX/forecast (OMF) file with
         no EffectiveQC group at all, which would otherwise leave every
         point "un-QC'd" and vulnerable to a missing HofX silently turning
         ObsValue - HofX into a huge, meaningless innovation.
    """
    missing = _hofx_fill_mask(hofx_arr)
    base = np.zeros(missing.shape, dtype=np.float64) if qc is None else np.asarray(qc, dtype=np.float64)
    if remap_background_check:
        base = np.where(base == 19, 0.0, base)
    return np.where(missing, np.nan, base)


def _pass_flags_to_zero(qc: Optional[np.ndarray], pass_flags) -> Optional[np.ndarray]:
    """Map every QC flag in `pass_flags` to 0 (the pipeline's single
    "passed" value); other flags are kept as-is (rejected)."""
    if qc is None or pass_flags is None:
        return qc
    qc = np.asarray(qc)
    return np.where(np.isin(qc, list(pass_flags)), 0, qc)


def _read_obs_type(f, diag_var: Union[str, int]) -> np.ndarray:
    """Read the ObsType values for one variable.

    IODA normally stores ObsType as a single 1-D ``(nobs,)`` dataset, but
    some writers expose it as a group keyed by variable name. Support both
    instead of assuming the group layout, which would make the filter raise
    (and, via the broad error handler in load_observation_data(), silently
    disable the whole sensor).
    """
    obs_type = f['ObsType']
    if isinstance(obs_type, h5py.Group):
        return obs_type[diag_var][:]
    return obs_type[:]


# Variables whose HofX group already holds the innovation itself (O-B in
# percent for GNSS-RO), so it is used as-is instead of ObsValue - HofX.
# (Before 2026-11 this list misspelled "bendingAngle", so bending angle
# silently took the ObsValue - HofX path.)
_HOFX_IS_INNOVATION = ["atmosphericRefractivity", "bendingAngle"]


def _ensemble_spread(members: np.ndarray, ddof: int, finite_correction: bool) -> np.ndarray:
    """Per-observation ensemble spread over axis 0 (NaN members ignored):
    sqrt of the variance with ``ddof``, times (N+1)/N when
    ``finite_correction`` (N = valid members at that observation)."""
    if not ddof and not finite_correction:
        return np.nanstd(members, axis=0, ddof=0)
    n = np.sum(np.isfinite(members), axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(members, axis=0)
        var = np.nansum((members - mean) ** 2, axis=0)
        var = np.where(n > ddof, var / (n - ddof), np.nan)
        if finite_correction:
            var = var * (n + 1.0) / n
    return np.sqrt(var)


def load_observation_data(filename: str, diag_obs: str, diag_var: Union[str, int],
                        bin_method: int, outerloop: int, nmember: int,
                        settings: Optional[Dict] = None):
    """Load observation data from HDF5 file for OMB and OMA respectively.

    Radiances (bin_method == 3) read channel ``diag_var`` (1-based) of each
    group's 'brightnessTemperature' dataset; every other sensor reads the
    group's ``diag_var`` dataset. Everything else is shared:

    * Innovation: the precomputed 'ombg' (outerloop 0) / 'oman' (outerloop
      N) group when present, else ObsValue minus the first available HofX
      group -- 'hofx', 'hofx0', 'hofx0_mean' for outerloop 0, or
      'hofx<N>', 'hofx0_mean' for outerloop N.
    * QC: 'EffectiveQC<outerloop>'; for ensembles without it (current
      LETKF/GETKF output) a pass/fail flag derived from 'ObsError'.

    Innovations computed from a HofX field (rather than a precomputed
    ombg/oman group) are additionally QC'd against the HofX values
    THEMSELVES being missing -- see _hofx_fill_mask() / _apply_hofx_qc().
    This matters most for OMF (extended-forecast skill,
    process_forecast_experiment()): those files are plain HofX runs with no
    DA QC applied, so a HofX value missing at some observation location
    (most commonly: the observation falls outside a REGIONAL MPAS-JEDI
    domain's extent) would otherwise silently pollute the diagnosis with a
    huge, meaningless "innovation" -- and could easily have no
    EffectiveQC group at all to catch it. The same masking is applied
    anywhere else a HofX field feeds the innovation (e.g. "hofx"-typed
    OMB, or an outerloop's hofx{N} fallback when 'oman' is absent) since
    it is the identical risk. Precomputed 'ombg'/'oman' innovations are
    NOT touched here -- those come from the DA system's own QC'd output.

    ``settings`` (from mpas_stat.obs.common.resolve_statistics_settings)
    selects the QC flags that count as passed (qc_pass_flags, applied to
    every innovation source alike) and the ensemble-spread definition
    (spread_ddof, spread_finite_ensemble_correction). None, or the legacy
    settings, keeps the pre-2026-11 behaviour: flag 0 passes, and flag 19
    additionally passes only for HofX-derived innovations; spread uses ddof 0.

    Returns a dict (lat, lon, hgt, pres, time, innov, qc, spread, totSpread),
    or None if the file cannot be read. ``time`` is MetaData/dateTime when
    present (used to match observations across files), else None.
    """
    pass_flags = settings.get("qc_pass_flags") if settings else None
    spread_ddof = int(settings.get("spread_ddof", 0)) if settings else 0
    spread_correction = bool(settings.get("spread_finite_ensemble_correction", False)) if settings else False

    obs_vector = {
        'lat': None,
        'lon': None,
        'hgt': None,
        'pres': None,
        'time': None,
        'innov': None,
        'qc': None,
        'spread': None,
        'totSpread': None,
    }

    try:
        with h5py.File(filename, 'r') as f:
            if bin_method == 3:  # radiances: one channel of brightnessTemperature
                def read(group):
                    return f[group]['brightnessTemperature'][:, diag_var-1]
            else:
                def read(group):
                    return f[group][diag_var][:]

            # --------------------------------------------------
            # MetaData
            # --------------------------------------------------
            obs_vector['pres'] = f['MetaData']['pressure'][:] / 100. if 'pressure' in f['MetaData'] else None
            obs_vector['hgt']  = f['MetaData']['height'][:] / 1000. if 'height' in f['MetaData'] else None
            obs_vector['lat']  = f['MetaData']['latitude'][:]
            obs_vector['lon']  = f['MetaData']['longitude'][:]
            obs_vector['lon']  = np.where(obs_vector['lon'] > 180., obs_vector['lon']-360., obs_vector['lon'])
            obs_vector['time'] = f['MetaData']['dateTime'][:] if 'dateTime' in f['MetaData'] else None

            # --------------------------------------------------
            # Innovation: precomputed ombg/oman, else ObsValue - HofX
            # --------------------------------------------------
            if outerloop == 0:
                precomputed, hofx_groups = 'ombg', ['hofx', 'hofx0', 'hofx0_mean']
            else:
                precomputed, hofx_groups = 'oman', [f'hofx{outerloop}', 'hofx0_mean']

            hofx_used = None
            if precomputed in f:
                obs_vector['innov'] = read(precomputed)
            else:
                for group in hofx_groups:
                    if group in f:
                        hofx_used = read(group)
                        if diag_var in _HOFX_IS_INNOVATION:
                            obs_vector['innov'] = hofx_used
                        else:
                            obs_vector['innov'] = read('ObsValue') - hofx_used
                        break

            # --------------------------------------------------
            # Quality control flags (0 == pass)
            # --------------------------------------------------
            if f'EffectiveQC{outerloop}' in f:
                obs_vector['qc'] = read(f'EffectiveQC{outerloop}')
            elif (nmember > 1) and ('ObsError' in f):
                # Current LETKF/GETKF output has no EffectiveQC group.
                obserr = read('ObsError')
                if bin_method == 3:
                    obs_vector['qc'] = np.where(np.isnan(obserr), 99, 0)
                else:
                    obs_vector['qc'] = np.where((obserr > 0.) & (obserr <= 10000.), 0, np.nan)

            # A missing HofX value (see _hofx_fill_mask) is excluded from
            # the diagnosis regardless of whatever EffectiveQC/ObsError
            # above did or did not already flag -- most relevant for OMF
            # (process_forecast_experiment), whose plain HofX files often
            # have no EffectiveQC group at all.
            obs_vector['qc'] = _pass_flags_to_zero(obs_vector['qc'], pass_flags)
            if hofx_used is not None:
                obs_vector['qc'] = _apply_hofx_qc(obs_vector['qc'], hofx_used,
                                                  remap_background_check=pass_flags is None)

            # --------------------------------------------------
            # Ensemble spread -- missing members (see _hofx_fill_mask) are
            # excluded via NaN, same convention nanstd already uses.
            # --------------------------------------------------
            if nmember > 1:
                hofx_ens = []
                for iens in range(nmember):
                    hofx = read(f'hofx0_{iens+1}')
                    hofx_ens.append(np.where(_hofx_fill_mask(hofx), np.nan, hofx))
                hofx_ens = np.stack(hofx_ens, axis=0)
                obs_vector['spread'] = _ensemble_spread(hofx_ens, spread_ddof, spread_correction)
                # total spread
                obserr = read('ObsError')
                obs_vector['totSpread'] = np.sqrt(np.nan_to_num(obs_vector['spread'], nan=0.)**2+
                    np.nan_to_num(obserr, nan=0.0)**2.0)

            # Some sensors share one IODA file with another sensor and are
            # distinguished by their ObsType code (e.g. sondes vs. profiler
            # both live in obsout_*_sondes.h5) -- see
            # mpas_stat.obs.sensor_config.ObservationConfig.obs_type_filter.
            obs_type_filter = get_obs_config(diag_obs).obs_type_filter
            if obs_type_filter is not None:
                op, threshold = obs_type_filter
                obs_type = _read_obs_type(f, diag_var)
                if op == 'lt':
                    keep = obs_type < threshold
                elif op == 'eq':
                    keep = obs_type == threshold
                else:
                    raise ValueError(f"Unsupported obs_type_filter operator: {op}")
                obs_vector['qc'] = np.where(keep, obs_vector['qc'], np.nan)

            if diag_var == "specificHumidity":
                obs_vector['innov'] *= 1000.
                if nmember > 1:
                    obs_vector['spread'] *= 1000.
                    obs_vector['totSpread'] *= 1000.

            return obs_vector

    except Exception as e:
        print(f"Error loading {filename}: {str(e)}", file=sys.stderr)
        return None
