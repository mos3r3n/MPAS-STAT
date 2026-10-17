#!/usr/bin/env python3
"""
Observation-space verification -- STATISTICS stage.

Reads IODA obs-space HDF5 files, computes statistics
  - domain-averaged                       ("0D"),
  - vertical/latitude binned profiles      ("1D"),
  - lat-lon gridded maps                   ("2D/lat_lon", domain-independent,
                                             covers plot_settings.latlon_domain's
                                             box -- global by default),
  - latitude x level cross-sections        ("2D/lat_vert", domain-independent,
                                             same latitude box as lat_lon;
                                             skipped for radiances),
and writes them to observation-space "diagnosis" netCDF files, using the
SAME group convention MPAS-STATS uses for its own mpasstat.<date>.nc files
(see mpas_stats_diag_driver.write_stats_results):
    <var>/0D/<region>/<stat>
    <var>/1D/<profile_key>/<stat>
    <var>/2D/lat_lon/<stat>
    <var>/2D/lat_vert/<stat>       (pressure/height-binned sensors only)

Two independent verification modes are supported, matching two distinct
experiment "types" in the config's experiments.types list:

  - OMB/OMA (background/analysis, cycling DA). exp_type "da" (OMA too) or
    "hofx" (OMB only). Reads
        <da_dir>/<exp>/CyclingDA/<cycle>/dbOut/obsout_da_<obs>.h5
        <hofx_dir>/<exp>/<cycle>/dbOut/obsout_hofx_<obs>.h5
    time-resolved per assimilation cycle, writing one diagnosis file per
    (experiment, cycle, obs type) to
        <diag_dir>/<experiment>/VerifyObs/<cycle>/obsstat_<obs>.<date>.nc
    -- see process_experiment() / write_obs_diag_files(). The experiment
    sits at the top of the tree, exactly mirroring MPAS-STATS' own
    model-space <exp>/VerifyModel/<cycle>/mpasstat.<date>.nc convention --
    obs type is not a separate directory level, since the filename itself
    already identifies the sensor; see the path-convention comment above
    obs_diag_file() in obsstat_common.py for the full rationale.

  - OMF (obs-minus-forecast, extended-forecast skill). exp_type
    "hofx_forecast". Reads
        <hofx_dir>/<exp>/HofX/<init_cycle>/f<HHH>/dbOut/obsout_hofx_<obs>.h5
    for every configured forecast lead hour (config["forecast"]: length,
    interval, start -- same schema as the model-space graphic.yaml), writing
    one diagnosis file per (experiment, init cycle, lead hour, obs type) to
        <diag_dir>/<experiment>/VerifyObs/<init_cycle>/f<HHH>/obsstat_<obs>.<valid_date>.nc
    -- mirroring mpas_stats_graphic_io.get_mpasstat_forecast_file exactly.
    See process_forecast_experiment() / write_obs_forecast_diag_files().

An experiments list may freely mix "da"/"hofx" and "hofx_forecast" entries;
each is processed and written by the function above that understands it.

This stage has NO plotting code and does not import matplotlib or any
MPAS-STATS graphic module -- see plot_ioda_obsout_stats.py for the separate
graphics stage, which reads these files back and draws figures via
mpas_stats_graphic_functions.py.

This script must live in the same directory as obsstat_common.py,
obs_sensor_config.py (per-sensor variables/units/labels/bin-method/quirks
registry), and the MPAS-STATS modules (mpas_stats_diag_kernels.py,
mpas_stats_graphic_io.py) so it can import them directly -- it is meant to
sit inside the MPAS-STATS package folder, next to run_verify_model.ksh.
"""
import os
import sys
import glob
import numpy as np
import h5py
import netCDF4 as nc4
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse MPAS-STATS' own binning kernel for the lat-lon gridded statistics
# (same grouped mean/RMS/STD approach used for MPAS model grid-cell stats).
from mpas_stats_diag_kernels import grouped_stats_1d

from obsstat_common import (
    load_config, get_levels,
    profile_key_for_domain, nc_group_name, STAT_KEYS, STAT_KEYS_2D,
    resolve_domains, resolve_time_settings, generate_cycles,
    resolve_diag_base_dir, obs_diag_file,
    generate_forecast_leads, obs_diag_forecast_file,
    resolve_latlon_bounds, lead_valid_time_allowed,
)
# Sensor metadata (variables/units/labels/bin method + per-sensor quirks:
# file_alias, obs_type_filter, level_bin_kwargs) all come from the single
# dedicated sensor-info module -- see obs_sensor_config.py.
from obs_sensor_config import get_obs_config, get_bin_method, source_obs_name

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
    # obs_sensor_config.ObservationConfig.file_alias).
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

def _apply_hofx_qc(qc: Optional[np.ndarray], hofx_arr: np.ndarray) -> np.ndarray:
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
    base = np.where(base == 19, 0.0, base)
    return np.where(missing, np.nan, base)

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


def load_observation_data(filename: str, diag_obs: str, diag_var: Union[str, int],
                        bin_method: int, outerloop: int, nmember: int):
    """Load observation data from HDF5 file for OMB and OMA respectively.

    Innovations computed from a HofX field (rather than a precomputed
    ombg/oman group -- i.e. any "hofx"/"hofx0"/"hofx{outerloop}"/
    "hofx0_mean" branch below) are additionally QC'd against the HofX
    values THEMSELVES being missing -- see _hofx_fill_mask() /
    _apply_hofx_qc(). This matters most for OMF (extended-forecast skill,
    process_forecast_experiment()): those files are plain HofX runs with no
    DA QC applied, so a HofX value missing at some observation location
    (most commonly: the observation falls outside a REGIONAL MPAS-JEDI
    domain's extent) would otherwise silently pollute the diagnosis with a
    huge, meaningless "innovation" -- and could easily have no
    EffectiveQC group at all to catch it. The same masking is applied
    anywhere else a HofX field feeds the innovation (e.g. "hofx"-typed
    OMB, or an outerloop's hofx{N} fallback when 'oman' is absent) since
    it is the identical risk. Precomputed 'ombg'/'oman' innovations are
    NOT touched here -- those come from the DA system's own QC'd output."""

    obs_vector = {
        'lat': None,
        'lon': None,
        'hgt': None,
        'pres': None,
        'innov': None,
        'qc': None,
        'spread': None,
        'totSpread': None,
    }

    try:
        with h5py.File(filename, 'r') as f:
            # --------------------------------------------------
            # MetaData
            # --------------------------------------------------
            obs_vector['pres'] = f['MetaData']['pressure'][:] / 100. if 'pressure' in f['MetaData'] else None
            obs_vector['hgt']  = f['MetaData']['height'][:] / 1000. if 'height' in f['MetaData'] else None
            obs_vector['lat']  = f['MetaData']['latitude'][:]
            obs_vector['lon']  = f['MetaData']['longitude'][:]
            obs_vector['lon']  = np.where(obs_vector['lon'] > 180., obs_vector['lon']-360., obs_vector['lon'])

            # Data assimilation files have ombg/oman
            if bin_method == 3:  # radiances
                ## Innovation
                hofx_used = None
                if outerloop == 0:
                    if 'ombg' in f:
                        obs_vector['innov'] = f['ombg']['brightnessTemperature'][:, diag_var-1]
                    elif 'hofx' in f:
                        hofx_used = f['hofx']['brightnessTemperature'][:, diag_var-1]
                        obs_vector['innov'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - hofx_used
                    elif 'hofx0' in f:
                        hofx_used = f['hofx0']['brightnessTemperature'][:, diag_var-1]
                        obs_vector['innov'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - hofx_used
                    elif 'hofx0_mean' in f:
                        hofx_used = f['hofx0_mean']['brightnessTemperature'][:, diag_var-1]
                        obs_vector['innov'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - hofx_used
                else:
                    if 'oman' in f:
                        obs_vector['innov'] = f['oman']['brightnessTemperature'][:, diag_var-1]
                    elif f'hofx{outerloop}' in f:
                        hofx_used = f[f'hofx{outerloop}']['brightnessTemperature'][:, diag_var-1]
                        obs_vector['innov'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - hofx_used
                    elif 'hofx0_mean' in f:
                        hofx_used = f['hofx0_mean']['brightnessTemperature'][:, diag_var-1]
                        obs_vector['innov'] = f['ObsValue']['brightnessTemperature'][:, diag_var-1] - hofx_used

                ## quality control flags
                if f'EffectiveQC{outerloop}' in f:
                    obs_vector['qc'] = f[f'EffectiveQC{outerloop}']['brightnessTemperature'][:, diag_var-1]
                elif (nmember > 1) and ('ObsError' in f): #### this is for current LETKF/GETKF cuz not EffectiveQC is available.
                    obs_vector['qc'] = f['ObsError']['brightnessTemperature'][:, diag_var-1]
                    obs_vector['qc'] = np.where(np.isnan(obs_vector['qc']), 99, 0)

                # A missing HofX value (see _hofx_fill_mask) is excluded from
                # the diagnosis regardless of whatever EffectiveQC/ObsError
                # above did or did not already flag -- most relevant for OMF
                # (process_forecast_experiment), whose plain HofX files often
                # have no EffectiveQC group at all.
                if hofx_used is not None:
                    obs_vector['qc'] = _apply_hofx_qc(obs_vector['qc'], hofx_used)

                ## spread
                if nmember > 1:
                    # Spread -- missing members (see _hofx_fill_mask) are
                    # excluded via NaN, same convention nanstd already uses.
                    hofx_ens = []
                    for iens in range(nmember):
                        hofx = f[f'hofx0_{iens+1}']['brightnessTemperature'][:, diag_var-1]
                        hofx_ens.append(np.where(_hofx_fill_mask(hofx), np.nan, hofx))
                    hofx_ens = np.stack(hofx_ens, axis=0)
                    obs_vector['spread'] = np.nanstd(hofx_ens, axis=0, ddof=0)
                    # total spread
                    obserr = f['ObsError']['brightnessTemperature'][:, diag_var-1]
                    obs_vector['totSpread'] = np.sqrt(np.nan_to_num(obs_vector['spread'], nan=0.)**2+
                        np.nan_to_num(obserr, nan=0.0)**2.0)

            else: # non-radiance
                ## Innovations
                hofx_used = None
                if outerloop == 0:
                    if 'ombg' in f:
                        obs_vector['innov'] = f['ombg'][diag_var][:]
                    elif 'hofx' in f:
                        hofx_used = f['hofx'][diag_var][:]
                        if diag_var in ["atmosphericRefractivity", "beningAngle"]:
                            obs_vector['innov'] = hofx_used
                        else:
                            obs_vector['innov'] = f['ObsValue'][diag_var][:] - hofx_used
                    elif 'hofx0' in f:
                        hofx_used = f['hofx0'][diag_var][:]
                        if diag_var in ["atmosphericRefractivity", "beningAngle"]:
                            obs_vector['innov'] = hofx_used
                        else:
                            obs_vector['innov'] = f['ObsValue'][diag_var][:] - hofx_used
                    elif 'hofx0_mean' in f:
                        hofx_used = f['hofx0_mean'][diag_var][:]
                        if diag_var in ["atmosphericRefractivity", "beningAngle"]:
                            obs_vector['innov'] = hofx_used
                        else:
                            obs_vector['innov'] = f['ObsValue'][diag_var][:] - hofx_used
                else:
                    if 'oman' in f:
                        obs_vector['innov'] = f['oman'][diag_var][:]
                    elif f'hofx{outerloop}' in f:
                        hofx_used = f[f'hofx{outerloop}'][diag_var][:]
                        if diag_var in ["atmosphericRefractivity", "beningAngle"]:
                           obs_vector['innov'] = hofx_used
                        else:
                            obs_vector['innov'] = f['ObsValue'][diag_var][:] - hofx_used
                    elif 'hofx0_mean' in f:
                        hofx_used = f['hofx0_mean'][diag_var][:]
                        if diag_var in ["atmosphericRefractivity", "beningAngle"]:
                            obs_vector['innov'] = hofx_used
                        else:
                            obs_vector['innov'] = f['ObsValue'][diag_var][:] - hofx_used

                ## quality control flags
                if f'EffectiveQC{outerloop}' in f:
                    obs_vector['qc'] = f[f'EffectiveQC{outerloop}'][diag_var][:]
                elif (nmember > 1) and ('ObsError' in f): #### this is for current LETKF/GETKF cuz not EffectiveQC is available.
                    obs_vector['qc'] = f['ObsError'][diag_var][:]
                    obs_vector['qc'] = np.where(
                            (obs_vector['qc'] > 0.) & (obs_vector['qc'] <= 10000.),
                            0,
                            np.nan
                    )

                # A missing HofX value (see _hofx_fill_mask) is excluded from
                # the diagnosis regardless of whatever EffectiveQC/ObsError
                # above did or did not already flag -- most relevant for OMF
                # (process_forecast_experiment), whose plain HofX files often
                # have no EffectiveQC group at all.
                if hofx_used is not None:
                    obs_vector['qc'] = _apply_hofx_qc(obs_vector['qc'], hofx_used)

                # Spread -- missing members (see _hofx_fill_mask) are
                # excluded via NaN, same convention nanstd already uses.
                if nmember > 1:
                    hofx_ens = []
                    for iens in range(nmember):
                        hofx = f[f'hofx0_{iens+1}'][diag_var][:]
                        hofx_ens.append(np.where(_hofx_fill_mask(hofx), np.nan, hofx))
                    hofx_ens = np.stack(hofx_ens, axis=0)
                    obs_vector['spread'] = np.nanstd(hofx_ens, axis=0, ddof=0)
                    # total spread
                    obserr = f['ObsError'][diag_var][:]
                    obs_vector['totSpread'] = np.sqrt(np.nan_to_num(obs_vector['spread'], nan=0.)**2+
                        np.nan_to_num(obserr, nan=0.0)**2.0)
            #### Some sensors share one IODA file with another sensor and are
            #### distinguished by their ObsType code (e.g. sondes vs.
            #### profiler both live in obsout_*_sondes.h5) -- see
            #### obs_sensor_config.ObservationConfig.obs_type_filter.
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

def calculate_stats(data: np.ndarray, qc: np.ndarray,
                  bin_values: np.ndarray, levels: np.ndarray,
                  bin_method: int, interval: float = None) -> Dict:
    """Calculate statistics for each level.

    For bin_method != 1 (pressure), each `levels[ilev]` is a bin CENTER and
    `interval` is the bin's full width (bot_bin/top_bin = level -+
    interval/2, below) -- so `interval` must match however `levels` itself
    was spaced, or the bins either gap (interval too small -- some
    bin_values fall between bins, silently dropped) or overlap (interval
    too large -- some bin_values double-counted across neighboring bins).
    When not given explicitly, `interval` is derived directly from the
    spacing between consecutive `levels` entries -- get_levels() always
    builds `levels` with np.arange() at a fixed step for bin_method != 1
    (its hardcoded pressure/height/latitude defaults, an explicit
    obs_sensor_config.ObservationConfig.level_bin_kwargs override, or a
    domain-derived latitude range/width for radiances -- see get_levels()'s
    docstring), so this is always the width that actually produced
    `levels`, rather than a second, independently-hardcoded guess that can
    silently drift out of sync with it (as the old fixed 10.0 (bin_method
    == 3) / 2.0 (bin_method == 2) constants did, e.g. for a sensor whose
    level_bin_kwargs sets bin_width to something else, or radiances' new
    domain-derived width)."""
    if interval is None:
        if bin_method == 1:
            interval = None  # pressure bins use per-level log-midpoint edges instead (below)
        else:
            levels_arr = np.asarray(levels, dtype=np.float64)
            interval = (float(levels_arr[1] - levels_arr[0]) if levels_arr.size >= 2
                       else (10.0 if bin_method == 3 else 2.0))

    results = {
        'rmse': np.zeros(len(levels)+1),
        'mean': np.zeros(len(levels)+1),
        'std': np.zeros(len(levels)+1),
        'count': np.zeros(len(levels)+1),
    }

    # For total
    if qc is not None:
        valid_mask = (qc == 0)
    else:
        # No QC information available for this observation (no
        # EffectiveQC<outerloop> group, not an ensemble ObsError case, and no
        # HofX-missing fallback triggered -- see load_observation_data()) --
        # treat it as "nothing to filter", the same convention
        # compute_stats_2d_latlon()/compute_stats_2d_lat_vert() already use.
        # Before this guard, qc=None fell through to the bare comparison
        # `(None == 0)`, which Python/numpy evaluates as the scalar `False`
        # rather than a boolean array -- `data[False]` silently returns a
        # LENGTH-0 slice (not "keep everything", and no error), so every
        # stat below came out NaN/0 with no crash: this was the exact bug
        # behind an obs type with no EffectiveQC group at all (e.g. sfc)
        # producing an empty diagnosis file.
        valid_mask = np.ones(np.asarray(data).shape, dtype=bool)
    valid_data = data[valid_mask]
    results['rmse'][-1] = np.sqrt(np.nanmean(valid_data**2)) if valid_data.size else np.nan
    results['mean'][-1] = np.nanmean(valid_data) if valid_data.size else np.nan
    results['std'][-1] = np.nanstd(valid_data) if valid_data.size else np.nan
    results['count'][-1] = np.sum(~np.isnan(valid_data))

    for ilev, level in enumerate(levels):
        if bin_method == 1:  # pressure
            log_levels = np.log(levels)
            if ilev == 0:
                top_bin = 200000.
                bot_bin = np.exp((log_levels[ilev] + log_levels[ilev + 1]) / 2.0)
            elif ilev == (len(levels) - 1):
                top_bin = np.exp((log_levels[ilev] + log_levels[ilev - 1]) / 2.0)
                bot_bin = 0.
            else:
                top_bin = np.exp((log_levels[ilev] + log_levels[ilev - 1]) / 2.0)
                bot_bin = np.exp((log_levels[ilev] + log_levels[ilev + 1]) / 2.0)
        else:
            bot_bin = level - interval/2.
            top_bin = level + interval/2.

        mask = (bin_values > bot_bin) & (bin_values <= top_bin) & (valid_mask)
        valid_data = data[mask]

        if np.any(~np.isnan(valid_data)):
            results['rmse'][ilev] = np.sqrt(np.nanmean(valid_data**2))
            results['mean'][ilev] = np.nanmean(valid_data)
            results['std'][ilev] = np.nanstd(valid_data)
            results['count'][ilev] = np.sum(~np.isnan(valid_data))
        else:
            results['rmse'][ilev] = np.nan
            results['mean'][ilev] = np.nan
            results['std'][ilev] = np.nan
            results['count'][ilev] = 0

    return results

def _apply_domain_mask(qc: Optional[np.ndarray], lat: np.ndarray, lon: np.ndarray,
                       domain_name: str, domain_coords) -> Optional[np.ndarray]:
    """Combine one domain's lat/lon selection with an optional QC array.

    Returns a QC-style array (0 == pass, NaN == reject), or None when no
    filtering is needed at all (the global domain AND no QC information).
    The domain selection is applied even when `qc` is None: folding it into
    the QC array must not silently disable it, or a non-global product
    (e.g. NXTro) would be computed from every observation in the file.
    """
    if domain_name == "global":
        return qc

    lat_min, lat_max, lon_min, lon_max = domain_coords
    in_domain = (
        (lat >= lat_min) & (lat <= lat_max) &
        (lon >= lon_min) & (lon <= lon_max)
    )
    if qc is None:
        return np.where(in_domain, 0.0, np.nan)
    return np.where(in_domain, qc, np.nan)


def compute_stats_2d_latlon(values: np.ndarray, qc: Optional[np.ndarray],
                           lat: np.ndarray, lon: np.ndarray,
                           lat_bins: np.ndarray, lon_bins: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Bin scattered obs-space values (OMB or OMA innovations) into a lat-lon
    grid and compute mean/RMS/STD/count per cell.

    Uses MPAS-STATS' own grouped_stats_1d kernel (mpas_stats_diag_kernels),
    the same grouped-binning routine MPAS-STATS uses to build its model
    grid-cell "2D lat_lon" statistics -- so obs-space and model-space maps
    are built with identical binning math.

    lat_bins/lon_bins need not cover the whole globe -- see
    obsstat_common.resolve_latlon_bounds() for a REGIONAL application.
    Observations outside [lat_bins[0], lat_bins[-1]] x [lon_bins[0],
    lon_bins[-1]] are dropped entirely rather than folded into the edge
    bins, so a regional box's boundary cells aren't contaminated by
    out-of-domain data.
    """
    nlat = len(lat_bins) - 1
    nlon = len(lon_bins) - 1

    in_bounds = (
        (lat >= lat_bins[0]) & (lat <= lat_bins[-1]) &
        (lon >= lon_bins[0]) & (lon <= lon_bins[-1])
    )

    lat_idx = np.clip(np.digitize(lat, lat_bins) - 1, 0, nlat - 1)
    lon_idx = np.clip(np.digitize(lon, lon_bins) - 1, 0, nlon - 1)
    group_ids = (lat_idx * nlon + lon_idx).astype(np.int64)

    if qc is not None:
        masked = np.where(qc == 0, values, np.nan)
    else:
        masked = values
    masked = np.where(in_bounds, masked, np.nan)

    mean, rms, std = grouped_stats_1d(masked, group_ids, nlat * nlon)

    valid = np.isfinite(masked)
    count = np.bincount(group_ids[valid], minlength=nlat * nlon).astype(np.float64)

    shape = (nlat, nlon)
    return {
        "mean": mean.reshape(shape),
        "rmse": rms.reshape(shape),
        "std": std.reshape(shape),
        "count": count.reshape(shape),
    }

def compute_stats_2d_lat_vert(values: np.ndarray, qc: Optional[np.ndarray],
                             bin_values: np.ndarray, lat: np.ndarray,
                             levels: np.ndarray, bin_method: int,
                             lat_bins: np.ndarray, interval: float = None) -> Dict[str, np.ndarray]:
    """
    Bin scattered obs-space values jointly by vertical level (pressure or
    height) AND latitude, producing a (level x latitude) 2-D cross-section --
    the obs-space counterpart of MPAS-STATS' own lat_lev/lat_hgt/lat_pres
    2-D grids.

    Per-level bin edges are computed EXACTLY as in calculate_stats() (log-
    midpoint bins for pressure, +/-interval/2 for height), so a lat_vert
    slice collapsed over all latitude bins reproduces the same per-level
    numbers as the domain="global" 1-D profile -- kept as a cross-check in
    testing. Not meaningful for bin_method == 3 (radiances, whose "level"
    axis already IS latitude) -- callers must skip that case themselves.

    Domain-independent (like compute_stats_2d_latlon): lat_bins spans
    whatever the caller passes in -- see resolve_latlon_bounds() usage in
    process_experiment(), which reuses the SAME latlon_domain-bounded
    latitude range as the lat-lon map for consistency, rather than
    introducing a second, separately-configured latitude box.

    `interval`, when not given explicitly, is derived from `levels`'
    spacing exactly like calculate_stats() -- see that function's
    docstring for why a second, independently-hardcoded guess can
    silently drift out of sync with however `levels` was actually built.
    Never called for bin_method == 3 (radiances) -- see this function's
    "Not meaningful for..." note above -- so there is no domain-derived
    latitude case to handle here.
    """
    if interval is None:
        if bin_method == 2:
            levels_arr = np.asarray(levels, dtype=np.float64)
            interval = float(levels_arr[1] - levels_arr[0]) if levels_arr.size >= 2 else 2.0
        else:
            interval = None

    nlev = len(levels)
    nlat = len(lat_bins) - 1

    valid_mask = (qc == 0) if qc is not None else np.ones_like(values, dtype=bool)

    lat_in_bounds = (lat >= lat_bins[0]) & (lat <= lat_bins[-1])
    lat_idx = np.clip(np.digitize(lat, lat_bins) - 1, 0, nlat - 1)

    mean  = np.full((nlev, nlat), np.nan)
    rmse  = np.full((nlev, nlat), np.nan)
    std   = np.full((nlev, nlat), np.nan)
    count = np.zeros((nlev, nlat))

    log_levels = np.log(levels) if bin_method == 1 else None

    for ilev, level in enumerate(levels):
        if bin_method == 1:  # pressure -- same log-midpoint bins as calculate_stats
            if ilev == 0:
                top_bin = 200000.
                bot_bin = np.exp((log_levels[ilev] + log_levels[ilev + 1]) / 2.0)
            elif ilev == nlev - 1:
                top_bin = np.exp((log_levels[ilev] + log_levels[ilev - 1]) / 2.0)
                bot_bin = 0.
            else:
                top_bin = np.exp((log_levels[ilev] + log_levels[ilev - 1]) / 2.0)
                bot_bin = np.exp((log_levels[ilev] + log_levels[ilev + 1]) / 2.0)
        else:  # height
            bot_bin = level - interval / 2.
            top_bin = level + interval / 2.

        lev_mask = (bin_values > bot_bin) & (bin_values <= top_bin) & valid_mask & lat_in_bounds
        if not np.any(lev_mask):
            continue

        for ilat in range(nlat):
            cell_vals = values[lev_mask & (lat_idx == ilat)]
            if cell_vals.size and np.any(~np.isnan(cell_vals)):
                mean[ilev, ilat]  = np.nanmean(cell_vals)
                rmse[ilev, ilat]  = np.sqrt(np.nanmean(cell_vals**2))
                std[ilev, ilat]   = np.nanstd(cell_vals)
                count[ilev, ilat] = np.sum(~np.isnan(cell_vals))

    return {"mean": mean, "rmse": rmse, "std": std, "count": count}

def _accumulate_valid_latlon_bounds(obs_vec: Optional[Dict], bounds: List[Optional[float]]) -> None:
    """Update `bounds` (a mutable [lat_min, lat_max, lon_min, lon_max] list,
    entries None until the first valid observation is seen) in place from
    one file's observations -- used by auto_detect_latlon_bounds() below.

    "Valid" mirrors exactly what would earn a real (non-NaN/Inf) statistic
    downstream: a finite innovation that also passes QC (EffectiveQC/
    ObsError/hofx-missing masking -- see load_observation_data()), i.e.
    QC-flagged and interpolation-failed observations are excluded from the
    bounding box the same way they're excluded from every statistic.

    Does NOT handle a regional domain that straddles the +/-180 antimeridian
    (min/max longitude breaks there) -- the same limitation
    resolve_latlon_bounds()'s plain [lat_min, lat_max, lon_min, lon_max] box
    already has, so "auto" is no worse than a hand-specified box here.
    """
    if obs_vec is None or obs_vec.get("innov") is None:
        return
    innov = np.asarray(obs_vec["innov"], dtype=np.float64)
    valid = np.isfinite(innov)
    qc = obs_vec.get("qc")
    if qc is not None:
        valid &= (np.asarray(qc) == 0)
    if not np.any(valid):
        return
    lat = np.asarray(obs_vec["lat"], dtype=np.float64)[valid]
    lon = np.asarray(obs_vec["lon"], dtype=np.float64)[valid]
    if lat.size == 0:
        return
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    bounds[0] = lat_min if bounds[0] is None else min(bounds[0], lat_min)
    bounds[1] = lat_max if bounds[1] is None else max(bounds[1], lat_max)
    bounds[2] = lon_min if bounds[2] is None else min(bounds[2], lon_min)
    bounds[3] = lon_max if bounds[3] is None else max(bounds[3], lon_max)

def auto_detect_latlon_bounds(config: Dict, diag_obs: str, bin_method: int,
                             file_kind: str) -> Optional[Tuple[float, float, float, float]]:
    """(lat_min, lat_max, lon_min, lon_max) bounding box containing every
    observation with a finite, QC-passed innovation, scanned directly from
    the obsout files -- i.e. exactly the locations that would otherwise
    earn a real (non-NaN/Inf) statistic. Used by process_experiment() /
    process_forecast_experiment() when plot_settings.latlon_domain ==
    "auto", in place of a hand-guessed domains.<name> box, for a REGIONAL
    MPAS-JEDI application whose mesh extent isn't known up front.

    file_kind selects which files to scan, mirroring the two process_*
    functions' own file conventions: "cycling" walks every "da"/"hofx"
    experiment's background (OMB) file at every configured cycle
    (build_filename); "forecast" walks every "hofx_forecast" experiment's
    HofX file at every configured (init cycle, lead hour) pair
    (build_hofx_forecast_filename).

    Only the FIRST variable configured for this sensor
    (get_obs_config(diag_obs).variables[0]) is scanned, not every variable:
    an observation is invalid here because the model background/mesh
    doesn't cover its location, a property of the interpolation rather
    than of the physical field being verified, so one representative
    variable's valid/invalid footprint stands in for every other
    variable's. This keeps the scan a small fraction of the cost of the
    full statistics computation that follows it, rather than doubling
    every file read.

    Returns None if no configured cycle/experiment yielded any valid
    observation for this sensor (caller falls back to full-globe bounds --
    see the "auto" branches in process_experiment()/
    process_forecast_experiment()).
    """
    obs_config = get_obs_config(diag_obs)
    if not obs_config.variables:
        return None
    diag_var = obs_config.variables[0]
    nmembers = config["experiments"]["n_member"]

    bounds: List[Optional[float]] = [None, None, None, None]

    adjusted_start, end_time, interval = resolve_time_settings(
        config, diag_obs, for_forecast=(file_kind == "forecast"))
    times = generate_cycles(adjusted_start, end_time, interval)

    if file_kind == "forecast":
        hofx_dir = config["paths"].get("hofx_dir")
        leads = generate_forecast_leads(config)
        for exp_name, exp_type, nmember in zip(
            config["experiments"]["names"], config["experiments"]["types"], nmembers,
        ):
            if exp_type != "hofx_forecast":
                continue
            for init_time in times:
                for lead in leads:
                    # Skip exactly the (init, lead) pairs the real statistics
                    # computation will also skip (see lead_valid_time_allowed's
                    # docstring) -- otherwise the auto-detected domain could be
                    # inflated by a lead whose valid time never earns a real
                    # statistic, or (more subtly) miss extent that a *counted*
                    # lead would have contributed.
                    if not lead_valid_time_allowed(diag_obs, init_time, lead):
                        continue
                    filename = build_hofx_forecast_filename(hofx_dir, exp_name, init_time, lead, diag_obs)
                    omf_vec = load_observation_data(
                        filename, diag_obs, diag_var, bin_method, outerloop=0, nmember=nmember,
                    )
                    _accumulate_valid_latlon_bounds(omf_vec, bounds)
    else:  # "cycling" -- OMB/OMA experiments (exp_type in "da"/"hofx")
        for exp_name, exp_type, nmember in zip(
            config["experiments"]["names"], config["experiments"]["types"], nmembers,
        ):
            if exp_type == "hofx_forecast":
                continue
            base_dir = config["paths"]["da_dir"] if exp_type == "da" else config["paths"]["hofx_dir"]
            for cycle_time in times:
                filedict = build_filename(base_dir, exp_name, exp_type, nmember, cycle_time, diag_obs)
                omb_vec = load_observation_data(
                    filedict["bak"], diag_obs, diag_var, bin_method, outerloop=0, nmember=nmember,
                )
                _accumulate_valid_latlon_bounds(omb_vec, bounds)

    if bounds[0] is None:
        return None
    return tuple(bounds)  # type: ignore[return-value]

def _resolve_latlon_bounds_for(config: Dict, diag_obs: str, bin_method: int,
                              file_kind: str) -> Tuple[float, float, float, float]:
    """resolve_latlon_bounds(config), except when plot_settings.latlon_domain
    == "auto": that sentinel has no file access from obsstat_common.py, so
    it's resolved here instead via auto_detect_latlon_bounds() (see that
    function; file_kind picks which files it scans). Falls back to
    full-globe bounds if the scan finds no valid observation at all for
    this sensor (e.g. no diagnosis files exist yet), with a warning --
    same "nothing to show yet" behaviour as every other empty-input case
    in this module, rather than raising."""
    if config["plot_settings"].get("latlon_domain") != "auto":
        return resolve_latlon_bounds(config)
    bounds = auto_detect_latlon_bounds(config, diag_obs, bin_method, file_kind)
    if bounds is None:
        print(f"WARNING: latlon_domain: auto found no valid observations for {diag_obs} "
              f"({file_kind}); falling back to full-globe bounds.")
        return -90.0, 90.0, -180.0, 180.0
    return bounds

def process_experiment(config: Dict, diag_obs: str) -> Dict:
    """Process diagnostics for a single observation type"""

    obs_config = get_obs_config(diag_obs)
    bin_method = get_bin_method(diag_obs)

    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs)
    times = generate_cycles(adjusted_start, end_time, interval)

    domains = resolve_domains(config)

    # --------------------------------------------------
    # Lat-lon grid (2-D statistics). Covers the whole globe by default, or
    # just the plot_settings.latlon_domain box for a REGIONAL application
    # -- see resolve_latlon_bounds() (a hand-configured domains.<name> box),
    # or, when latlon_domain == "auto", auto_detect_latlon_bounds() (the
    # tightest box containing every observation with a valid, QC-passed
    # OMB innovation -- see _resolve_latlon_bounds_for()). Independent of
    # the 0D/1D `domains` masking loop below (mirrors MPAS-STATS' own
    # domain-independent "2D lat_lon" product), though a hand-configured
    # box may reuse one of the same `domains` entries as its bounding box.
    #
    # Resolved BEFORE `levels` below (rather than after, as in an earlier
    # version of this function) because bin_method == 3 (radiances) reuses
    # this SAME lat_min_2d/lat_max_2d/lat_bin_width for its "1D latitude
    # profile" levels -- see the get_levels() call below.
    # --------------------------------------------------
    gen_2d = bool(config["plot_settings"].get("generate_lat_lon_2d", True))
    lat_bin_width = float(config["plot_settings"].get("latlon_lat_bin_width", 10.0))
    lon_bin_width = float(config["plot_settings"].get("latlon_lon_bin_width", 10.0))
    lat_min_2d, lat_max_2d, lon_min_2d, lon_max_2d = _resolve_latlon_bounds_for(
        config, diag_obs, bin_method, file_kind="cycling")
    lat_bins_2d = np.arange(lat_min_2d, lat_max_2d + lat_bin_width, lat_bin_width)
    lon_bins_2d = np.arange(lon_min_2d, lon_max_2d + lon_bin_width, lon_bin_width)
    lat_centers_2d = 0.5 * (lat_bins_2d[:-1] + lat_bins_2d[1:])
    lon_centers_2d = 0.5 * (lon_bins_2d[:-1] + lon_bins_2d[1:])
    nlat2d, nlon2d = len(lat_centers_2d), len(lon_centers_2d)

    # --------------------------------------------------
    # Lat-vert grid (2-D level x latitude cross-section). Reuses the SAME
    # latlon_domain-bounded latitude range/bin width as the lat-lon map
    # above (lat_bins_2d), rather than a second, separately-configured
    # latitude box -- see compute_stats_2d_lat_vert(). Not generated for
    # radiances (bin_method == 3), whose "level" axis already IS latitude.
    # --------------------------------------------------
    gen_lat_vert = bool(config["plot_settings"].get("generate_lat_vert_2d", True)) and bin_method != 3

    # Vertical/latitude bin levels for the 1D profile (and, for
    # pressure/height sensors, the 2D lat-vert level axis too). For
    # radiances (bin_method == 3) this axis IS latitude -- reuse the
    # domain-aware lat_min_2d/lat_max_2d/lat_bin_width resolved just above
    # instead of get_levels()'s hardcoded global -80..80 default, UNLESS
    # the sensor has its own explicit level_bin_kwargs override
    # (obs_sensor_config.py) -- an explicit per-sensor override always
    # wins over the domain-derived default.
    level_bin_kwargs = obs_config.level_bin_kwargs or {}
    if bin_method == 3 and not level_bin_kwargs:
        levels = get_levels(bin_method, bin_width=lat_bin_width,
                            domain_range=(lat_min_2d, lat_max_2d))
    else:
        levels = get_levels(bin_method, **level_bin_kwargs)

    num_exp = len(config["experiments"]["names"])
    num_dom = len(domains)
    num_time = len(times)
    num_var = len(obs_config.variables)
    num_lev = len(levels)

    def nan_array(*shape):
        return np.full(shape, np.nan)

    results = {
        # OMB
        "rmse_omb": nan_array(num_exp, num_dom, num_time, num_var),
        "mean_omb": nan_array(num_exp, num_dom, num_time, num_var),
        "std_omb":  nan_array(num_exp, num_dom, num_time, num_var),
        "count_omb": nan_array(num_exp, num_dom, num_time, num_var),
        "spread_omb": nan_array(num_exp, num_dom, num_time, num_var),
        "totSpread_omb": nan_array(num_exp, num_dom, num_time, num_var),

        "rmse_omb_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "mean_omb_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "std_omb_bin":  nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "count_omb_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "spread_omb_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "totSpread_omb_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),

        # OMA
        "rmse_oma": nan_array(num_exp, num_dom, num_time, num_var),
        "mean_oma": nan_array(num_exp, num_dom, num_time, num_var),
        "std_oma":  nan_array(num_exp, num_dom, num_time, num_var),
        "count_oma": nan_array(num_exp, num_dom, num_time, num_var),
        "spread_oma": nan_array(num_exp, num_dom, num_time, num_var),
        "totSpread_oma": nan_array(num_exp, num_dom, num_time, num_var),

        "rmse_oma_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "mean_oma_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "std_oma_bin":  nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "count_oma_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "spread_oma_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),
        "totSpread_oma_bin": nan_array(num_exp, num_dom, num_time, num_lev, num_var),

        # Metadata
        "times": times,
        "levels": levels,
        "variables": obs_config.variables,
        "units": obs_config.units,
        "labels": obs_config.labels,
        "domain_names": list(domains.keys()),
        "domain_coords": list(domains.values()),
    }

    if gen_2d:
        results.update({
            "rmse_omb_latlon": nan_array(num_exp, num_time, nlat2d, nlon2d, num_var),
            "mean_omb_latlon": nan_array(num_exp, num_time, nlat2d, nlon2d, num_var),
            "std_omb_latlon":  nan_array(num_exp, num_time, nlat2d, nlon2d, num_var),
            "count_omb_latlon": nan_array(num_exp, num_time, nlat2d, nlon2d, num_var),
            "rmse_oma_latlon": nan_array(num_exp, num_time, nlat2d, nlon2d, num_var),
            "mean_oma_latlon": nan_array(num_exp, num_time, nlat2d, nlon2d, num_var),
            "std_oma_latlon":  nan_array(num_exp, num_time, nlat2d, nlon2d, num_var),
            "count_oma_latlon": nan_array(num_exp, num_time, nlat2d, nlon2d, num_var),
            "latlon_lat_centers": lat_centers_2d,
            "latlon_lon_centers": lon_centers_2d,
        })

    if gen_lat_vert:
        results.update({
            "rmse_omb_latvert": nan_array(num_exp, num_time, num_lev, nlat2d, num_var),
            "mean_omb_latvert": nan_array(num_exp, num_time, num_lev, nlat2d, num_var),
            "std_omb_latvert":  nan_array(num_exp, num_time, num_lev, nlat2d, num_var),
            "count_omb_latvert": nan_array(num_exp, num_time, num_lev, nlat2d, num_var),
            "rmse_oma_latvert": nan_array(num_exp, num_time, num_lev, nlat2d, num_var),
            "mean_oma_latvert": nan_array(num_exp, num_time, num_lev, nlat2d, num_var),
            "std_oma_latvert":  nan_array(num_exp, num_time, num_lev, nlat2d, num_var),
            "count_oma_latvert": nan_array(num_exp, num_time, num_lev, nlat2d, num_var),
            "latvert_lat_centers": lat_centers_2d,
        })

    # ==================================================
    # Main loop
    # ==================================================
    for iexp, (exp_name, exp_type, nmember) in enumerate(
        zip(
            config["experiments"]["names"],
            config["experiments"]["types"],
            config["experiments"]["n_member"],
        )
    ):
        if exp_type == "hofx_forecast":
            # OMF (extended-forecast) experiments are processed separately
            # by process_forecast_experiment() -- they don't fit the
            # single-file-per-cycle convention this loop assumes.
            continue

        base_dir = (
            config["paths"]["da_dir"]
            if exp_type == "da"
            else config["paths"]["hofx_dir"]
        )

        outerloop_ana = config["experiments"]["outerloop"][iexp]

        for it, cycle_time in enumerate(times):

            filedict = build_filename(
                base_dir, exp_name, exp_type, nmember, cycle_time, diag_obs
            )

            for iv, diag_var in enumerate(obs_config.variables):

                # ---------------------------
                # Load OMB / OMA
                # ---------------------------
                omb_vec = load_observation_data(
                    filedict["bak"],
                    diag_obs,
                    diag_var,
                    bin_method,
                    outerloop=0,
                    nmember=nmember,
                )

                if filedict["ana"]:
                    oma_vec = load_observation_data(
                        filedict["ana"],
                        diag_obs,
                        diag_var,
                        bin_method,
                        outerloop=outerloop_ana,
                        nmember=nmember,
                    )
                else:
                    oma_vec = None

                if omb_vec is None:
                    continue

                # ---------------------------
                # Bin coordinate
                # ---------------------------
                bin_values = (
                    omb_vec["pres"] if bin_method == 1 else
                    omb_vec["hgt"]  if bin_method == 2 else
                    omb_vec["lat"]
                )

                # ---------------------------
                # 2-D lat-lon gridded stats (global, domain-independent)
                # ---------------------------
                if gen_2d:
                    latlon_omb = compute_stats_2d_latlon(
                        omb_vec["innov"], omb_vec["qc"],
                        omb_vec["lat"], omb_vec["lon"],
                        lat_bins_2d, lon_bins_2d,
                    )
                    for stat_disp, prefix in STAT_KEYS_2D:
                        results[f"{prefix}_omb_latlon"][iexp, it, :, :, iv] = latlon_omb[prefix]

                    if oma_vec is not None and oma_vec["innov"] is not None:
                        latlon_oma = compute_stats_2d_latlon(
                            oma_vec["innov"], oma_vec["qc"],
                            oma_vec["lat"], oma_vec["lon"],
                            lat_bins_2d, lon_bins_2d,
                        )
                        for stat_disp, prefix in STAT_KEYS_2D:
                            results[f"{prefix}_oma_latlon"][iexp, it, :, :, iv] = latlon_oma[prefix]

                # ---------------------------
                # 2-D lat-vert (level x latitude) cross-section --
                # domain-independent, skipped for radiances
                # ---------------------------
                if gen_lat_vert:
                    latvert_omb = compute_stats_2d_lat_vert(
                        omb_vec["innov"], omb_vec["qc"],
                        bin_values, omb_vec["lat"],
                        levels, bin_method, lat_bins_2d,
                    )
                    for stat_disp, prefix in STAT_KEYS_2D:
                        results[f"{prefix}_omb_latvert"][iexp, it, :, :, iv] = latvert_omb[prefix]

                    if oma_vec is not None and oma_vec["innov"] is not None:
                        oma_bin_values = (
                            oma_vec["pres"] if bin_method == 1 else
                            oma_vec["hgt"]  if bin_method == 2 else
                            oma_vec["lat"]
                        )
                        latvert_oma = compute_stats_2d_lat_vert(
                            oma_vec["innov"], oma_vec["qc"],
                            oma_bin_values, oma_vec["lat"],
                            levels, bin_method, lat_bins_2d,
                        )
                        for stat_disp, prefix in STAT_KEYS_2D:
                            results[f"{prefix}_oma_latvert"][iexp, it, :, :, iv] = latvert_oma[prefix]

                # ---------------------------
                # Domain loop
                # ---------------------------
                for idom, (dom_name, dom_coords) in enumerate(domains.items()):

                    omb_qc = _apply_domain_mask(
                        omb_vec["qc"], omb_vec["lat"], omb_vec["lon"],
                        dom_name, dom_coords,
                    )
                    oma_qc = (
                        _apply_domain_mask(
                            oma_vec["qc"], oma_vec["lat"], oma_vec["lon"],
                            dom_name, dom_coords,
                        )
                        if oma_vec is not None
                        else None
                    )

                    # =======================
                    # OMB stats
                    # =======================
                    omb_stats = calculate_stats(
                        omb_vec["innov"],
                        omb_qc,
                        bin_values,
                        levels,
                        bin_method,
                    )

                    if omb_stats["count"][num_lev] > 0:
                        results["rmse_omb"][iexp, idom, it, iv] = omb_stats["rmse"][num_lev]
                        results["mean_omb"][iexp, idom, it, iv] = omb_stats["mean"][num_lev]
                        results["std_omb"][iexp, idom, it, iv]  = omb_stats["std"][num_lev]
                        results["count_omb"][iexp, idom, it, iv] = omb_stats["count"][num_lev]

                    valid = omb_stats["count"][:num_lev] > 0
                    results["rmse_omb_bin"][iexp, idom, it, valid, iv] = omb_stats["rmse"][:num_lev][valid]
                    results["mean_omb_bin"][iexp, idom, it, valid, iv] = omb_stats["mean"][:num_lev][valid]
                    results["std_omb_bin"][iexp, idom, it, valid, iv]  = omb_stats["std"][:num_lev][valid]
                    results["count_omb_bin"][iexp, idom, it, valid, iv] = omb_stats["count"][:num_lev][valid]

                    if omb_vec["spread"] is not None:
                        omb_stats = calculate_stats(
                            omb_vec["spread"],
                            omb_qc,
                            bin_values,
                            levels,
                            bin_method,
                        )
                        if omb_stats["count"][num_lev] > 0:
                            results["spread_omb"][iexp, idom, it, iv] = omb_stats["mean"][num_lev]
                        valid = omb_stats["count"][:num_lev] > 0
                        results["spread_omb_bin"][iexp, idom, it, valid, iv] = omb_stats["mean"][:num_lev][valid]

                    if omb_vec["totSpread"] is not None:
                        omb_stats = calculate_stats(
                            omb_vec["totSpread"],
                            omb_qc,
                            bin_values,
                            levels,
                            bin_method,
                        )
                        if omb_stats["count"][num_lev] > 0:
                            results["totSpread_omb"][iexp, idom, it, iv] = omb_stats["mean"][num_lev]
                        valid = omb_stats["count"][:num_lev] > 0
                        results["totSpread_omb_bin"][iexp, idom, it, valid, iv] = omb_stats["mean"][:num_lev][valid]

                    # =======================
                    # OMA stats
                    # =======================
                    if oma_vec is not None and oma_vec["innov"] is not None:
                        # OMA may come from a separate `_ana.h5` file whose
                        # observation ordering/length is not guaranteed to
                        # match OMB, so bin it with its own coordinates.
                        oma_bin_values = (
                            oma_vec["pres"] if bin_method == 1 else
                            oma_vec["hgt"]  if bin_method == 2 else
                            oma_vec["lat"]
                        )
                        oma_stats = calculate_stats(
                            oma_vec["innov"],
                            oma_qc,
                            oma_bin_values,
                            levels,
                            bin_method,
                        )

                        if oma_stats["count"][num_lev] > 0:
                            results["rmse_oma"][iexp, idom, it, iv] = oma_stats["rmse"][num_lev]
                            results["mean_oma"][iexp, idom, it, iv] = oma_stats["mean"][num_lev]
                            results["std_oma"][iexp, idom, it, iv]  = oma_stats["std"][num_lev]
                            results["count_oma"][iexp, idom, it, iv] = oma_stats["count"][num_lev]

                        valid = oma_stats["count"][:num_lev] > 0
                        results["rmse_oma_bin"][iexp, idom, it, valid, iv] = oma_stats["rmse"][:num_lev][valid]
                        results["mean_oma_bin"][iexp, idom, it, valid, iv] = oma_stats["mean"][:num_lev][valid]
                        results["std_oma_bin"][iexp, idom, it, valid, iv]  = oma_stats["std"][:num_lev][valid]
                        results["count_oma_bin"][iexp, idom, it, valid, iv] = oma_stats["count"][:num_lev][valid]

                        if oma_vec["spread"] is not None:
                            oma_stats = calculate_stats(
                                oma_vec["spread"],
                                oma_qc,
                                oma_bin_values,
                                levels,
                                bin_method,
                            )
                            if oma_stats["count"][num_lev] > 0:
                                results["spread_oma"][iexp, idom, it, iv] = oma_stats["mean"][num_lev]
                            valid = oma_stats["count"][:num_lev] > 0
                            results["spread_oma_bin"][iexp, idom, it, valid, iv] = oma_stats["mean"][:num_lev][valid]

                        if oma_vec["totSpread"] is not None:
                            oma_stats = calculate_stats(
                                oma_vec["totSpread"],
                                oma_qc,
                                oma_bin_values,
                                levels,
                                bin_method,
                            )
                            if oma_stats["count"][num_lev] > 0:
                                results["totSpread_oma"][iexp, idom, it, iv] = oma_stats["mean"][num_lev]
                            valid = oma_stats["count"][:num_lev] > 0
                            results["totSpread_oma_bin"][iexp, idom, it, valid, iv] = oma_stats["mean"][:num_lev][valid]

    return results

# ============================================================================
# Observation-space diagnosis files
# (same <var>/0D/<region>/<stat>, <var>/1D/<profile_key>/<stat>,
#  <var>/2D/lat_lon/<stat> convention as mpas_stats_diag_driver.py)
# ============================================================================

def write_obs_diag_files(results: Dict, config: Dict, diag_obs: str, diag_base_dir: str) -> None:
    """
    Write one observation-space diagnosis netCDF file per (experiment, cycle),
    following the same 0D/1D/2D group convention MPAS-STATS uses for its own
    mpasstat.<date>.nc model-space statistics files.
    """
    bin_method = get_bin_method(diag_obs)
    var_names = list(results["variables"])
    domains = results["domain_names"]
    levels = np.asarray(results["levels"], dtype=np.float64)
    times = results["times"]
    lat_centers = results.get("latlon_lat_centers")
    lon_centers = results.get("latlon_lon_centers")
    latvert_lat_centers = results.get("latvert_lat_centers")

    level_units = {1: "hPa", 2: "km", 3: "degrees_north"}[bin_method]

    exp_names = config["experiments"]["names"]
    exp_types = config["experiments"].get("types", ["da"] * len(exp_names))
    plot_oma = config["plot_settings"].get("plot_oma", True)

    global_idx = domains.index("global") if "global" in domains else None

    for iexp, exp_name in enumerate(exp_names):
        if exp_types[iexp] == "hofx_forecast":
            # written separately by write_obs_forecast_diag_files()
            continue

        diagtypes = ["omb"]
        if exp_types[iexp] == "da" and plot_oma:
            diagtypes.append("oma")

        for it, cycle_time in enumerate(times):
            out_file = obs_diag_file(diag_base_dir, exp_name, diag_obs, cycle_time)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

            with nc4.Dataset(out_file, "w") as ds:
                ds.createDimension("nLevels", len(levels))
                lev_var = ds.createVariable("Levels", "f4", ("nLevels",))
                lev_var[:] = levels.astype(np.float32)
                lev_var.units = level_units

                have_2d = lat_centers is not None and lon_centers is not None
                have_latvert = latvert_lat_centers is not None
                # lat_vert reuses the SAME latitude centers as the lat-lon
                # map (see resolve_latlon_bounds()), so a single nLatBins2D
                # dimension/LatBins2D variable serves both -- created if
                # either product is enabled, from whichever centers array
                # is available (they are numerically identical).
                if have_2d or have_latvert:
                    any_lat_centers = lat_centers if have_2d else latvert_lat_centers
                    ds.createDimension("nLatBins2D", len(any_lat_centers))
                    lb = ds.createVariable("LatBins2D", "f4", ("nLatBins2D",))
                    lb[:] = np.asarray(any_lat_centers, dtype=np.float32)
                    lb.units = "degrees_north"
                if have_2d:
                    ds.createDimension("nLonBins2D", len(lon_centers))
                    lo = ds.createVariable("LonBins2D", "f4", ("nLonBins2D",))
                    lo[:] = np.asarray(lon_centers, dtype=np.float32)
                    lo.units = "degrees_east"

                for diagtype in diagtypes:
                    diag_grp = ds.createGroup(diagtype)

                    for iv, var in enumerate(var_names):
                        var_grp = diag_grp.createGroup(nc_group_name(var))

                        # ---- 0D: domain-averaged scalar stats ----
                        d0 = var_grp.createGroup("0D")
                        for idom, domain in enumerate(domains):
                            region_grp = d0.createGroup(domain)
                            for stat_disp, prefix in STAT_KEYS:
                                key = f"{prefix}_{diagtype}"
                                if key not in results:
                                    continue
                                val = results[key][iexp, idom, it, iv]
                                var_out = region_grp.createVariable(stat_disp, "f4", ())
                                var_out[:] = np.float32(val)

                        # ---- 1D: binned profile stats ----
                        d1 = var_grp.createGroup("1D")
                        if bin_method == 3:
                            profile_entries = ([("lat_bin_profile", global_idx)]
                                                if global_idx is not None else [])
                        else:
                            profile_entries = [
                                (profile_key_for_domain(bin_method, domain), idom)
                                for idom, domain in enumerate(domains)
                            ]
                        for pkey, idom in profile_entries:
                            prof_grp = d1.createGroup(pkey)
                            for stat_disp, prefix in STAT_KEYS:
                                key = f"{prefix}_{diagtype}_bin"
                                if key not in results:
                                    continue
                                arr = results[key][iexp, idom, it, :, iv]
                                var_out = prof_grp.createVariable(stat_disp, "f4", ("nLevels",))
                                var_out[:] = arr.astype(np.float32)

                        # ---- 2D: lat-lon gridded stats (global) ----
                        if have_2d:
                            d2 = var_grp.createGroup("2D").createGroup("lat_lon")
                            for stat_disp, prefix in STAT_KEYS_2D:
                                key = f"{prefix}_{diagtype}_latlon"
                                if key not in results:
                                    continue
                                arr = results[key][iexp, it, :, :, iv]
                                var_out = d2.createVariable(
                                    stat_disp, "f4", ("nLatBins2D", "nLonBins2D")
                                )
                                var_out[:] = arr.astype(np.float32)

                        # ---- 2D: lat-vert (level x latitude) cross-section
                        #      -- domain-independent, skipped for radiances ----
                        if have_latvert:
                            d2v = var_grp.groups.get("2D") or var_grp.createGroup("2D")
                            d2v = d2v.createGroup("lat_vert")
                            for stat_disp, prefix in STAT_KEYS_2D:
                                key = f"{prefix}_{diagtype}_latvert"
                                if key not in results:
                                    continue
                                arr = results[key][iexp, it, :, :, iv]
                                var_out = d2v.createVariable(
                                    stat_disp, "f4", ("nLevels", "nLatBins2D")
                                )
                                var_out[:] = arr.astype(np.float32)

            print(f"Wrote obs diagnosis file: {out_file}")

# ============================================================================
# OMF (obs-minus-forecast, extended-forecast skill) statistics
#
# Same 0D/1D/2D group convention as process_experiment() /
# write_obs_diag_files() above, but resolved at every (init cycle, forecast
# lead hour) pair instead of just per assimilation cycle -- mirroring how
# the model-space package's run_verify_model_fcst.ksh points
# mpas_stats_diag_model.py at ExtendedFC/<cycle>/mpasout.<valid_date>.nc
# files instead of CyclingFC ones, one call per lead hour.
# ============================================================================

def process_forecast_experiment(config: Dict, diag_obs: str) -> Dict:
    """Process OMF (extended-forecast) diagnostics for a single observation
    type, across every configured forecast lead hour. Only experiments with
    type == "hofx_forecast" contribute -- see build_hofx_forecast_filename()
    for the expected HofX directory layout, and config["forecast"] (length/
    interval/start, same schema as the model-space graphic.yaml) for the
    lead-hour list."""

    obs_config = get_obs_config(diag_obs)
    bin_method = get_bin_method(diag_obs)

    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs, for_forecast=True)
    times = generate_cycles(adjusted_start, end_time, interval)
    leads = generate_forecast_leads(config)

    domains = resolve_domains(config)

    # See resolve_latlon_bounds() -- defaults to full-globe coverage, or a
    # REGIONAL application's own extent when plot_settings.latlon_domain
    # points at a bounded `domains` entry, or (latlon_domain == "auto")
    # the tightest box containing every observation with a valid,
    # QC-passed OMF innovation -- see auto_detect_latlon_bounds() /
    # _resolve_latlon_bounds_for().
    #
    # Resolved BEFORE `levels` below (rather than after, as in an earlier
    # version of this function) because bin_method == 3 (radiances) reuses
    # this SAME lat_min_2d/lat_max_2d/lat_bin_width for its "1D latitude
    # profile" levels -- see the get_levels() call below.
    gen_2d = bool(config["plot_settings"].get("generate_lat_lon_2d", True))
    lat_bin_width = float(config["plot_settings"].get("latlon_lat_bin_width", 10.0))
    lon_bin_width = float(config["plot_settings"].get("latlon_lon_bin_width", 10.0))
    lat_min_2d, lat_max_2d, lon_min_2d, lon_max_2d = _resolve_latlon_bounds_for(
        config, diag_obs, bin_method, file_kind="forecast")
    lat_bins_2d = np.arange(lat_min_2d, lat_max_2d + lat_bin_width, lat_bin_width)
    lon_bins_2d = np.arange(lon_min_2d, lon_max_2d + lon_bin_width, lon_bin_width)
    lat_centers_2d = 0.5 * (lat_bins_2d[:-1] + lat_bins_2d[1:])
    lon_centers_2d = 0.5 * (lon_bins_2d[:-1] + lon_bins_2d[1:])
    nlat2d, nlon2d = len(lat_centers_2d), len(lon_centers_2d)

    # 2-D lat-vert (level x latitude) cross-section, at every forecast lead
    # -- reuses the SAME latlon_domain-bounded latitude range/bin width as
    # the lat-lon map above (lat_bins_2d), exactly like process_experiment()'s
    # own gen_lat_vert handling for OMB/OMA. Not generated for radiances
    # (bin_method == 3), whose "level" axis already IS latitude.
    gen_lat_vert = bool(config["plot_settings"].get("generate_lat_vert_2d", True)) and bin_method != 3

    # Vertical/latitude bin levels for the 1D profile -- see
    # process_experiment()'s identical get_levels() call for why bin_method
    # == 3 (radiances) reuses lat_min_2d/lat_max_2d/lat_bin_width instead of
    # a hardcoded global default, unless the sensor has its own explicit
    # level_bin_kwargs override.
    level_bin_kwargs = obs_config.level_bin_kwargs or {}
    if bin_method == 3 and not level_bin_kwargs:
        levels = get_levels(bin_method, bin_width=lat_bin_width,
                            domain_range=(lat_min_2d, lat_max_2d))
    else:
        levels = get_levels(bin_method, **level_bin_kwargs)

    num_exp = len(config["experiments"]["names"])
    num_dom = len(domains)
    num_time = len(times)
    num_lead = len(leads)
    num_var = len(obs_config.variables)
    num_lev = len(levels)

    def nan_array(*shape):
        return np.full(shape, np.nan)

    results = {
        "rmse_omf": nan_array(num_exp, num_dom, num_time, num_lead, num_var),
        "mean_omf": nan_array(num_exp, num_dom, num_time, num_lead, num_var),
        "std_omf":  nan_array(num_exp, num_dom, num_time, num_lead, num_var),
        "count_omf": nan_array(num_exp, num_dom, num_time, num_lead, num_var),
        "spread_omf": nan_array(num_exp, num_dom, num_time, num_lead, num_var),
        "totSpread_omf": nan_array(num_exp, num_dom, num_time, num_lead, num_var),

        "rmse_omf_bin": nan_array(num_exp, num_dom, num_time, num_lead, num_lev, num_var),
        "mean_omf_bin": nan_array(num_exp, num_dom, num_time, num_lead, num_lev, num_var),
        "std_omf_bin":  nan_array(num_exp, num_dom, num_time, num_lead, num_lev, num_var),
        "count_omf_bin": nan_array(num_exp, num_dom, num_time, num_lead, num_lev, num_var),
        "spread_omf_bin": nan_array(num_exp, num_dom, num_time, num_lead, num_lev, num_var),
        "totSpread_omf_bin": nan_array(num_exp, num_dom, num_time, num_lead, num_lev, num_var),

        # Metadata
        "times": times,
        "leads": leads,
        "levels": levels,
        "variables": obs_config.variables,
        "units": obs_config.units,
        "labels": obs_config.labels,
        "domain_names": list(domains.keys()),
        "domain_coords": list(domains.values()),
    }

    if gen_2d:
        results.update({
            "rmse_omf_latlon": nan_array(num_exp, num_time, num_lead, nlat2d, nlon2d, num_var),
            "mean_omf_latlon": nan_array(num_exp, num_time, num_lead, nlat2d, nlon2d, num_var),
            "std_omf_latlon":  nan_array(num_exp, num_time, num_lead, nlat2d, nlon2d, num_var),
            "count_omf_latlon": nan_array(num_exp, num_time, num_lead, nlat2d, nlon2d, num_var),
            "latlon_lat_centers": lat_centers_2d,
            "latlon_lon_centers": lon_centers_2d,
        })

    if gen_lat_vert:
        results.update({
            "rmse_omf_latvert": nan_array(num_exp, num_time, num_lead, num_lev, nlat2d, num_var),
            "mean_omf_latvert": nan_array(num_exp, num_time, num_lead, num_lev, nlat2d, num_var),
            "std_omf_latvert":  nan_array(num_exp, num_time, num_lead, num_lev, nlat2d, num_var),
            "count_omf_latvert": nan_array(num_exp, num_time, num_lead, num_lev, nlat2d, num_var),
            "latvert_lat_centers": lat_centers_2d,
        })

    hofx_dir = config["paths"].get("hofx_dir")

    for iexp, (exp_name, exp_type, nmember) in enumerate(
        zip(
            config["experiments"]["names"],
            config["experiments"]["types"],
            config["experiments"]["n_member"],
        )
    ):
        if exp_type != "hofx_forecast":
            continue

        for it, init_time in enumerate(times):
            for il, lead in enumerate(leads):

                # The INIT time list above is deliberately unrestricted (see
                # resolve_time_settings()'s for_forecast=True docstring) --
                # a forecast can launch at any hour the model cycles at. What
                # a sensor's cycle_valid_hours actually restricts is whether
                # THIS LEAD's valid time (init_time + lead) lands on an hour
                # that sensor actually observes at (e.g. sondes only launch
                # at 00Z/12Z). A disallowed lead is simply skipped here,
                # leaving its slot at the NaN fill value already in `results`
                # -- same convention as any other kind of missing data.
                if not lead_valid_time_allowed(diag_obs, init_time, lead):
                    continue

                filename = build_hofx_forecast_filename(
                    hofx_dir, exp_name, init_time, lead, diag_obs
                )

                for iv, diag_var in enumerate(obs_config.variables):

                    omf_vec = load_observation_data(
                        filename, diag_obs, diag_var, bin_method,
                        outerloop=0, nmember=nmember,
                    )
                    if omf_vec is None:
                        continue

                    bin_values = (
                        omf_vec["pres"] if bin_method == 1 else
                        omf_vec["hgt"]  if bin_method == 2 else
                        omf_vec["lat"]
                    )

                    if gen_2d:
                        latlon_omf = compute_stats_2d_latlon(
                            omf_vec["innov"], omf_vec["qc"],
                            omf_vec["lat"], omf_vec["lon"],
                            lat_bins_2d, lon_bins_2d,
                        )
                        for stat_disp, prefix in STAT_KEYS_2D:
                            results[f"{prefix}_omf_latlon"][iexp, it, il, :, :, iv] = latlon_omf[prefix]

                    if gen_lat_vert:
                        latvert_omf = compute_stats_2d_lat_vert(
                            omf_vec["innov"], omf_vec["qc"],
                            bin_values, omf_vec["lat"],
                            levels, bin_method, lat_bins_2d,
                        )
                        for stat_disp, prefix in STAT_KEYS_2D:
                            results[f"{prefix}_omf_latvert"][iexp, it, il, :, :, iv] = latvert_omf[prefix]

                    for idom, (dom_name, dom_coords) in enumerate(domains.items()):

                        omf_qc = _apply_domain_mask(
                            omf_vec["qc"], omf_vec["lat"], omf_vec["lon"],
                            dom_name, dom_coords,
                        )

                        omf_stats = calculate_stats(
                            omf_vec["innov"], omf_qc, bin_values, levels, bin_method,
                        )

                        if omf_stats["count"][num_lev] > 0:
                            results["rmse_omf"][iexp, idom, it, il, iv] = omf_stats["rmse"][num_lev]
                            results["mean_omf"][iexp, idom, it, il, iv] = omf_stats["mean"][num_lev]
                            results["std_omf"][iexp, idom, it, il, iv]  = omf_stats["std"][num_lev]
                            results["count_omf"][iexp, idom, it, il, iv] = omf_stats["count"][num_lev]

                        valid = omf_stats["count"][:num_lev] > 0
                        results["rmse_omf_bin"][iexp, idom, it, il, valid, iv] = omf_stats["rmse"][:num_lev][valid]
                        results["mean_omf_bin"][iexp, idom, it, il, valid, iv] = omf_stats["mean"][:num_lev][valid]
                        results["std_omf_bin"][iexp, idom, it, il, valid, iv]  = omf_stats["std"][:num_lev][valid]
                        results["count_omf_bin"][iexp, idom, it, il, valid, iv] = omf_stats["count"][:num_lev][valid]

                        if omf_vec["spread"] is not None:
                            spread_stats = calculate_stats(
                                omf_vec["spread"], omf_qc, bin_values, levels, bin_method,
                            )
                            if spread_stats["count"][num_lev] > 0:
                                results["spread_omf"][iexp, idom, it, il, iv] = spread_stats["mean"][num_lev]
                            valid = spread_stats["count"][:num_lev] > 0
                            results["spread_omf_bin"][iexp, idom, it, il, valid, iv] = spread_stats["mean"][:num_lev][valid]

                        if omf_vec["totSpread"] is not None:
                            total_stats = calculate_stats(
                                omf_vec["totSpread"], omf_qc, bin_values, levels, bin_method,
                            )
                            if total_stats["count"][num_lev] > 0:
                                results["totSpread_omf"][iexp, idom, it, il, iv] = total_stats["mean"][num_lev]
                            valid = total_stats["count"][:num_lev] > 0
                            results["totSpread_omf_bin"][iexp, idom, it, il, valid, iv] = total_stats["mean"][:num_lev][valid]

    return results

def write_obs_forecast_diag_files(results: Dict, config: Dict, diag_obs: str, diag_base_dir: str) -> None:
    """
    Write one OMF diagnosis netCDF file per (experiment, init cycle, forecast
    lead hour), following the same 0D/1D/2D-lat_lon/2D-lat_vert group
    convention as write_obs_diag_files() -- just under a single top-level
    "omf" group and the f<HHH>-nested forecast path
    (obs_diag_forecast_file()).
    """
    bin_method = get_bin_method(diag_obs)
    var_names = list(results["variables"])
    domains = results["domain_names"]
    levels = np.asarray(results["levels"], dtype=np.float64)
    times = results["times"]
    leads = results["leads"]
    lat_centers = results.get("latlon_lat_centers")
    lon_centers = results.get("latlon_lon_centers")
    latvert_lat_centers = results.get("latvert_lat_centers")

    level_units = {1: "hPa", 2: "km", 3: "degrees_north"}[bin_method]

    exp_names = config["experiments"]["names"]
    exp_types = config["experiments"].get("types", [])

    global_idx = domains.index("global") if "global" in domains else None

    for iexp, exp_name in enumerate(exp_names):
        if exp_types[iexp] != "hofx_forecast":
            continue

        for it, init_time in enumerate(times):
            for il, lead in enumerate(leads):
                out_file = obs_diag_forecast_file(diag_base_dir, exp_name, diag_obs, init_time, lead)
                os.makedirs(os.path.dirname(out_file), exist_ok=True)

                with nc4.Dataset(out_file, "w") as ds:
                    ds.createDimension("nLevels", len(levels))
                    lev_var = ds.createVariable("Levels", "f4", ("nLevels",))
                    lev_var[:] = levels.astype(np.float32)
                    lev_var.units = level_units

                    have_2d = lat_centers is not None and lon_centers is not None
                    have_latvert = latvert_lat_centers is not None
                    # lat_vert reuses the SAME latitude centers as the
                    # lat-lon map (see resolve_latlon_bounds()), so a single
                    # nLatBins2D dimension/LatBins2D variable serves both --
                    # created if either product is enabled, exactly
                    # mirroring write_obs_diag_files()'s OMB/OMA handling.
                    if have_2d or have_latvert:
                        any_lat_centers = lat_centers if have_2d else latvert_lat_centers
                        ds.createDimension("nLatBins2D", len(any_lat_centers))
                        lb = ds.createVariable("LatBins2D", "f4", ("nLatBins2D",))
                        lb[:] = np.asarray(any_lat_centers, dtype=np.float32)
                        lb.units = "degrees_north"
                    if have_2d:
                        ds.createDimension("nLonBins2D", len(lon_centers))
                        lo = ds.createVariable("LonBins2D", "f4", ("nLonBins2D",))
                        lo[:] = np.asarray(lon_centers, dtype=np.float32)
                        lo.units = "degrees_east"

                    diag_grp = ds.createGroup("omf")

                    for iv, var in enumerate(var_names):
                        var_grp = diag_grp.createGroup(nc_group_name(var))

                        # ---- 0D: domain-averaged scalar stats ----
                        d0 = var_grp.createGroup("0D")
                        for idom, domain in enumerate(domains):
                            region_grp = d0.createGroup(domain)
                            for stat_disp, prefix in STAT_KEYS:
                                key = f"{prefix}_omf"
                                if key not in results:
                                    continue
                                val = results[key][iexp, idom, it, il, iv]
                                var_out = region_grp.createVariable(stat_disp, "f4", ())
                                var_out[:] = np.float32(val)

                        # ---- 1D: binned profile stats ----
                        d1 = var_grp.createGroup("1D")
                        if bin_method == 3:
                            profile_entries = ([("lat_bin_profile", global_idx)]
                                                if global_idx is not None else [])
                        else:
                            profile_entries = [
                                (profile_key_for_domain(bin_method, domain), idom)
                                for idom, domain in enumerate(domains)
                            ]
                        for pkey, idom in profile_entries:
                            prof_grp = d1.createGroup(pkey)
                            for stat_disp, prefix in STAT_KEYS:
                                key = f"{prefix}_omf_bin"
                                if key not in results:
                                    continue
                                arr = results[key][iexp, idom, it, il, :, iv]
                                var_out = prof_grp.createVariable(stat_disp, "f4", ("nLevels",))
                                var_out[:] = arr.astype(np.float32)

                        # ---- 2D: lat-lon gridded stats (global) ----
                        if have_2d:
                            d2 = var_grp.createGroup("2D").createGroup("lat_lon")
                            for stat_disp, prefix in STAT_KEYS_2D:
                                key = f"{prefix}_omf_latlon"
                                if key not in results:
                                    continue
                                arr = results[key][iexp, it, il, :, :, iv]
                                var_out = d2.createVariable(
                                    stat_disp, "f4", ("nLatBins2D", "nLonBins2D")
                                )
                                var_out[:] = arr.astype(np.float32)

                        # ---- 2D: lat-vert (level x latitude) cross-section
                        #      -- domain-independent, skipped for radiances ----
                        if have_latvert:
                            d2v = var_grp.groups.get("2D") or var_grp.createGroup("2D")
                            d2v = d2v.createGroup("lat_vert")
                            for stat_disp, prefix in STAT_KEYS_2D:
                                key = f"{prefix}_omf_latvert"
                                if key not in results:
                                    continue
                                arr = results[key][iexp, it, il, :, :, iv]
                                var_out = d2v.createVariable(
                                    stat_disp, "f4", ("nLevels", "nLatBins2D")
                                )
                                var_out[:] = arr.astype(np.float32)

                print(f"Wrote obs forecast diagnosis file: {out_file}")

def main(config_path: Optional[str] = None) -> None:
    """Compute observation-space statistics and write diagnosis files."""
    config = load_config(config_path)

    graphics_base_dir = config["paths"].get("graphic_dir", "graphics")
    os.makedirs(graphics_base_dir, exist_ok=True)

    diag_base_dir = resolve_diag_base_dir(config)
    os.makedirs(diag_base_dir, exist_ok=True)

    exp_types = config["experiments"].get("types", [])
    has_cycling = any(t in ("da", "hofx") for t in exp_types)
    has_forecast = "hofx_forecast" in exp_types

    for diag_obs in config["sensors"]:
        # diag_base_dir itself (not a per-sensor subdirectory of it) is
        # passed straight through to write_obs_diag_files() /
        # write_obs_forecast_diag_files() -- the experiment sits at the top
        # of the tree, and the obs type is only ever a filename suffix
        # (obsstat_<obs>.<date>.nc), never its own directory level (see
        # obs_diag_file() / obs_diag_forecast_file() in obsstat_common.py),
        # so there is no "sensor directory" to pre-create here at all; each
        # out_file's own os.makedirs(os.path.dirname(...)) below handles it.
        if has_cycling:
            adjusted_start, end_time, _ = resolve_time_settings(config, diag_obs)
            print(f"Processing {diag_obs} OMB/OMA from {adjusted_start} to {end_time}")
            results = process_experiment(config, diag_obs)
            write_obs_diag_files(results, config, diag_obs, diag_base_dir)

        if has_forecast:
            leads = generate_forecast_leads(config)
            print(f"Processing {diag_obs} OMF (extended forecast) leads {leads}")
            fresults = process_forecast_experiment(config, diag_obs)
            write_obs_forecast_diag_files(fresults, config, diag_obs, diag_base_dir)

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
