"""Lat-lon box resolution for 2-D obs-space statistics, including
latlon_domain: auto (tightest box around valid, QC-passed observations).
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import (
    generate_cycles,
    generate_forecast_leads,
    lead_valid_time_allowed,
    resolve_latlon_bounds,
    resolve_statistics_settings,
    resolve_time_settings,
)
from .reader import (
    build_filename,
    build_hofx_forecast_filename,
    load_observation_data,
)
from .sensor_config import get_obs_config


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
    settings = resolve_statistics_settings(config)
    reader_settings = None if settings["method"] == "legacy" else settings

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
                        settings=reader_settings,
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
                    settings=reader_settings,
                )
                _accumulate_valid_latlon_bounds(omb_vec, bounds)

    if bounds[0] is None:
        return None
    return tuple(bounds)  # type: ignore[return-value]


def _resolve_latlon_bounds_for(config: Dict, diag_obs: str, bin_method: int,
                              file_kind: str) -> Tuple[float, float, float, float]:
    """resolve_latlon_bounds(config), except when plot_settings.latlon_domain
    == "auto": that sentinel has no file access from src/mpas_stat/obs/common.py, so
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
