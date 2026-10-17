#!/usr/bin/env python3
"""
Shared configuration, metadata, and file-path conventions for observation-
space verification.

This module has NO plotting or HDF5-obs-reading code in it -- it is the
common layer imported by BOTH:
  - diag_ioda_obsout_stats.py   (reads IODA obs-space HDF5, computes stats,
                                  writes obs-space diagnosis netCDF files)
  - plot_ioda_obsout_stats.py   (reads those diagnosis netCDF files back and
                                  draws figures via MPAS-STATS' graphic
                                  functions)

Keeping the config schema, obs-type metadata (variables/units/labels), bin
method, netCDF group-naming and file-path conventions in one place is what
lets the two stages run as fully independent programs while still agreeing
on where files live and how they are organized.
"""
import copy
import glob
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import yaml

# This module's own directory (the MPAS-STATS package folder) must be
# importable so mpas_stats_graphic_io's cycle/date helpers can be reused.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpas_stats_graphic_io import (  # noqa: E402,F401
    datetime_to_cycle, datetime_to_datefile, generate_forecast_leads,
)

# Sensor/observation metadata (variables, units, labels, bin method, and
# per-sensor quirks) now lives entirely in obs_sensor_config.py -- this
# module only re-exports get_obs_config/get_bin_method for callers that
# import it from here for convenience; diag_ioda_obsout_stats.py and
# plot_ioda_obsout_stats.py import the rest (source_obs_name,
# display_var_names, ObservationConfig, ...) directly from
# obs_sensor_config.
from obs_sensor_config import get_obs_config, get_bin_method  # noqa: E402,F401

# ============================================================================
# Configuration
# ============================================================================
DEFAULT_CONFIG = {
    # Render figures in parallel across this many worker processes in
    # plot_ioda_obsout_stats.py. The unit of parallel work is one SENSOR:
    # each worker reads that sensor's own diagnosis files and renders all
    # of its figures, so different sensors' statistics are never all
    # collected into memory at once and figure rendering for one sensor
    # can start before another sensor's files have even been read (see
    # _process_sensor() in plot_ioda_obsout_stats.py). 1 = sequential
    # (default, unchanged behaviour, sensors processed one at a time).
    # With many sensors configured this can be set close to the number of
    # CPUs available to the job -- same top-level "n_workers" key as the
    # model-space package's own graphic.yaml/mpas_stats_graphic_driver.py,
    # just applied at sensor granularity here since obs-space has a
    # natural per-sensor split that model-space doesn't.
    "n_workers": 1,
    "paths": {
        "da_dir": "/path/to/regional_mpasjedi",
        "hofx_dir": "/path/to/regional_mpasjedi",
        # Observation-space diagnosis netCDF files (statistics stage output /
        # graphics stage input). Defaults to "<graphic_dir>/diag_files" when
        # not set -- see resolve_diag_base_dir().
        "diag_dir": None,
        "graphic_dir": "/path/to/verify_obs",
    },
    "experiments": {
        "names": [
            "exp_control",
            "exp_test",
        ],
        "labels": ["Control", "Test"],
        "types": ["da", "da"],
        "outerloop": [2, 2],
        "n_member": [1, 1],
    },
    "sensors": [
        "sondes", "profiler", "aircraft", "satwind", "satwnd", "sfc", "gnssrorefncep",
        "amsua_n18", "amsua_n19", "amsua_metop-b", "amsua_metop-c",
        "mhs_n19", "mhs_metop-b", "mhs_metop-c",
    ],
    "domains": {
        "global": [None, None, None, None],  # No masking
        "NXTro": [30.0, 90.0, -180.0, 180.0],
        "SXTro": [-90.0, -30.0, -180.0, 180.0],
        "NTro": [5.0, 30.0, -180.0, 180.0],
        "STro": [-30.0, -5.0, -180.0, 180.0],
        "ITCZ": [-5.0, 5.0, -180.0, 180.0],
        "Tro": [-23.5,  23.5, -180.0, 180.0]
    },
    "plot_settings": {
        "plot_oma": True,
        "apply_domains": ["global"],
        "figure_format": "png",
        "max_time_ticks": 12,
        "generate_lat_lon_2d": True,
        # Cycle x vertical-level Hovmoller (2-D), reusing the SAME per-cycle
        # 1D profile groups (vert_pres_profile_<domain> / vert_hgt_profile_
        # <domain> / lat_bin_profile) the 1D profile plots already read --
        # no statistics-stage computation needed, purely a graphics-stage
        # reshaping. See plot_ioda_obsout_stats.generate_time_level_graphics().
        "generate_time_level_2d": True,
        # Which `domains` entry bounds the 2-D lat-lon grid (see
        # resolve_latlon_bounds()). Defaults to full-globe coverage; point
        # this at a domain covering the mesh's own extent for a REGIONAL
        # MPAS-JEDI application instead of wasting resolution gridding the
        # whole globe when observations only cover a small regional box.
        # Set to "auto" instead of a `domains` key to have
        # diag_ioda_obsout_stats.py determine that box itself, from the
        # actual lat/lon of observations with a finite, QC-passed
        # innovation -- see auto_detect_latlon_bounds() there -- rather
        # than hand-guessing a domains.<name> box for the mesh's extent.
        "latlon_domain": "global",
        "latlon_lat_bin_width": 10.0,
        "latlon_lon_bin_width": 10.0,
        # Latitude x vertical-level cross-section (2-D), the obs-space
        # counterpart of MPAS-STATS' own lat_lev/lat_hgt/lat_pres 2-D grids.
        # Domain-independent, like the lat-lon map -- reuses that SAME
        # latlon_domain-bounded latitude range (and latlon_lat_bin_width) as
        # its own latitude axis, so no separate bin-width knob is needed.
        # Not generated for radiance sensors (bin_method LATITUDE), whose
        # "level" axis already IS latitude -- see diag_ioda_obsout_stats.py's
        # compute_stats_2d_lat_vert().
        "generate_lat_vert_2d": True,
        # Which of the (RMS/Bias/STD/Count) 2-D-map statistics are drawn with
        # the cross-experiment comparison convention -- experiments.names[0]
        # plotted as the absolute/full-state reference, every other
        # configured experiment plotted as a bootstrap % reduction vs. it
        # (mirroring the model-space package's own
        # relative_reduction_statistics). Applies to the lat-lon map and the
        # new lat-vert cross-section only -- the 0D time series and 1D
        # profiles keep their existing OMB-solid/OMA-dashed, absolute-valued
        # convention regardless of this setting.
        "comparison_relative_reduction_statistics": ["RMS", "STD"],
    },
    "time_settings": {
        "start_time": "2024-09-13 06:00:00",
        "end_time": "2024-09-21 12:00:00"
    }
}

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge `override` on top of a copy of `base`."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from file, merged over the built-in defaults.

    A path that was supplied but does not exist is an error: silently
    falling back to the hard-coded defaults would run the wrong experiment
    with no indication that the requested config was never read.
    """
    if config_path:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
        if not isinstance(user_config, dict):
            raise ValueError(f"Configuration file must contain a YAML mapping: {config_path}")
        return _deep_merge(DEFAULT_CONFIG, user_config)
    return copy.deepcopy(DEFAULT_CONFIG)

def get_levels(bin_method: int, bin_width=None, bin_bottom=None, bin_top=None,
               domain_range=None):
    """Get vertical/latitude levels based on binning method.

    bin_width/bin_bottom/bin_top (all three set) -- explicit per-sensor
    override (obs_sensor_config.ObservationConfig.level_bin_kwargs):
    np.arange(bin_bottom, bin_top + bin_width/2, bin_width). Takes priority
    over domain_range below when both are somehow given.

    bin_width + domain_range=(lo, hi) (bin_bottom/bin_top left unset) -- use
    an already-resolved (lo, hi) box as (bin_bottom, bin_top) instead of a
    hardcoded default. This is for bin_method == LATITUDE (radiances): a
    caller can pass the SAME lat_min_2d/lat_max_2d/lat_bin_width a sensor's
    2D lat-lon/lat-vert products already resolved (resolve_latlon_bounds() /
    auto_detect_latlon_bounds(), see _resolve_latlon_bounds_for() in
    diag_ioda_obsout_stats.py) so the "1D latitude profile" (lat_bin_profile)
    spans the SAME domain-aware latitude range/bin width as those products
    -- for a REGIONAL application (a hand-configured `domains` box, or
    latlon_domain: "auto"), rather than the unconditional global -80..80
    default below, which ignores the domain entirely.

    With neither override, falls back to the fixed pressure/height/latitude
    defaults below (the pre-existing, domain-independent behaviour)."""
    import numpy as np
    if bin_width is not None and bin_bottom is not None and bin_top is not None:
        return np.arange(bin_bottom, bin_top+bin_width/2., bin_width)
    if bin_width is not None and domain_range is not None:
        lo, hi = domain_range
        return np.arange(lo, hi+bin_width/2., bin_width)
    if bin_method == 1:  # pressure
        return np.array([1000, 925, 850, 700, 500, 300, 250, 200, 100])
    if bin_method == 2:  # height
        interval = 2.0
        return np.arange(interval/2., 15*interval+1.0, interval)
    # latitude bins for radiances
    interval = 10.0
    return np.arange(-80.0, 80.0+interval, interval)

def profile_key_for_domain(bin_method: int, domain: str) -> str:
    """netCDF/plot profile-key naming, matching MPAS-STATS' own
    vert_pres_profile_<region> / vert_hgt_profile_<region> / lat_bin_profile
    convention (mpas_stats_diag_kernels.compute_1d_vert_*_profile,
    mpas_stats_graphic_io.get_profile_coord)."""
    if bin_method == 1:
        return f"vert_pres_profile_{domain}"
    if bin_method == 2:
        return f"vert_hgt_profile_{domain}"
    return "lat_bin_profile"  # bin_method == 3, region-independent

def nc_group_name(var: Union[str, int]) -> str:
    """A valid, human-readable netCDF group name for an obs variable/channel."""
    import numpy as np
    return f"channel_{var}" if isinstance(var, (int, np.integer)) else str(var)

# ============================================================================
# Domains, statistic names, time/cycle handling
# ============================================================================
STAT_KEYS = [
    ("RMS", "rmse"),
    ("Bias", "mean"),
    ("STD", "std"),
    ("Count", "count"),
    ("Spread", "spread"),
    ("TotalSpread", "totSpread"),
]
# Lat-lon (2-D) gridded statistics only cover the basic moments -- ensemble
# spread on a lat-lon grid is intentionally out of scope (same as 3-D
# products).
STAT_KEYS_2D = [
    ("RMS", "rmse"),
    ("Bias", "mean"),
    ("STD", "std"),
    ("Count", "count"),
]

def resolve_domains(config: Dict) -> Dict[str, List]:
    """The domains a given config actually applies, in a fixed, shared order
    so the statistics and graphics stages always agree."""
    domain_keys = config["plot_settings"].get("apply_domains", ["global"])
    domains = {k: config["domains"][k] for k in domain_keys if k in config["domains"]}
    if not domains:
        domains = {"global": [None, None, None, None]}
    return domains

def resolve_latlon_bounds(config: Dict) -> Tuple[float, float, float, float]:
    """(lat_min, lat_max, lon_min, lon_max) bounding box for the 2-D lat-lon
    gridded statistics, taken from plot_settings.latlon_domain -- a key into
    the SAME `domains` map used for 0D/1D region masking (default "global",
    i.e. the full globe, [None, None, None, None]).

    For a REGIONAL MPAS-JEDI application, point latlon_domain at a domain
    entry covering the mesh's own extent (mirroring the model-space
    package's MPASConfig.regional flag, which limits ITS lat/lon binning
    box the same way) so the 2-D map is gridded over the area that actually
    has observations, rather than the whole globe at a coarse bin width.
    This bounding box is FIXED across every cycle a config processes (it
    does not adapt to the extent of any one cycle's data) -- required so
    every diagnosis file the statistics stage writes shares an identical
    grid shape, which the graphics stage's cycle-to-cycle averaging
    (np.nanmean over stacked per-cycle maps -- see
    plot_ioda_obsout_stats.collect_obs_lat_lon_2d) depends on.

    latlon_domain: "auto" is NOT handled here -- this function has no file
    access to determine it, and callers must branch on that sentinel
    themselves and call diag_ioda_obsout_stats.auto_detect_latlon_bounds()
    instead (both process_experiment() and process_forecast_experiment()
    do). Raises ValueError if called with "auto" directly, rather than
    silently falling back to full-globe bounds, so a caller that forgets
    the branch fails loudly instead of quietly gridding the whole globe.
    """
    domain_key = config["plot_settings"].get("latlon_domain", "global")
    if domain_key == "auto":
        raise ValueError(
            "resolve_latlon_bounds() cannot resolve latlon_domain: \"auto\" itself -- "
            "call diag_ioda_obsout_stats.auto_detect_latlon_bounds() instead, which "
            "has the file access this needs."
        )
    bounds = config.get("domains", {}).get(domain_key)
    if not bounds or any(b is None for b in bounds):
        return -90.0, 90.0, -180.0, 180.0
    lat_min, lat_max, lon_min, lon_max = (float(b) for b in bounds)
    return lat_min, lat_max, lon_min, lon_max

def _next_valid_cycle_start(start_time: datetime, valid_hours: List[int]) -> datetime:
    """Smallest datetime >= start_time whose hour is in `valid_hours`,
    preserving minute/second -- rolls into the next day if every listed
    hour today has already passed. Used by resolve_time_settings() for a
    sensor with cycle_valid_hours set (e.g. sondes: [0, 12], since
    radiosondes only launch at 00Z/12Z). Unlike the older
    cycle_start_trigger_hours/cycle_start_offset_hours mechanism below
    (which only shifts from specific listed trigger hours), this "snap to
    the nearest valid hour" works for ANY configured start hour, so the
    generated cycle list can never include an hour outside `valid_hours`
    regardless of what time_settings.start_time happens to be."""
    valid = sorted(set(int(h) % 24 for h in valid_hours))
    h = start_time.hour
    for v in valid:
        if v >= h:
            return start_time + timedelta(hours=(v - h))
    # Every valid hour today has already passed -- roll to the first valid
    # hour tomorrow.
    return start_time + timedelta(hours=(24 - h + valid[0]))

def resolve_time_settings(config: Dict, diag_obs: str, for_forecast: bool = False
                         ) -> Tuple[datetime, datetime, timedelta]:
    """(adjusted_start, end_time, interval) -- the CYCLE list this returns
    means something different depending on `for_forecast`, and a sensor's
    launch-schedule quirks apply to only ONE of the two meanings:

    for_forecast=False (default -- cycling/OMB-OMA mode): the generated
    times ARE observation times, so a sensor's own network cadence applies
    here. A sensor may override the config-wide interval
    (cycle_interval_hours) and restrict/snap its start time to only fall on
    specific UTC hours -- see obs_sensor_config.ObservationConfig's
    cycle_valid_hours (preferred: works for any configured start hour) /
    cycle_start_trigger_hours / cycle_start_offset_hours (older, only
    shifts from specific listed trigger hours) fields -- e.g. sondes only
    launch at 00Z/12Z, so every OMB/OMA cycle for sondes is snapped to one
    of those two hours (_next_valid_cycle_start()).

    for_forecast=True (OMF/extended-forecast mode): the generated times are
    INIT (forecast launch) times -- a property of the MODEL/DA system, not
    of any one sensor's observation network. A forecast can be launched
    whenever the model actually cycles (e.g. 00/06/12/18Z) regardless of
    which sensors have observations at that hour, and a launch at an
    "invalid" hour for some sensor can still validate against it later --
    a sondes-verifying forecast launched at 06Z has no sonde to compare
    against at its own init time, but its +6h lead validates at 12Z (a real
    sonde launch) and its +18h lead validates at the next day's 00Z. So
    for_forecast=True ALWAYS uses the plain config-wide
    time_settings.interval/start_time, ignoring cycle_interval_hours /
    cycle_valid_hours / cycle_start_trigger_hours entirely -- restricting
    the INIT time the same way cycling mode restricts observation times
    would silently drop every one of those still-valid lead comparisons.
    The restriction instead applies per (init, lead) pair, to the LEAD's
    VALID time -- see lead_valid_time_allowed(), used by
    process_forecast_experiment() in diag_ioda_obsout_stats.py.

    Kept identical between the statistics and graphics stages (both must
    pass the same for_forecast value in matching contexts -- cycling
    functions pass False, OMF functions pass True) so both walk the exact
    same cycle list.

    Once a cycling adjusted_start's hour is in cycle_valid_hours, EVERY
    subsequent cycle stays on a valid hour too, forever, with no
    re-checking needed -- generate_cycles() just keeps adding a fixed
    `interval`, and 24 hours is an exact multiple of any of the interval
    values this pipeline uses (6h/12h), so the phase relative to 00Z never
    drifts."""
    start_time = datetime.strptime(config["time_settings"]["start_time"], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(config["time_settings"]["end_time"], "%Y-%m-%d %H:%M:%S")
    interval_hour = config["time_settings"]["interval"]

    if for_forecast:
        return start_time, end_time, timedelta(hours=interval_hour)

    obs_config = get_obs_config(diag_obs)

    if obs_config.cycle_interval_hours is not None:
        interval = timedelta(hours=obs_config.cycle_interval_hours)
    else:
        interval = timedelta(hours=interval_hour)

    valid_hours = obs_config.cycle_valid_hours
    if valid_hours:
        adjusted_start = _next_valid_cycle_start(start_time, valid_hours)
    else:
        trigger_hours = obs_config.cycle_start_trigger_hours or []
        if trigger_hours and start_time.hour in trigger_hours:
            adjusted_start = start_time + timedelta(hours=obs_config.cycle_start_offset_hours or 0)
        else:
            adjusted_start = start_time

    return adjusted_start, end_time, interval

def lead_valid_time_allowed(diag_obs: str, init_time: datetime, lead_hour: int) -> bool:
    """True if this sensor's cycle_valid_hours (if any) allows an OMF
    (extended-forecast) lead's VALID time (init_time + lead_hour). E.g. for
    sondes (cycle_valid_hours=[0, 12]), a forecast launched at 06Z has its
    +6h lead validate at 12Z (a real sonde launch -> allowed) but its +12h
    lead validate at 18Z (no sonde there -> not allowed). A sensor with no
    cycle_valid_hours set (the common case) always returns True -- the
    restriction is opt-in, matching cycle_valid_hours' own
    "no restriction unless set" default elsewhere. Used by
    process_forecast_experiment() (diag_ioda_obsout_stats.py) to skip
    (leave NaN) a disallowed (init, lead) pair -- see
    resolve_time_settings()'s for_forecast docstring for why the INIT time
    itself is deliberately NOT restricted the same way."""
    obs_config = get_obs_config(diag_obs)
    valid_hours = obs_config.cycle_valid_hours
    if not valid_hours:
        return True
    valid_time = init_time + timedelta(hours=int(lead_hour))
    return valid_time.hour in set(int(h) % 24 for h in valid_hours)

def generate_cycles(start_time: datetime, end_time: datetime, interval: timedelta) -> List[datetime]:
    times = []
    t = start_time
    while t <= end_time:
        times.append(t)
        t += interval
    return times

def resolve_diag_base_dir(config: Dict) -> str:
    """Root directory for obs-space diagnosis netCDF files. One
    <experiment>/VerifyObs/... subtree is created underneath it per
    configured experiment -- see obs_diag_file()/obs_diag_forecast_file()
    below."""
    graphics_base_dir = config["paths"].get("graphic_dir", "graphics")
    diag_dir = config["paths"].get("diag_dir")
    return diag_dir if diag_dir else os.path.join(graphics_base_dir, "diag_files")

# ============================================================================
# Diagnosis-file path convention
#
# <diag_base_dir>/<experiment>/VerifyObs/<cycle>/obsstat_<obs>.<date>.nc
#
# The experiment sits at the TOP of the tree, exactly mirroring MPAS-STATS'
# own model-space convention (mpas_stats_graphic_io.get_mpasstat_file):
# <base_dir>/<experiment>/VerifyModel/<cycle>/mpasstat.<date>.nc -- with no
# extra "obs type" directory level, since the filename itself
# (obsstat_<obs>.<date>.nc) already identifies the sensor. Every sensor's
# files for one (experiment, cycle) pair land side by side in the SAME
# VerifyObs/<cycle>/ directory, distinguished only by filename -- there is
# no collision risk (each sensor writes its own obsstat_<obs>.*.nc name).
# This lets every diagnosis file for one experiment be found/archived/
# rsynced as a single <diag_base_dir>/<experiment>/ subtree, matching how
# run_verify_obs.ksh/run_verify_obs_fcst.ksh jobs are typically organized
# per experiment.
#
# NOTE: earlier revisions of this convention put the obs type in its own
# directory level, either above the experiment
# (<diag_base_dir>/<obs_type>/<experiment>/VerifyObs/...) or below it
# (<diag_base_dir>/<experiment>/VerifyObs/<obs_type>/...) -- if migrating an
# existing diag_dir from either older layout, either re-run
# diag_ioda_obsout_stats.py to regenerate files under the current layout, or
# move files by hand; none of these layouts are compatible with each other
# on disk.
# ============================================================================
def obs_diag_file(diag_base_dir: str, exp_name: str, diag_obs: str, cycle_time: datetime) -> str:
    cycle = datetime_to_cycle(cycle_time)
    date_file = datetime_to_datefile(cycle_time)
    return os.path.join(
        diag_base_dir, exp_name, "VerifyObs", cycle,
        f"obsstat_{diag_obs}.{date_file}.nc",
    )

# ============================================================================
# Extended-forecast (OMF, obs-minus-forecast) diagnosis-file path convention
#
# <diag_base_dir>/<experiment>/VerifyObs/<init_cycle>/f<HHH>/obsstat_<obs>.<valid_date>.nc
#
# mirrors MPAS-STATS' own extended-forecast model-space convention exactly
# (mpas_stats_graphic_io.get_mpasstat_forecast_file):
# <base_dir>/<experiment>/VerifyModel/<init_cycle>/f<HHH>/mpasstat.<valid_date>.nc
# -- again, no extra obs-type directory level; see obs_diag_file() above.
# ============================================================================
def obs_diag_forecast_file(diag_base_dir: str, exp_name: str, diag_obs: str, init_time: datetime,
                          lead_hour: int) -> str:
    init_cycle = datetime_to_cycle(init_time)
    valid_time = init_time + timedelta(hours=int(lead_hour))
    date_file = datetime_to_datefile(valid_time)
    return os.path.join(
        diag_base_dir, exp_name, "VerifyObs", init_cycle,
        f"f{int(lead_hour):03d}",
        f"obsstat_{diag_obs}.{date_file}.nc",
    )

def obs_diag_sensor_has_files(diag_base_dir: str, exp_names: List[str], diag_obs: str) -> bool:
    """True if at least one configured experiment has ANY obsstat_<diag_obs>.*.nc
    file under its VerifyObs/ subtree -- used by plot_ioda_obsout_stats.py's
    main() to skip a sensor entirely (with one clear warning) when
    diag_ioda_obsout_stats.py hasn't been run for it yet. A glob (not a
    directory check) is required because a sensor no longer has its own
    directory level -- every sensor's files for one (experiment, cycle)
    share the same VerifyObs/<cycle>/ directory, distinguished only by
    filename (see obs_diag_file())."""
    return any(
        glob.glob(os.path.join(diag_base_dir, exp_name, "VerifyObs", "**",
                                f"obsstat_{diag_obs}.*.nc"), recursive=True)
        for exp_name in exp_names
    )

# ============================================================================
# Radar composite-reflectivity diagnosis-file path convention
#
# <diag_base_dir>/<experiment>/VerifyObs/<init_cycle>/f<HHH>/radarstat_<obs>.<valid_date>.nc
#
# Same directory convention as obs_diag_forecast_file() above (one
# <experiment>/VerifyObs/<init_cycle>/f<HHH>/ subtree per (experiment, init
# cycle, lead) triple -- radar composite reflectivity is an OMF-only,
# extended-forecast product, with no cycling/OMB-OMA counterpart), but with
# a DISTINCT "radarstat_" filename prefix instead of "obsstat_". This is
# deliberate, not cosmetic: a radar sensor listed in BOTH the top-level
# `sensors:` list (point-wise pipeline, e.g. "radar_zh*"'s HEIGHT-binned
# products) and `radar.sensors` (this gridded/neighborhood-based product)
# for the SAME diag_obs name would otherwise collide on
# obsstat_<diag_obs>.<date>.nc in the exact same directory, silently
# clobbering whichever pipeline wrote second. The two products' diagnosis
# files carry completely different schemas (see
# diag_radar_reflectivity.py's _write_radar_diag_file() vs.
# write_obs_forecast_diag_files() in diag_ioda_obsout_stats.py) so they
# must never share a filename even incidentally.
# ============================================================================
def radar_composite_diag_file(diag_base_dir: str, exp_name: str, diag_obs: str,
                             init_time: datetime, lead_hour: int) -> str:
    init_cycle = datetime_to_cycle(init_time)
    valid_time = init_time + timedelta(hours=int(lead_hour))
    date_file = datetime_to_datefile(valid_time)
    return os.path.join(
        diag_base_dir, exp_name, "VerifyObs", init_cycle,
        f"f{int(lead_hour):03d}",
        f"radarstat_{diag_obs}.{date_file}.nc",
    )

def radar_diag_sensor_has_files(diag_base_dir: str, exp_names: List[str], diag_obs: str) -> bool:
    """radar_composite_diag_file() counterpart of obs_diag_sensor_has_files()
    -- used by plot_radar_reflectivity.py's main() to skip a radar sensor
    (with one clear warning) when diag_radar_reflectivity.py hasn't been
    run for it yet."""
    return any(
        glob.glob(os.path.join(diag_base_dir, exp_name, "VerifyObs", "**",
                                f"radarstat_{diag_obs}.*.nc"), recursive=True)
        for exp_name in exp_names
    )
