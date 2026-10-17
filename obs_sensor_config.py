#!/usr/bin/env python3
"""
Single source of truth for observation-type ("sensor") metadata used by the
observation-space verification pipeline (diag_ioda_obsout_stats.py /
plot_ioda_obsout_stats.py).

To add or adjust a sensor, edit ONLY the `OBSERVATION_CONFIGS` registry
below -- nothing else in the pipeline needs to change, as long as the
sensor's IODA obsout_*.h5 layout matches the common case already handled by
diag_ioda_obsout_stats.load_observation_data() (MetaData/ombg/oman/hofx*/
EffectiveQC*/ObsError groups, standard for JEDI obs-space output).

This module has NO dependency on obsstat_common.py, matplotlib, h5py, or
netCDF4 -- it is pure, dependency-free metadata, importable from anywhere
(the statistics stage, the graphics stage, a notebook, a unit test) without
pulling in the rest of the pipeline.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import re

# ============================================================================
# Bin methods: how one sensor's innovations are grouped for the "1-D profile"
# statistic and for domain (lat/lon box) masking.
# ============================================================================
PRESSURE = 1   # conventional obs            -> pressure-level bins
HEIGHT   = 2   # GNSS-RO / radar              -> height-level bins
LATITUDE = 3   # radiances                    -> latitude bins (zonal profile)

@dataclass
class ObservationConfig:
    """Everything the verification pipeline needs to know about one obs type."""

    # ---- Identity (required) -------------------------------------------
    variables: List[Union[str, int]]
    """Physical variable names (IODA group-dataset names, e.g.
    'windNorthward'), or radiance channel numbers (1-based, e.g. 5)."""

    units: List[str]
    """One entry per variable, for documentation/labelling."""

    labels: List[str]
    """Panel labels, e.g. '(a) U' or '(a) Channel 5' -- the leading
    '(a) ' panel marker is stripped automatically for plot titles."""

    bin_method: int
    """PRESSURE, HEIGHT, or LATITUDE (see above)."""

    # ---- Optional per-sensor quirks (all default to "nothing special") --
    file_alias: Optional[str] = None
    """Read this obs type's IODA files instead of the sensor's own name
    (e.g. profiler reads the sondes obsout file and is distinguished by
    ObsType -- see obs_type_filter)."""

    obs_type_filter: Optional[Tuple[str, int]] = None
    """('lt' | 'eq', threshold) -- keep only observations whose top-level
    'ObsType' dataset satisfies this comparison; used when two sensors
    share one obsout file (sondes vs. profiler)."""

    level_bin_kwargs: Optional[Dict[str, float]] = None
    """Override the default vertical/latitude bin levels for this sensor:
    {'bin_width':, 'bin_bottom':, 'bin_top':} passed to get_levels()."""

    cycle_interval_hours: Optional[int] = None
    """Override the config-wide time_settings.interval for this sensor
    (e.g. sondes are 12-hourly regardless of the model DA cycle length)."""

    cycle_start_trigger_hours: Optional[List[int]] = None
    cycle_start_offset_hours: Optional[int] = None
    """If the configured start hour is one of cycle_start_trigger_hours,
    shift the start time forward by cycle_start_offset_hours before
    generating this sensor's cycle list. Superseded by cycle_valid_hours
    below for any sensor that has one -- kept only for a sensor that wants
    a plain fixed-hour shift instead of "snap to the nearest valid hour"."""

    cycle_valid_hours: Optional[List[int]] = None
    """Restrict this sensor's cycles to ONLY these UTC hours (e.g. [0, 12]
    for sondes -- radiosondes only launch at 00Z/12Z, regardless of
    whatever cadence the model DA cycles at or what hour the configured
    time_settings.start_time happens to fall on). When set, the configured
    start time is rounded forward to the next hour in this list (rolling
    into the next day if every listed hour today has already passed) --
    see resolve_time_settings()'s _next_valid_cycle_start(). This is a
    general "snap to nearest valid hour" and works for ANY configured
    start hour, unlike cycle_start_trigger_hours/cycle_start_offset_hours
    above, which only shift from specific listed trigger hours. Takes
    priority over cycle_start_trigger_hours when both are set (they
    shouldn't both need to be)."""

# ============================================================================
# Sensor registry -- add new obs types here.
# ============================================================================
OBSERVATION_CONFIGS: Dict[str, ObservationConfig] = {
    # ---------------- Conventional observations ----------------
    "sondes": ObservationConfig(
        variables=['windNorthward', 'windEastward', 'airTemperature', 'specificHumidity'],
        units=['$\mathrm{m \cdot s^{-1}}$', '$\mathrm{m \cdot s^{-1}}$', '$\mathrm{K}$', '$\mathrm{g \cdot kg^{-1}}$'],
        labels=['(a) U', '(b) V', '(c) T', '(d) Q'],
        bin_method=PRESSURE,
        obs_type_filter=('lt', 221),
        cycle_interval_hours=12,
        # Radiosondes only launch at 00Z/12Z -- only ever count cycles at
        # those two hours for statistics, regardless of the model DA cycle
        # length or what hour time_settings.start_time happens to be (see
        # cycle_valid_hours / resolve_time_settings._next_valid_cycle_start
        # in obsstat_common.py).
        cycle_valid_hours=[0, 12],
    ),
    "profiler": ObservationConfig(
        variables=['windNorthward', 'windEastward'],
        units=['$\mathrm{m \cdot s^{-1}}$', '$\mathrm{m \cdot s^{-1}}$'],
        labels=['(a) U', '(b) V'],
        bin_method=PRESSURE,
        file_alias='sondes',
        obs_type_filter=('eq', 221),
    ),
    "aircraft": ObservationConfig(
        variables=['windNorthward', 'windEastward', 'airTemperature'],
        units=['$\mathrm{m \cdot s^{-1}}$', '$\mathrm{m \cdot s^{-1}}$', '$\mathrm{K}$'],
        labels=['(a) U', '(b) V', '(c) T'],
        bin_method=PRESSURE,
    ),
    "satwind": ObservationConfig(
        variables=['windNorthward', 'windEastward'],
        units=['$\mathrm{m \cdot s^{-1}}$', '$\mathrm{m \cdot s^{-1}}$'],
        labels=['(a) U', '(b) V'],
        bin_method=PRESSURE,
    ),
    "satwnd": ObservationConfig(
        variables=['windNorthward', 'windEastward'],
        units=['$\mathrm{m \cdot s^{-1}}$', '$\mathrm{m \cdot s^{-1}}$'],
        labels=['(a) U', '(b) V'],
        bin_method=PRESSURE,
    ),
    "sfc": ObservationConfig(
        variables=['windNorthward', 'windEastward', 'airTemperature', 'specificHumidity', 'stationPressure'],
        units=['$\mathrm{m \cdot s^{-1}}$', '$\mathrm{m \cdot s^{-1}}$', '$\mathrm{K}$', '$\mathrm{g \cdot kg^{-1}}$', 'Pa'],
        labels=['(a) U10', '(b) V10', '(c) T2', '(d) Q2', '(e) Ps'],
        bin_method=PRESSURE,
    ),
    "gnssrobndropp1d*": ObservationConfig(
        variables=['bendingAngle'],
        units=['%'],
        labels=['Bending angle'],
        bin_method=HEIGHT,
    ),
    "gnssrorefncep*": ObservationConfig(
        variables=['atmosphericRefractivity'],
        units=['%'],
        labels=['Refractivity'],
        bin_method=HEIGHT,
    ),
    "radar_rw": ObservationConfig(
        variables=['radialVelocity'],
        units=['$\mathrm{m \cdot s^{-1}}$'],
        labels=['Radial velocity'],
        bin_method=HEIGHT,
        level_bin_kwargs={'bin_width': 0.5, 'bin_bottom': 0.5, 'bin_top': 12.0},
    ),
    "radar_zh*": ObservationConfig(
        variables=['equivalentReflectivityFactor'],
        units=['dBZ'],
        labels=['Reflectivity'],
        bin_method=HEIGHT,
        level_bin_kwargs={'bin_width': 0.5, 'bin_bottom': 0.5, 'bin_top': 12.0},
    ),
    # ---------------- Radiance sensors ----------------
    "amsua_aqua": ObservationConfig(
        variables=[8, 9], units=['K', 'K'],
        labels=['(a) Channel 8', '(b) Channel 9'], bin_method=LATITUDE,
    ),
    "amsua_metop-a": ObservationConfig(
        variables=[5, 6, 9], units=['K', 'K', 'K'],
        labels=['(a) Channel 5', '(b) Channel 6', '(c) Channel 9'], bin_method=LATITUDE,
    ),
    "amsua_metop-b": ObservationConfig(
        variables=[8, 9], units=['K', 'K', 'K'],
        labels=['(a) Channel 8', '(b) Channel 9'], bin_method=LATITUDE,
    ),
    "amsua_metop-c": ObservationConfig(
        variables=[4, 5, 6, 7, 8, 9, 10, 11],
        units=['K', 'K', 'K', 'K', 'K', 'K', 'K', 'K'],
        labels=['(a) Channel 4', '(b) Channel 5', '(c) Channel 6', '(d) Channel 7',
                '(e) Channel 8', '(f) Channel 9', '(g) Channel 10', '(h) Channel 11'],
        bin_method=LATITUDE,
    ),
    "amsua_n15": ObservationConfig(
        variables=[5, 7, 8, 9], units=['K', 'K', 'K', 'K'],
        labels=['(a) Channel 5', '(b) Channel 7', '(c) Channel 8', '(d) Channel 9'],
        bin_method=LATITUDE,
    ),
    "amsua_n18": ObservationConfig(
        variables=[5, 6, 7, 8, 9], units=['K', 'K', 'K', 'K', 'K'],
        labels=['(a) Channel 5', '(b) Channel 6', '(c) Channel 7', '(d) Channel 8', '(e) Channel 9'],
        bin_method=LATITUDE,
    ),
    "amsua_n19": ObservationConfig(
        variables=[5, 6, 7, 9], units=['K', 'K', 'K', 'K'],
        labels=['(a) Channel 5', '(b) Channel 6', '(c) Channel 7', '(d) Channel 9'],
        bin_method=LATITUDE,
    ),
    "mhs_n19": ObservationConfig(
        variables=[4, 5], units=['K', 'K'],
        labels=['(a) Channel 4', '(b) Channel 5'], bin_method=LATITUDE,
    ),
    "mhs_n18": ObservationConfig(
        variables=[3, 4, 5], units=['K', 'K', 'K'],
        labels=['(a) Channel 3', '(b) Channel 4', '(c) Channel 5'], bin_method=LATITUDE,
    ),
    "mhs_metop-a": ObservationConfig(
        variables=[3, 4, 5], units=['K', 'K', 'K'],
        labels=['(a) Channel 3', '(b) Channel 4', '(c) Channel 5'], bin_method=LATITUDE,
    ),
    "mhs_metop-b": ObservationConfig(
        variables=[3, 4, 5], units=['K', 'K', 'K'],
        labels=['(a) Channel 3', '(b) Channel 4', '(c) Channel 5'], bin_method=LATITUDE,
    ),
    "mhs_metop-c": ObservationConfig(
        variables=[3, 4, 5], units=['K', 'K', 'K'],
        labels=['(a) Channel 3', '(b) Channel 4', '(c) Channel 5'], bin_method=LATITUDE,
    ),
    "amsua-cld": ObservationConfig(
        variables=[1, 2, 3, 4, 15], units=['K', 'K', 'K', 'K', 'K'],
        labels=['(a) Channel 1', '(b) Channel 2', '(c) Channel 3', '(d) Channel 4', '(e) Channel 15'],
        bin_method=LATITUDE,
    ),
}

# ============================================================================
# Lookups
# ============================================================================
def get_obs_config(diag_obs: str) -> ObservationConfig:
    """Exact match first, then a '<prefix>*' wildcard match (e.g. individual
    GNSS-RO satellites/radar volumes sharing one entry)."""
    if diag_obs in OBSERVATION_CONFIGS:
        return OBSERVATION_CONFIGS[diag_obs]

    for key, cfg in OBSERVATION_CONFIGS.items():
        if key.endswith('*') and diag_obs.startswith(key[:-1]):
            return cfg

    raise ValueError(f"No ObservationConfig found for {diag_obs}")

def get_bin_method(diag_obs: str) -> int:
    return get_obs_config(diag_obs).bin_method

def source_obs_name(diag_obs: str) -> str:
    """The obs-type name to use when building IODA filenames -- handles
    `file_alias` (e.g. profiler -> sondes)."""
    return get_obs_config(diag_obs).file_alias or diag_obs

def _clean_panel_label(label: str) -> str:
    """Strip a leading '(a) '-style panel marker from an ObservationConfig label."""
    return re.sub(r'^\([^)]*\)\s*', '', str(label)).strip()

def display_var_names(diag_obs: str) -> List[str]:
    """Human-readable per-variable names (e.g. 'U', 'Channel 5') used as
    panel titles / dict keys by the graphics stage -- decoupled from the
    raw `variables` identifiers (which may be plain channel integers) used
    as netCDF group names by the statistics stage."""
    return [_clean_panel_label(lbl) for lbl in get_obs_config(diag_obs).labels]
