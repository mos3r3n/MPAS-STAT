"""Observation-space verification -- GRAPHICS stage.

Reads back the observation-space "diagnosis" netCDF files written by
mpas_stat.obs.diag (the separate STATISTICS stage) -- one per
(experiment, cycle, obs type), at
    <diag_dir>/<experiment>/VerifyObs/<cycle>/obsstat_<obs>.<date>.nc
-- and draws figures for each obs type by calling MPAS-STAT's own graphic
functions directly (mpas_stat.graphics.functions.plot_time_series_0d /
plot_profiles_1d / plot_lat_lon_2d / plot_forecast_lead_0d), exactly as used
for model-space verification. This mirrors MPAS-STAT's own diag/graphic
split: this stage plays the role of mpas_stat.graphics.driver +
mpas_stat.graphics.io's collect_* functions, just pointed at obs-space
diagnosis files instead of mpasstat.<date>.nc model-space files.

Two products are generated, matching the two statistics-stage modes (see
mpas_stat.obs.diag's module docstring):
  - OMB/OMA time series and profiles, for "da"/"hofx" experiments --
    generate_time_series_graphics / generate_profile_graphics. These use
    MPAS-STAT's "pseudo-experiment" trick (see _build_pseudo_experiments):
    one figure per (domain, statistic), OMB solid / OMA dashed, real
    experiments sharing a colour.
  - OMB/OMA 2-D products -- lat-lon maps, lev-time (cycle x level)
    cross-sections, and lev-lat (latitude x level) cross-sections -- for
    "da"/"hofx" experiments -- generate_lat_lon_graphics /
    generate_time_level_graphics / generate_lat_vert_graphics. Unlike the
    0D/1D products above, these use the REAL experiment axis (not OMB/OMA
    pseudo-experiments), mirroring the model-space package's own
    plot_lat_lon_2d/plot_time_level_2d/plot_lat_vert_2d convention exactly:
    experiments.names[0] plotted as the absolute/full-state reference,
    every other configured experiment plotted as a bootstrap % RMS/STD
    reduction vs. it. Because OMB and OMA can't both share this same
    real-experiment axis, each is drawn as its own separate set of figures,
    under an "omb/" or "oma/" subfolder of the sensor's graphics directory.
  - OMF "forecast skill" (statistic vs. forecast lead time, bootstrap-
    aggregated over initialization cycles), for "hofx_forecast" experiments
    -- generate_forecast_lead_graphics, via plot_forecast_lead_0d -- the
    exact function the model-space package uses for its own extended-
    forecast skill plots. OMF also has profile, lat-lon, lev-lat and
    lev-time counterparts -- see docs/USER_GUIDE.md.

This stage has NO obs-space HDF5 reading code and does not import h5py --
see mpas_stat.obs.diag for the separate statistics stage that produces the
diagnosis files this module reads.

Run AFTER mpas_stat.obs.diag has produced the diagnosis files for the obs
types you want plotted (the same config file works for both stages).
"""
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..core.bootstrap import get_bootstrap_settings
from ..core.jobs import lead_cfg as _lead_cfg
from ..core.jobs import resolve_n_workers, run_job
from ..graphics.functions import (
    plot_forecast_lead_0d,
    plot_forecast_time_level_2d,
    plot_lat_lon_2d,
    plot_lat_vert_2d,
    plot_profile_reduction_1d,
    plot_profiles_1d,
    plot_time_level_2d,
    plot_time_series_0d,
    plot_time_series_reduction_0d,
)
from .collect import (
    collect_obs_forecast_lat_lon_2d,
    collect_obs_forecast_lat_vert_2d,
    collect_obs_forecast_lead_0d,
    collect_obs_forecast_profiles_1d,
    collect_obs_forecast_time_level_2d,
    collect_obs_lat_lon_2d,
    collect_obs_lat_vert_2d,
    collect_obs_profiles_1d,
    collect_obs_stats_0d,
    collect_obs_time_level_2d,
)
from .common import (
    STAT_KEYS,
    STAT_KEYS_2D,
    generate_cycles,
    generate_forecast_leads,
    load_config,
    obs_diag_sensor_has_files,
    profile_key_for_domain,
    resolve_diag_base_dir,
    resolve_domains,
    resolve_statistics_settings,
    resolve_time_settings,
)
from .experiments import (
    _base_graphic_cfg,
    _build_forecast_experiments,
    _comparison_graphic_cfg,
    _experiments_for_diagtype,
    _profile_axis_meta,
)
from .sensor_config import display_var_names, get_bin_method, get_obs_config

# ============================================================================
# Graphics -- generated via MPAS-STAT's own graphic functions
#
# Every generate_*_graphics function below builds a flat list of "jobs" --
# (plot_func, args, kwargs) tuples -- instead of calling plot_* directly.
# Each job is entirely self-contained: its data was already read from disk
# by the (cheap) collect_obs_* call that built its args, and running it does
# nothing but create, draw, and save one matplotlib figure.
#
# Parallelism is applied at the SENSOR level, not the individual-figure
# level: _process_sensor() below collects (reads diagnosis netCDF files)
# AND renders every figure for ONE sensor, and main() runs one
# _process_sensor() call per sensor -- sequentially by default
# (config["n_workers"] unset or 1), or spread across a process pool when
# config["n_workers"] > 1. This is a deliberate choice over a single flat
# job list spanning every sensor: with many sensors configured, collecting
# every sensor's statistics into this one process before any figure starts
# rendering is itself slow and memory-heavy, and sensor-level parallelism
# avoids both -- each worker reads only the one sensor's diagnosis files it
# is responsible for, and different sensors' collection + rendering overlap
# in time instead of collection-then-rendering being two fully separate
# phases. Process-based (not thread-based) parallelism is required: each
# job creates and saves its own matplotlib figure, and a single process's
# pyplot state is not safe for concurrent figure creation.
# ============================================================================

def _time_aggregation(config):
    """How cycles are combined: "pooled" or "mean" (statistics.time_aggregation)."""
    return resolve_statistics_settings(config)["time_aggregation"]


def _process_sensor(config, diag_obs, sensor_graphic_dir, diag_base_dir):
    """Collect and render every figure for ONE sensor, entirely within
    whichever process calls this (the main process when running
    sequentially, or one worker process in the pool otherwise). This makes
    one sensor's (collection + rendering) the unit of parallel work -- see
    main() and the comment block above -- so no process ever needs more
    than one sensor's statistics in memory at a time, and a slow/large
    sensor's collection doesn't hold up figures for every other sensor."""
    print(f"[{diag_obs}] collecting figures from diagnosis files in {diag_base_dir}")
    jobs = generate_plots(config, diag_obs, sensor_graphic_dir, diag_base_dir)
    for job in jobs:
        ok, err = run_job(job)
        if not ok:
            print(f"ERROR in {job[0].__name__} ({diag_obs}):\n{err}")
    print(f"[{diag_obs}] finished {len(jobs)} figures")
    return diag_obs, len(jobs)


def generate_time_series_graphics(config, diag_obs, graphic_out_dir, diag_base_dir):
    """Build one time-series job per (domain, statistic), via
    plot_time_series_0d -- returns [] when the config has no "da"/"hofx"-
    typed experiments (e.g. an all-"hofx_forecast" config -- see
    generate_forecast_lead_graphics()). Also builds one extra per-domain
    job (plot_time_series_reduction_0d) showing % RMS reduction from OMB to
    OMA -- the SAME two RMS numbers as the corresponding absolute figure's
    OMB/OMA lines, just plotted as a single derived percent-change line per
    "da"-typed experiment (only those actually produce an OMA -- see
    _experiments_for_diagtype()) instead of two absolute ones. Skipped
    entirely when no experiment has OMA."""
    cfg, index_map = _base_graphic_cfg(config, graphic_out_dir)
    pseudo_names = cfg["experiments"]["names"]
    if not pseudo_names:
        return []

    display_vars = display_var_names(diag_obs)
    raw_vars = get_obs_config(diag_obs).variables
    domains = list(resolve_domains(config).keys())
    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs)
    cycles = generate_cycles(adjusted_start, end_time, interval)

    cfg["variables"] = display_vars

    # Read once per pseudo-experiment (covers every domain/stat), reused by
    # every job below -- cheap, and keeps this collection off the worker
    # processes entirely.
    per_exp_stats = {
        pseudo_name: collect_obs_stats_0d(
            diag_base_dir, exp_name, diagtype, diag_obs, cycles,
            display_vars, raw_vars, domains,
        )
        for pseudo_name, (exp_name, diagtype) in zip(pseudo_names, index_map)
    }

    # Real "da"-typed experiments that produce an OMA (see
    # _experiments_for_diagtype) -- these are the only ones with anything
    # to reduce; re-keyed from the OMB/OMA pseudo-experiment dicts already
    # read above rather than reading the diagnosis files a second time.
    oma_names, oma_labels, oma_colors = _experiments_for_diagtype(config, "oma")
    stats_omb = {name: per_exp_stats.get(f"{name}__omb", {}) for name in oma_names}
    stats_oma = {name: per_exp_stats.get(f"{name}__oma", {}) for name in oma_names}

    jobs = []
    for domain in domains:
        cfg_dom = dict(cfg)
        cfg_dom["region"] = domain
        for stat_disp, _ in STAT_KEYS:
            jobs.append((plot_time_series_0d, (per_exp_stats, stat_disp, cfg_dom), {}))

        if oma_names:
            cfg_red = dict(cfg_dom)
            cfg_red["experiments"] = {
                "names": oma_names, "labels": oma_labels, "colors": oma_colors,
                "linestyles": ["-"] * len(oma_names),
            }
            jobs.append((plot_time_series_reduction_0d, (stats_omb, stats_oma, cfg_red), {}))
    return jobs


def generate_profile_graphics(config, diag_obs, graphic_out_dir, diag_base_dir):
    """Build one profile job per (domain, statistic), via plot_profiles_1d
    -- returns [] when the config has no "da"/"hofx"-typed experiments.
    Also builds one extra per-domain job (plot_profile_reduction_1d)
    showing % RMS reduction from OMB to OMA vs. level -- the SAME RMS
    profiles as the corresponding absolute figure's OMB/OMA lines, just
    plotted as a single derived percent-change line per "da"-typed
    experiment (only those actually produce an OMA -- see
    _experiments_for_diagtype()) instead of two absolute ones. Skipped
    entirely when no experiment has OMA."""
    cfg, index_map = _base_graphic_cfg(config, graphic_out_dir)
    pseudo_names = cfg["experiments"]["names"]
    if not pseudo_names:
        return []

    bin_method = get_bin_method(diag_obs)
    display_vars = display_var_names(diag_obs)
    raw_vars = get_obs_config(diag_obs).variables
    domains = list(resolve_domains(config).keys())
    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs)
    cycles = generate_cycles(adjusted_start, end_time, interval)

    cfg["variables"] = display_vars

    coord_label, invert_y = _profile_axis_meta(bin_method)

    # Real "da"-typed experiments that produce an OMA (see
    # _experiments_for_diagtype) -- these are the only ones with anything
    # to reduce.
    oma_names, oma_labels, oma_colors = _experiments_for_diagtype(config, "oma")

    # Radiances (bin_method 3) only have a meaningful, region-independent
    # latitude-binned profile (computed once, using the "global" domain's
    # data), matching the original script's behaviour.
    if bin_method == 3:
        if "global" not in domains:
            return []
        domain_loop = ["global"]
    else:
        domain_loop = domains

    jobs = []
    for domain in domain_loop:
        profile_key = profile_key_for_domain(bin_method, domain)

        per_exp_profiles = {
            pseudo_name: collect_obs_profiles_1d(
                diag_base_dir, exp_name, diagtype, diag_obs, cycles,
                display_vars, raw_vars, profile_key, coord_label, invert_y,
                time_aggregation=_time_aggregation(config))
            for pseudo_name, (exp_name, diagtype) in zip(pseudo_names, index_map)
        }

        for stat_disp, _ in STAT_KEYS:
            Profiles = {name: per_exp_profiles[name].get(stat_disp, {}) for name in pseudo_names}
            if any(Profiles.values()):
                jobs.append((plot_profiles_1d, (Profiles, profile_key, stat_disp, cfg), {}))

        if oma_names:
            profiles_omb = {name: per_exp_profiles.get(f"{name}__omb", {}).get("RMS", {})
                            for name in oma_names}
            profiles_oma = {name: per_exp_profiles.get(f"{name}__oma", {}).get("RMS", {})
                            for name in oma_names}
            if any(profiles_omb.values()) and any(profiles_oma.values()):
                cfg_red = dict(cfg)
                cfg_red["experiments"] = {
                    "names": oma_names, "labels": oma_labels, "colors": oma_colors,
                    "linestyles": ["-"] * len(oma_names),
                }
                jobs.append((plot_profile_reduction_1d,
                            (profiles_omb, profiles_oma, profile_key, cfg_red), {}))
    return jobs


def generate_lat_lon_graphics(config, diag_obs, graphic_out_dir, diag_base_dir):
    """Build one lat-lon map job per (diagtype, statistic), via
    plot_lat_lon_2d, with the REAL configured experiments on the panel axis
    -- experiments.names[0] plotted as the absolute/full-state reference,
    every other experiment as a bootstrap % RMS/STD reduction vs. it (see
    _comparison_graphic_cfg). OMB and OMA are drawn as two separate sets of
    figures, under "omb/lat_lon_2d/" and "oma/lat_lon_2d/" respectively,
    since they can't share this real-experiment panel axis with the
    reference/reduction convention at the same time. A diagtype with no
    contributing experiments (see _experiments_for_diagtype) contributes no
    jobs."""
    display_vars = display_var_names(diag_obs)
    raw_vars = get_obs_config(diag_obs).variables
    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs)
    cycles = generate_cycles(adjusted_start, end_time, interval)

    jobs = []
    for diagtype in ("omb", "oma"):
        names, labels, colors = _experiments_for_diagtype(config, diagtype)
        if not names:
            continue

        cfg = _comparison_graphic_cfg(
            config, os.path.join(graphic_out_dir, diagtype), names, labels, colors)
        cfg["variables"] = display_vars

        per_exp_latlon = {
            name: collect_obs_lat_lon_2d(
                diag_base_dir, name, diagtype, diag_obs, cycles,
                display_vars, raw_vars,
                time_aggregation=_time_aggregation(config))
            for name in names
        }

        for stat_disp, _ in STAT_KEYS_2D:
            LatLon = {name: per_exp_latlon[name].get(stat_disp, {}) for name in names}
            if any(LatLon.values()):
                jobs.append((plot_lat_lon_2d, (LatLon, stat_disp, cfg), {}))
    return jobs


def generate_time_level_graphics(config, diag_obs, graphic_out_dir, diag_base_dir):
    """Build one lev-time (cycle x level) Hovmoller job per (diagtype,
    domain, statistic), via plot_time_level_2d, with the REAL configured
    experiments on the panel axis (see generate_lat_lon_graphics for why).
    Returns [] when plot_settings.generate_time_level_2d is false; a
    diagtype with no contributing experiments contributes no jobs."""
    if not config["plot_settings"].get("generate_time_level_2d", True):
        return []

    bin_method = get_bin_method(diag_obs)
    display_vars = display_var_names(diag_obs)
    raw_vars = get_obs_config(diag_obs).variables
    domains = list(resolve_domains(config).keys())
    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs)
    cycles = generate_cycles(adjusted_start, end_time, interval)
    coord_label, invert_y = _profile_axis_meta(bin_method)

    # Radiances (bin_method 3): only the region-independent lat_bin_profile
    # exists, same special-casing as generate_profile_graphics -- this
    # makes lev-time effectively a latitude x time Hovmoller for radiances.
    if bin_method == 3:
        if "global" not in domains:
            return []
        domain_loop = ["global"]
    else:
        domain_loop = domains

    jobs = []
    for diagtype in ("omb", "oma"):
        names, labels, colors = _experiments_for_diagtype(config, diagtype)
        if not names:
            continue

        cfg = _comparison_graphic_cfg(
            config, os.path.join(graphic_out_dir, diagtype), names, labels, colors)
        cfg["variables"] = display_vars

        for domain in domain_loop:
            profile_key = profile_key_for_domain(bin_method, domain)

            per_exp_timelev = {
                name: collect_obs_time_level_2d(
                    diag_base_dir, name, diagtype, diag_obs, cycles,
                    display_vars, raw_vars, profile_key, coord_label, invert_y,
                )
                for name in names
            }

            for stat_disp, _ in STAT_KEYS:
                TimeLev = {name: per_exp_timelev[name].get(stat_disp, {}) for name in names}
                if any(TimeLev.values()):
                    jobs.append((plot_time_level_2d, (TimeLev, profile_key, stat_disp, cfg), {}))
    return jobs


def generate_lat_vert_graphics(config, diag_obs, graphic_out_dir, diag_base_dir):
    """Build one lev-lat (latitude x level) cross-section job per
    (diagtype, statistic), via plot_lat_vert_2d, with the REAL configured
    experiments on the panel axis (see generate_lat_lon_graphics for why).
    Returns [] entirely for radiances (bin_method LATITUDE -- the "level"
    axis would already be latitude, a degenerate lat-vs-lat plot) and when
    plot_settings.generate_lat_vert_2d is false."""
    if not config["plot_settings"].get("generate_lat_vert_2d", True):
        return []

    bin_method = get_bin_method(diag_obs)
    if bin_method == 3:
        return []

    display_vars = display_var_names(diag_obs)
    raw_vars = get_obs_config(diag_obs).variables
    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs)
    cycles = generate_cycles(adjusted_start, end_time, interval)
    coord_label, invert_y = _profile_axis_meta(bin_method)
    boot = get_bootstrap_settings(config)

    jobs = []
    for diagtype in ("omb", "oma"):
        names, labels, colors = _experiments_for_diagtype(config, diagtype)
        if not names:
            continue

        cfg = _comparison_graphic_cfg(
            config, os.path.join(graphic_out_dir, diagtype), names, labels, colors)
        cfg["variables"] = display_vars

        per_exp_latvert = {
            name: collect_obs_lat_vert_2d(
                diag_base_dir, name, diagtype, diag_obs, cycles,
                display_vars, raw_vars, coord_label, invert_y, boot,
                time_aggregation=_time_aggregation(config))
            for name in names
        }

        for stat_disp, _ in STAT_KEYS_2D:
            LatVert = {name: per_exp_latvert[name].get(stat_disp, {}) for name in names}
            if any(LatVert.values()):
                jobs.append((plot_lat_vert_2d, (LatVert, "lat_vert", stat_disp, cfg), {}))
    return jobs


def generate_forecast_lead_graphics(config, diag_obs, graphic_out_dir, diag_base_dir):
    """"Forecast skill": build one job per (domain, statistic), via
    plot_forecast_lead_0d -- reading OMF diagnosis files written by
    process_forecast_experiment() / write_obs_forecast_diag_files().
    Returns [] when the config has no "hofx_forecast"-typed experiments."""
    names, labels, colors, linestyles = _build_forecast_experiments(config)
    if not names:
        return []

    display_vars = display_var_names(diag_obs)
    raw_vars = get_obs_config(diag_obs).variables
    domains = list(resolve_domains(config).keys())
    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs, for_forecast=True)
    cycles = generate_cycles(adjusted_start, end_time, interval)
    leads = generate_forecast_leads(config)
    boot = get_bootstrap_settings(config)

    cfg = {
        "variables": display_vars,
        "experiments": {
            "names": names,
            "labels": labels,
            "colors": colors,
            "linestyles": linestyles,
        },
        "paths": {"out_dir": graphic_out_dir},
        "figure_format": config["plot_settings"].get("figure_format", "png"),
        "bootstrap": config.get("bootstrap", {}),
    }

    jobs = []
    for domain in domains:
        cfg_dom = dict(cfg)
        cfg_dom["region"] = domain
        for stat_disp, _ in STAT_KEYS:
            ForecastLead = {
                exp_name: collect_obs_forecast_lead_0d(
                    diag_base_dir, exp_name, diag_obs, cycles, leads,
                    display_vars, raw_vars, domain, stat_disp, boot,
                    time_aggregation=_time_aggregation(config))
                for exp_name in names
            }
            if any(ForecastLead.values()):
                jobs.append((plot_forecast_lead_0d, (ForecastLead, stat_disp, cfg_dom), {}))
    return jobs


# ----------------------------------------------------------------------------
# OMF forecast-lead counterparts of the profile/lat-lon/time-level/lat-vert
# products above -- extending obs-space verification to the SAME figure set
# the model-space package draws for its own extended-forecast skill (see
# src/mpas_stat/graphics/driver.py's forecast_profiles_1d / forecast_lat_lon_2d /
# forecast_time_level_2d / forecast_lat_vert_2d plot types). Each reuses the
# shared plot_* function completely unmodified, exactly like
# generate_forecast_lead_graphics already does for plot_forecast_lead_0d.
# ----------------------------------------------------------------------------

def generate_forecast_profile_graphics(config, diag_obs, graphic_out_dir, diag_base_dir):
    """Build one profile job per (domain, forecast lead hour, statistic),
    via plot_profiles_1d reused unmodified -- the OMF counterpart of
    generate_profile_graphics, mirroring the model-space package's own
    forecast_profiles_1d plot type. Returns [] when the config has no
    "hofx_forecast"-typed experiments, for sfc sensors (no vertical
    profile -- same as generate_plots' own sfc early-return), or (for
    radiances) when "global" isn't a configured domain."""
    names, labels, colors, linestyles = _build_forecast_experiments(config)
    if not names:
        return []

    bin_method = get_bin_method(diag_obs)
    display_vars = display_var_names(diag_obs)
    raw_vars = get_obs_config(diag_obs).variables
    domains = list(resolve_domains(config).keys())
    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs, for_forecast=True)
    cycles = generate_cycles(adjusted_start, end_time, interval)
    leads = generate_forecast_leads(config)
    boot = get_bootstrap_settings(config)
    coord_label, invert_y = _profile_axis_meta(bin_method)

    if bin_method == 3:
        if "global" not in domains:
            return []
        domain_loop = ["global"]
    else:
        domain_loop = domains

    cfg = {
        "variables": display_vars,
        "experiments": {
            "names": names,
            "labels": labels,
            "colors": colors,
            "linestyles": linestyles,
        },
        "paths": {"out_dir": graphic_out_dir},
        "figure_format": config["plot_settings"].get("figure_format", "png"),
    }

    jobs = []
    for domain in domain_loop:
        profile_key = profile_key_for_domain(bin_method, domain)
        for lead in leads:
            per_exp_profiles = {
                name: collect_obs_forecast_profiles_1d(
                    diag_base_dir, name, diag_obs, cycles, lead,
                    display_vars, raw_vars, profile_key, coord_label, invert_y, boot,
                    time_aggregation=_time_aggregation(config))
                for name in names
            }
            lead_cfg = _lead_cfg(cfg, lead)
            for stat_disp, _ in STAT_KEYS:
                Profiles = {name: per_exp_profiles[name].get(stat_disp, {}) for name in names}
                if any(Profiles.values()):
                    jobs.append((plot_profiles_1d, (Profiles, profile_key, stat_disp, lead_cfg), {}))
    return jobs


def generate_forecast_lat_lon_graphics(config, diag_obs, graphic_out_dir, diag_base_dir):
    """Build one lat-lon map job per (forecast lead hour, statistic), via
    plot_lat_lon_2d reused unmodified -- the OMF counterpart of
    generate_lat_lon_graphics, mirroring the model-space package's own
    forecast_lat_lon_2d plot type. Real OMF experiments sit on the panel
    axis with the SAME reference/bootstrap-%-reduction comparison
    convention as OMB/OMA's own lat-lon maps (_comparison_graphic_cfg);
    unlike OMB/OMA there's no diagtype to split on, so figures land
    directly under graphic_out_dir/lat_lon_2d/ (matching
    generate_forecast_lead_graphics' own un-prefixed output folder).
    Returns [] when the config has no "hofx_forecast"-typed experiments, or
    when plot_settings.generate_lat_lon_2d is false."""
    if not config["plot_settings"].get("generate_lat_lon_2d", True):
        return []
    names, labels, colors = _build_forecast_experiments(config)[:3]
    if not names:
        return []

    display_vars = display_var_names(diag_obs)
    raw_vars = get_obs_config(diag_obs).variables
    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs, for_forecast=True)
    cycles = generate_cycles(adjusted_start, end_time, interval)
    leads = generate_forecast_leads(config)
    boot = get_bootstrap_settings(config)

    cfg = _comparison_graphic_cfg(config, graphic_out_dir, names, labels, colors)
    cfg["variables"] = display_vars

    jobs = []
    for lead in leads:
        per_exp_latlon = {
            name: collect_obs_forecast_lat_lon_2d(
                diag_base_dir, name, diag_obs, cycles, lead, display_vars, raw_vars, boot,
                time_aggregation=_time_aggregation(config))
            for name in names
        }
        lead_cfg = _lead_cfg(cfg, lead)
        for stat_disp, _ in STAT_KEYS_2D:
            LatLon = {name: per_exp_latlon[name].get(stat_disp, {}) for name in names}
            if any(LatLon.values()):
                jobs.append((plot_lat_lon_2d, (LatLon, stat_disp, lead_cfg), {}))
    return jobs


def generate_forecast_lat_vert_graphics(config, diag_obs, graphic_out_dir, diag_base_dir):
    """Build one lev-lat (latitude x level) cross-section job per
    (forecast lead hour, statistic), via plot_lat_vert_2d reused unmodified
    -- the OMF counterpart of generate_lat_vert_graphics, mirroring the
    model-space package's own forecast_lat_vert_2d plot type. Returns []
    entirely for radiances (bin_method LATITUDE) and when
    plot_settings.generate_lat_vert_2d is false, same as
    generate_lat_vert_graphics."""
    if not config["plot_settings"].get("generate_lat_vert_2d", True):
        return []
    bin_method = get_bin_method(diag_obs)
    if bin_method == 3:
        return []
    names, labels, colors = _build_forecast_experiments(config)[:3]
    if not names:
        return []

    display_vars = display_var_names(diag_obs)
    raw_vars = get_obs_config(diag_obs).variables
    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs, for_forecast=True)
    cycles = generate_cycles(adjusted_start, end_time, interval)
    leads = generate_forecast_leads(config)
    coord_label, invert_y = _profile_axis_meta(bin_method)
    boot = get_bootstrap_settings(config)

    cfg = _comparison_graphic_cfg(config, graphic_out_dir, names, labels, colors)
    cfg["variables"] = display_vars

    jobs = []
    for lead in leads:
        per_exp_latvert = {
            name: collect_obs_forecast_lat_vert_2d(
                diag_base_dir, name, diag_obs, cycles, lead,
                display_vars, raw_vars, coord_label, invert_y, boot,
                time_aggregation=_time_aggregation(config))
            for name in names
        }
        lead_cfg = _lead_cfg(cfg, lead)
        for stat_disp, _ in STAT_KEYS_2D:
            LatVert = {name: per_exp_latvert[name].get(stat_disp, {}) for name in names}
            if any(LatVert.values()):
                jobs.append((plot_lat_vert_2d, (LatVert, "lat_vert", stat_disp, lead_cfg), {}))
    return jobs


def generate_forecast_time_level_graphics(config, diag_obs, graphic_out_dir, diag_base_dir):
    """Build one forecast-lead x level Hovmoller job per (domain,
    statistic), via plot_forecast_time_level_2d -- the OMF counterpart of
    generate_time_level_graphics, with forecast lead hour on the x-axis
    instead of assimilation cycle, mirroring the model-space package's own
    forecast_time_level_2d plot type exactly. Each lead-hour column already
    averages over every initialization cycle, so (unlike the OMB/OMA
    lev-time Hovmoller) bootstrap significance vs. the reference/control
    experiment IS available here, drawn automatically by the shared
    _draw_2d_panels machinery. Returns [] when the config has no
    "hofx_forecast"-typed experiments, or when
    plot_settings.generate_time_level_2d is false."""
    if not config["plot_settings"].get("generate_time_level_2d", True):
        return []
    names, labels, colors = _build_forecast_experiments(config)[:3]
    if not names:
        return []

    bin_method = get_bin_method(diag_obs)
    display_vars = display_var_names(diag_obs)
    raw_vars = get_obs_config(diag_obs).variables
    domains = list(resolve_domains(config).keys())
    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs, for_forecast=True)
    cycles = generate_cycles(adjusted_start, end_time, interval)
    leads = generate_forecast_leads(config)
    coord_label, invert_y = _profile_axis_meta(bin_method)
    boot = get_bootstrap_settings(config)

    if bin_method == 3:
        if "global" not in domains:
            return []
        domain_loop = ["global"]
    else:
        domain_loop = domains

    cfg = _comparison_graphic_cfg(config, graphic_out_dir, names, labels, colors)
    cfg["variables"] = display_vars

    jobs = []
    for domain in domain_loop:
        profile_key = profile_key_for_domain(bin_method, domain)
        for stat_disp, _ in STAT_KEYS:
            TimeLev = {
                name: collect_obs_forecast_time_level_2d(
                    diag_base_dir, name, diag_obs, cycles, leads,
                    display_vars, raw_vars, profile_key, coord_label, invert_y,
                    stat_disp, boot,
                    time_aggregation=_time_aggregation(config))
                for name in names
            }
            if any(TimeLev.values()):
                jobs.append((plot_forecast_time_level_2d, (TimeLev, profile_key, stat_disp, cfg), {}))
    return jobs


def generate_plots(config, diag_obs, graphic_out_dir, diag_base_dir):
    """Collect every obs-space figure job for one obs type using MPAS-STAT's
    graphic functions, reading statistics back from the diagnosis files
    src/mpas_stat/obs/diag.py already wrote. Returns a flat list of
    (plot_func, args, kwargs) jobs -- nothing is drawn or saved here; see
    _process_sensor() / main()."""
    jobs = []
    jobs += generate_time_series_graphics(config, diag_obs, graphic_out_dir, diag_base_dir)

    # Forecast skill (OMF vs. lead time) is domain-averaged only, so it
    # applies regardless of whether this sensor gets profile/map figures.
    jobs += generate_forecast_lead_graphics(config, diag_obs, graphic_out_dir, diag_base_dir)

    if diag_obs.startswith("sfc"):
        return jobs

    jobs += generate_profile_graphics(config, diag_obs, graphic_out_dir, diag_base_dir)
    jobs += generate_lat_lon_graphics(config, diag_obs, graphic_out_dir, diag_base_dir)
    jobs += generate_time_level_graphics(config, diag_obs, graphic_out_dir, diag_base_dir)
    jobs += generate_lat_vert_graphics(config, diag_obs, graphic_out_dir, diag_base_dir)

    # OMF counterparts of the four products just above -- same figures,
    # same shared plot_* functions, one extra "forecast lead hour" axis.
    jobs += generate_forecast_profile_graphics(config, diag_obs, graphic_out_dir, diag_base_dir)
    jobs += generate_forecast_lat_lon_graphics(config, diag_obs, graphic_out_dir, diag_base_dir)
    jobs += generate_forecast_time_level_graphics(config, diag_obs, graphic_out_dir, diag_base_dir)
    jobs += generate_forecast_lat_vert_graphics(config, diag_obs, graphic_out_dir, diag_base_dir)
    return jobs


def main(config_path=None):
    """Read obs-space diagnosis files and generate figures for every sensor.

    Parallelism is per-SENSOR, not per-figure: each sensor's own statistics
    collection (reading its diagnosis netCDF files) and figure rendering
    both happen together, inside whichever process handles that sensor
    (see _process_sensor()). Concretely, with config["n_workers"] > 1,
    different sensors' collection + rendering run concurrently across
    worker processes, instead of every sensor's statistics first being read
    into this one main process before any figure starts rendering -- so
    peak memory is bounded by n_workers sensors' worth of collected
    statistics at a time, not every configured sensor's at once.
    n_workers <= 1 (the default) processes sensors one at a time in this
    process, unchanged from before parallelization was added. This mirrors
    the model-space package's own config["n_workers"] knob
    (src/mpas_stat/graphics/driver.py), just applied at sensor granularity here
    since obs-space (unlike model-space) has a natural per-sensor split,
    with each sensor's diagnosis files independent of every other's."""
    config = load_config(config_path)

    graphics_base_dir = config["paths"].get("graphic_dir", "graphics")
    os.makedirs(graphics_base_dir, exist_ok=True)
    diag_base_dir = resolve_diag_base_dir(config)

    exp_names = config["experiments"]["names"]
    n_workers = resolve_n_workers(config)

    sensor_tasks = []
    for diag_obs in config["sensors"]:
        # Diagnosis files now live under <diag_base_dir>/<experiment>/
        # VerifyObs/... (one subtree per experiment, obs type identified
        # only by filename) rather than a single <diag_base_dir>/<diag_obs>/
        # directory -- see obs_diag_file()/obs_diag_forecast_file() in
        # src/mpas_stat/obs/common.py. obs_diag_sensor_has_files() checks across every
        # configured experiment so this warning still fires when NONE of
        # them have been processed yet for this sensor.
        if not obs_diag_sensor_has_files(diag_base_dir, exp_names, diag_obs):
            print(f"WARNING: no diagnosis files found for {diag_obs} under {diag_base_dir}; "
                  f"run src/mpas_stat/obs/diag.py first. Skipping.")
            continue
        sensor_graphic_dir = os.path.join(graphics_base_dir, diag_obs)
        os.makedirs(sensor_graphic_dir, exist_ok=True)
        sensor_tasks.append((diag_obs, sensor_graphic_dir))

    if not sensor_tasks:
        return

    if n_workers == 1 or len(sensor_tasks) == 1:
        for diag_obs, sensor_graphic_dir in sensor_tasks:
            _process_sensor(config, diag_obs, sensor_graphic_dir, diag_base_dir)
        return

    n_workers = min(n_workers, len(sensor_tasks))
    print(f"Processing {len(sensor_tasks)} sensors across {n_workers} worker processes "
          f"(each sensor's statistics collection + figure rendering runs together in one worker)...")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_sensor, config, diag_obs, sensor_graphic_dir, diag_base_dir): diag_obs
            for diag_obs, sensor_graphic_dir in sensor_tasks
        }
        for future in as_completed(futures):
            diag_obs = futures[future]
            try:
                future.result()
            except Exception:
                print(f"ERROR processing sensor {diag_obs}:\n{traceback.format_exc()}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
