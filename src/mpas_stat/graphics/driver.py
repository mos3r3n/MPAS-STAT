"""Model-space graphics driver: turn a graphic.yaml config into figure jobs.

Each job pairs one collect_* call (reading mpasstat.<date>.nc files) with the
matching plot_* call; jobs then run sequentially or across a process pool.
"""
from ..core.jobs import lead_cfg as _lead_cfg
from ..core.jobs import resolve_n_workers, run_jobs
from ..core.timeutils import generate_forecast_leads
from .collect import (
    collect_lat_lon_2d,
    collect_lat_lon_slice_2d,
    collect_lat_vert_2d,
    collect_profiles_1d,
    collect_stats_0d,
    collect_time_level_2d,
)
from .collect_forecast import (
    collect_forecast_lat_lon_2d,
    collect_forecast_lat_vert_2d,
    collect_forecast_lead_0d,
    collect_forecast_profiles_1d,
    collect_forecast_scorecard,
    collect_forecast_time_level_2d,
)
from .plots_0d import (
    plot_forecast_lead_0d,
    plot_forecast_lead_diff_0d,
    plot_time_series_0d,
)
from .plots_1d import plot_profile_diff_1d, plot_profiles_1d
from .plots_2d import (
    plot_forecast_time_level_2d,
    plot_lat_lon_2d,
    plot_lat_lon_slice_2d,
    plot_lat_vert_2d,
    plot_time_level_2d,
)
from .scorecard import plot_forecast_scorecard

DEFAULT_PROFILES = [
    "lat_bin_profile",
    "vert_pres_profile",
    "vert_hgt_profile",
    "vert_bin_profile",
]

DEFAULT_LAT_VERT_SECTIONS = [
    "lat_lev",
    "lat_hgt",
    "lat_pres",
]


def _cfg_with_variables(cfg, variables):
    """Return a shallow cfg copy with a reduced variable list."""
    new_cfg = dict(cfg)
    new_cfg["variables"] = list(variables)
    return new_cfg

def _configured_vertical_variables(cfg):
    """Return user-declared 3-D/vertical variables when available."""
    for key in ("vertical_variables", "variables_3d", "three_d_variables", "upper_air_variables"):
        vals = cfg.get(key, None)
        if vals:
            return list(vals)
    return list(cfg.get("variables", []))


def _vertical_cfg_for_collection(cfg):
    """Use only declared vertical variables for vertical-related collectors."""
    return _cfg_with_variables(cfg, _configured_vertical_variables(cfg))


def _available_variables(data_store, requested_variables):
    """Return requested variables that have data in at least one experiment."""
    available = []
    for var_name in requested_variables:
        for exp_store in data_store.values():
            if isinstance(exp_store, dict) and var_name in exp_store:
                available.append(var_name)
                break
    return available


def _plot_if_has_variables(plot_func, data_store, cfg_for_collection, **kwargs):
    """Call a plot function only after removing unavailable variables."""
    valid_vars = _available_variables(data_store, cfg_for_collection.get("variables", []))
    if not valid_vars:
        return
    plot_func(**kwargs, cfg=_cfg_with_variables(cfg_for_collection, valid_vars))


def _regions(cfg):
    """Return profile regions. Backward-compatible with single 'region'."""
    regs = cfg.get("regions", None)
    if regs is None:
        regs = [cfg.get("region", "Global")]
    elif isinstance(regs, str):
        regs = [regs]
    return [str(r).strip() for r in regs if str(r).strip()]


def _is_region_dependent_profile(profile_key):
    """Only vertical profile diagnostics are region dependent."""
    return str(profile_key).startswith((
        "vert_pres_profile",
        "vert_hgt_profile",
        "vert_bin_profile",
        "vert_profile",
    ))


def _profile_nc_key(profile_key, region):
    """Return the NetCDF group name for a profile diagnostic."""
    if not _is_region_dependent_profile(profile_key):
        return profile_key
    if not region:
        return profile_key
    if str(profile_key).endswith(f"_{region}"):
        return profile_key
    return f"{profile_key}_{region}"


def _profile_plot_key(profile_key, region):
    """Return the clean figure-name key for a profile diagnostic."""
    if not _is_region_dependent_profile(profile_key):
        return profile_key
    if str(region).lower() == "global":
        return profile_key
    return f"{profile_key}_{region}"


def _profile_tasks(cfg):
    """Return (profile_nc_key, profile_plot_key) pairs.

    Region-independent profiles such as lat_bin_profile are included once.
    Region-dependent vertical profiles are expanded over all configured regions.
    """
    tasks = []
    for profile_key in cfg.get("profiles", DEFAULT_PROFILES):
        if _is_region_dependent_profile(profile_key):
            for region in _regions(cfg):
                tasks.append((_profile_nc_key(profile_key, region),
                              _profile_plot_key(profile_key, region)))
        else:
            tasks.append((profile_key, profile_key))
    return tasks


def _slice_sections(cfg):
    """Return selected 3-D lat-lon slice configurations."""
    sections = cfg.get("level_slices", cfg.get("lat_lon_3d_slices", []))
    if isinstance(sections, dict):
        sections = [sections]
    return list(sections)


def _levels_from_section(section_cfg):
    """Return list of requested levels from a section configuration."""
    levels = section_cfg.get("levels", section_cfg.get("level", None))
    if levels is None:
        return []
    if isinstance(levels, (list, tuple)):
        return list(levels)
    return [levels]


def _level_name(vertical_type, level):
    """Compact level suffix for output names."""
    level_text = str(level).replace(".", "p")
    return f"{vertical_type}{level_text}"



# ---------------------------------------------------------------------------
# Parallel plotting jobs
#
# Each block in run_graphics() used to call its collect_* + plot_* pair
# directly. That work is now packaged as a "job" -- (function, args, kwargs)
# -- and collected into one flat list so it can be run either sequentially
# (the default, byte-for-byte the same order/behaviour as before) or spread
# across a process pool (cfg["n_workers"] > 1). Every job function below is
# a plain, top-level function taking only picklable arguments (cfg dicts,
# strings, ints, small dicts) so it can be sent to a worker process.
#
# Matplotlib figures themselves are never shared across jobs or processes:
# each job collects its own data and creates/saves/closes its own figure,
# so the jobs are fully independent and safe to run concurrently. Process
# based parallelism (not threads) is used deliberately, since a single
# matplotlib/pyplot process is not safe for concurrent figure creation.
# ---------------------------------------------------------------------------
def _job_time_series_0d(cfg, stats):
    Stats = collect_stats_0d(cfg)
    for stat_name in stats:
        plot_time_series_0d(Stats=Stats, stat_name=stat_name, cfg=cfg)


def _job_forecast_lead_0d(cfg, stat_name):
    ForecastLead = collect_forecast_lead_0d(cfg, stat_name=stat_name)
    _plot_if_has_variables(
        plot_forecast_lead_0d, ForecastLead, cfg,
        ForecastLead=ForecastLead, stat_name=stat_name,
    )


def _job_forecast_lead_diff_0d(cfg, stat_name):
    ForecastLead = collect_forecast_lead_0d(cfg, stat_name=stat_name)
    _plot_if_has_variables(
        plot_forecast_lead_diff_0d, ForecastLead, cfg,
        ForecastLead=ForecastLead, stat_name=stat_name,
    )


def _job_forecast_scorecard(cfg, stat_name):
    ScoreCard, row_labels = collect_forecast_scorecard(cfg, stat_name=stat_name)
    plot_forecast_scorecard(
        ScoreCard=ScoreCard, row_labels=row_labels,
        stat_name=stat_name, cfg=cfg,
    )


def _job_profiles_1d(vcfg, profile_nc_key, profile_plot_key, stat_name):
    Profiles = collect_profiles_1d(cfg=vcfg, profile_key=profile_nc_key, stat_name=stat_name)
    _plot_if_has_variables(
        plot_profiles_1d, Profiles, vcfg,
        Profiles=Profiles, profile_key=profile_plot_key, stat_name=stat_name,
    )


def _job_forecast_profiles_1d(vcfg, lead_vcfg, profile_nc_key, profile_plot_key, stat_name, lead):
    Profiles = collect_forecast_profiles_1d(
        cfg=vcfg, profile_key=profile_nc_key, stat_name=stat_name, lead_hour=lead)
    _plot_if_has_variables(
        plot_profiles_1d, Profiles, lead_vcfg,
        Profiles=Profiles, profile_key=profile_plot_key, stat_name=stat_name,
    )


def _job_profile_diff_1d(vcfg, profile_nc_key, profile_plot_key, stat_name):
    Profiles = collect_profiles_1d(cfg=vcfg, profile_key=profile_nc_key, stat_name=stat_name)
    _plot_if_has_variables(
        plot_profile_diff_1d, Profiles, vcfg,
        Profiles=Profiles, profile_key=profile_plot_key, stat_name=stat_name,
    )


def _job_forecast_profile_diff_1d(vcfg, lead_vcfg, profile_nc_key, profile_plot_key, stat_name, lead):
    Profiles = collect_forecast_profiles_1d(
        cfg=vcfg, profile_key=profile_nc_key, stat_name=stat_name, lead_hour=lead)
    _plot_if_has_variables(
        plot_profile_diff_1d, Profiles, lead_vcfg,
        Profiles=Profiles, profile_key=profile_plot_key, stat_name=stat_name,
    )


def _job_time_level_2d(vcfg, profile_nc_key, profile_plot_key, stat_name):
    TimeLev = collect_time_level_2d(cfg=vcfg, profile_key=profile_nc_key, stat_name=stat_name)
    _plot_if_has_variables(
        plot_time_level_2d, TimeLev, vcfg,
        TimeLev=TimeLev, profile_key=profile_plot_key, stat_name=stat_name,
    )


def _job_forecast_time_level_2d(vcfg, profile_nc_key, profile_plot_key, stat_name):
    ForecastTimeLev = collect_forecast_time_level_2d(
        cfg=vcfg, profile_key=profile_nc_key, stat_name=stat_name)
    _plot_if_has_variables(
        plot_forecast_time_level_2d, ForecastTimeLev, vcfg,
        TimeLev=ForecastTimeLev, profile_key=profile_plot_key, stat_name=stat_name,
    )


def _job_lat_vert_2d(vcfg, section_key, stat_name):
    LatVert = collect_lat_vert_2d(cfg=vcfg, section_key=section_key, stat_name=stat_name)
    _plot_if_has_variables(
        plot_lat_vert_2d, LatVert, vcfg,
        LatVert=LatVert, section_key=section_key, stat_name=stat_name,
    )


def _job_forecast_lat_vert_2d(vcfg, lead_vcfg, section_key, stat_name, lead):
    LatVert = collect_forecast_lat_vert_2d(
        cfg=vcfg, section_key=section_key, stat_name=stat_name, lead_hour=lead)
    _plot_if_has_variables(
        plot_lat_vert_2d, LatVert, lead_vcfg,
        LatVert=LatVert, section_key=section_key, stat_name=stat_name,
    )


def _job_lat_lon_2d(cfg, stat_name):
    LatLon = collect_lat_lon_2d(cfg=cfg, stat_name=stat_name)
    valid_vars = _available_variables(LatLon, cfg.get("variables", []))
    if not valid_vars:
        return
    plot_lat_lon_2d(LatLon=LatLon, stat_name=stat_name, cfg=_cfg_with_variables(cfg, valid_vars))


def _job_forecast_lat_lon_2d(cfg, lead_cfg, stat_name, lead):
    LatLon = collect_forecast_lat_lon_2d(cfg=cfg, stat_name=stat_name, lead_hour=lead)
    valid_vars = _available_variables(LatLon, cfg.get("variables", []))
    if not valid_vars:
        return
    plot_lat_lon_2d(LatLon=LatLon, stat_name=stat_name, cfg=_cfg_with_variables(lead_cfg, valid_vars))


def _job_lat_lon_slice_2d(vcfg, stat_name, one_section_cfg):
    LatLonLev = collect_lat_lon_slice_2d(cfg=vcfg, stat_name=stat_name, section_cfg=one_section_cfg)
    _plot_if_has_variables(
        plot_lat_lon_slice_2d, LatLonLev, vcfg,
        LatLonLev=LatLonLev, stat_name=stat_name, section_cfg=one_section_cfg,
    )


def run_graphics(cfg):
    """Build every requested figure's collect+plot job, then run them all.

    Sequential by default (cfg["n_workers"] unset or 1) -- same order and
    behaviour as before this was parallelized. Set cfg["n_workers"] to the
    number of CPU cores to use (e.g. matching a PBS job's ncpus) to render
    figures across multiple processes instead.
    """
    plot_types = cfg.get("plot_types", ["time_series_0d", "profiles_1d"])
    stats = cfg.get("statistics", [])
    n_workers = resolve_n_workers(cfg)

    jobs = []

    if "time_series_0d" in plot_types:
        jobs.append((_job_time_series_0d, (cfg, stats), {}))

    if "forecast_lead_0d" in plot_types:
        for stat_name in stats:
            jobs.append((_job_forecast_lead_0d, (cfg, stat_name), {}))

    if ("forecast_lead_diff_0d" in plot_types) or ("forecast_lead_reduction_0d" in plot_types):
        for stat_name in stats:
            jobs.append((_job_forecast_lead_diff_0d, (cfg, stat_name), {}))

    if "forecast_scorecard" in plot_types:
        for stat_name in stats:
            jobs.append((_job_forecast_scorecard, (cfg, stat_name), {}))

    vcfg = _vertical_cfg_for_collection(cfg)

    if "profiles_1d" in plot_types:
        for profile_nc_key, profile_plot_key in _profile_tasks(cfg):
            for stat_name in stats:
                jobs.append((_job_profiles_1d,
                            (vcfg, profile_nc_key, profile_plot_key, stat_name), {}))

    if "forecast_profiles_1d" in plot_types:
        # One figure per forecast lead hour (see graphic.yaml "forecast:").
        for lead in generate_forecast_leads(cfg):
            lead_vcfg = _lead_cfg(vcfg, lead)
            for profile_nc_key, profile_plot_key in _profile_tasks(vcfg):
                for stat_name in stats:
                    jobs.append((_job_forecast_profiles_1d,
                                (vcfg, lead_vcfg, profile_nc_key, profile_plot_key, stat_name, lead), {}))

    if ("profile_diff_1d" in plot_types) or ("profile_reduction_1d" in plot_types):
        diff_stats = cfg.get("profile_diff_statistics",
                             cfg.get("profile_reduction_statistics", ["rms"]))
        for profile_nc_key, profile_plot_key in _profile_tasks(cfg):
            for stat_name in diff_stats:
                jobs.append((_job_profile_diff_1d,
                            (vcfg, profile_nc_key, profile_plot_key, stat_name), {}))

    if ("forecast_profile_diff_1d" in plot_types) or ("forecast_profile_reduction_1d" in plot_types):
        # One figure per forecast lead hour (see graphic.yaml "forecast:").
        diff_stats = cfg.get("profile_diff_statistics",
                             cfg.get("profile_reduction_statistics", ["rms"]))
        for lead in generate_forecast_leads(cfg):
            lead_vcfg = _lead_cfg(vcfg, lead)
            for profile_nc_key, profile_plot_key in _profile_tasks(vcfg):
                for stat_name in diff_stats:
                    jobs.append((_job_forecast_profile_diff_1d,
                                (vcfg, lead_vcfg, profile_nc_key, profile_plot_key, stat_name, lead), {}))

    if "time_level_2d" in plot_types:
        for profile_nc_key, profile_plot_key in _profile_tasks(cfg):
            for stat_name in stats:
                jobs.append((_job_time_level_2d,
                            (vcfg, profile_nc_key, profile_plot_key, stat_name), {}))

    if "forecast_time_level_2d" in plot_types:
        # Single figure per profile/stat: x-axis is forecast lead time
        # (not assimilation cycle), with bootstrap significance vs. the
        # reference/control experiment since each lead-time column already
        # averages over all initialization cycles.
        for profile_nc_key, profile_plot_key in _profile_tasks(vcfg):
            for stat_name in stats:
                jobs.append((_job_forecast_time_level_2d,
                            (vcfg, profile_nc_key, profile_plot_key, stat_name), {}))

    if "lat_vert_2d" in plot_types:
        for section_key in cfg.get("lat_vert_sections", DEFAULT_LAT_VERT_SECTIONS):
            for stat_name in stats:
                jobs.append((_job_lat_vert_2d, (vcfg, section_key, stat_name), {}))

    if "forecast_lat_vert_2d" in plot_types:
        # One figure per forecast lead hour (see graphic.yaml "forecast:").
        for lead in generate_forecast_leads(cfg):
            lead_vcfg = _lead_cfg(vcfg, lead)
            for section_key in vcfg.get("lat_vert_sections", DEFAULT_LAT_VERT_SECTIONS):
                for stat_name in stats:
                    jobs.append((_job_forecast_lat_vert_2d,
                                (vcfg, lead_vcfg, section_key, stat_name, lead), {}))

    if "lat_lon_2d" in plot_types:
        for stat_name in stats:
            jobs.append((_job_lat_lon_2d, (cfg, stat_name), {}))

    if "forecast_lat_lon_2d" in plot_types:
        # One figure per forecast lead hour (see graphic.yaml "forecast:").
        for lead in generate_forecast_leads(cfg):
            lead_cfg = _lead_cfg(cfg, lead)
            for stat_name in stats:
                jobs.append((_job_forecast_lat_lon_2d, (cfg, lead_cfg, stat_name, lead), {}))

    if "lat_lon_slice_2d" in plot_types:
        for section_cfg in _slice_sections(cfg):
            for level in _levels_from_section(section_cfg):
                one_section_cfg = dict(section_cfg)
                one_section_cfg["level"] = level
                vertical_type = one_section_cfg.get("vertical_type", "level")
                one_section_cfg["name"] = _level_name(vertical_type, level)

                for stat_name in stats:
                    jobs.append((_job_lat_lon_slice_2d, (vcfg, stat_name, one_section_cfg), {}))

    run_jobs(jobs, n_workers)
