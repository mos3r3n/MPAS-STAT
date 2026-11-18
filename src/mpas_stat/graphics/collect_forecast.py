"""Collectors for extended-forecast model-space statistics (per lead hour)
and the forecast scorecard.
"""
import os

import netCDF4 as nc
import numpy as np

from ..core.bootstrap import bootstrap_mean_ci, get_bootstrap_settings
from ..core.timeutils import generate_cycles, generate_forecast_leads
from .ncread import (
    _has_expected_ndim,
    _maybe_print_missing,
    apply_region_suffix,
    get_lat_vert_coord,
    get_mpasstat_forecast_file,
    get_profile_coord,
    read_nc_path,
)


def collect_forecast_lead_0d(cfg, stat_name):
    """Collect 0-D statistics as a function of extended-forecast lead time.

    For each experiment/variable, one row is built per initialization cycle,
    holding that cycle's statistic at every configured forecast lead hour.
    The rows are then bootstrap-aggregated over cycles (samples) at each
    lead time, exactly as collect_profiles_1d aggregates over cycles at each
    vertical level -- forecast lead hour simply takes the place of the
    vertical coordinate.
    """
    base_dir    = cfg["paths"]["base_dir"]
    exp_names   = cfg["experiments"]["names"]
    var_names   = cfg["variables"]
    region_name = cfg.get("region", "global")
    boot        = get_bootstrap_settings(cfg)
    cycles      = generate_cycles(cfg["time"]["start"], cfg["time"]["end"],
                                  cfg["time"]["interval_hours"])
    leads       = generate_forecast_leads(cfg)

    ForecastLead = {}
    for exp_name in exp_names:
        ForecastLead[exp_name] = {}
        for var_name in var_names:
            samples = []
            for dt in cycles:
                row = []
                for lead in leads:
                    ncfile = get_mpasstat_forecast_file(base_dir, exp_name, dt, lead)
                    if not os.path.exists(ncfile):
                        _maybe_print_missing(cfg, f"WARNING: missing file: {ncfile}")
                        row.append(np.nan)
                        continue
                    path = f"{var_name}/0D/{region_name}/{stat_name}"
                    with nc.Dataset(ncfile, "r") as ds:
                        try:
                            row.append(float(read_nc_path(ds, path)))
                        except Exception:
                            _maybe_print_missing(cfg, f"WARNING: missing {path} in {ncfile}")
                            row.append(np.nan)
                if any(np.isfinite(v) for v in row):
                    samples.append(row)

            if not samples:
                _maybe_print_missing(
                    cfg, f"WARNING: no valid forecast-lead data for {exp_name} {var_name}")
                continue

            samples = np.asarray(samples, dtype=np.float64)
            mean_data, ci_low, ci_high = bootstrap_mean_ci(
                samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])

            ForecastLead[exp_name][var_name] = {
                "mean_profile": mean_data,
                "data":         mean_data,
                "samples":      samples,
                "ci_low":       ci_low,
                "ci_high":      ci_high,
                "leads":        np.asarray(leads, dtype=np.float64),
                "n_cycles":     samples.shape[0],
            }
    return ForecastLead


def collect_forecast_profiles_1d(cfg, profile_key, stat_name, lead_hour):
    """Like collect_profiles_1d, but at one fixed extended-forecast lead hour.

    Reads files via get_mpasstat_forecast_file instead of get_mpasstat_file,
    and bootstrap-aggregates over initialization cycles at that lead. The
    returned structure is identical to collect_profiles_1d's, so it can be
    fed directly into plot_profiles_1d / plot_profile_diff_1d unchanged.
    """
    base_dir  = cfg["paths"]["base_dir"]
    exp_names = cfg["experiments"]["names"]
    var_names = cfg["variables"]
    boot      = get_bootstrap_settings(cfg)
    cycles    = generate_cycles(cfg["time"]["start"], cfg["time"]["end"],
                                cfg["time"]["interval_hours"])

    Profiles = {}
    for exp_name in exp_names:
        Profiles[exp_name] = {}
        for var_name in var_names:
            all_profiles = []
            coord = coord_label = None
            invert_y = False

            for dt in cycles:
                ncfile = get_mpasstat_forecast_file(base_dir, exp_name, dt, lead_hour)
                if not os.path.exists(ncfile):
                    _maybe_print_missing(cfg, f"WARNING: missing file: {ncfile}")
                    continue
                path = f"{var_name}/1D/{profile_key}/{stat_name}"
                with nc.Dataset(ncfile, "r") as ds:
                    try:
                        arr = read_nc_path(ds, path)
                        if not _has_expected_ndim(arr, 1):
                            continue
                        all_profiles.append(np.asarray(arr, dtype=np.float64))
                        if coord is None:
                            coord, coord_label, invert_y = get_profile_coord(ds, profile_key)
                    except Exception:
                        _maybe_print_missing(cfg, f"WARNING: missing/invalid {path} in {ncfile}")

            if not all_profiles:
                _maybe_print_missing(
                    cfg, f"WARNING: no valid profiles for {exp_name} {var_name} at lead {lead_hour}")
                continue

            samples = np.asarray(all_profiles)
            mean_data, ci_low, ci_high = bootstrap_mean_ci(
                samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])

            Profiles[exp_name][var_name] = {
                "mean_profile": mean_data,
                "data":         mean_data,
                "samples":      samples,
                "ci_low":       ci_low,
                "ci_high":      ci_high,
                "coord":        coord,
                "coord_label":  coord_label,
                "invert_y":     invert_y,
                "lead_hour":    lead_hour,
            }
    return Profiles


def collect_forecast_time_level_2d(cfg, profile_key, stat_name):
    """Lead-time x level cross-section for extended-forecast verification.

    Unlike collect_time_level_2d (one column per assimilation cycle, with no
    repeated samples), each column here is one forecast lead hour, itself
    averaged over all initialization cycles -- exactly as
    collect_forecast_lead_0d averages 0-D statistics over cycles at each
    lead hour. Every initialization cycle contributes one (n_leads,
    n_levels) block (NaN-filled where a given lead is missing for that
    cycle), so the resulting (n_cycles, n_leads, n_levels) sample cube
    supports the same bootstrap significance testing (vs. the reference
    experiment) used for the other 2-D cross sections.
    """
    base_dir  = cfg["paths"]["base_dir"]
    exp_names = cfg["experiments"]["names"]
    var_names = cfg["variables"]
    boot      = get_bootstrap_settings(cfg)
    cycles    = generate_cycles(cfg["time"]["start"], cfg["time"]["end"],
                                cfg["time"]["interval_hours"])
    leads     = generate_forecast_leads(cfg)

    TimeLev = {}
    for exp_name in exp_names:
        TimeLev[exp_name] = {}
        for var_name in var_names:
            coord = coord_label = None
            invert_y = False
            nlev = None
            blocks = []  # one (n_leads, n_levels) block per initialization cycle

            for dt in cycles:
                row = None
                got_any = False
                for ilead, lead in enumerate(leads):
                    ncfile = get_mpasstat_forecast_file(base_dir, exp_name, dt, lead)
                    if not os.path.exists(ncfile):
                        _maybe_print_missing(cfg, f"WARNING: missing file: {ncfile}")
                        continue
                    path = f"{var_name}/1D/{profile_key}/{stat_name}"
                    with nc.Dataset(ncfile, "r") as ds:
                        try:
                            arr = read_nc_path(ds, path)
                            if not _has_expected_ndim(arr, 1):
                                continue
                            arr = np.asarray(arr, dtype=np.float64)
                            if coord is None:
                                coord, coord_label, invert_y = get_profile_coord(ds, profile_key)
                                nlev = arr.shape[0]
                            if row is None:
                                row = np.full((len(leads), nlev), np.nan, dtype=np.float64)
                            row[ilead, :] = arr
                            got_any = True
                        except Exception:
                            _maybe_print_missing(cfg, f"WARNING: missing/invalid {path} in {ncfile}")

                if got_any:
                    blocks.append(row)

            if not blocks or coord is None:
                _maybe_print_missing(
                    cfg, f"WARNING: no valid forecast time-level data for {exp_name} {var_name}")
                continue

            samples = np.asarray(blocks, dtype=np.float64)  # (n_cycles, n_leads, n_levels)
            mean_data, ci_low, ci_high = bootstrap_mean_ci(
                samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])

            TimeLev[exp_name][var_name] = {
                "data":        mean_data,
                "samples":     samples,
                "ci_low":      ci_low,
                "ci_high":     ci_high,
                "leads":       np.asarray(leads, dtype=np.float64),
                "coord":       coord,
                "coord_label": coord_label,
                "invert_y":    invert_y,
            }
    return TimeLev


def collect_forecast_lat_vert_2d(cfg, section_key, stat_name, lead_hour):
    """Like collect_lat_vert_2d, but at one fixed extended-forecast lead hour."""
    base_dir  = cfg["paths"]["base_dir"]
    exp_names = cfg["experiments"]["names"]
    var_names = cfg["variables"]
    boot      = get_bootstrap_settings(cfg)
    cycles    = generate_cycles(cfg["time"]["start"], cfg["time"]["end"],
                                cfg["time"]["interval_hours"])

    LatVert = {}
    for exp_name in exp_names:
        LatVert[exp_name] = {}
        for var_name in var_names:
            sections = []
            lat = vert = vert_label = None
            invert_y = False

            for dt in cycles:
                ncfile = get_mpasstat_forecast_file(base_dir, exp_name, dt, lead_hour)
                if not os.path.exists(ncfile):
                    _maybe_print_missing(cfg, f"WARNING: missing file: {ncfile}")
                    continue
                path = f"{var_name}/2D/{section_key}/{stat_name}"
                with nc.Dataset(ncfile, "r") as ds:
                    try:
                        arr = read_nc_path(ds, path)
                        if not _has_expected_ndim(arr, 2):
                            continue
                        sections.append(np.asarray(arr, dtype=np.float64))
                        if lat is None:
                            lat, vert, vert_label, invert_y = get_lat_vert_coord(ds, section_key)
                    except Exception:
                        _maybe_print_missing(cfg, f"WARNING: missing/invalid {path} in {ncfile}")

            if not sections:
                _maybe_print_missing(
                    cfg, f"WARNING: no valid sections for {exp_name} {var_name} at lead {lead_hour}")
                continue

            samples = np.asarray(sections)
            mean_data, ci_low, ci_high = bootstrap_mean_ci(
                samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])

            LatVert[exp_name][var_name] = {
                "data":       mean_data,
                "samples":    samples,
                "ci_low":     ci_low,
                "ci_high":    ci_high,
                "lat":        lat,
                "vert":       vert,
                "vert_label": vert_label,
                "invert_y":   invert_y,
                "lead_hour":  lead_hour,
            }
    return LatVert


def collect_forecast_lat_lon_2d(cfg, stat_name, lead_hour):
    """Like collect_lat_lon_2d, but at one fixed extended-forecast lead hour."""
    base_dir  = cfg["paths"]["base_dir"]
    exp_names = cfg["experiments"]["names"]
    var_names = cfg["variables"]
    boot      = get_bootstrap_settings(cfg)
    cycles    = generate_cycles(cfg["time"]["start"], cfg["time"]["end"],
                                cfg["time"]["interval_hours"])

    LatLon = {}
    for exp_name in exp_names:
        LatLon[exp_name] = {}
        for var_name in var_names:
            maps = []
            lat = lon = None

            for dt in cycles:
                ncfile = get_mpasstat_forecast_file(base_dir, exp_name, dt, lead_hour)
                if not os.path.exists(ncfile):
                    _maybe_print_missing(cfg, f"WARNING: missing file: {ncfile}")
                    continue
                path = f"{var_name}/2D/lat_lon/{stat_name}"
                with nc.Dataset(ncfile, "r") as ds:
                    try:
                        arr = read_nc_path(ds, path)
                        if not _has_expected_ndim(arr, 2):
                            continue
                        maps.append(np.asarray(arr, dtype=np.float64))
                        if lat is None:
                            lat = np.asarray(ds["LatBins"][:])
                            lon = np.asarray(ds["LonBins"][:])
                    except Exception:
                        _maybe_print_missing(cfg, f"WARNING: missing/invalid {path} in {ncfile}")

            if not maps:
                _maybe_print_missing(
                    cfg, f"WARNING: no valid lat-lon maps for {exp_name} {var_name} at lead {lead_hour}")
                continue

            samples = np.asarray(maps)
            mean_data, ci_low, ci_high = bootstrap_mean_ci(
                samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])

            LatLon[exp_name][var_name] = {
                "data":      mean_data,
                "samples":   samples,
                "ci_low":    ci_low,
                "ci_high":   ci_high,
                "lat":       lat,
                "lon":       lon,
                "lead_hour": lead_hour,
            }
    return LatLon


# ---------------------------------------------------------------------------
# Extended-forecast scorecard
# ---------------------------------------------------------------------------
def collect_forecast_scorecard(cfg, stat_name):
    """Collect scorecard rows (variable, optionally @ level) x forecast lead time.

    Controlled by cfg["scorecard"]:
        profile_key : 1-D profile group holding the vertical levels used by
                      level-based rows (default "vert_pres_profile", i.e.
                      pressure levels -- see PressureLevels in the stat files)
        rows        : ordered list of {"variable": <name>, "levels": [...]}.
                      Omit "levels" (or leave it empty) for a surface / 0-D
                      row (e.g. Ps), which is read from
                      "<var>/0D/<region>/<stat>" instead of the profile group.

    Each row is an experiment -> list (ordered exactly like cfg["scorecard"]
    ["rows"], expanded one entry per level) of {"leads", "samples"}, where
    "samples" has shape (n_cycles, n_leads) so plot_forecast_scorecard can
    bootstrap-test each cell against the reference/control experiment
    exactly like collect_forecast_lead_0d does for a single 0-D variable.

    Returns (ScoreCard, row_labels).
    """
    base_dir    = cfg["paths"]["base_dir"]
    exp_names   = cfg["experiments"]["names"]
    region_name = cfg.get("region", "global")
    cycles      = generate_cycles(cfg["time"]["start"], cfg["time"]["end"],
                                  cfg["time"]["interval_hours"])
    leads       = generate_forecast_leads(cfg)

    sc_cfg      = cfg.get("scorecard", {})
    # Vertical-profile NetCDF groups are written per region (e.g.
    # "vert_pres_profile_Global"); apply the same suffixing _profile_tasks
    # uses elsewhere so a plain "vert_pres_profile" in cfg["scorecard"]
    # resolves to the group that actually exists in the stat files.
    profile_key = apply_region_suffix(
        sc_cfg.get("profile_key", "vert_pres_profile"), cfg)
    row_specs   = sc_cfg.get("rows", [])

    ScoreCard  = {exp_name: [] for exp_name in exp_names}
    row_labels = []

    for row_spec in row_specs:
        var_name  = row_spec["variable"]
        raw_levels = row_spec.get("levels", None)
        is_surface = not raw_levels
        levels = [None] if is_surface else list(raw_levels)

        for exp_name in exp_names:
            rows_by_level = {lvl: [] for lvl in levels}

            for dt in cycles:
                cols_by_level = {lvl: [] for lvl in levels}
                for lead in leads:
                    ncfile = get_mpasstat_forecast_file(base_dir, exp_name, dt, lead)
                    if not os.path.exists(ncfile):
                        _maybe_print_missing(cfg, f"WARNING: missing file: {ncfile}")
                        for lvl in levels:
                            cols_by_level[lvl].append(np.nan)
                        continue

                    with nc.Dataset(ncfile, "r") as ds:
                        if is_surface:
                            path = f"{var_name}/0D/{region_name}/{stat_name}"
                            try:
                                cols_by_level[None].append(float(read_nc_path(ds, path)))
                            except Exception:
                                _maybe_print_missing(cfg, f"WARNING: missing {path} in {ncfile}")
                                cols_by_level[None].append(np.nan)
                        else:
                            path = f"{var_name}/1D/{profile_key}/{stat_name}"
                            try:
                                arr = read_nc_path(ds, path)
                                if not _has_expected_ndim(arr, 1):
                                    raise ValueError(f"unexpected shape for {path}")
                                arr = np.asarray(arr, dtype=np.float64)
                                coord, _, _ = get_profile_coord(ds, profile_key)
                                coord = np.asarray(coord, dtype=np.float64)
                                for lvl in levels:
                                    idx = int(np.nanargmin(np.abs(coord - float(lvl))))
                                    cols_by_level[lvl].append(float(arr[idx]))
                            except Exception:
                                _maybe_print_missing(cfg, f"WARNING: missing/invalid {path} in {ncfile}")
                                for lvl in levels:
                                    cols_by_level[lvl].append(np.nan)

                for lvl in levels:
                    row = cols_by_level[lvl]
                    if any(np.isfinite(v) for v in row):
                        rows_by_level[lvl].append(row)

            for lvl in levels:
                samples = (np.asarray(rows_by_level[lvl], dtype=np.float64)
                          if rows_by_level[lvl] else np.zeros((0, len(leads))))
                ScoreCard[exp_name].append({
                    "leads":   np.asarray(leads, dtype=np.float64),
                    "samples": samples,
                })

        for lvl in levels:
            row_labels.append(var_name if lvl is None else f"{var_name} {int(lvl)} hPa")

    return ScoreCard, row_labels
