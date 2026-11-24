"""Readers for the observation-space graphics stage.

Each collect_obs_* function pulls one (experiment, OMB|OMA|OMF) worth of
statistics back out of the diagnosis netCDF files written by
mpas_stat.obs.diag, in the dict shape the matching
mpas_stat.graphics plot_* function expects.
"""
import os

import netCDF4 as nc
import numpy as np

from ..core.pooling import (
    aggregate,
    missing_components,
    pool_from_rows,
    pooled_value,
    read_components,
    stat_kind,
)
from ..core.timeutils import datetime_to_cycle
from .common import (
    ERROR_DIAG_KEYS,
    STAT_KEYS,
    SPREAD_SKILL_KEYS,
    STAT_KEYS_2D,
    nc_group_name,
    obs_diag_file,
    obs_diag_forecast_file,
)

# ============================================================================
# Readers -- pull one (experiment, OMB|OMA) worth of data back out of the
# diagnosis netCDF files written by src/mpas_stat/obs/diag.py.
# ============================================================================

def _subgroup(grp, path):
    """grp.groups[path[0]].groups[path[1]]..., or None if any level is missing."""
    for name in path:
        if name not in grp.groups:
            return None
        grp = grp.groups[name]
    return grp


def _gather_stat_arrays(ncfiles, diagtype, display_vars, raw_vars, group_path,
                        stat_keys, coord_vars, coord_trigger=None):
    """Scan diagnosis files and gather one array per (statistic, variable, file).

    For every existing file in `ncfiles` this reads, from
    <diagtype>/<var>/<group_path...>/<stat>, each statistic in `stat_keys`
    for each display variable, and the coordinate variables `coord_vars`
    from the first file that has all of `coord_trigger` (default:
    `coord_vars`).

    Returns ``(coords, per_stat_per_var, hits)``: coords is a list of float64
    arrays (or None if no file had them), per_stat_per_var[stat][var] is the
    list of gathered float64 arrays in file order, and hits lists the indices
    into `ncfiles` of files that contributed at least one array.
    """
    per_stat_per_var = {stat_disp: {v: [] for v in display_vars} for stat_disp, _ in stat_keys}
    coords = None
    trigger = coord_vars if coord_trigger is None else coord_trigger
    hits = []

    for ifile, ncfile in enumerate(ncfiles):
        if not os.path.exists(ncfile):
            continue
        with nc.Dataset(ncfile, "r") as ds:
            if coords is None and all(c in ds.variables for c in trigger):
                coords = [np.asarray(ds.variables[c][:], dtype=np.float64) for c in coord_vars]
            if diagtype not in ds.groups:
                continue
            diag_grp = ds.groups[diagtype]
            got_any = False
            for var, raw in zip(display_vars, raw_vars):
                stat_grp = _subgroup(diag_grp, (nc_group_name(raw),) + tuple(group_path))
                if stat_grp is None:
                    continue
                for stat_disp, _ in stat_keys:
                    if stat_disp in stat_grp.variables:
                        arr = np.asarray(stat_grp.variables[stat_disp][...], dtype=np.float64)
                        per_stat_per_var[stat_disp][var].append(arr)
                        got_any = True
            if got_any:
                hits.append(ifile)
    return coords, per_stat_per_var, hits


def _obs_pool(per_stat_per_var, stat_disp, var, transpose=None):
    """Pooling components (see mpas_stat.core.pooling) for one statistic of
    one variable, from the per-file arrays _gather_stat_arrays() gathered for
    ALL statistics: weights from Count, means from Bias, mean squares from
    RMS (or the spread itself). None if the companions are unavailable or not
    aligned file-by-file."""
    kind = stat_kind(stat_disp)
    if kind == "plain":
        return None
    def stack(name):
        arrs = per_stat_per_var.get(name, {}).get(var) or []
        return np.stack(arrs, axis=0) if arrs else None
    value = stack(stat_disp)
    count = stack("Count")
    if value is None or count is None or count.shape != value.shape:
        return None
    w = np.where(np.isfinite(value) & np.isfinite(count), count, 0.0)
    nan = np.full(value.shape, np.nan)
    if kind == "mean":
        m1, m2 = value, nan
    elif kind == "rms":
        m1, m2 = nan, value * value
    else:  # std
        bias, rms = stack("Bias"), stack("RMS")
        if bias is None or rms is None or bias.shape != value.shape or rms.shape != value.shape:
            return None
        m1, m2 = bias, rms * rms
    pool = {"kind": kind, "w": w, "m1": m1, "m2": m2}
    if transpose is not None:
        pool = {k: (v.transpose(transpose) if k != "kind" else v) for k, v in pool.items()}
    return pool


def collect_obs_stats_0d(diag_base_dir, exp_name, diagtype, diag_obs, cycles,
                        display_vars, raw_vars, domains):
    """Stats[cycle][display_var]['0D'][domain][stat] for one (exp, diagtype),
    read from disk -- the same shape mpas_stat.graphics.functions.plot_time_series_0d
    expects for one entry of its top-level `Stats` dict."""
    out = {}
    for cycle_time in cycles:
        ncfile = obs_diag_file(diag_base_dir, exp_name, diag_obs, cycle_time)
        if not os.path.exists(ncfile):
            print(f"WARNING: missing file: {ncfile}")
            continue
        cycle = datetime_to_cycle(cycle_time)
        with nc.Dataset(ncfile, "r") as ds:
            if diagtype not in ds.groups:
                continue
            diag_grp = ds.groups[diagtype]
            cycle_entry = {}
            for var, raw in zip(display_vars, raw_vars):
                gname = nc_group_name(raw)
                if gname not in diag_grp.groups:
                    continue
                var_grp = diag_grp.groups[gname]
                if "0D" not in var_grp.groups:
                    continue
                d0 = var_grp.groups["0D"]
                var_entry = {}
                for domain in domains:
                    if domain not in d0.groups:
                        continue
                    region_grp = d0.groups[domain]
                    stat_vals = {}
                    for stat_disp, _ in STAT_KEYS:
                        if stat_disp in region_grp.variables:
                            val = float(region_grp.variables[stat_disp][...])
                            if np.isfinite(val):
                                stat_vals[stat_disp] = val
                    if stat_vals:
                        var_entry[domain] = stat_vals
                if var_entry:
                    cycle_entry[var] = {"0D": var_entry}
            if cycle_entry:
                out[cycle] = cycle_entry
    return out


def collect_obs_profiles_1d(diag_base_dir, exp_name, diagtype, diag_obs, cycles,
                           display_vars, raw_vars, profile_key, coord_label, invert_y,
                           time_aggregation="mean"):
    """{stat_disp: {display_var: entry}} time-averaged profiles for one
    (exp, diagtype), read from disk -- entry shape matches what
    mpas_stat.graphics.functions.plot_profiles_1d expects."""
    ncfiles = [obs_diag_file(diag_base_dir, exp_name, diag_obs, c) for c in cycles]
    coords, per_stat_per_var, _ = _gather_stat_arrays(
        ncfiles, diagtype, display_vars, raw_vars, ("1D", profile_key), STAT_KEYS, ["Levels"])

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS}
    if coords is None:
        return result
    coord, = coords

    for stat_disp, _ in STAT_KEYS:
        for var in display_vars:
            profiles = per_stat_per_var[stat_disp][var]
            if not profiles:
                continue
            pool = (_obs_pool(per_stat_per_var, stat_disp, var)
                    if time_aggregation == "pooled" else None)
            mean_profile = (pooled_value(pool) if pool is not None
                            else np.nanmean(np.stack(profiles, axis=0), axis=0))
            if not np.any(np.isfinite(mean_profile)):
                continue
            result[stat_disp][var] = {
                "mean_profile": mean_profile,
                "ci_low": None,
                "ci_high": None,
                "coord": coord,
                "coord_label": coord_label,
                "invert_y": invert_y,
            }
    return result


def collect_obs_lat_lon_2d(diag_base_dir, exp_name, diagtype, diag_obs, cycles,
                          display_vars, raw_vars, time_aggregation="mean"):
    """{stat_disp: {display_var: entry}} time-averaged lat-lon maps for one
    (exp, diagtype), read from disk -- entry shape matches what
    mpas_stat.graphics.functions.plot_lat_lon_2d expects."""
    ncfiles = [obs_diag_file(diag_base_dir, exp_name, diag_obs, c) for c in cycles]
    coords, per_stat_per_var, _ = _gather_stat_arrays(
        ncfiles, diagtype, display_vars, raw_vars, ("2D", "lat_lon"), STAT_KEYS_2D,
        ["LatBins2D", "LonBins2D"], coord_trigger=["LatBins2D"])

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS_2D}
    if coords is None:
        return result
    lat, lon = coords

    for stat_disp, _ in STAT_KEYS_2D:
        for var in display_vars:
            maps = per_stat_per_var[stat_disp][var]
            if not maps:
                continue
            samples = np.stack(maps, axis=0)
            pool = (_obs_pool(per_stat_per_var, stat_disp, var)
                    if time_aggregation == "pooled" else None)
            data = pooled_value(pool) if pool is not None else np.nanmean(samples, axis=0)
            if not np.any(np.isfinite(data)):
                continue
            result[stat_disp][var] = {
                "data": data,
                # Required by mpas_stat.graphics.functions._draw_2d_panels
                # even though relative_reduction_statistics is empty below
                # (so it is never actually used for a % difference).
                "samples": samples,
                "pool": pool,
                "lat": lat,
                "lon": lon,
            }
    return result


def collect_obs_time_level_2d(diag_base_dir, exp_name, diagtype, diag_obs, cycles,
                             display_vars, raw_vars, profile_key, coord_label, invert_y):
    """{stat_disp: {display_var: entry}} cycle-resolved (level x cycle)
    Hovmoller data for one (exp, diagtype), read from disk -- entry shape
    matches what mpas_stat.graphics.functions.plot_time_level_2d expects.

    Reuses the SAME per-cycle 1D profile groups collect_obs_profiles_1d
    reads (vert_pres_profile_<domain> / vert_hgt_profile_<domain> /
    lat_bin_profile) -- no new statistics-stage computation, just kept
    cycle-resolved (stacked, not time-averaged) instead of collapsed to one
    mean profile. Mirrors mpas_stat.graphics.io.collect_time_level_2d
    exactly (each cycle's profile is a single value, so no bootstrap CI is
    computed here -- plot_time_level_2d draws it as-is)."""
    ncfiles = [obs_diag_file(diag_base_dir, exp_name, diag_obs, c) for c in cycles]
    coords, per_stat_per_var, hits = _gather_stat_arrays(
        ncfiles, diagtype, display_vars, raw_vars, ("1D", profile_key), STAT_KEYS, ["Levels"])
    cycle_labels = [datetime_to_cycle(cycles[i]) for i in hits]

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS}
    if coords is None or not cycle_labels:
        return result
    coord, = coords

    for stat_disp, _ in STAT_KEYS:
        for var in display_vars:
            profiles = per_stat_per_var[stat_disp][var]
            if len(profiles) != len(cycle_labels):
                # A variable missing from some cycles can't form a clean
                # (n_cycles, n_levels) matrix -- skip rather than
                # misalign cycles and levels.
                continue
            data = np.stack(profiles, axis=0)
            if not np.any(np.isfinite(data)):
                continue
            result[stat_disp][var] = {
                "data":        data,
                "samples":     data,
                "times":       cycle_labels,
                "coord":       coord,
                "coord_label": coord_label,
                "invert_y":    invert_y,
            }
    return result


def collect_obs_lat_vert_2d(diag_base_dir, exp_name, diagtype, diag_obs, cycles,
                           display_vars, raw_vars, coord_label, invert_y, boot,
                           time_aggregation="mean"):
    """{stat_disp: {display_var: entry}} time-averaged (level x latitude)
    cross-sections for one (exp, diagtype), read from disk -- entry shape
    matches what mpas_stat.graphics.functions.plot_lat_vert_2d expects
    (x_key="lat", y_key="vert", y_label_key="vert_label",
    invert_y_key="invert_y"). Bootstrap-averaged across cycles exactly like
    mpas_stat.graphics.io.collect_lat_vert_2d does for the model-space
    lat_lev/lat_hgt/lat_pres product.

    Not meaningful for radiances (bin_method LATITUDE, no 2D/lat_vert group
    written) -- callers must skip that case; see
    mpas_stat.obs.diag.compute_stats_2d_lat_vert()."""
    ncfiles = [obs_diag_file(diag_base_dir, exp_name, diag_obs, c) for c in cycles]
    coords, per_stat_per_var, _ = _gather_stat_arrays(
        ncfiles, diagtype, display_vars, raw_vars, ("2D", "lat_vert"), STAT_KEYS_2D,
        ["LatBins2D", "Levels"])

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS_2D}
    if coords is None:
        return result
    lat, levels = coords

    for stat_disp, _ in STAT_KEYS_2D:
        for var in display_vars:
            sections = per_stat_per_var[stat_disp][var]
            if not sections:
                continue
            # On disk (2D/lat_vert/<stat>), arrays are (nLevels, nLatBins2D)
            # -- level-major, matching compute_stats_2d_lat_vert() and the
            # 1D vertical profiles' own level-first convention.mpas_stats_
            # graphic_functions._draw_2d_panels (x_key="lat", y_key="vert",
            # transpose_data=True) instead expects (nlat, nlev) -- it
            # transposes internally before pcolormesh(x=lat, y=vert, ...).
            # Swap axes here, once, at read time.
            samples = np.stack(sections, axis=0).transpose(0, 2, 1)
            pool = (_obs_pool(per_stat_per_var, stat_disp, var, transpose=(0, 2, 1))
                    if time_aggregation == "pooled" else None)
            mean_data, ci_low, ci_high = aggregate(samples, pool, boot)
            if not np.any(np.isfinite(mean_data)):
                continue
            result[stat_disp][var] = {
                "data":       mean_data,
                "samples":    samples,
                "pool":    pool,
                "ci_low":     ci_low,
                "ci_high":    ci_high,
                "lat":        lat,
                "vert":       levels,
                "vert_label": coord_label,
                "invert_y":   invert_y,
            }
    return result


def collect_obs_forecast_lead_0d(diag_base_dir, exp_name, diag_obs, cycles, leads,
                                display_vars, raw_vars, domain, stat_disp, boot,
                                time_aggregation="mean"):
    """{display_var: {leads, data, samples, ci_low, ci_high}} -- domain-
    averaged OMF statistic vs. forecast lead time, read from the per-(init
    cycle, lead hour) obs-space forecast diagnosis files
    (obs_diag_forecast_file()) and bootstrap-aggregated over initialization
    cycles at each lead, exactly like
    mpas_stat.graphics.io.collect_forecast_lead_0d does for the model-space
    ExtendedFC files -- so the result can be fed straight into
    mpas_stat.graphics.functions.plot_forecast_lead_0d unmodified."""
    out = {}
    for var, raw in zip(display_vars, raw_vars):
        gname = nc_group_name(raw)
        samples = []
        comp_rows = []
        for cycle_time in cycles:
            row = []
            crow = []
            got_any = False
            for lead in leads:
                ncfile = obs_diag_forecast_file(diag_base_dir, exp_name, diag_obs, cycle_time, lead)
                val = np.nan
                comps = missing_components()
                if os.path.exists(ncfile):
                    with nc.Dataset(ncfile, "r") as ds:
                        try:
                            grp = ds.groups["omf"].groups[gname].groups["0D"].groups[domain]
                            comps = read_components(grp, stat_disp)
                            val = float(comps[0])
                            got_any = True
                        except Exception:
                            comps = missing_components()
                row.append(val)
                crow.append(comps)
            if got_any:
                samples.append(row)
                comp_rows.append(crow)

        if not samples:
            continue

        samples = np.asarray(samples, dtype=np.float64)
        pool = (pool_from_rows(stat_disp, comp_rows)
                if time_aggregation == "pooled" and stat_kind(stat_disp) != "plain" else None)
        mean_data, ci_low, ci_high = aggregate(samples, pool, boot)

        out[var] = {
            "data": mean_data,
            "mean_profile": mean_data,
            "samples": samples,
            "pool": pool,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "leads": np.asarray(leads, dtype=np.float64),
        }
    return out


# ----------------------------------------------------------------------------
# OMF "extended-forecast skill" readers, beyond forecast_lead_0d above --
# the obs-space counterparts of mpas_stat.graphics.io's
# collect_forecast_profiles_1d / collect_forecast_lat_lon_2d /
# collect_forecast_lat_vert_2d / collect_forecast_time_level_2d, reading the
# same OMF diagnosis files (obs_diag_forecast_file(), "omf" group) as
# collect_obs_forecast_lead_0d rather than model-space mpasstat forecast
# files. Each is bootstrap-aggregated over initialization cycles, exactly
# like their model-space counterparts (unlike the plain time-mean OMB/OMA
# 1D/2D collectors above, which intentionally skip bootstrap CI -- see
# MPAS-STAT/docs/USER_GUIDE.md's "Known limitations").
# ----------------------------------------------------------------------------

def collect_obs_forecast_profiles_1d(diag_base_dir, exp_name, diag_obs, cycles, lead,
                                     display_vars, raw_vars, profile_key, coord_label,
                                     invert_y, boot, time_aggregation="mean"):
    """{stat_disp: {display_var: entry}} at one fixed extended-forecast lead
    hour, bootstrap-aggregated over initialization cycles -- mirrors
    mpas_stat.graphics.io.collect_forecast_profiles_1d; entry shape matches
    what plot_profiles_1d expects (same as collect_obs_profiles_1d's, plus
    a bootstrap ci_low/ci_high and a lead_hour tag)."""
    ncfiles = [obs_diag_forecast_file(diag_base_dir, exp_name, diag_obs, c, lead) for c in cycles]
    coords, per_stat_per_var, _ = _gather_stat_arrays(
        ncfiles, "omf", display_vars, raw_vars, ("1D", profile_key), STAT_KEYS, ["Levels"])

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS}
    if coords is None:
        return result
    coord, = coords

    for stat_disp, _ in STAT_KEYS:
        for var in display_vars:
            profiles = per_stat_per_var[stat_disp][var]
            if not profiles:
                continue
            samples = np.stack(profiles, axis=0)
            if not np.any(np.isfinite(samples)):
                continue
            pool = (_obs_pool(per_stat_per_var, stat_disp, var, transpose=None)
                    if time_aggregation == "pooled" else None)
            mean_profile, ci_low, ci_high = aggregate(samples, pool, boot)
            result[stat_disp][var] = {
                "mean_profile": mean_profile,
                "data":         mean_profile,
                "samples":      samples,
                "pool":      pool,
                "ci_low":       ci_low,
                "ci_high":      ci_high,
                "coord":        coord,
                "coord_label":  coord_label,
                "invert_y":     invert_y,
                "lead_hour":    lead,
            }
    return result


def collect_obs_forecast_lat_lon_2d(diag_base_dir, exp_name, diag_obs, cycles, lead,
                                    display_vars, raw_vars, boot, time_aggregation="mean"):
    """{stat_disp: {display_var: entry}} lat-lon map at one fixed
    extended-forecast lead hour, bootstrap-aggregated over initialization
    cycles -- mirrors mpas_stat.graphics.io.collect_forecast_lat_lon_2d;
    entry shape matches what plot_lat_lon_2d expects (same as
    collect_obs_lat_lon_2d's, plus bootstrap ci_low/ci_high and a
    lead_hour tag)."""
    ncfiles = [obs_diag_forecast_file(diag_base_dir, exp_name, diag_obs, c, lead) for c in cycles]
    coords, per_stat_per_var, _ = _gather_stat_arrays(
        ncfiles, "omf", display_vars, raw_vars, ("2D", "lat_lon"), STAT_KEYS_2D,
        ["LatBins2D", "LonBins2D"], coord_trigger=["LatBins2D"])

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS_2D}
    if coords is None:
        return result
    lat, lon = coords

    for stat_disp, _ in STAT_KEYS_2D:
        for var in display_vars:
            maps = per_stat_per_var[stat_disp][var]
            if not maps:
                continue
            samples = np.stack(maps, axis=0)
            if not np.any(np.isfinite(samples)):
                continue
            pool = (_obs_pool(per_stat_per_var, stat_disp, var, transpose=None)
                    if time_aggregation == "pooled" else None)
            mean_data, ci_low, ci_high = aggregate(samples, pool, boot)
            result[stat_disp][var] = {
                "data":      mean_data,
                "samples":   samples,
                "pool":   pool,
                "ci_low":    ci_low,
                "ci_high":   ci_high,
                "lat":       lat,
                "lon":       lon,
                "lead_hour": lead,
            }
    return result


def collect_obs_forecast_lat_vert_2d(diag_base_dir, exp_name, diag_obs, cycles, lead,
                                     display_vars, raw_vars, coord_label, invert_y, boot,
                                     time_aggregation="mean"):
    """{stat_disp: {display_var: entry}} (level x latitude) cross-section at
    one fixed extended-forecast lead hour, bootstrap-aggregated over
    initialization cycles -- mirrors
    mpas_stat.graphics.io.collect_forecast_lat_vert_2d; entry shape matches
    what plot_lat_vert_2d expects (same as collect_obs_lat_vert_2d's, plus
    a lead_hour tag). Not meaningful for radiances (bin_method LATITUDE, no
    2D/lat_vert group written) -- callers must skip that case."""
    ncfiles = [obs_diag_forecast_file(diag_base_dir, exp_name, diag_obs, c, lead) for c in cycles]
    coords, per_stat_per_var, _ = _gather_stat_arrays(
        ncfiles, "omf", display_vars, raw_vars, ("2D", "lat_vert"), STAT_KEYS_2D,
        ["LatBins2D", "Levels"])

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS_2D}
    if coords is None:
        return result
    lat, levels = coords

    for stat_disp, _ in STAT_KEYS_2D:
        for var in display_vars:
            sections = per_stat_per_var[stat_disp][var]
            if not sections:
                continue
            # On disk (2D/lat_vert/<stat>), arrays are (nLevels, nLatBins2D)
            # -- _draw_2d_panels (x_key="lat", transpose_data=True) needs
            # (nlat, nlev) pre-transpose. Swap axes here, once, at read
            # time -- see collect_obs_lat_vert_2d's identical note.
            samples = np.stack(sections, axis=0).transpose(0, 2, 1)
            if not np.any(np.isfinite(samples)):
                continue
            pool = (_obs_pool(per_stat_per_var, stat_disp, var, transpose=(0, 2, 1))
                    if time_aggregation == "pooled" else None)
            mean_data, ci_low, ci_high = aggregate(samples, pool, boot)
            result[stat_disp][var] = {
                "data":       mean_data,
                "samples":    samples,
                "pool":    pool,
                "ci_low":     ci_low,
                "ci_high":    ci_high,
                "lat":        lat,
                "vert":       levels,
                "vert_label": coord_label,
                "invert_y":   invert_y,
                "lead_hour":  lead,
            }
    return result


def collect_obs_forecast_time_level_2d(diag_base_dir, exp_name, diag_obs, cycles, leads,
                                       display_vars, raw_vars, profile_key, coord_label,
                                       invert_y, stat_disp, boot, time_aggregation="mean"):
    """{display_var: entry} lead-time x level cross-section for ONE
    statistic, spanning every configured forecast lead hour -- mirrors
    mpas_stat.graphics.io.collect_forecast_time_level_2d exactly: each
    initialization cycle contributes one (n_leads, n_levels) block
    (NaN-filled where a lead is missing for that cycle), stacked into an
    (n_cycles, n_leads, n_levels) sample cube and bootstrap-averaged over
    cycles -- so, unlike the OMB/OMA lev-time Hovmoller (plain per-cycle
    values, no bootstrap), this supports the same significance-vs-reference
    testing as the other 2-D comparison products. Feeds
    plot_forecast_time_level_2d directly."""
    out = {}
    for var, raw in zip(display_vars, raw_vars):
        gname = nc_group_name(raw)
        coord = None
        nlev = None
        blocks = []  # one (n_leads, n_levels) block per initialization cycle
        comp_blocks = []

        for cycle_time in cycles:
            row = None
            crow = None
            got_any = False
            for ilead, lead in enumerate(leads):
                ncfile = obs_diag_forecast_file(diag_base_dir, exp_name, diag_obs, cycle_time, lead)
                if not os.path.exists(ncfile):
                    continue
                with nc.Dataset(ncfile, "r") as ds:
                    if coord is None and "Levels" in ds.variables:
                        coord = np.asarray(ds.variables["Levels"][:], dtype=np.float64)
                        nlev = coord.shape[0]
                    if "omf" not in ds.groups or gname not in ds.groups["omf"].groups:
                        continue
                    var_grp = ds.groups["omf"].groups[gname]
                    if "1D" not in var_grp.groups or profile_key not in var_grp.groups["1D"].groups:
                        continue
                    prof_grp = var_grp.groups["1D"].groups[profile_key]
                    if stat_disp not in prof_grp.variables:
                        continue
                    comps = read_components(prof_grp, stat_disp)
                    arr = np.asarray(comps[0], dtype=np.float64)
                    if nlev is None:
                        nlev = arr.shape[0]
                    if row is None:
                        row = np.full((len(leads), nlev), np.nan, dtype=np.float64)
                        crow = [np.full((len(leads), nlev), fill) for fill in (np.nan, 0.0, np.nan, np.nan)]
                    row[ilead, :] = arr
                    for comp, value in zip(crow, comps):
                        comp[ilead, :] = value
                    got_any = True

            if got_any:
                blocks.append(row)
                comp_blocks.append(crow)

        if not blocks or coord is None:
            continue

        samples = np.asarray(blocks, dtype=np.float64)  # (n_cycles, n_leads, n_levels)
        if not np.any(np.isfinite(samples)):
            continue
        pool = None
        if time_aggregation == "pooled" and stat_kind(stat_disp) != "plain":
            pool = {"kind": stat_kind(stat_disp),
                    "w": np.asarray([b[1] for b in comp_blocks]),
                    "m1": np.asarray([b[2] for b in comp_blocks]),
                    "m2": np.asarray([b[3] for b in comp_blocks])}
        mean_data, ci_low, ci_high = aggregate(samples, pool, boot)

        out[var] = {
            "data":        mean_data,
            "samples":     samples,
            "pool":        pool,
            "ci_low":      ci_low,
            "ci_high":     ci_high,
            "leads":       np.asarray(leads, dtype=np.float64),
            "coord":       coord,
            "coord_label": coord_label,
            "invert_y":    invert_y,
        }
    return out


def _error_diag_entries(ncfiles, display_vars, raw_vars, group_path, coord_vars,
                        time_aggregation):
    """{var: {name: value}} of the error-consistency quantities in the omb
    group at `group_path` (see collect_obs_error_profiles), plus "Count".
    Returns (coords, entries); coords is None when no file had the group's
    coordinates."""
    names = [k for k, _ in ERROR_DIAG_KEYS] + ["RMS", "TotalSpread", "Count", "Bias"]
    keys = [(n, n) for n in names]
    coords, per_stat_per_var, _ = _gather_stat_arrays(
        ncfiles, "omb", display_vars, raw_vars, group_path, keys, coord_vars)

    def value(name, var):
        arrs = per_stat_per_var[name][var]
        if not arrs:
            return None
        if name == "Count":
            return np.nansum(np.stack(arrs, axis=0), axis=0)
        pool = _obs_pool(per_stat_per_var, name, var) if time_aggregation == "pooled" else None
        if pool is not None:
            return pooled_value(pool)
        return np.nanmean(np.stack(arrs, axis=0), axis=0)

    out = {}
    for var in display_vars:
        entry = {}
        for name, _ in ERROR_DIAG_KEYS:
            v = value(name, var)
            if v is not None and np.any(np.isfinite(v)):
                entry[name] = v
        rms, tot = value("RMS", var), value("TotalSpread", var)
        if rms is not None and tot is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                cr = np.where(rms > 0, tot / rms, np.nan)
            if np.any(np.isfinite(cr)):
                entry["ConsistencyRatio"] = cr
        if entry:
            count = value("Count", var)
            if count is not None:
                entry["Count"] = count
            out[var] = entry
    return coords, out


def collect_obs_error_profiles(diag_base_dir, exp_name, diag_obs, cycles, display_vars,
                               raw_vars, profile_key, coord_label, invert_y,
                               time_aggregation="pooled"):
    """Observation-error consistency profiles of one "da" experiment from its
    OMB statistics: ObsErrorAssumed, ObsErrorDesroziers, BkgErrorDesroziers
    (when the statistics stage wrote them), Count and, when an ensemble
    spread was written, ConsistencyRatio = TotalSpread / RMS. Each is
    aggregated over cycles like the other statistics (pooled by Count by
    default; Count is summed)."""
    ncfiles = [obs_diag_file(diag_base_dir, exp_name, diag_obs, c) for c in cycles]
    coords, entries = _error_diag_entries(
        ncfiles, display_vars, raw_vars, ("1D", profile_key), ["Levels"], time_aggregation)
    if coords is None:
        return {}
    coord, = coords
    return {var: dict(entry, coord=coord, coord_label=coord_label, invert_y=invert_y)
            for var, entry in entries.items()}


def collect_obs_error_totals(diag_base_dir, exp_name, diag_obs, cycles, display_vars,
                             raw_vars, domain, time_aggregation="pooled"):
    """Same quantities as collect_obs_error_profiles, for the whole domain
    (all levels together, the 0D group): {var: {name: scalar}}."""
    ncfiles = [obs_diag_file(diag_base_dir, exp_name, diag_obs, c) for c in cycles]
    _, entries = _error_diag_entries(
        ncfiles, display_vars, raw_vars, ("0D", domain), [], time_aggregation)
    return {var: {k: float(np.asarray(v).reshape(-1)[0]) for k, v in e.items()}
            for var, e in entries.items()}


# ----------------------------------------------------------------------------
# Ensemble reliability (rank histograms, spread-skill) -- OMB of experiments
# with n_member > 1, written by the statistics stage when
# statistics.ensemble_diagnostics is on. Counts and bin moments are summed /
# count-weighted over cycles.
# ----------------------------------------------------------------------------
def _read_group_vars(ncfile, path, names):
    """{name: array} for the variables of netCDF group `path` that exist."""
    out = {}
    if not os.path.exists(ncfile):
        return out
    with nc.Dataset(ncfile, "r") as ds:
        grp = _subgroup(ds, path)
        if grp is None:
            return out
        for n in names:
            if n in grp.variables:
                out[n] = np.asarray(grp.variables[n][...], dtype=np.float64)
        if "SpreadBinEdges" in ds.variables:
            out["_edges"] = np.asarray(ds.variables["SpreadBinEdges"][:], dtype=np.float64)
        if "Levels" in ds.variables:
            out["_levels"] = np.asarray(ds.variables["Levels"][:], dtype=np.float64)
    return out


def collect_obs_rank_histograms(diag_base_dir, exp_name, diag_obs, cycles, display_vars,
                                raw_vars, domain, profile_key=None):
    """{var: {"hist": counts (n_rank,), "hist_levels": (n_lev, n_rank) or
    None, "coord": levels or None}} summed over cycles."""
    out = {}
    files = [obs_diag_file(diag_base_dir, exp_name, diag_obs, c) for c in cycles]
    for var, raw in zip(display_vars, raw_vars):
        g = nc_group_name(raw)
        total, levels_total, coord = None, None, None
        for f in files:
            d0 = _read_group_vars(f, ("omb", g, "0D", domain), ["RankHistogram"])
            if "RankHistogram" in d0:
                h = np.nan_to_num(d0["RankHistogram"])
                total = h if total is None else total + h
            if profile_key:
                d1 = _read_group_vars(f, ("omb", g, "1D", profile_key), ["RankHistogram"])
                if "RankHistogram" in d1:
                    h = np.nan_to_num(d1["RankHistogram"])
                    levels_total = h if levels_total is None else levels_total + h
                    coord = d1.get("_levels", coord)
        if total is not None and total.sum() > 0:
            out[var] = {"hist": total, "hist_levels": levels_total, "coord": coord}
    return out


def merge_spread_bins(count, moments, n_groups=10, min_count=50):
    """Merge fine spread bins into up to `n_groups` groups of roughly equal
    observation count. `moments` maps name -> per-bin mean (count-weighted
    when merged). Returns (group counts, {name: group means})."""
    count = np.nan_to_num(np.asarray(count, dtype=np.float64))
    total = count.sum()
    if total <= 0:
        return np.zeros(0), {k: np.zeros(0) for k in moments}
    cum = np.cumsum(count)
    group = np.minimum((n_groups * (cum - 0.5 * count) / total).astype(int), n_groups - 1)
    out_c, out_m = [], {k: [] for k in moments}
    for gi in range(n_groups):
        sel = (group == gi) & (count > 0)
        c = count[sel].sum()
        if c < min_count:
            continue
        out_c.append(c)
        for k, m in moments.items():
            mm = np.asarray(m, dtype=np.float64)[sel]
            ok = np.isfinite(mm)
            out_m[k].append(np.sum(count[sel][ok] * mm[ok]) / max(count[sel][ok].sum(), 1.0)
                            if ok.any() else np.nan)
    return np.asarray(out_c), {k: np.asarray(v) for k, v in out_m.items()}


def collect_obs_spread_skill(diag_base_dir, exp_name, diag_obs, cycles, display_vars,
                             raw_vars, domain, n_groups=10, min_count=50):
    """{var: {"count", "total_spread", "spread", "rms", "std"}} -- one value
    per spread group (see merge_spread_bins), pooled over cycles: total
    spread, ensemble spread and RMS / STD of OMB of the observations whose
    total spread falls in each group."""
    names = [k for k, _ in SPREAD_SKILL_KEYS]
    files = [obs_diag_file(diag_base_dir, exp_name, diag_obs, c) for c in cycles]
    out = {}
    for var, raw in zip(display_vars, raw_vars):
        g = nc_group_name(raw)
        cnt, sums = None, {}
        for f in files:
            d = _read_group_vars(f, ("omb", g, "0D", domain), names)
            if "SpreadSkillCount" not in d:
                continue
            c = np.nan_to_num(d["SpreadSkillCount"])
            cnt = c if cnt is None else cnt + c
            for k in names[1:]:
                if k in d:
                    sums[k] = sums.get(k, 0.0) + c * np.nan_to_num(d[k])
        if cnt is None or cnt.sum() <= 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            means = {k: np.where(cnt > 0, s / cnt, np.nan) for k, s in sums.items()}
        gc, gm = merge_spread_bins(cnt, means, n_groups=n_groups, min_count=min_count)
        if gc.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            ms, mean = gm.get("SpreadSkillOMBMS"), gm.get("SpreadSkillOMBMean")
            out[var] = {
                "count": gc,
                "total_spread": np.sqrt(gm["SpreadSkillTotalSpreadMS"]),
                "spread": np.sqrt(gm["SpreadSkillSpreadMS"]) if "SpreadSkillSpreadMS" in gm else None,
                "rms": np.sqrt(ms),
                "std": np.sqrt(np.maximum(ms - mean * mean, 0.0)),
            }
    return out


# ----------------------------------------------------------------------------
# Normalized departures (statistics.normalized_departures): departures
# divided by the assumed observation error ("NormDep*", omb and oma) and,
# for ensembles, by the total spread ("NormDepTotal*", omb). Pooled over
# cycles with their own counts; histograms summed.
# ----------------------------------------------------------------------------
def _pool_rms_mean(rms_list, mean_list, count_list):
    """Count-weighted pooled (rms, mean, count) over cycles (axis 0)."""
    c = np.nan_to_num(np.stack(count_list, axis=0))
    r = np.stack(rms_list, axis=0)
    m = np.stack(mean_list, axis=0)
    okr = np.isfinite(r) & (c > 0)
    okm = np.isfinite(m) & (c > 0)
    wr = np.where(okr, c, 0.0).sum(axis=0)
    wm = np.where(okm, c, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(np.where(okr, c * r * r, 0.0).sum(axis=0) / wr)
        mean = np.where(okm, c * m, 0.0).sum(axis=0) / wm
    rms = np.where(wr > 0, rms, np.nan)
    mean = np.where(wm > 0, mean, np.nan)
    return rms, mean, c.sum(axis=0)


def collect_obs_normalized_departures(diag_base_dir, exp_name, diag_obs, cycles, display_vars,
                                      raw_vars, domain, profile_key=None):
    """{var: {kind: {"rms", "mean", "count", "hist", "rms_profile",
    "mean_profile", "count_profile"}}, plus "coord" and "edges"} with kind
    in "omb" (O-B / sigma_o), "oma" (O-A / sigma_o) and "omb_total"
    ((O-B) / total spread, ensembles), pooled over cycles."""
    files = [obs_diag_file(diag_base_dir, exp_name, diag_obs, c) for c in cycles]
    kinds = (("omb", "omb", "NormDep"), ("oma", "oma", "NormDep"),
             ("omb_total", "omb", "NormDepTotal"))
    out = {}
    for var, raw in zip(display_vars, raw_vars):
        g = nc_group_name(raw)
        entry, coord, edges = {}, None, None
        for kind, diagtype, base in kinds:
            names = [f"{base}RMS", f"{base}Mean", f"{base}Count", f"{base}Hist"]
            acc = {"r0": [], "m0": [], "c0": [], "r1": [], "m1": [], "c1": [], "h": None}
            for f in files:
                d0 = _read_group_vars(f, (diagtype, g, "0D", domain), names)
                if f"{base}RMS" in d0 and f"{base}Count" in d0:
                    acc["r0"].append(np.atleast_1d(d0[f"{base}RMS"]))
                    acc["m0"].append(np.atleast_1d(d0.get(f"{base}Mean", np.nan)))
                    acc["c0"].append(np.atleast_1d(d0[f"{base}Count"]))
                    if f"{base}Hist" in d0:
                        h = np.nan_to_num(d0[f"{base}Hist"])
                        acc["h"] = h if acc["h"] is None else acc["h"] + h
                if profile_key:
                    d1 = _read_group_vars(f, (diagtype, g, "1D", profile_key), names)
                    if f"{base}RMS" in d1 and f"{base}Count" in d1:
                        acc["r1"].append(d1[f"{base}RMS"])
                        acc["m1"].append(d1.get(f"{base}Mean", np.full_like(d1[f"{base}RMS"], np.nan)))
                        acc["c1"].append(d1[f"{base}Count"])
                        coord = d1.get("_levels", coord)
            if not acc["r0"]:
                continue
            rms, mean, count = _pool_rms_mean(acc["r0"], acc["m0"], acc["c0"])
            if not (count[0] > 0):
                continue
            e = {"rms": float(rms[0]), "mean": float(mean[0]), "count": float(count[0]),
                 "hist": acc["h"]}
            if acc["r1"]:
                e["rms_profile"], e["mean_profile"], e["count_profile"] = _pool_rms_mean(
                    acc["r1"], acc["m1"], acc["c1"])
            entry[kind] = e
            if edges is None and files:
                for f in files:
                    if os.path.exists(f):
                        with nc.Dataset(f, "r") as ds:
                            if "NormDepBinEdges" in ds.variables:
                                edges = np.asarray(ds.variables["NormDepBinEdges"][:])
                                break
        if entry:
            entry["coord"] = coord
            entry["edges"] = edges
            out[var] = entry
    return out


# ----------------------------------------------------------------------------
# Radiance channel x channel covariances (statistics.channel_correlations):
# count-weighted pooling of the per-cycle mean cross-products.
# ----------------------------------------------------------------------------
def collect_channel_covariance(diag_base_dir, exp_name, diag_obs, cycles, domain):
    """{"channels", "R", "B", "D", "count", "n_cycles"} pooled over cycles,
    or {} when no file has the channel_covariance group."""
    from .channel_cov import CHANNEL_COV_KEYS
    sums, count, channels, ncyc = {}, None, None, 0
    for c in cycles:
        f = obs_diag_file(diag_base_dir, exp_name, diag_obs, c)
        if not os.path.exists(f):
            continue
        with nc.Dataset(f, "r") as ds:
            grp = _subgroup(ds, ("channel_covariance", domain))
            if grp is None:
                continue
            ch = np.asarray(ds["channel_covariance"].variables["Channels"][:])
            if channels is None:
                channels = ch
            elif not np.array_equal(ch, channels):
                continue
            n = np.nan_to_num(np.asarray(grp.variables["PairCount"][:], dtype=np.float64))
            count = n if count is None else count + n
            for name, prefix in CHANNEL_COV_KEYS:
                if prefix == "chcov_n":
                    continue
                m = np.asarray(grp.variables[name][:], dtype=np.float64)
                sums[name] = sums.get(name, 0.0) + np.where(np.isfinite(m), n * m, 0.0)
            ncyc += 1
    if count is None or not np.any(count > 0):
        return {}
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = {k: np.where(count > 0, v / count, np.nan) for k, v in sums.items()}
    return {"channels": channels, "R": mean["ObsErrorCovDesroziers"],
            "B": mean["BkgErrorCovDesroziers"], "D": mean["OMBCov"], "count": count,
            "n_cycles": ncyc}
