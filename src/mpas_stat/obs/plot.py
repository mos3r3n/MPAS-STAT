#!/usr/bin/env python3
"""
Observation-space verification -- GRAPHICS stage.

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
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import netCDF4 as nc

from ..graphics.functions import (
    plot_time_series_0d,
    plot_time_series_reduction_0d,
    plot_profiles_1d,
    plot_profile_reduction_1d,
    plot_lat_lon_2d,
    plot_time_level_2d,
    plot_lat_vert_2d,
    plot_forecast_lead_0d,
    plot_forecast_time_level_2d,
)
from ..graphics.io import (
    datetime_to_cycle, get_bootstrap_settings, bootstrap_mean_ci,
)

from .common import (
    load_config,
    profile_key_for_domain, nc_group_name, STAT_KEYS, STAT_KEYS_2D,
    resolve_domains, resolve_time_settings, generate_cycles,
    resolve_diag_base_dir, obs_diag_file, obs_diag_sensor_has_files,
    generate_forecast_leads, obs_diag_forecast_file,
)
# Sensor metadata (variables/units/labels/bin method + display names) all
# come from the dedicated sensor registry.
from .sensor_config import get_obs_config, get_bin_method, display_var_names

# ============================================================================
# Readers -- pull one (experiment, OMB|OMA) worth of data back out of the
# diagnosis netCDF files written by src/mpas_stat/obs/diag.py.
# ============================================================================

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
                           display_vars, raw_vars, profile_key, coord_label, invert_y):
    """{stat_disp: {display_var: entry}} time-averaged profiles for one
    (exp, diagtype), read from disk -- entry shape matches what
    mpas_stat.graphics.functions.plot_profiles_1d expects."""
    per_stat_per_var = {stat_disp: {v: [] for v in display_vars} for stat_disp, _ in STAT_KEYS}
    coord = None

    for cycle_time in cycles:
        ncfile = obs_diag_file(diag_base_dir, exp_name, diag_obs, cycle_time)
        if not os.path.exists(ncfile):
            continue
        with nc.Dataset(ncfile, "r") as ds:
            if coord is None and "Levels" in ds.variables:
                coord = np.asarray(ds.variables["Levels"][:], dtype=np.float64)
            if diagtype not in ds.groups:
                continue
            diag_grp = ds.groups[diagtype]
            for var, raw in zip(display_vars, raw_vars):
                gname = nc_group_name(raw)
                if gname not in diag_grp.groups:
                    continue
                var_grp = diag_grp.groups[gname]
                if "1D" not in var_grp.groups or profile_key not in var_grp.groups["1D"].groups:
                    continue
                prof_grp = var_grp.groups["1D"].groups[profile_key]
                for stat_disp, _ in STAT_KEYS:
                    if stat_disp in prof_grp.variables:
                        arr = np.asarray(prof_grp.variables[stat_disp][:], dtype=np.float64)
                        per_stat_per_var[stat_disp][var].append(arr)

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS}
    if coord is None:
        return result

    for stat_disp, _ in STAT_KEYS:
        for var in display_vars:
            profiles = per_stat_per_var[stat_disp][var]
            if not profiles:
                continue
            mean_profile = np.nanmean(np.stack(profiles, axis=0), axis=0)
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
                          display_vars, raw_vars):
    """{stat_disp: {display_var: entry}} time-averaged lat-lon maps for one
    (exp, diagtype), read from disk -- entry shape matches what
    mpas_stat.graphics.functions.plot_lat_lon_2d expects."""
    per_stat_per_var = {stat_disp: {v: [] for v in display_vars} for stat_disp, _ in STAT_KEYS_2D}
    lat = lon = None

    for cycle_time in cycles:
        ncfile = obs_diag_file(diag_base_dir, exp_name, diag_obs, cycle_time)
        if not os.path.exists(ncfile):
            continue
        with nc.Dataset(ncfile, "r") as ds:
            if lat is None and "LatBins2D" in ds.variables:
                lat = np.asarray(ds.variables["LatBins2D"][:], dtype=np.float64)
                lon = np.asarray(ds.variables["LonBins2D"][:], dtype=np.float64)
            if diagtype not in ds.groups:
                continue
            diag_grp = ds.groups[diagtype]
            for var, raw in zip(display_vars, raw_vars):
                gname = nc_group_name(raw)
                if gname not in diag_grp.groups:
                    continue
                var_grp = diag_grp.groups[gname]
                if "2D" not in var_grp.groups or "lat_lon" not in var_grp.groups["2D"].groups:
                    continue
                d2 = var_grp.groups["2D"].groups["lat_lon"]
                for stat_disp, _ in STAT_KEYS_2D:
                    if stat_disp in d2.variables:
                        arr = np.asarray(d2.variables[stat_disp][:, :], dtype=np.float64)
                        per_stat_per_var[stat_disp][var].append(arr)

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS_2D}
    if lat is None:
        return result

    for stat_disp, _ in STAT_KEYS_2D:
        for var in display_vars:
            maps = per_stat_per_var[stat_disp][var]
            if not maps:
                continue
            samples = np.stack(maps, axis=0)
            data = np.nanmean(samples, axis=0)
            if not np.any(np.isfinite(data)):
                continue
            result[stat_disp][var] = {
                "data": data,
                # Required by mpas_stat.graphics.functions._draw_2d_panels
                # even though relative_reduction_statistics is empty below
                # (so it is never actually used for a % difference).
                "samples": samples,
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
    per_stat_per_var = {stat_disp: {v: [] for v in display_vars} for stat_disp, _ in STAT_KEYS}
    cycle_labels = []
    coord = None

    for cycle_time in cycles:
        ncfile = obs_diag_file(diag_base_dir, exp_name, diag_obs, cycle_time)
        if not os.path.exists(ncfile):
            continue
        cycle = datetime_to_cycle(cycle_time)
        with nc.Dataset(ncfile, "r") as ds:
            if coord is None and "Levels" in ds.variables:
                coord = np.asarray(ds.variables["Levels"][:], dtype=np.float64)
            if diagtype not in ds.groups:
                continue
            diag_grp = ds.groups[diagtype]
            got_any_this_cycle = False
            for var, raw in zip(display_vars, raw_vars):
                gname = nc_group_name(raw)
                if gname not in diag_grp.groups:
                    continue
                var_grp = diag_grp.groups[gname]
                if "1D" not in var_grp.groups or profile_key not in var_grp.groups["1D"].groups:
                    continue
                prof_grp = var_grp.groups["1D"].groups[profile_key]
                for stat_disp, _ in STAT_KEYS:
                    if stat_disp in prof_grp.variables:
                        arr = np.asarray(prof_grp.variables[stat_disp][:], dtype=np.float64)
                        per_stat_per_var[stat_disp][var].append(arr)
                        got_any_this_cycle = True
            if got_any_this_cycle:
                cycle_labels.append(cycle)

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS}
    if coord is None or not cycle_labels:
        return result

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
                           display_vars, raw_vars, coord_label, invert_y, boot):
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
    per_stat_per_var = {stat_disp: {v: [] for v in display_vars} for stat_disp, _ in STAT_KEYS_2D}
    lat = levels = None

    for cycle_time in cycles:
        ncfile = obs_diag_file(diag_base_dir, exp_name, diag_obs, cycle_time)
        if not os.path.exists(ncfile):
            continue
        with nc.Dataset(ncfile, "r") as ds:
            if lat is None and "LatBins2D" in ds.variables and "Levels" in ds.variables:
                lat = np.asarray(ds.variables["LatBins2D"][:], dtype=np.float64)
                levels = np.asarray(ds.variables["Levels"][:], dtype=np.float64)
            if diagtype not in ds.groups:
                continue
            diag_grp = ds.groups[diagtype]
            for var, raw in zip(display_vars, raw_vars):
                gname = nc_group_name(raw)
                if gname not in diag_grp.groups:
                    continue
                var_grp = diag_grp.groups[gname]
                if "2D" not in var_grp.groups or "lat_vert" not in var_grp.groups["2D"].groups:
                    continue
                d2 = var_grp.groups["2D"].groups["lat_vert"]
                for stat_disp, _ in STAT_KEYS_2D:
                    if stat_disp in d2.variables:
                        arr = np.asarray(d2.variables[stat_disp][:, :], dtype=np.float64)
                        per_stat_per_var[stat_disp][var].append(arr)

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS_2D}
    if lat is None:
        return result

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
            mean_data, ci_low, ci_high = bootstrap_mean_ci(
                samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])
            if not np.any(np.isfinite(mean_data)):
                continue
            result[stat_disp][var] = {
                "data":       mean_data,
                "samples":    samples,
                "ci_low":     ci_low,
                "ci_high":    ci_high,
                "lat":        lat,
                "vert":       levels,
                "vert_label": coord_label,
                "invert_y":   invert_y,
            }
    return result

def collect_obs_forecast_lead_0d(diag_base_dir, exp_name, diag_obs, cycles, leads,
                                display_vars, raw_vars, domain, stat_disp, boot):
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
        for cycle_time in cycles:
            row = []
            got_any = False
            for lead in leads:
                ncfile = obs_diag_forecast_file(diag_base_dir, exp_name, diag_obs, cycle_time, lead)
                val = np.nan
                if os.path.exists(ncfile):
                    with nc.Dataset(ncfile, "r") as ds:
                        try:
                            val = float(
                                ds.groups["omf"].groups[gname].groups["0D"]
                                  .groups[domain].variables[stat_disp][...]
                            )
                            got_any = True
                        except Exception:
                            pass
                row.append(val)
            if got_any:
                samples.append(row)

        if not samples:
            continue

        samples = np.asarray(samples, dtype=np.float64)
        mean_data, ci_low, ci_high = bootstrap_mean_ci(
            samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])

        out[var] = {
            "data": mean_data,
            "mean_profile": mean_data,
            "samples": samples,
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
                                     invert_y, boot):
    """{stat_disp: {display_var: entry}} at one fixed extended-forecast lead
    hour, bootstrap-aggregated over initialization cycles -- mirrors
    mpas_stat.graphics.io.collect_forecast_profiles_1d; entry shape matches
    what plot_profiles_1d expects (same as collect_obs_profiles_1d's, plus
    a bootstrap ci_low/ci_high and a lead_hour tag)."""
    per_stat_per_var = {stat_disp: {v: [] for v in display_vars} for stat_disp, _ in STAT_KEYS}
    coord = None

    for cycle_time in cycles:
        ncfile = obs_diag_forecast_file(diag_base_dir, exp_name, diag_obs, cycle_time, lead)
        if not os.path.exists(ncfile):
            continue
        with nc.Dataset(ncfile, "r") as ds:
            if coord is None and "Levels" in ds.variables:
                coord = np.asarray(ds.variables["Levels"][:], dtype=np.float64)
            if "omf" not in ds.groups:
                continue
            diag_grp = ds.groups["omf"]
            for var, raw in zip(display_vars, raw_vars):
                gname = nc_group_name(raw)
                if gname not in diag_grp.groups:
                    continue
                var_grp = diag_grp.groups[gname]
                if "1D" not in var_grp.groups or profile_key not in var_grp.groups["1D"].groups:
                    continue
                prof_grp = var_grp.groups["1D"].groups[profile_key]
                for stat_disp, _ in STAT_KEYS:
                    if stat_disp in prof_grp.variables:
                        arr = np.asarray(prof_grp.variables[stat_disp][:], dtype=np.float64)
                        per_stat_per_var[stat_disp][var].append(arr)

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS}
    if coord is None:
        return result

    for stat_disp, _ in STAT_KEYS:
        for var in display_vars:
            profiles = per_stat_per_var[stat_disp][var]
            if not profiles:
                continue
            samples = np.stack(profiles, axis=0)
            if not np.any(np.isfinite(samples)):
                continue
            mean_profile, ci_low, ci_high = bootstrap_mean_ci(
                samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])
            result[stat_disp][var] = {
                "mean_profile": mean_profile,
                "data":         mean_profile,
                "samples":      samples,
                "ci_low":       ci_low,
                "ci_high":      ci_high,
                "coord":        coord,
                "coord_label":  coord_label,
                "invert_y":     invert_y,
                "lead_hour":    lead,
            }
    return result

def collect_obs_forecast_lat_lon_2d(diag_base_dir, exp_name, diag_obs, cycles, lead,
                                    display_vars, raw_vars, boot):
    """{stat_disp: {display_var: entry}} lat-lon map at one fixed
    extended-forecast lead hour, bootstrap-aggregated over initialization
    cycles -- mirrors mpas_stat.graphics.io.collect_forecast_lat_lon_2d;
    entry shape matches what plot_lat_lon_2d expects (same as
    collect_obs_lat_lon_2d's, plus bootstrap ci_low/ci_high and a
    lead_hour tag)."""
    per_stat_per_var = {stat_disp: {v: [] for v in display_vars} for stat_disp, _ in STAT_KEYS_2D}
    lat = lon = None

    for cycle_time in cycles:
        ncfile = obs_diag_forecast_file(diag_base_dir, exp_name, diag_obs, cycle_time, lead)
        if not os.path.exists(ncfile):
            continue
        with nc.Dataset(ncfile, "r") as ds:
            if lat is None and "LatBins2D" in ds.variables:
                lat = np.asarray(ds.variables["LatBins2D"][:], dtype=np.float64)
                lon = np.asarray(ds.variables["LonBins2D"][:], dtype=np.float64)
            if "omf" not in ds.groups:
                continue
            diag_grp = ds.groups["omf"]
            for var, raw in zip(display_vars, raw_vars):
                gname = nc_group_name(raw)
                if gname not in diag_grp.groups:
                    continue
                var_grp = diag_grp.groups[gname]
                if "2D" not in var_grp.groups or "lat_lon" not in var_grp.groups["2D"].groups:
                    continue
                d2 = var_grp.groups["2D"].groups["lat_lon"]
                for stat_disp, _ in STAT_KEYS_2D:
                    if stat_disp in d2.variables:
                        arr = np.asarray(d2.variables[stat_disp][:, :], dtype=np.float64)
                        per_stat_per_var[stat_disp][var].append(arr)

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS_2D}
    if lat is None:
        return result

    for stat_disp, _ in STAT_KEYS_2D:
        for var in display_vars:
            maps = per_stat_per_var[stat_disp][var]
            if not maps:
                continue
            samples = np.stack(maps, axis=0)
            if not np.any(np.isfinite(samples)):
                continue
            mean_data, ci_low, ci_high = bootstrap_mean_ci(
                samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])
            result[stat_disp][var] = {
                "data":      mean_data,
                "samples":   samples,
                "ci_low":    ci_low,
                "ci_high":   ci_high,
                "lat":       lat,
                "lon":       lon,
                "lead_hour": lead,
            }
    return result

def collect_obs_forecast_lat_vert_2d(diag_base_dir, exp_name, diag_obs, cycles, lead,
                                     display_vars, raw_vars, coord_label, invert_y, boot):
    """{stat_disp: {display_var: entry}} (level x latitude) cross-section at
    one fixed extended-forecast lead hour, bootstrap-aggregated over
    initialization cycles -- mirrors
    mpas_stat.graphics.io.collect_forecast_lat_vert_2d; entry shape matches
    what plot_lat_vert_2d expects (same as collect_obs_lat_vert_2d's, plus
    a lead_hour tag). Not meaningful for radiances (bin_method LATITUDE, no
    2D/lat_vert group written) -- callers must skip that case."""
    per_stat_per_var = {stat_disp: {v: [] for v in display_vars} for stat_disp, _ in STAT_KEYS_2D}
    lat = levels = None

    for cycle_time in cycles:
        ncfile = obs_diag_forecast_file(diag_base_dir, exp_name, diag_obs, cycle_time, lead)
        if not os.path.exists(ncfile):
            continue
        with nc.Dataset(ncfile, "r") as ds:
            if lat is None and "LatBins2D" in ds.variables and "Levels" in ds.variables:
                lat = np.asarray(ds.variables["LatBins2D"][:], dtype=np.float64)
                levels = np.asarray(ds.variables["Levels"][:], dtype=np.float64)
            if "omf" not in ds.groups:
                continue
            diag_grp = ds.groups["omf"]
            for var, raw in zip(display_vars, raw_vars):
                gname = nc_group_name(raw)
                if gname not in diag_grp.groups:
                    continue
                var_grp = diag_grp.groups[gname]
                if "2D" not in var_grp.groups or "lat_vert" not in var_grp.groups["2D"].groups:
                    continue
                d2 = var_grp.groups["2D"].groups["lat_vert"]
                for stat_disp, _ in STAT_KEYS_2D:
                    if stat_disp in d2.variables:
                        arr = np.asarray(d2.variables[stat_disp][:, :], dtype=np.float64)
                        per_stat_per_var[stat_disp][var].append(arr)

    result = {stat_disp: {} for stat_disp, _ in STAT_KEYS_2D}
    if lat is None:
        return result

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
            mean_data, ci_low, ci_high = bootstrap_mean_ci(
                samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])
            result[stat_disp][var] = {
                "data":       mean_data,
                "samples":    samples,
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
                                       invert_y, stat_disp, boot):
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

        for cycle_time in cycles:
            row = None
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
                    arr = np.asarray(prof_grp.variables[stat_disp][:], dtype=np.float64)
                    if nlev is None:
                        nlev = arr.shape[0]
                    if row is None:
                        row = np.full((len(leads), nlev), np.nan, dtype=np.float64)
                    row[ilead, :] = arr
                    got_any = True

            if got_any:
                blocks.append(row)

        if not blocks or coord is None:
            continue

        samples = np.asarray(blocks, dtype=np.float64)  # (n_cycles, n_leads, n_levels)
        if not np.any(np.isfinite(samples)):
            continue
        mean_data, ci_low, ci_high = bootstrap_mean_ci(
            samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])

        out[var] = {
            "data":        mean_data,
            "samples":     samples,
            "ci_low":      ci_low,
            "ci_high":     ci_high,
            "leads":       np.asarray(leads, dtype=np.float64),
            "coord":       coord,
            "coord_label": coord_label,
            "invert_y":    invert_y,
        }
    return out

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

def _run_job(job):
    """Run one (func, args, kwargs) job, catching any error so a single bad
    figure never aborts the rest of the sensor's batch -- this matters even
    more once sensors run in a process pool, where an unhandled exception
    would otherwise just silently drop that figure and any jobs still
    queued behind it."""
    func, args, kwargs = job
    try:
        func(*args, **kwargs)
        return (True, None)
    except Exception:
        return (False, traceback.format_exc())

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
        ok, err = _run_job(job)
        if not ok:
            print(f"ERROR in {job[0].__name__} ({diag_obs}):\n{err}")
    print(f"[{diag_obs}] finished {len(jobs)} figures")
    return diag_obs, len(jobs)

def _build_pseudo_experiments(config):
    """
    MPAS-STAT's graphic functions plot one line/column per "experiment".
    Obs-space verification additionally distinguishes OMB (background) vs.
    OMA (analysis) *within* one experiment/cycle.  To reuse the graphic
    functions unmodified, each (experiment, OMB|OMA) combination is exposed
    to them as its own pseudo-experiment -- "<exp> - OMB" (solid) and
    "<exp> - OMA" (dashed) -- sharing the experiment's colour, matching the
    solid/dashed OMB/OMA convention of the original plots.

    Only "da"/"hofx" experiments contribute -- "hofx_forecast" experiments
    have no cycling diagnosis files to read (see _build_forecast_experiments()
    / generate_forecast_lead_graphics() for their own OMF product instead).

    Returns (names, labels, colors, linestyles, index_map) where
    index_map[j] = (exp_name, "omb"|"oma") for pseudo-experiment j.
    """
    plot_oma = config["plot_settings"].get("plot_oma", True)
    exp_names = config["experiments"]["names"]
    exp_labels = config["experiments"].get("labels", exp_names)
    exp_types = config["experiments"].get("types", ["da"] * len(exp_names))
    palette = [
        'black', 'blue', 'red', 'green', 'orange', 'brown', 'cyan', 'purple',
        'magenta', 'olive', 'teal', 'darkred', 'gold', 'navy', 'darkgreen'
    ]

    names, labels, colors, linestyles, index_map = [], [], [], [], []
    for iexp, (name, label, etype) in enumerate(zip(exp_names, exp_labels, exp_types)):
        if etype == "hofx_forecast":
            continue

        color = palette[iexp % len(palette)]

        names.append(f"{name}__omb")
        labels.append(f"{label} - OMB")
        colors.append(color)
        linestyles.append("-")
        index_map.append((name, "omb"))

        if etype == "da" and plot_oma:
            names.append(f"{name}__oma")
            labels.append(f"{label} - OMA")
            colors.append(color)
            linestyles.append("--")
            index_map.append((name, "oma"))

    return names, labels, colors, linestyles, index_map

def _build_forecast_experiments(config):
    """The subset of configured experiments with type == "hofx_forecast",
    in cfg["experiments"] shape for plot_forecast_lead_0d. Unlike
    _build_pseudo_experiments(), no OMB/OMA doubling is needed here -- OMF
    is the only diagtype an extended-forecast HofX run produces, so each
    "hofx_forecast" experiment maps to exactly one line.

    Returns (names, labels, colors, linestyles) where `names` are the real
    experiment names (no pseudo-experiment renaming)."""
    exp_names = config["experiments"]["names"]
    exp_labels = config["experiments"].get("labels", exp_names)
    exp_types = config["experiments"].get("types", ["da"] * len(exp_names))
    palette = [
        'black', 'blue', 'red', 'green', 'orange', 'brown', 'cyan', 'purple',
        'magenta', 'olive', 'teal', 'darkred', 'gold', 'navy', 'darkgreen'
    ]

    names, labels, colors = [], [], []
    for iexp, (name, label, etype) in enumerate(zip(exp_names, exp_labels, exp_types)):
        if etype != "hofx_forecast":
            continue
        names.append(name)
        labels.append(label)
        colors.append(palette[iexp % len(palette)])

    linestyles = ["-"] * len(names)
    return names, labels, colors, linestyles

def _experiments_for_diagtype(config, diagtype):
    """The real (non-pseudo) "da"/"hofx" experiments that contribute to one
    diagtype ("omb" or "oma") -- every da/hofx experiment has OMB, but only
    "da"-typed experiments (with plot_oma) have OMA. Used by the 2-D
    comparison products (lat-lon map, lev-time, lev-lat), which put the
    REAL experiment on the panel axis instead of MPAS-STAT's
    OMB/OMA-pseudo-experiment trick -- see _build_pseudo_experiments()'s
    docstring for why 0D/1D products still need that trick and these don't.

    Returns (names, labels, colors) -- experiments.names[0] among these is
    the comparison reference (absolute/full-state) for whichever of these
    figures enable a relative-reduction statistic."""
    exp_names = config["experiments"]["names"]
    exp_labels = config["experiments"].get("labels", exp_names)
    exp_types = config["experiments"].get("types", ["da"] * len(exp_names))
    plot_oma = config["plot_settings"].get("plot_oma", True)
    palette = [
        'black', 'blue', 'red', 'green', 'orange', 'brown', 'cyan', 'purple',
        'magenta', 'olive', 'teal', 'darkred', 'gold', 'navy', 'darkgreen'
    ]

    names, labels, colors = [], [], []
    for iexp, (name, label, etype) in enumerate(zip(exp_names, exp_labels, exp_types)):
        if etype == "hofx_forecast":
            continue
        if diagtype == "oma" and not (etype == "da" and plot_oma):
            continue
        names.append(name)
        labels.append(label)
        colors.append(palette[iexp % len(palette)])
    return names, labels, colors

def _comparison_graphic_cfg(config, out_dir, names, labels, colors):
    """cfg dict for the 2-D REAL-experiment comparison products (lat-lon
    map, lev-time, lev-lat) -- unlike _base_graphic_cfg(), enables
    relative_reduction_statistics so experiments.names[0] draws as the
    absolute reference and every other experiment draws as a bootstrap %
    reduction vs. it, exactly like the model-space package's own 2-D
    figures. A no-op visually when `names` has fewer than 2 entries (there
    is nothing to reduce against)."""
    comparison_stats = config["plot_settings"].get(
        "comparison_relative_reduction_statistics", ["RMS", "STD"])
    return {
        "variables": [],  # filled in per call site
        "experiments": {
            "names": names,
            "labels": labels,
            "colors": colors,
            "linestyles": ["-"] * len(names),
        },
        "paths": {"out_dir": out_dir},
        "figure_format": config["plot_settings"].get("figure_format", "png"),
        "max_time_ticks": config["plot_settings"].get("max_time_ticks", 12),
        "relative_reduction_statistics": comparison_stats,
        "bootstrap": config.get("bootstrap", {}),
    }

def _profile_axis_meta(bin_method):
    """(coord_label, invert_y) for a sensor's vertical/latitude coordinate --
    shared by generate_profile_graphics, generate_time_level_graphics, and
    generate_lat_vert_graphics so all three agree on axis labelling."""
    if bin_method == 1:
        return "Pressure (hPa)", True
    if bin_method == 2:
        return "Height (km)", False
    return "Latitude", False

def _base_graphic_cfg(config, out_dir):
    """cfg dict consumed by mpas_stat.graphics.functions.plot_*."""
    names, labels, colors, linestyles, index_map = _build_pseudo_experiments(config)
    cfg = {
        "variables": [],  # filled in per call site
        "experiments": {
            "names": names,
            "labels": labels,
            "colors": colors,
            "linestyles": linestyles,
        },
        "paths": {"out_dir": out_dir},
        "figure_format": config["plot_settings"].get("figure_format", "png"),
        "max_time_ticks": config["plot_settings"].get("max_time_ticks", 12),
        # OMB vs. OMA are plotted as absolute values on the same panel (like
        # the original script), not as a percent-difference product -- so
        # disable MPAS-STAT's relative/percent-reduction plotting mode.
        "relative_reduction_statistics": [],
    }
    return cfg, index_map

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
            )
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
            )
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
            )
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
                )
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

def _lead_cfg(cfg, lead):
    """Shallow cfg copy carrying a per-lead filename suffix (e.g. "_f006"),
    mirroring mpas_stat.graphics.driver._lead_cfg exactly -- _save_fig
    appends this suffix to the base filename so each forecast lead hour's
    figure gets its own file (instead of overwriting the previous lead's)
    while landing in the SAME output folder as the non-forecast figures
    plot_profiles_1d/plot_lat_lon_2d/plot_lat_vert_2d already write, e.g.
    profiles_1d/profile_vert_pres_profile_global_RMS_f006.png next to
    profiles_1d/profile_vert_pres_profile_global_RMS.png."""
    new_cfg = dict(cfg)
    new_cfg["_lead_suffix"] = f"_f{int(lead):03d}"
    return new_cfg

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
                )
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
            )
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
            )
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
                )
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
    n_workers = max(1, int(config.get("n_workers", config.get("parallel", {}).get("n_workers", 1))))

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
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
