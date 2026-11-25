"""Collectors for cycling (background/analysis) model-space statistics.

Each collect_* function reads mpasstat.<date>.nc files for every experiment
and cycle and returns the nested dict its matching plot_* function expects.
"""
import os

import netCDF4 as nc
import numpy as np

from ..core.bootstrap import get_bootstrap_settings
from ..core.pooling import PoolBuilder, aggregate, resolve_time_aggregation
from ..core.timeutils import datetime_to_cycle, generate_cycles
from .ncread import (
    _has_expected_ndim,
    _maybe_print_missing,
    get_lat_vert_coord,
    get_mpasstat_file,
    get_profile_coord,
    read_nc_path,
    read_stat,
)


# ---------------------------------------------------------------------------
# Data collectors
# ---------------------------------------------------------------------------
def collect_stats_0d(cfg):
    """Read the region averages of every experiment, variable and statistic, by cycle.

    Returns ``{exp: {cycle: {var: {stat: value}}}}``. Missing files are
    reported and skipped.
    """
    base_dir    = cfg["paths"]["base_dir"]
    exp_names   = cfg["experiments"]["names"]
    var_names   = cfg["variables"]
    stat_names  = cfg["statistics"]
    region_name = cfg.get("region", "global")
    cycles      = generate_cycles(cfg["time"]["start"], cfg["time"]["end"],
                                  cfg["time"]["interval_hours"])

    Stats = {}
    for exp_name in exp_names:
        Stats[exp_name] = {}
        for dt in cycles:
            cycle  = datetime_to_cycle(dt)
            ncfile = get_mpasstat_file(base_dir, exp_name, dt)
            if not os.path.exists(ncfile):
                print(f"WARNING: missing file: {ncfile}")
                continue

            Stats[exp_name][cycle] = {}
            with nc.Dataset(ncfile, "r") as ds:
                for var_name in var_names:
                    d = (Stats[exp_name][cycle]
                         .setdefault(var_name, {})
                         .setdefault("0D", {})
                         .setdefault(region_name, {}))
                    for stat_name in stat_names:
                        path = f"{var_name}/0D/{region_name}/{stat_name}"
                        try:
                            d[stat_name] = float(read_nc_path(ds, path))
                        except Exception:
                            print(f"WARNING: missing {path} in {ncfile}")
    return Stats


def collect_profiles_1d(cfg, profile_key, stat_name):
    """Read one profile type (``profile_key``) of every experiment and variable and pool it over cycles.

    Each entry holds the pooled profile and its bootstrap interval, plus the
    per-cycle samples and pool that the comparison figures use for the
    paired difference from the reference.
    """
    base_dir  = cfg["paths"]["base_dir"]
    exp_names = cfg["experiments"]["names"]
    var_names = cfg["variables"]
    boot      = get_bootstrap_settings(cfg)
    pooled    = resolve_time_aggregation(cfg) == "pooled"
    cycles    = generate_cycles(cfg["time"]["start"], cfg["time"]["end"],
                                cfg["time"]["interval_hours"])

    Profiles = {}
    for exp_name in exp_names:
        Profiles[exp_name] = {}
        for var_name in var_names:
            all_profiles = []
            pool = PoolBuilder(stat_name)
            coord = coord_label = None
            invert_y = False

            for dt in cycles:
                ncfile = get_mpasstat_file(base_dir, exp_name, dt)
                if not os.path.exists(ncfile):
                    print(f"WARNING: missing file: {ncfile}")
                    continue
                path = f"{var_name}/1D/{profile_key}/{stat_name}"
                with nc.Dataset(ncfile, "r") as ds:
                    try:
                        comps = read_stat(ds, path)
                        arr = comps[0]
                        if not _has_expected_ndim(arr, 1):
                            continue
                        all_profiles.append(np.asarray(arr, dtype=np.float64))
                        pool.add(comps)
                        if coord is None:
                            coord, coord_label, invert_y = get_profile_coord(ds, profile_key)
                    except Exception:
                        _maybe_print_missing(cfg, f"WARNING: missing/invalid {path} in {ncfile}")

            if not all_profiles:
                _maybe_print_missing(cfg, f"WARNING: no valid profiles for {exp_name} {var_name}")
                continue

            samples = np.asarray(all_profiles)
            pool_arr = pool.build() if pooled else None
            mean_data, ci_low, ci_high = aggregate(samples, pool_arr, boot)

            Profiles[exp_name][var_name] = {
                "mean_profile": mean_data,
                "data":         mean_data,
                "samples":      samples,
                "pool":      pool_arr,
                "ci_low":       ci_low,
                "ci_high":      ci_high,
                "coord":        coord,
                "coord_label":  coord_label,
                "invert_y":     invert_y,
            }
    return Profiles


def collect_time_level_2d(cfg, profile_key, stat_name):
    """Read one profile type of every experiment and variable as a cycle × level array (no pooling)."""
    base_dir  = cfg["paths"]["base_dir"]
    exp_names = cfg["experiments"]["names"]
    var_names = cfg["variables"]
    cycles    = generate_cycles(cfg["time"]["start"], cfg["time"]["end"],
                                cfg["time"]["interval_hours"])

    TimeLev = {}
    for exp_name in exp_names:
        TimeLev[exp_name] = {}
        for var_name in var_names:
            profiles = []
            times    = []
            coord = coord_label = None
            invert_y = False

            for dt in cycles:
                cycle  = datetime_to_cycle(dt)
                ncfile = get_mpasstat_file(base_dir, exp_name, dt)
                if not os.path.exists(ncfile):
                    print(f"WARNING: missing file: {ncfile}")
                    continue
                path = f"{var_name}/1D/{profile_key}/{stat_name}"
                with nc.Dataset(ncfile, "r") as ds:
                    try:
                        arr = read_nc_path(ds, path)
                        if not _has_expected_ndim(arr, 1):
                            continue
                        profiles.append(np.asarray(arr, dtype=np.float64))
                        times.append(cycle)
                        if coord is None:
                            coord, coord_label, invert_y = get_profile_coord(ds, profile_key)
                    except Exception:
                        _maybe_print_missing(cfg, f"WARNING: missing/invalid {path} in {ncfile}")

            if not profiles:
                continue

            TimeLev[exp_name][var_name] = {
                "data":        np.asarray(profiles),
                "samples":     np.asarray(profiles),
                "times":       times,
                "coord":       coord,
                "coord_label": coord_label,
                "invert_y":    invert_y,
            }
    return TimeLev


def collect_lat_vert_2d(cfg, section_key, stat_name):
    """Read a latitude–level section (``lat_lev``, ``lat_hgt`` or ``lat_pres``) and pool it over cycles, with bootstrap intervals."""
    base_dir  = cfg["paths"]["base_dir"]
    exp_names = cfg["experiments"]["names"]
    var_names = cfg["variables"]
    boot      = get_bootstrap_settings(cfg)
    pooled    = resolve_time_aggregation(cfg) == "pooled"
    cycles    = generate_cycles(cfg["time"]["start"], cfg["time"]["end"],
                                cfg["time"]["interval_hours"])

    LatVert = {}
    for exp_name in exp_names:
        LatVert[exp_name] = {}
        for var_name in var_names:
            sections = []
            pool = PoolBuilder(stat_name)
            lat = vert = vert_label = None
            invert_y = False

            for dt in cycles:
                ncfile = get_mpasstat_file(base_dir, exp_name, dt)
                if not os.path.exists(ncfile):
                    print(f"WARNING: missing file: {ncfile}")
                    continue
                path = f"{var_name}/2D/{section_key}/{stat_name}"
                with nc.Dataset(ncfile, "r") as ds:
                    try:
                        comps = read_stat(ds, path)
                        arr = comps[0]
                        if not _has_expected_ndim(arr, 2):
                            continue
                        sections.append(np.asarray(arr, dtype=np.float64))
                        pool.add(comps)
                        if lat is None:
                            lat, vert, vert_label, invert_y = get_lat_vert_coord(ds, section_key)
                    except Exception:
                        _maybe_print_missing(cfg, f"WARNING: missing/invalid {path} in {ncfile}")

            if not sections:
                _maybe_print_missing(cfg, f"WARNING: no valid sections for {exp_name} {var_name}")
                continue

            samples = np.asarray(sections)
            pool_arr = pool.build() if pooled else None
            mean_data, ci_low, ci_high = aggregate(samples, pool_arr, boot)

            LatVert[exp_name][var_name] = {
                "data":       mean_data,
                "samples":    samples,
                "pool":    pool_arr,
                "ci_low":     ci_low,
                "ci_high":    ci_high,
                "lat":        lat,
                "vert":       vert,
                "vert_label": vert_label,
                "invert_y":   invert_y,
            }
    return LatVert


def collect_lat_lon_2d(cfg, stat_name):
    """Read the lat-lon maps of every experiment and variable and pool them over cycles, with bootstrap intervals."""
    base_dir  = cfg["paths"]["base_dir"]
    exp_names = cfg["experiments"]["names"]
    var_names = cfg["variables"]
    boot      = get_bootstrap_settings(cfg)
    pooled    = resolve_time_aggregation(cfg) == "pooled"
    cycles    = generate_cycles(cfg["time"]["start"], cfg["time"]["end"],
                                cfg["time"]["interval_hours"])

    LatLon = {}
    for exp_name in exp_names:
        LatLon[exp_name] = {}
        for var_name in var_names:
            maps = []
            pool = PoolBuilder(stat_name)
            lat = lon = None

            for dt in cycles:
                ncfile = get_mpasstat_file(base_dir, exp_name, dt)
                if not os.path.exists(ncfile):
                    print(f"WARNING: missing file: {ncfile}")
                    continue
                path = f"{var_name}/2D/lat_lon/{stat_name}"
                with nc.Dataset(ncfile, "r") as ds:
                    try:
                        comps = read_stat(ds, path)
                        arr = comps[0]
                        if not _has_expected_ndim(arr, 2):
                            continue
                        maps.append(np.asarray(arr, dtype=np.float64))
                        pool.add(comps)
                        if lat is None:
                            lat = np.asarray(ds["LatBins"][:])
                            lon = np.asarray(ds["LonBins"][:])
                    except Exception:
                        _maybe_print_missing(cfg, f"WARNING: missing/invalid {path} in {ncfile}")

            if not maps:
                _maybe_print_missing(cfg, f"WARNING: no valid lat-lon maps for {exp_name} {var_name}")
                continue

            samples = np.asarray(maps)
            pool_arr = pool.build() if pooled else None
            mean_data, ci_low, ci_high = aggregate(samples, pool_arr, boot)

            LatLon[exp_name][var_name] = {
                "data":    mean_data,
                "samples": samples,
                "pool": pool_arr,
                "ci_low":  ci_low,
                "ci_high": ci_high,
                "lat":     lat,
                "lon":     lon,
            }
    return LatLon


# ---------------------------------------------------------------------------
# 3-D latitude-longitude slices at selected vertical levels
# ---------------------------------------------------------------------------
def _canonical_3d_lat_lon_section(vertical_type):
    """Return the MPAS-stat 3-D lat-lon group suffix.

    The NetCDF files use short names:
      lat_lon_hgt, lat_lon_pres, lat_lon_lev
    while the YAML may use more readable aliases such as height/pressure/level.
    """
    vtype = str(vertical_type).lower()
    if vtype in ("pressure", "pres", "p"):
        return "pres"
    if vtype in ("height", "hgt", "z"):
        return "hgt"
    if vtype in ("level", "lev", "model_level", "model-level", "modellevel"):
        return "lev"
    raise ValueError(f"Unsupported vertical_type: {vertical_type}")


def _get_3d_lat_lon_vertical_coord(ds, vertical_type):
    """Return vertical coordinate values and label for 3-D lat-lon diagnostics."""
    section = _canonical_3d_lat_lon_section(vertical_type)
    if section == "pres":
        return np.asarray(ds["PressureLevels"][:]), "Pressure (hPa)"
    if section == "hgt":
        return np.asarray(ds["HeightLevels"][:]), "Height (m)"
    if section == "lev":
        return np.asarray(ds["LevelBins"][:]), "Model level bin"
    raise ValueError(f"Unsupported vertical_type: {vertical_type}")


def _nearest_level_index(levels, requested_level):
    """Return nearest vertical index and selected level value."""
    if requested_level is None:
        return None, None
    if levels is None:
        # Treat requested_level as 1-based model-level index when levels are unavailable.
        idx = int(requested_level) - 1
        return idx, int(requested_level)
    vals = np.asarray(levels, dtype=np.float64)
    idx = int(np.nanargmin(np.abs(vals - float(requested_level))))
    return idx, vals[idx]


def _slice_3d_lat_lon_field(arr, level_index, nlat, nlon):
    """Slice one vertical level from a 3-D lat-lon diagnostic field.

    Expected layout:
      (lat, lon, vertical)

    Returns a 2-D array with shape (lat, lon).
    """
    a = np.asarray(arr, dtype=np.float64)
    a = np.squeeze(a)

    if a.ndim != 3:
        raise ValueError(f"Expected 3-D field after squeeze, got shape {a.shape}")

    if a.shape[0] != nlat or a.shape[1] != nlon:
        raise ValueError(
            f"Expected field shape (lat, lon, vertical), got {a.shape}; "
            f"expected lat={nlat}, lon={nlon}"
        )

    if level_index < 0 or level_index >= a.shape[2]:
        raise IndexError(
            f"Selected level index {level_index} is outside vertical axis length {a.shape[2]}"
        )

    return a[:, :, level_index]


def collect_lat_lon_slice_2d(cfg, stat_name, section_cfg):
    """Collect time-averaged lat-lon maps from one selected 3-D vertical level.

    Expected default NetCDF path:
      {var}/3D/lat_lon_{section}/{stat}

    This can be overridden with section_cfg["path_template"], e.g.
      "{var}/3D/lat_lon_pres/{stat}"
    """
    base_dir  = cfg["paths"]["base_dir"]
    exp_names = cfg["experiments"]["names"]
    var_names = cfg["variables"]
    boot      = get_bootstrap_settings(cfg)
    pooled    = resolve_time_aggregation(cfg) == "pooled"
    cycles    = generate_cycles(cfg["time"]["start"], cfg["time"]["end"],
                                cfg["time"]["interval_hours"])

    vertical_type = section_cfg.get("vertical_type", "pressure")
    section = section_cfg.get("section", _canonical_3d_lat_lon_section(vertical_type))
    requested_level = section_cfg.get("level", None)
    section_name = section_cfg.get("name", f"{section}_{requested_level}")
    path_template = section_cfg.get(
        "path_template",
        "{var}/3D/lat_lon_{section}/{stat}",
    )

    LatLonLev = {}
    selected_level_value = None
    selected_level_index = None
    vertical_label = None

    for exp_name in exp_names:
        LatLonLev[exp_name] = {}
        for var_name in var_names:
            maps = []
            pool = PoolBuilder(stat_name)
            lat = lon = None
            levs = None

            for dt in cycles:
                ncfile = get_mpasstat_file(base_dir, exp_name, dt)
                if not os.path.exists(ncfile):
                    print(f"WARNING: missing file: {ncfile}")
                    continue

                path = path_template.format(
                    var=var_name,
                    stat=stat_name,
                    vertical_type=vertical_type,
                    section=section,
                    name=section_name,
                )

                with nc.Dataset(ncfile, "r") as ds:
                    try:
                        comps = read_stat(ds, path)
                        field3d = comps[0]
                        if not _has_expected_ndim(field3d, 3):
                            continue
                        if lat is None:
                            lat = np.asarray(ds["LatBins"][:])
                            lon = np.asarray(ds["LonBins"][:])
                            levs, vertical_label = _get_3d_lat_lon_vertical_coord(ds, vertical_type)
                            selected_level_index, selected_level_value = _nearest_level_index(
                                levs, requested_level
                            )
                            if selected_level_index is None:
                                raise ValueError("lat_lon_3d_slices requires a selected level")
                        maps.append(_slice_3d_lat_lon_field(
                            field3d, selected_level_index, len(lat), len(lon)
                        ))
                        pool.add(tuple(_slice_3d_lat_lon_field(c, selected_level_index, len(lat), len(lon))
                                       for c in comps))
                    except Exception as e:
                        _maybe_print_missing(cfg, f"WARNING: missing/invalid {path} in {ncfile}: {e}")

            if not maps:
                _maybe_print_missing(cfg, f"WARNING: no valid 3-D lat-lon maps for {exp_name} {var_name} {section_name}")
                continue

            samples = np.asarray(maps)
            pool_arr = pool.build() if pooled else None
            mean_data, ci_low, ci_high = aggregate(samples, pool_arr, boot)

            LatLonLev[exp_name][var_name] = {
                "data": mean_data,
                "samples": samples,
                "pool": pool_arr,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "lat": lat,
                "lon": lon,
                "vertical_type": vertical_type,
                "vertical_label": vertical_label,
                "selected_level": selected_level_value,
                "selected_level_index": selected_level_index,
                "section_name": section_name,
            }
    return LatLonLev
