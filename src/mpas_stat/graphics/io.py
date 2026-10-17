import os
import warnings
from datetime import datetime, timedelta
import netCDF4 as nc
import numpy as np

# ---------------------------------------------------------------------------
# Date / cycle helpers
# ---------------------------------------------------------------------------
def yyyymmddhh_to_datetime(date_str):
    return datetime.strptime(str(date_str), "%Y%m%d%H")

def datetime_to_cycle(dt):
    return dt.strftime("%Y%m%d%H")

def datetime_to_datefile(dt):
    return dt.strftime("%Y-%m-%d_%H.00.00")

def generate_cycles(start_date, end_date, interval_hours):
    start_dt = yyyymmddhh_to_datetime(start_date)
    end_dt   = yyyymmddhh_to_datetime(end_date)
    cycles, dt = [], start_dt
    while dt <= end_dt:
        cycles.append(dt)
        dt += timedelta(hours=int(interval_hours))
    return cycles

def generate_forecast_leads(cfg):
    """Return forecast lead hours (int) for extended forecast verification.

    Controlled by cfg["forecast"]:
        length   : maximum forecast lead hour (required)
        interval : lead-time spacing in hours (required)
        start    : first lead hour (default = interval; use 0 to include
                   the initialization/analysis time itself)
    """
    fcfg     = cfg.get("forecast", {})
    length   = int(fcfg["length"])
    interval = int(fcfg["interval"])
    start    = int(fcfg.get("start", interval))

    leads, lead = [], start
    while lead <= length:
        leads.append(lead)
        lead += interval
    return leads

# ---------------------------------------------------------------------------
# NetCDF helpers
# ---------------------------------------------------------------------------
def read_nc_path(ds, path):
    obj = ds
    for p in path.strip("/").split("/"):
        obj = obj[p]
    return np.asarray(obj[...], dtype=np.float64)

def safe_nanmean(a, axis=0):
    """np.nanmean without RuntimeWarning for all-NaN slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)

def safe_nanpercentile(a, q, axis=0):
    """np.nanpercentile without RuntimeWarning for all-NaN slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanpercentile(a, q, axis=axis)

def _has_expected_ndim(arr, expected_ndim):
    """Return True when the squeezed array has the expected number of dimensions."""
    return np.squeeze(np.asarray(arr)).ndim == int(expected_ndim)

def _maybe_print_missing(cfg, message):
    """Print optional missing-diagnostic messages only when requested."""
    if bool(cfg.get("verbose_missing", False)):
        print(message)

def get_mpasstat_file(base_dir, exp_name, cycle_dt):
    cycle     = datetime_to_cycle(cycle_dt)
    date_file = datetime_to_datefile(cycle_dt)
    return os.path.join(
        base_dir, exp_name, "VerifyModel", cycle,
        f"mpasstat.{date_file}.nc",
    )

def get_mpasstat_forecast_file(base_dir, exp_name, init_dt, lead_hour):
    """Path to an extended-forecast statistics file for one init cycle/lead.

    Mirrors the directory layout written by scripts/run_verify_model_fcst.ksh in
    verify_mode=forecast:
        <base_dir>/<exp_name>/VerifyModel/<init_cycle>/f<HHH>/mpasstat.<valid_date>.nc
    """
    init_cycle = datetime_to_cycle(init_dt)
    valid_dt   = init_dt + timedelta(hours=int(lead_hour))
    date_file  = datetime_to_datefile(valid_dt)
    return os.path.join(
        base_dir, exp_name, "VerifyModel", init_cycle,
        f"f{int(lead_hour):03d}",
        f"mpasstat.{date_file}.nc",
    )

# ---------------------------------------------------------------------------
# Bootstrap settings  (single authoritative location)
# ---------------------------------------------------------------------------
def get_bootstrap_settings(cfg):
    bcfg = cfg.get("bootstrap", {})
    return {
        "n_boot":             int(bcfg.get("n_boot", 1000)),
        "ci":                 float(bcfg.get("ci", 95)),
        "seed":               int(bcfg.get("seed", 0)),
        "plot_significance":  bool(bcfg.get("plot_significance", True)),
    }

# ---------------------------------------------------------------------------
# Bootstrap statistics
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(samples, n_boot=1000, ci=95, seed=0):
    samples = np.asarray(samples, dtype=np.float64)
    ntime   = samples.shape[0]
    mean    = safe_nanmean(samples, axis=0)

    if ntime < 2:
        nan = np.full_like(mean, np.nan)
        return mean, nan, nan

    rng        = np.random.default_rng(seed)
    boot_means = [
        safe_nanmean(samples[rng.integers(0, ntime, size=ntime), ...], axis=0)
        for _ in range(int(n_boot))
    ]
    boot_means = np.asarray(boot_means)

    alpha    = (100.0 - ci) / 2.0
    ci_low   = safe_nanpercentile(boot_means, alpha,         axis=0)
    ci_high  = safe_nanpercentile(boot_means, 100.0 - alpha, axis=0)
    return mean, ci_low, ci_high

def bootstrap_reduction_ci(ref_samples, exp_samples, n_boot=1000, ci=95, seed=0):
    """Bootstrap CI for percent reduction = 100 * (ref - exp) / ref.
    Positive values indicate improvement."""
    ref_samples = np.asarray(ref_samples, dtype=np.float64)
    exp_samples = np.asarray(exp_samples, dtype=np.float64)

    ntime       = min(ref_samples.shape[0], exp_samples.shape[0])
    ref_samples = ref_samples[:ntime, ...]
    exp_samples = exp_samples[:ntime, ...]

    def _reduction(ref, exp):
        red   = np.full_like(ref, np.nan, dtype=np.float64)
        valid = np.isfinite(ref) & np.isfinite(exp) & (np.abs(ref) > 1.0e-12)
        red[valid] = 100.0 * (ref[valid] - exp[valid]) / ref[valid]
        return red

    reduction = _reduction(safe_nanmean(ref_samples, axis=0),
                           safe_nanmean(exp_samples, axis=0))

    if ntime < 2:
        nan = np.full_like(reduction, np.nan)
        return reduction, nan, nan

    rng            = np.random.default_rng(seed)
    boot_reductions = []
    for _ in range(int(n_boot)):
        ind = rng.integers(0, ntime, size=ntime)
        boot_reductions.append(
            _reduction(safe_nanmean(ref_samples[ind, ...], axis=0),
                       safe_nanmean(exp_samples[ind, ...], axis=0))
        )
    boot_reductions = np.asarray(boot_reductions)

    alpha   = (100.0 - ci) / 2.0
    ci_low  = safe_nanpercentile(boot_reductions, alpha,         axis=0)
    ci_high = safe_nanpercentile(boot_reductions, 100.0 - alpha, axis=0)
    return reduction, ci_low, ci_high

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def get_profile_coord(ds, profile_key):
    """
    Return coordinate, label, and whether to invert y-axis.
    Model levels are treated as categorical indices.
    """

    # --- Latitude ---
    if profile_key == "lat_bin_profile":
        return np.asarray(ds["LatBins"][:]), "Latitude", False

    # --- Model level (categorical) ---
    if profile_key.startswith("vert_profile_"):
        nlev = ds["ModelLevels"].shape[0]
        coord = np.arange(1, nlev + 1)   # categorical index
        return coord, "Model level", False

    # --- Level bins (categorical) ---
    if profile_key.startswith("vert_bin_profile_"):
        return np.asarray(ds["LevelBins"][:]), "Model level bin", False

    # --- Height ---
    if profile_key.startswith("vert_hgt_profile_"):
        return np.asarray(ds["HeightLevels"][:]), "Height (m)", False

    # --- Pressure ---
    if profile_key.startswith("vert_pres_profile_"):
        return np.asarray(ds["PressureLevels"][:]), "Pressure (hPa)", True

    raise ValueError(f"Unknown profile key: {profile_key}")

def get_lat_vert_coord(ds, section_key):
    lat = np.asarray(ds["LatBins"][:])
    _map = {
        "lat_lev":  (ds["LevelBins"][:],    "Model level bin",  False),
        "lat_hgt":  (ds["HeightLevels"][:], "Height (m)",       False),
        "lat_pres": (ds["PressureLevels"][:], "Pressure (hPa)", False),
    }
    if section_key not in _map:
        raise ValueError(f"Unsupported latitude-vertical section key: {section_key}")
    raw_vert, vert_label, invert_y = _map[section_key]
    return lat, np.asarray(raw_vert), vert_label, invert_y

def apply_region_suffix(name, cfg):
    """Append region suffix if needed."""
    region = str(cfg.get("region", "")).strip()

    if not region:
        return name

    if name.endswith(f"_{region}"):
        return name

    return f"{name}_{region}"

# ---------------------------------------------------------------------------
# Data collectors
# ---------------------------------------------------------------------------
def collect_stats_0d(cfg):
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

def collect_profiles_1d(cfg, profile_key, stat_name):
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
                        all_profiles.append(np.asarray(arr, dtype=np.float64))
                        if coord is None:
                            coord, coord_label, invert_y = get_profile_coord(ds, profile_key)
                    except Exception:
                        _maybe_print_missing(cfg, f"WARNING: missing/invalid {path} in {ncfile}")

            if not all_profiles:
                _maybe_print_missing(cfg, f"WARNING: no valid profiles for {exp_name} {var_name}")
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
            }
    return Profiles

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

def collect_time_level_2d(cfg, profile_key, stat_name):
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
                ncfile = get_mpasstat_file(base_dir, exp_name, dt)
                if not os.path.exists(ncfile):
                    print(f"WARNING: missing file: {ncfile}")
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
                _maybe_print_missing(cfg, f"WARNING: no valid sections for {exp_name} {var_name}")
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
            }
    return LatVert

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

def collect_lat_lon_2d(cfg, stat_name):
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
                ncfile = get_mpasstat_file(base_dir, exp_name, dt)
                if not os.path.exists(ncfile):
                    print(f"WARNING: missing file: {ncfile}")
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
                _maybe_print_missing(cfg, f"WARNING: no valid lat-lon maps for {exp_name} {var_name}")
                continue

            samples = np.asarray(maps)
            mean_data, ci_low, ci_high = bootstrap_mean_ci(
                samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])

            LatLon[exp_name][var_name] = {
                "data":    mean_data,
                "samples": samples,
                "ci_low":  ci_low,
                "ci_high": ci_high,
                "lat":     lat,
                "lon":     lon,
            }
    return LatLon

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
                        field3d = read_nc_path(ds, path)
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
                    except Exception as e:
                        _maybe_print_missing(cfg, f"WARNING: missing/invalid {path} in {ncfile}: {e}")

            if not maps:
                _maybe_print_missing(cfg, f"WARNING: no valid 3-D lat-lon maps for {exp_name} {var_name} {section_name}")
                continue

            samples = np.asarray(maps)
            mean_data, ci_low, ci_high = bootstrap_mean_ci(
                samples, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])

            LatLonLev[exp_name][var_name] = {
                "data": mean_data,
                "samples": samples,
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
