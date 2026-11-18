"""Per-sensor processing for the obs-space statistics stage: OMB/OMA for
cycling experiments, OMF for extended forecasts, and the stage's main().
"""
import os
from typing import Dict, Optional, Tuple

import numpy as np

from .binning import (
    _apply_domain_mask,
    calculate_stats,
    compute_stats_2d_lat_vert,
    compute_stats_2d_latlon,
)
from .bounds import _resolve_latlon_bounds_for
from .common import (
    STAT_KEYS_2D,
    generate_cycles,
    generate_forecast_leads,
    get_levels,
    lead_valid_time_allowed,
    load_config,
    resolve_diag_base_dir,
    resolve_domains,
    resolve_time_settings,
)
from .reader import (
    build_filename,
    build_hofx_forecast_filename,
    load_observation_data,
)
from .sensor_config import get_bin_method, get_obs_config
from .writer import write_obs_diag_files, write_obs_forecast_diag_files

# ============================================================================
# Shared pieces of process_experiment() / process_forecast_experiment()
# ============================================================================

# Domain-averaged (0D) and level-binned (1D) statistics stored per diagtype.
_INNOV_STATS = ("rmse", "mean", "std", "count")


_SPREAD_FIELDS = ("spread", "totSpread")


def _bin_values(obs_vec: Dict, bin_method: int) -> np.ndarray:
    """The coordinate a sensor's innovations are binned by for 1-D profiles:
    pressure (bin_method 1), height (2) or latitude (3, radiances)."""
    return (
        obs_vec["pres"] if bin_method == 1 else
        obs_vec["hgt"]  if bin_method == 2 else
        obs_vec["lat"]
    )


def _resolve_stat_grids(config: Dict, diag_obs: str, obs_config, bin_method: int,
                        file_kind: str) -> Dict:
    """Lat-lon, lat-vert and vertical/latitude bin grids for one sensor.

    Lat-lon grid (2-D statistics): covers the whole globe by default, or
    just the plot_settings.latlon_domain box for a REGIONAL application --
    see resolve_latlon_bounds() (a hand-configured domains.<name> box), or,
    when latlon_domain == "auto", auto_detect_latlon_bounds() (the tightest
    box containing every observation with a valid, QC-passed innovation --
    see _resolve_latlon_bounds_for()). Independent of the 0D/1D `domains`
    masking loop (mirrors MPAS-STAT's own domain-independent "2D lat_lon"
    product), though a hand-configured box may reuse one of the same
    `domains` entries as its bounding box.

    The lat-lon grid is resolved BEFORE `levels` because bin_method == 3
    (radiances) reuses the SAME lat range / lat_bin_width for its "1D
    latitude profile" levels, UNLESS the sensor has its own explicit
    level_bin_kwargs override (src/mpas_stat/obs/sensor_config.py) -- an
    explicit per-sensor override always wins over the domain-derived default.

    Lat-vert grid (2-D level x latitude cross-section): reuses the SAME
    latlon_domain-bounded latitude bins as the lat-lon map rather than a
    second, separately-configured latitude box -- see
    compute_stats_2d_lat_vert(). Not generated for radiances, whose "level"
    axis already IS latitude.
    """
    plot_settings = config["plot_settings"]
    gen_2d = bool(plot_settings.get("generate_lat_lon_2d", True))
    lat_bin_width = float(plot_settings.get("latlon_lat_bin_width", 10.0))
    lon_bin_width = float(plot_settings.get("latlon_lon_bin_width", 10.0))
    lat_min_2d, lat_max_2d, lon_min_2d, lon_max_2d = _resolve_latlon_bounds_for(
        config, diag_obs, bin_method, file_kind=file_kind)
    lat_bins_2d = np.arange(lat_min_2d, lat_max_2d + lat_bin_width, lat_bin_width)
    lon_bins_2d = np.arange(lon_min_2d, lon_max_2d + lon_bin_width, lon_bin_width)

    gen_lat_vert = bool(plot_settings.get("generate_lat_vert_2d", True)) and bin_method != 3

    level_bin_kwargs = obs_config.level_bin_kwargs or {}
    if bin_method == 3 and not level_bin_kwargs:
        levels = get_levels(bin_method, bin_width=lat_bin_width,
                            domain_range=(lat_min_2d, lat_max_2d))
    else:
        levels = get_levels(bin_method, **level_bin_kwargs)

    return {
        "gen_2d": gen_2d,
        "gen_lat_vert": gen_lat_vert,
        "lat_bins_2d": lat_bins_2d,
        "lon_bins_2d": lon_bins_2d,
        "lat_centers_2d": 0.5 * (lat_bins_2d[:-1] + lat_bins_2d[1:]),
        "lon_centers_2d": 0.5 * (lon_bins_2d[:-1] + lon_bins_2d[1:]),
        "levels": levels,
    }


def _allocate_results(tags, lead_shape: Tuple[int, ...], grids: Dict,
                      num_exp: int, num_dom: int, num_time: int, num_var: int) -> Dict:
    """NaN-filled result arrays for every diagtype in `tags` ("omb"/"oma"
    for cycling DA, "omf" for extended forecasts).

    `lead_shape` is () for cycling statistics and (num_lead,) for forecast
    statistics, inserted right after the cycle/init-time axis.  Keys are
    created in the same order the statistics stage has always used:
    <stat>_<tag> (0D), <stat>_<tag>_bin (1D), then <stat>_<tag>_latlon /
    <stat>_<tag>_latvert when the 2-D products are enabled.
    """
    num_lev = len(grids["levels"])
    nlat2d, nlon2d = len(grids["lat_centers_2d"]), len(grids["lon_centers_2d"])
    zero_d = (num_exp, num_dom, num_time) + lead_shape + (num_var,)
    one_d = (num_exp, num_dom, num_time) + lead_shape + (num_lev, num_var)
    latlon = (num_exp, num_time) + lead_shape + (nlat2d, nlon2d, num_var)
    latvert = (num_exp, num_time) + lead_shape + (num_lev, nlat2d, num_var)

    results = {}
    for tag in tags:
        for stat in _INNOV_STATS + _SPREAD_FIELDS:
            results[f"{stat}_{tag}"] = np.full(zero_d, np.nan)
        for stat in _INNOV_STATS + _SPREAD_FIELDS:
            results[f"{stat}_{tag}_bin"] = np.full(one_d, np.nan)

    if grids["gen_2d"]:
        for tag in tags:
            for stat in _INNOV_STATS:
                results[f"{stat}_{tag}_latlon"] = np.full(latlon, np.nan)
    if grids["gen_lat_vert"]:
        for tag in tags:
            for stat in _INNOV_STATS:
                results[f"{stat}_{tag}_latvert"] = np.full(latvert, np.nan)
    return results


def _store_2d_stats(results: Dict, tag: str, idx: Tuple, iv: int, obs_vec: Dict,
                    bin_values: np.ndarray, bin_method: int, grids: Dict) -> None:
    """Lat-lon and lat-vert gridded statistics for one innovation vector,
    stored at results[<stat>_<tag>_latlon|latvert][idx..., :, :, iv]."""
    if grids["gen_2d"]:
        latlon = compute_stats_2d_latlon(
            obs_vec["innov"], obs_vec["qc"], obs_vec["lat"], obs_vec["lon"],
            grids["lat_bins_2d"], grids["lon_bins_2d"],
        )
        for _, prefix in STAT_KEYS_2D:
            results[f"{prefix}_{tag}_latlon"][idx + (slice(None), slice(None), iv)] = latlon[prefix]

    if grids["gen_lat_vert"]:
        latvert = compute_stats_2d_lat_vert(
            obs_vec["innov"], obs_vec["qc"], bin_values, obs_vec["lat"],
            grids["levels"], bin_method, grids["lat_bins_2d"],
        )
        for _, prefix in STAT_KEYS_2D:
            results[f"{prefix}_{tag}_latvert"][idx + (slice(None), slice(None), iv)] = latvert[prefix]


def _store_domain_stats(results: Dict, tag: str, idx: Tuple, iv: int, obs_vec: Dict,
                        qc: Optional[np.ndarray], bin_values: np.ndarray,
                        levels: np.ndarray, bin_method: int) -> None:
    """Domain-total (0D) and level-binned (1D) statistics of one innovation
    vector -- plus the mean ensemble spread / total spread when present --
    stored at results[<stat>_<tag>][idx..., iv] and
    results[<stat>_<tag>_bin][idx..., level, iv]. Empty totals/bins keep
    their NaN fill value."""
    num_lev = len(levels)

    stats = calculate_stats(obs_vec["innov"], qc, bin_values, levels, bin_method)
    if stats["count"][num_lev] > 0:
        for stat in _INNOV_STATS:
            results[f"{stat}_{tag}"][idx + (iv,)] = stats[stat][num_lev]
    valid = stats["count"][:num_lev] > 0
    for stat in _INNOV_STATS:
        results[f"{stat}_{tag}_bin"][idx + (valid, iv)] = stats[stat][:num_lev][valid]

    for field in _SPREAD_FIELDS:
        if obs_vec[field] is None:
            continue
        stats = calculate_stats(obs_vec[field], qc, bin_values, levels, bin_method)
        if stats["count"][num_lev] > 0:
            results[f"{field}_{tag}"][idx + (iv,)] = stats["mean"][num_lev]
        valid = stats["count"][:num_lev] > 0
        results[f"{field}_{tag}_bin"][idx + (valid, iv)] = stats["mean"][:num_lev][valid]


def _metadata(times, levels, obs_config, domains: Dict) -> Dict:
    return {
        "times": times,
        "levels": levels,
        "variables": obs_config.variables,
        "units": obs_config.units,
        "labels": obs_config.labels,
        "domain_names": list(domains.keys()),
        "domain_coords": list(domains.values()),
    }


def _grid_metadata(grids: Dict) -> Dict:
    out = {}
    if grids["gen_2d"]:
        out["latlon_lat_centers"] = grids["lat_centers_2d"]
        out["latlon_lon_centers"] = grids["lon_centers_2d"]
    if grids["gen_lat_vert"]:
        out["latvert_lat_centers"] = grids["lat_centers_2d"]
    return out


def process_experiment(config: Dict, diag_obs: str) -> Dict:
    """Process OMB/OMA diagnostics for a single observation type, for every
    "da"/"hofx" experiment and cycle."""

    obs_config = get_obs_config(diag_obs)
    bin_method = get_bin_method(diag_obs)

    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs)
    times = generate_cycles(adjusted_start, end_time, interval)
    domains = resolve_domains(config)
    grids = _resolve_stat_grids(config, diag_obs, obs_config, bin_method, file_kind="cycling")
    levels = grids["levels"]

    results = _allocate_results(
        ("omb", "oma"), (), grids,
        num_exp=len(config["experiments"]["names"]), num_dom=len(domains),
        num_time=len(times), num_var=len(obs_config.variables),
    )
    results.update(_metadata(times, levels, obs_config, domains))
    results.update(_grid_metadata(grids))

    for iexp, (exp_name, exp_type, nmember) in enumerate(
        zip(
            config["experiments"]["names"],
            config["experiments"]["types"],
            config["experiments"]["n_member"],
        )
    ):
        if exp_type == "hofx_forecast":
            # OMF (extended-forecast) experiments are processed separately
            # by process_forecast_experiment() -- they don't fit the
            # single-file-per-cycle convention this loop assumes.
            continue

        base_dir = (
            config["paths"]["da_dir"]
            if exp_type == "da"
            else config["paths"]["hofx_dir"]
        )
        outerloop_ana = config["experiments"]["outerloop"][iexp]

        for it, cycle_time in enumerate(times):
            filedict = build_filename(
                base_dir, exp_name, exp_type, nmember, cycle_time, diag_obs
            )

            for iv, diag_var in enumerate(obs_config.variables):
                omb_vec = load_observation_data(
                    filedict["bak"], diag_obs, diag_var, bin_method,
                    outerloop=0, nmember=nmember,
                )
                oma_vec = (
                    load_observation_data(
                        filedict["ana"], diag_obs, diag_var, bin_method,
                        outerloop=outerloop_ana, nmember=nmember,
                    )
                    if filedict["ana"] else None
                )
                if omb_vec is None:
                    continue

                # OMA may come from a separate `_ana.h5` file whose
                # observation ordering/length is not guaranteed to match
                # OMB, so it is always binned with its own coordinates.
                has_oma = oma_vec is not None and oma_vec["innov"] is not None
                omb_bins = _bin_values(omb_vec, bin_method)
                oma_bins = _bin_values(oma_vec, bin_method) if has_oma else None

                # 2-D lat-lon / lat-vert gridded stats (domain-independent)
                _store_2d_stats(results, "omb", (iexp, it), iv, omb_vec,
                                omb_bins, bin_method, grids)
                if has_oma:
                    _store_2d_stats(results, "oma", (iexp, it), iv, oma_vec,
                                    oma_bins, bin_method, grids)

                for idom, (dom_name, dom_coords) in enumerate(domains.items()):
                    omb_qc = _apply_domain_mask(
                        omb_vec["qc"], omb_vec["lat"], omb_vec["lon"],
                        dom_name, dom_coords,
                    )
                    _store_domain_stats(results, "omb", (iexp, idom, it), iv,
                                        omb_vec, omb_qc, omb_bins, levels, bin_method)
                    if has_oma:
                        oma_qc = _apply_domain_mask(
                            oma_vec["qc"], oma_vec["lat"], oma_vec["lon"],
                            dom_name, dom_coords,
                        )
                        _store_domain_stats(results, "oma", (iexp, idom, it), iv,
                                            oma_vec, oma_qc, oma_bins, levels, bin_method)

    return results


# ============================================================================
# OMF (obs-minus-forecast, extended-forecast skill) statistics
#
# Same 0D/1D/2D group convention as process_experiment() /
# write_obs_diag_files() above, but resolved at every (init cycle, forecast
# lead hour) pair instead of just per assimilation cycle -- mirroring how
# the model-space package's scripts/run_verify_model_fcst.ksh points
# src/mpas_stat/cli/model_stats.py at ExtendedFC/<cycle>/mpasout.<valid_date>.nc
# files instead of CyclingFC ones, one call per lead hour.
# ============================================================================

def process_forecast_experiment(config: Dict, diag_obs: str) -> Dict:
    """Process OMF (extended-forecast) diagnostics for a single observation
    type, across every configured forecast lead hour. Only experiments with
    type == "hofx_forecast" contribute -- see build_hofx_forecast_filename()
    for the expected HofX directory layout, and config["forecast"] (length/
    interval/start, same schema as the model-space graphic.yaml) for the
    lead-hour list."""

    obs_config = get_obs_config(diag_obs)
    bin_method = get_bin_method(diag_obs)

    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs, for_forecast=True)
    times = generate_cycles(adjusted_start, end_time, interval)
    leads = generate_forecast_leads(config)
    domains = resolve_domains(config)
    grids = _resolve_stat_grids(config, diag_obs, obs_config, bin_method, file_kind="forecast")
    levels = grids["levels"]

    results = _allocate_results(
        ("omf",), (len(leads),), grids,
        num_exp=len(config["experiments"]["names"]), num_dom=len(domains),
        num_time=len(times), num_var=len(obs_config.variables),
    )
    results.update(_metadata(times, levels, obs_config, domains))
    results["leads"] = leads
    results.update(_grid_metadata(grids))

    hofx_dir = config["paths"].get("hofx_dir")

    for iexp, (exp_name, exp_type, nmember) in enumerate(
        zip(
            config["experiments"]["names"],
            config["experiments"]["types"],
            config["experiments"]["n_member"],
        )
    ):
        if exp_type != "hofx_forecast":
            continue

        for it, init_time in enumerate(times):
            for il, lead in enumerate(leads):

                # The INIT time list above is deliberately unrestricted (see
                # resolve_time_settings()'s for_forecast=True docstring) --
                # a forecast can launch at any hour the model cycles at. What
                # a sensor's cycle_valid_hours actually restricts is whether
                # THIS LEAD's valid time (init_time + lead) lands on an hour
                # that sensor actually observes at (e.g. sondes only launch
                # at 00Z/12Z). A disallowed lead is simply skipped here,
                # leaving its slot at the NaN fill value already in `results`
                # -- same convention as any other kind of missing data.
                if not lead_valid_time_allowed(diag_obs, init_time, lead):
                    continue

                filename = build_hofx_forecast_filename(
                    hofx_dir, exp_name, init_time, lead, diag_obs
                )

                for iv, diag_var in enumerate(obs_config.variables):
                    omf_vec = load_observation_data(
                        filename, diag_obs, diag_var, bin_method,
                        outerloop=0, nmember=nmember,
                    )
                    if omf_vec is None:
                        continue

                    bin_values = _bin_values(omf_vec, bin_method)
                    _store_2d_stats(results, "omf", (iexp, it, il), iv, omf_vec,
                                    bin_values, bin_method, grids)

                    for idom, (dom_name, dom_coords) in enumerate(domains.items()):
                        omf_qc = _apply_domain_mask(
                            omf_vec["qc"], omf_vec["lat"], omf_vec["lon"],
                            dom_name, dom_coords,
                        )
                        _store_domain_stats(results, "omf", (iexp, idom, it, il), iv,
                                            omf_vec, omf_qc, bin_values, levels, bin_method)

    return results


def main(config_path: Optional[str] = None) -> None:
    """Compute observation-space statistics and write diagnosis files."""
    config = load_config(config_path)

    graphics_base_dir = config["paths"].get("graphic_dir", "graphics")
    os.makedirs(graphics_base_dir, exist_ok=True)

    diag_base_dir = resolve_diag_base_dir(config)
    os.makedirs(diag_base_dir, exist_ok=True)

    exp_types = config["experiments"].get("types", [])
    has_cycling = any(t in ("da", "hofx") for t in exp_types)
    has_forecast = "hofx_forecast" in exp_types

    for diag_obs in config["sensors"]:
        # diag_base_dir itself (not a per-sensor subdirectory of it) is
        # passed straight through to write_obs_diag_files() /
        # write_obs_forecast_diag_files() -- the experiment sits at the top
        # of the tree, and the obs type is only ever a filename suffix
        # (obsstat_<obs>.<date>.nc), never its own directory level (see
        # obs_diag_file() / obs_diag_forecast_file() in src/mpas_stat/obs/common.py),
        # so there is no "sensor directory" to pre-create here at all; each
        # out_file's own os.makedirs(os.path.dirname(...)) below handles it.
        if has_cycling:
            adjusted_start, end_time, _ = resolve_time_settings(config, diag_obs)
            print(f"Processing {diag_obs} OMB/OMA from {adjusted_start} to {end_time}")
            results = process_experiment(config, diag_obs)
            write_obs_diag_files(results, config, diag_obs, diag_base_dir)

        if has_forecast:
            leads = generate_forecast_leads(config)
            print(f"Processing {diag_obs} OMF (extended forecast) leads {leads}")
            fresults = process_forecast_experiment(config, diag_obs)
            write_obs_forecast_diag_files(fresults, config, diag_obs, diag_base_dir)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
