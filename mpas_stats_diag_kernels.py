import numpy as np
from typing import Dict, Tuple, Any, Optional

from mpas_stats_diag_interp import (
    apply_interp_by_level,
)

NON3D_VARS = ['U10', 'V10', 'T2', 'Q2', 'Ps']


def make_stat_dict(shape, include_spread: bool) -> Dict[str, np.ndarray]:
    """Allocate arrays for statistics."""
    stats = {
        "mean": np.full(shape, np.nan, dtype=np.float64),
        "rms": np.full(shape, np.nan, dtype=np.float64),
        "std": np.full(shape, np.nan, dtype=np.float64),
    }
    if include_spread:
        stats["spread"] = np.full(shape, np.nan, dtype=np.float64)
    return stats


def grouped_stats_1d(values: np.ndarray, group_ids: np.ndarray, n_groups: int):
    """
    Compute mean, rms, std for 1D values grouped by integer group_ids.
    NaNs are ignored.
    """
    valid = np.isfinite(values)
    if not np.any(valid):
        shape = (n_groups,)
        return (
            np.full(shape, np.nan, dtype=np.float64),
            np.full(shape, np.nan, dtype=np.float64),
            np.full(shape, np.nan, dtype=np.float64),
        )

    gids = group_ids[valid]
    vals = values[valid]

    count = np.bincount(gids, minlength=n_groups).astype(np.float64)
    s1 = np.bincount(gids, weights=vals, minlength=n_groups).astype(np.float64)
    s2 = np.bincount(gids, weights=vals * vals, minlength=n_groups).astype(np.float64)

    mean = np.full(n_groups, np.nan, dtype=np.float64)
    rms = np.full(n_groups, np.nan, dtype=np.float64)
    std = np.full(n_groups, np.nan, dtype=np.float64)

    nonzero = count > 0
    mean[nonzero] = s1[nonzero] / count[nonzero]
    rms[nonzero] = np.sqrt(s2[nonzero] / count[nonzero])

    var = np.full(n_groups, np.nan, dtype=np.float64)
    var[nonzero] = s2[nonzero] / count[nonzero] - mean[nonzero] ** 2
    var[nonzero] = np.maximum(var[nonzero], 0.0)
    std[nonzero] = np.sqrt(var[nonzero])

    return mean, rms, std


def prepare_one_variable_for_stats(
    var_name: str,
    fields: Dict[str, Optional[np.ndarray]],
    shared_coeffs: Dict[str, Dict[str, np.ndarray]],
    use_numba: bool = True,
) -> Dict[str, Any]:
    """
    Prepare all arrays needed for statistics of one variable.

    This version:
    - does NOT build interpolation coefficients here
    - reuses shared coefficients across all variables
    - applies interpolation in parallel over target levels
    """

    init_var = fields["init"]
    fcst_mean = fields["fcst_mean"]
    fcst_sprd = fields["fcst_sprd"]

    diff = fcst_mean - init_var
    is_3d = (diff.ndim == 2)

    diff_hgt = None
    sprd_hgt = None
    diff_pres = None
    sprd_pres = None

    if is_3d:
        coeff_hgt = shared_coeffs["hgt_from_init_z"]
        coeff_pres_init = shared_coeffs["pres_from_init_p"]
        coeff_pres_fcst = shared_coeffs["pres_from_fcst_p_mean"]

        # Height interpolation
        diff_hgt = apply_interp_by_level(diff, coeff_hgt, use_numba=use_numba)

        if fcst_sprd is not None:
            sprd_hgt = apply_interp_by_level(fcst_sprd, coeff_hgt, use_numba=use_numba)

        # Pressure interpolation
        if var_name != "P":
            fcst_mean_pres = apply_interp_by_level(
                fcst_mean, coeff_pres_fcst, use_numba=use_numba
            )
            init_var_pres = apply_interp_by_level(
                init_var, coeff_pres_init, use_numba=use_numba
            )
            diff_pres = fcst_mean_pres - init_var_pres

            if fcst_sprd is not None:
                sprd_pres = apply_interp_by_level(
                    fcst_sprd, coeff_pres_fcst, use_numba=use_numba
                )

    return {
        "var_name": var_name,
        "init": init_var,
        "fcst_mean": fcst_mean,
        "fcst_sprd": fcst_sprd,
        "diff": diff,
        "is_3d": is_3d,
        "diff_hgt": diff_hgt,
        "sprd_hgt": sprd_hgt,
        "diff_pres": diff_pres,
        "sprd_pres": sprd_pres,
    }


def compute_0d_regions(
    diff: np.ndarray,
    spread: Optional[np.ndarray],
    lat: np.ndarray,
    regions: Dict[str, Tuple[float, float]],
    is_3d: bool,
) -> Dict[str, Dict[str, float]]:
    results = {}

    for region_name, (lat_min_r, lat_max_r) in regions.items():
        if region_name.lower() == "global":
            diff_region = diff
            sprd_region = spread
        else:
            region_mask = (lat >= lat_min_r) & (lat < lat_max_r)
            diff_region = diff[region_mask, :] if is_3d else diff[region_mask]
            sprd_region = (
                spread[region_mask, :] if (spread is not None and is_3d)
                else spread[region_mask] if spread is not None
                else None
            )

        results[region_name] = {
            "mean": np.nanmean(diff_region),
            "rms": np.sqrt(np.nanmean(diff_region**2)),
            "std": np.nanstd(diff_region),
        }
        if sprd_region is not None:
            results[region_name]["spread"] = np.nanmean(sprd_region)

    return results


def compute_1d_lat_bin_profile(
    diff: np.ndarray,
    spread: Optional[np.ndarray],
    lat_idx: np.ndarray,
    lat_bins: np.ndarray,
    is_3d: bool,
) -> Dict[str, np.ndarray]:
    nlat = len(lat_bins) - 1
    stats = make_stat_dict((nlat,), spread is not None)

    for i in range(nlat):
        mask = (lat_idx == i)
        if not np.any(mask):
            continue

        diff_bin = diff[mask, :] if is_3d else diff[mask]
        stats["mean"][i] = np.nanmean(diff_bin)
        stats["rms"][i] = np.sqrt(np.nanmean(diff_bin**2))
        stats["std"][i] = np.nanstd(diff_bin)

        if spread is not None:
            sprd_bin = spread[mask, :] if is_3d else spread[mask]
            stats["spread"][i] = np.nanmean(sprd_bin)

    return stats


def compute_1d_vert_profile(
    diff: np.ndarray,
    spread: Optional[np.ndarray],
    lat: np.ndarray,
    regions: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, np.ndarray]]:
    nVertLevels = diff.shape[1]
    results = {}

    for region_name, (lat_min_r, lat_max_r) in regions.items():
        if region_name.lower() == "global":
            diff_region = diff
            sprd_region = spread
        else:
            region_mask = (lat >= lat_min_r) & (lat < lat_max_r)
            diff_region = diff[region_mask, :]
            sprd_region = spread[region_mask, :] if spread is not None else None

        stats = make_stat_dict((nVertLevels,), spread is not None)
        stats["mean"] = np.nanmean(diff_region, axis=0)
        stats["rms"] = np.sqrt(np.nanmean(diff_region**2, axis=0))
        stats["std"] = np.nanstd(diff_region, axis=0)
        if sprd_region is not None:
            stats["spread"] = np.nanmean(sprd_region, axis=0)

        results[f"vert_profile_{region_name}"] = stats

    return results


def compute_1d_vert_bin_profile(
    diff: np.ndarray,
    spread: Optional[np.ndarray],
    lat: np.ndarray,
    level_bins: np.ndarray,
    regions: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, np.ndarray]]:
    nLevelBins = len(level_bins) - 1
    results = {}

    for region_name, (lat_min_r, lat_max_r) in regions.items():
        if region_name.lower() == "global":
            diff_region = diff
            sprd_region = spread
        else:
            region_mask = (lat >= lat_min_r) & (lat < lat_max_r)
            diff_region = diff[region_mask, :]
            sprd_region = spread[region_mask, :] if spread is not None else None

        stats = make_stat_dict((nLevelBins,), spread is not None)

        for k in range(nLevelBins):
            lev_slice = slice(level_bins[k], level_bins[k + 1])
            diff_bin = diff_region[:, lev_slice]
            stats["mean"][k] = np.nanmean(diff_bin)
            stats["rms"][k] = np.sqrt(np.nanmean(diff_bin**2))
            stats["std"][k] = np.nanstd(diff_bin)

            if sprd_region is not None:
                sprd_bin = sprd_region[:, lev_slice]
                stats["spread"][k] = np.nanmean(sprd_bin)

        results[f"vert_bin_profile_{region_name}"] = stats

    return results


def compute_1d_vert_hgt_profile(
    diff_hgt: np.ndarray,
    sprd_hgt: Optional[np.ndarray],
    lat: np.ndarray,
    hgt_levels: np.ndarray,
    regions: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, np.ndarray]]:
    results = {}

    for region_name, (lat_min_r, lat_max_r) in regions.items():
        if region_name.lower() == "global":
            diff_region = diff_hgt
            sprd_region = sprd_hgt
        else:
            region_mask = (lat >= lat_min_r) & (lat < lat_max_r)
            diff_region = diff_hgt[region_mask, :]
            sprd_region = sprd_hgt[region_mask, :] if sprd_hgt is not None else None

        stats = make_stat_dict((len(hgt_levels),), sprd_hgt is not None)
        stats["mean"] = np.nanmean(diff_region, axis=0)
        stats["rms"] = np.sqrt(np.nanmean(diff_region**2, axis=0))
        stats["std"] = np.nanstd(diff_region, axis=0)
        if sprd_region is not None:
            stats["spread"] = np.nanmean(sprd_region, axis=0)

        results[f"vert_hgt_profile_{region_name}"] = stats

    return results


def compute_1d_vert_pres_profile(
    diff_pres: np.ndarray,
    sprd_pres: Optional[np.ndarray],
    lat: np.ndarray,
    pres_levels: np.ndarray,
    regions: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, np.ndarray]]:
    results = {}

    for region_name, (lat_min_r, lat_max_r) in regions.items():
        if region_name.lower() == "global":
            diff_region = diff_pres
            sprd_region = sprd_pres
        else:
            region_mask = (lat >= lat_min_r) & (lat < lat_max_r)
            diff_region = diff_pres[region_mask, :]
            sprd_region = sprd_pres[region_mask, :] if sprd_pres is not None else None

        stats = make_stat_dict((len(pres_levels),), sprd_pres is not None)
        stats["mean"] = np.nanmean(diff_region, axis=0)
        stats["rms"] = np.sqrt(np.nanmean(diff_region**2, axis=0))
        stats["std"] = np.nanstd(diff_region, axis=0)
        if sprd_region is not None:
            stats["spread"] = np.nanmean(sprd_region, axis=0)

        results[f"vert_pres_profile_{region_name}"] = stats

    return results


def compute_2d_lat_lon(
    diff: np.ndarray,
    spread: Optional[np.ndarray],
    group_ids_latlon: np.ndarray,
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
    is_3d: bool,
) -> Dict[str, np.ndarray]:
    nlat = len(lat_bins) - 1
    nlon = len(lon_bins) - 1
    ngroups = nlat * nlon

    stats = make_stat_dict((nlat, nlon), spread is not None)

    diff_agg = np.nanmean(diff, axis=1) if is_3d else diff
    mean, rms, std = grouped_stats_1d(diff_agg, group_ids_latlon, ngroups)
    stats["mean"][:, :] = mean.reshape(nlat, nlon)
    stats["rms"][:, :] = rms.reshape(nlat, nlon)
    stats["std"][:, :] = std.reshape(nlat, nlon)

    if spread is not None:
        sprd_agg = np.nanmean(spread, axis=1) if is_3d else spread
        sprd_mean, _, _ = grouped_stats_1d(sprd_agg, group_ids_latlon, ngroups)
        stats["spread"][:, :] = sprd_mean.reshape(nlat, nlon)

    return stats


def compute_2d_lat_lev(
    diff: np.ndarray,
    spread: Optional[np.ndarray],
    lat_idx: np.ndarray,
    lat_bins: np.ndarray,
    level_bins: np.ndarray,
) -> Dict[str, np.ndarray]:
    nlat = len(lat_bins) - 1
    nLevelBins = len(level_bins) - 1

    stats = make_stat_dict((nlat, nLevelBins), spread is not None)

    for k in range(nLevelBins):
        lev_slice = slice(level_bins[k], level_bins[k + 1])

        diff_k = np.nanmean(diff[:, lev_slice], axis=1)
        mean, rms, std = grouped_stats_1d(diff_k, lat_idx, nlat)

        stats["mean"][:, k] = mean
        stats["rms"][:, k] = rms
        stats["std"][:, k] = std

        if spread is not None:
            sprd_k = np.nanmean(spread[:, lev_slice], axis=1)
            sprd_mean, _, _ = grouped_stats_1d(sprd_k, lat_idx, nlat)
            stats["spread"][:, k] = sprd_mean

    return stats


def compute_2d_lat_hgt(
    diff_hgt: np.ndarray,
    sprd_hgt: Optional[np.ndarray],
    lat_idx: np.ndarray,
    lat_bins: np.ndarray,
    hgt_levels: np.ndarray,
) -> Dict[str, np.ndarray]:
    nlat = len(lat_bins) - 1

    stats = make_stat_dict((nlat, len(hgt_levels)), sprd_hgt is not None)

    for k in range(len(hgt_levels)):
        diff_k = diff_hgt[:, k]
        mean, rms, std = grouped_stats_1d(diff_k, lat_idx, nlat)

        stats["mean"][:, k] = mean
        stats["rms"][:, k] = rms
        stats["std"][:, k] = std

        if sprd_hgt is not None:
            sprd_k = sprd_hgt[:, k]
            sprd_mean, _, _ = grouped_stats_1d(sprd_k, lat_idx, nlat)
            stats["spread"][:, k] = sprd_mean

    return stats


def compute_2d_lat_pres(
    diff_pres: np.ndarray,
    sprd_pres: Optional[np.ndarray],
    lat_idx: np.ndarray,
    lat_bins: np.ndarray,
    pres_levels: np.ndarray,
) -> Dict[str, np.ndarray]:
    nlat = len(lat_bins) - 1

    stats = make_stat_dict((nlat, len(pres_levels)), sprd_pres is not None)

    for k in range(len(pres_levels)):
        diff_k = diff_pres[:, k]
        mean, rms, std = grouped_stats_1d(diff_k, lat_idx, nlat)

        stats["mean"][:, k] = mean
        stats["rms"][:, k] = rms
        stats["std"][:, k] = std

        if sprd_pres is not None:
            sprd_k = sprd_pres[:, k]
            sprd_mean, _, _ = grouped_stats_1d(sprd_k, lat_idx, nlat)
            stats["spread"][:, k] = sprd_mean

    return stats


def compute_3d_lat_lon_lev(
    diff: np.ndarray,
    spread: Optional[np.ndarray],
    group_ids_latlon: np.ndarray,
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
    level_bins: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Third dimension is level bins, not original model levels.
    Shape = (nLatBins, nLonBins, nLevelBins)
    """
    nlat = len(lat_bins) - 1
    nlon = len(lon_bins) - 1
    nLevelBins = len(level_bins) - 1
    ngroups = nlat * nlon

    stats = make_stat_dict((nlat, nlon, nLevelBins), spread is not None)

    for k in range(nLevelBins):
        lev_slice = slice(level_bins[k], level_bins[k + 1])

        diff_k = np.nanmean(diff[:, lev_slice], axis=1)
        mean, rms, std = grouped_stats_1d(diff_k, group_ids_latlon, ngroups)

        stats["mean"][:, :, k] = mean.reshape(nlat, nlon)
        stats["rms"][:, :, k] = rms.reshape(nlat, nlon)
        stats["std"][:, :, k] = std.reshape(nlat, nlon)

        if spread is not None:
            sprd_k = np.nanmean(spread[:, lev_slice], axis=1)
            sprd_mean, _, _ = grouped_stats_1d(sprd_k, group_ids_latlon, ngroups)
            stats["spread"][:, :, k] = sprd_mean.reshape(nlat, nlon)

    return stats


def compute_3d_lat_lon_hgt(
    diff_hgt: np.ndarray,
    sprd_hgt: Optional[np.ndarray],
    group_ids_latlon: np.ndarray,
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
    hgt_levels: np.ndarray,
) -> Dict[str, np.ndarray]:
    nlat = len(lat_bins) - 1
    nlon = len(lon_bins) - 1
    ngroups = nlat * nlon

    stats = make_stat_dict((nlat, nlon, len(hgt_levels)), sprd_hgt is not None)

    for k in range(len(hgt_levels)):
        diff_k = diff_hgt[:, k]
        mean, rms, std = grouped_stats_1d(diff_k, group_ids_latlon, ngroups)

        stats["mean"][:, :, k] = mean.reshape(nlat, nlon)
        stats["rms"][:, :, k] = rms.reshape(nlat, nlon)
        stats["std"][:, :, k] = std.reshape(nlat, nlon)

        if sprd_hgt is not None:
            sprd_k = sprd_hgt[:, k]
            sprd_mean, _, _ = grouped_stats_1d(sprd_k, group_ids_latlon, ngroups)
            stats["spread"][:, :, k] = sprd_mean.reshape(nlat, nlon)

    return stats


def compute_3d_lat_lon_pres(
    diff_pres: np.ndarray,
    sprd_pres: Optional[np.ndarray],
    group_ids_latlon: np.ndarray,
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
    pres_levels: np.ndarray,
) -> Dict[str, np.ndarray]:
    nlat = len(lat_bins) - 1
    nlon = len(lon_bins) - 1
    ngroups = nlat * nlon

    stats = make_stat_dict((nlat, nlon, len(pres_levels)), sprd_pres is not None)

    for k in range(len(pres_levels)):
        diff_k = diff_pres[:, k]
        mean, rms, std = grouped_stats_1d(diff_k, group_ids_latlon, ngroups)

        stats["mean"][:, :, k] = mean.reshape(nlat, nlon)
        stats["rms"][:, :, k] = rms.reshape(nlat, nlon)
        stats["std"][:, :, k] = std.reshape(nlat, nlon)

        if sprd_pres is not None:
            sprd_k = sprd_pres[:, k]
            sprd_mean, _, _ = grouped_stats_1d(sprd_k, group_ids_latlon, ngroups)
            stats["spread"][:, :, k] = sprd_mean.reshape(nlat, nlon)

    return stats


def calculate_statistics_one_task(
    var_name: str,
    task_name: str,
    prepared: Dict[str, Any],
    meta: Dict[str, Any],
    config,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Compute one statistic product for one variable from cached arrays.
    Returns (var_name, dimension_group, result_dict)
    """
    diff = prepared["diff"]
    spread = prepared["fcst_sprd"]
    diff_hgt = prepared["diff_hgt"]
    sprd_hgt = prepared["sprd_hgt"]
    diff_pres = prepared["diff_pres"]
    sprd_pres = prepared["sprd_pres"]
    is_3d = prepared["is_3d"]

    lat = meta["lat"]
    lat_bins = meta["lat_bins"]
    lon_bins = meta["lon_bins"]
    lat_idx = meta["lat_idx"]
    group_ids_latlon = meta["group_ids_latlon"]
    level_bins = meta["level_bins"]

    hgt_levels = np.asarray(config.hgt_levels, dtype=np.float64)
    pres_levels = np.asarray(config.pres_levels, dtype=np.float64)

    if task_name == "0D_regions":
        return var_name, "0D", compute_0d_regions(diff, spread, lat, config.regions, is_3d)

    if task_name == "1D_lat_bin_profile":
        return var_name, "1D", {
            "lat_bin_profile": compute_1d_lat_bin_profile(diff, spread, lat_idx, lat_bins, is_3d)
        }

    if task_name == "1D_vert_profile":
        return var_name, "1D", compute_1d_vert_profile(diff, spread, lat, config.regions)

    if task_name == "1D_vert_bin_profile":
        return var_name, "1D", compute_1d_vert_bin_profile(diff, spread, lat, level_bins, config.regions)

    if task_name == "1D_vert_hgt_profile":
        return var_name, "1D", compute_1d_vert_hgt_profile(diff_hgt, sprd_hgt, lat, hgt_levels, config.regions)

    if task_name == "1D_vert_pres_profile":
        return var_name, "1D", compute_1d_vert_pres_profile(diff_pres, sprd_pres, lat, pres_levels, config.regions)

    if task_name == "2D_lat_lon":
        return var_name, "2D", {
            "lat_lon": compute_2d_lat_lon(diff, spread, group_ids_latlon, lat_bins, lon_bins, is_3d)
        }

    if task_name == "2D_lat_lev":
        return var_name, "2D", {
            "lat_lev": compute_2d_lat_lev(diff, spread, lat_idx, lat_bins, level_bins)
        }

    if task_name == "2D_lat_hgt":
        return var_name, "2D", {
            "lat_hgt": compute_2d_lat_hgt(diff_hgt, sprd_hgt, lat_idx, lat_bins, hgt_levels)
        }

    if task_name == "2D_lat_pres":
        return var_name, "2D", {
            "lat_pres": compute_2d_lat_pres(diff_pres, sprd_pres, lat_idx, lat_bins, pres_levels)
        }

    if task_name == "3D_lat_lon_lev":
        return var_name, "3D", {
            "lat_lon_lev": compute_3d_lat_lon_lev(diff, spread, group_ids_latlon, lat_bins, lon_bins, level_bins)
        }

    if task_name == "3D_lat_lon_hgt":
        return var_name, "3D", {
            "lat_lon_hgt": compute_3d_lat_lon_hgt(diff_hgt, sprd_hgt, group_ids_latlon, lat_bins, lon_bins, hgt_levels)
        }

    if task_name == "3D_lat_lon_pres":
        return var_name, "3D", {
            "lat_lon_pres": compute_3d_lat_lon_pres(diff_pres, sprd_pres, group_ids_latlon, lat_bins, lon_bins, pres_levels)
        }

    raise ValueError(f"Unknown task name: {task_name}")


def build_task_list(
    prepared_cache: Dict[str, Dict[str, Any]],
    config,
):
    """Build one task per statistic product."""
    tasks = []
    expected_dim_groups = {}

    for var_name in config.variables:
        prepared = prepared_cache[var_name]
        expected_dim_groups[var_name] = {"0D", "1D", "2D"}

        tasks.append((var_name, "0D_regions"))
        tasks.append((var_name, "1D_lat_bin_profile"))
        tasks.append((var_name, "2D_lat_lon"))

        if prepared["is_3d"]:
            tasks.append((var_name, "1D_vert_profile"))
            tasks.append((var_name, "1D_vert_bin_profile"))
            tasks.append((var_name, "1D_vert_hgt_profile"))
            tasks.append((var_name, "2D_lat_lev"))
            tasks.append((var_name, "2D_lat_hgt"))

            if prepared["diff_pres"] is not None:
                tasks.append((var_name, "1D_vert_pres_profile"))
                tasks.append((var_name, "2D_lat_pres"))

            if var_name not in NON3D_VARS:
                expected_dim_groups[var_name].add("3D")
                tasks.append((var_name, "3D_lat_lon_lev"))
                tasks.append((var_name, "3D_lat_lon_hgt"))
                if prepared["diff_pres"] is not None:
                    tasks.append((var_name, "3D_lat_lon_pres"))

    return tasks, expected_dim_groups
