import numpy as np
from typing import Dict, Tuple, Any, Optional

from ..core.grouping import grouped_stats_1d, grouped_weighted_stats
from .interp import (
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


def _region_subset(
    diff: np.ndarray,
    spread: Optional[np.ndarray],
    lat: np.ndarray,
    region_name: str,
    lat_min_r: float,
    lat_max_r: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Rows (cells) of diff/spread inside one latitude region; "global" is all rows."""
    if region_name.lower() == "global":
        return diff, spread
    region_mask = (lat >= lat_min_r) & (lat < lat_max_r)
    return diff[region_mask, :], (spread[region_mask, :] if spread is not None else None)


def _regional_column_profiles(
    diff: np.ndarray,
    spread: Optional[np.ndarray],
    lat: np.ndarray,
    nlev: int,
    regions: Dict[str, Tuple[float, float]],
    key_prefix: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-region mean/rms/std (and spread) of every column of a
    (nCells, nlev) field -- model levels, or values already interpolated to
    height or pressure levels. Keys are f"{key_prefix}_{region_name}"."""
    results = {}

    for region_name, (lat_min_r, lat_max_r) in regions.items():
        diff_region, sprd_region = _region_subset(diff, spread, lat, region_name, lat_min_r, lat_max_r)

        stats = make_stat_dict((nlev,), spread is not None)
        stats["mean"] = np.nanmean(diff_region, axis=0)
        stats["rms"] = np.sqrt(np.nanmean(diff_region**2, axis=0))
        stats["std"] = np.nanstd(diff_region, axis=0)
        if sprd_region is not None:
            stats["spread"] = np.nanmean(sprd_region, axis=0)

        results[f"{key_prefix}_{region_name}"] = stats

    return results


def _grouped_column_stats(
    diff: np.ndarray,
    spread: Optional[np.ndarray],
    group_ids: np.ndarray,
    group_shape: Tuple[int, ...],
    nlev: int,
) -> Dict[str, np.ndarray]:
    """Grouped (lat bins, or lat-lon boxes) mean/rms/std (and spread) of each
    column k of a (nCells, nlev) field. Output arrays are group_shape + (nlev,)."""
    ngroups = int(np.prod(group_shape))
    stats = make_stat_dict(tuple(group_shape) + (nlev,), spread is not None)

    for k in range(nlev):
        mean, rms, std = grouped_stats_1d(diff[:, k], group_ids, ngroups)

        stats["mean"][..., k] = mean.reshape(group_shape)
        stats["rms"][..., k] = rms.reshape(group_shape)
        stats["std"][..., k] = std.reshape(group_shape)

        if spread is not None:
            sprd_mean, _, _ = grouped_stats_1d(spread[:, k], group_ids, ngroups)
            stats["spread"][..., k] = sprd_mean.reshape(group_shape)

    return stats


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
    return _regional_column_profiles(diff, spread, lat, diff.shape[1], regions, "vert_profile")


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
        diff_region, sprd_region = _region_subset(diff, spread, lat, region_name, lat_min_r, lat_max_r)

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
    return _regional_column_profiles(diff_hgt, sprd_hgt, lat, len(hgt_levels), regions, "vert_hgt_profile")


def compute_1d_vert_pres_profile(
    diff_pres: np.ndarray,
    sprd_pres: Optional[np.ndarray],
    lat: np.ndarray,
    pres_levels: np.ndarray,
    regions: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, np.ndarray]]:
    return _regional_column_profiles(diff_pres, sprd_pres, lat, len(pres_levels), regions, "vert_pres_profile")


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
    return _grouped_column_stats(diff_hgt, sprd_hgt, lat_idx, (len(lat_bins) - 1,), len(hgt_levels))


def compute_2d_lat_pres(
    diff_pres: np.ndarray,
    sprd_pres: Optional[np.ndarray],
    lat_idx: np.ndarray,
    lat_bins: np.ndarray,
    pres_levels: np.ndarray,
) -> Dict[str, np.ndarray]:
    return _grouped_column_stats(diff_pres, sprd_pres, lat_idx, (len(lat_bins) - 1,), len(pres_levels))


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
    return _grouped_column_stats(diff_hgt, sprd_hgt, group_ids_latlon,
                                 (len(lat_bins) - 1, len(lon_bins) - 1), len(hgt_levels))


def compute_3d_lat_lon_pres(
    diff_pres: np.ndarray,
    sprd_pres: Optional[np.ndarray],
    group_ids_latlon: np.ndarray,
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
    pres_levels: np.ndarray,
) -> Dict[str, np.ndarray]:
    return _grouped_column_stats(diff_pres, sprd_pres, group_ids_latlon,
                                 (len(lat_bins) - 1, len(lon_bins) - 1), len(pres_levels))


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


# ===========================================================================
# Weighted, pooled statistics (statistics_method: "weighted", the default)
#
# Every product pools ALL samples that fall into one output cell -- grid
# cells in a latitude band / lat-lon box / region and, for products that
# span several model levels, every level in the column or level bin -- into
# a single weighted mean / RMS / STD:
#
#   * horizontal weight: MPAS cell area (areaCell), so refined regions of a
#     variable-resolution mesh do not dominate;
#   * vertical weight (model-level products only): layer pressure thickness
#     (mass), so thin upper layers do not dominate a column or level bin.
#
# Squares are pooled before the square root, so a 2-D lat-lon "RMS" of a
# 3-D variable is the RMS over the column, not the RMS of the column-mean
# (signed) error, and ensemble spread is pooled as sqrt(mean variance).
# Each product also stores its total "weight", used by the graphics stage to
# pool statistics across cycles.
# ===========================================================================

def _level_bin_index(level_bins: np.ndarray, n_levels: int) -> np.ndarray:
    """Level-bin index (0..nbins-1) of each model level 0..n_levels-1."""
    nbins = len(level_bins) - 1
    idx = np.searchsorted(level_bins, np.arange(n_levels), side="right") - 1
    return np.clip(idx, 0, nbins - 1)


def _pooled(values, weights, group_ids, shape, spread=None) -> Dict[str, np.ndarray]:
    """grouped_weighted_stats reshaped to the product's output shape."""
    n_groups = int(np.prod(shape)) if len(shape) else 1
    stats = grouped_weighted_stats(values, weights, group_ids, n_groups, spread=spread)
    if not len(shape):
        return {k: float(v[0]) for k, v in stats.items()}
    return {k: v.reshape(shape) for k, v in stats.items()}


def _region_cells(lat: np.ndarray, region_name: str, bounds) -> np.ndarray:
    if region_name.lower() == "global":
        return np.ones(lat.shape, dtype=bool)
    lat_min_r, lat_max_r = bounds
    return (lat >= lat_min_r) & (lat < lat_max_r)


def calculate_statistics_one_task_weighted(
    var_name: str,
    task_name: str,
    prepared: Dict[str, Any],
    meta: Dict[str, Any],
    config,
) -> Tuple[str, str, Dict[str, Any]]:
    """Weighted/pooled counterpart of calculate_statistics_one_task()."""
    diff = prepared["diff"]
    spread = prepared["fcst_sprd"]
    is_3d = prepared["is_3d"]

    lat = meta["lat"]
    area = meta["cell_weight"]                       # (nCells,)
    lat_idx = meta["lat_idx"]
    gid_ll = meta["group_ids_latlon"]
    nlat = len(meta["lat_bins"]) - 1
    nlon = len(meta["lon_bins"]) - 1
    level_bins = meta["level_bins"]
    nlb = len(level_bins) - 1

    if is_3d:
        nlev = diff.shape[1]
        w_col = area[:, None] * meta["level_weight"]  # area x layer mass
        w_lev = np.broadcast_to(area[:, None], diff.shape)
        lev = np.broadcast_to(np.arange(nlev)[None, :], diff.shape)
        lbin = np.broadcast_to(_level_bin_index(level_bins, nlev)[None, :], diff.shape)
        col = lambda ids: np.broadcast_to(np.asarray(ids)[:, None], diff.shape)  # noqa: E731
    else:
        w_col = area
        col = lambda ids: ids  # noqa: E731

    def interp_product(field, sprd, levels, grouping):
        """Per-level products on height/pressure levels (area weights only)."""
        nk = len(levels)
        k = np.broadcast_to(np.arange(nk)[None, :], field.shape)
        w = np.broadcast_to(area[:, None], field.shape)
        if grouping == "lat":
            return _pooled(field, w, lat_idx[:, None] * nk + k, (nlat, nk), sprd)
        return _pooled(field, w, gid_ll[:, None] * nk + k, (nlat, nlon, nk), sprd)

    def regional_levels(field, sprd, n_out, level_ids, weights, prefix):
        out = {}
        for region_name, bounds in config.regions.items():
            cells = _region_cells(lat, region_name, bounds)
            out[f"{prefix}_{region_name}"] = _pooled(
                field[cells], weights[cells], level_ids[cells], (n_out,),
                sprd[cells] if sprd is not None else None)
        return out

    hgt_levels = np.asarray(config.hgt_levels, dtype=np.float64)
    pres_levels = np.asarray(config.pres_levels, dtype=np.float64)

    if task_name == "0D_regions":
        out = {}
        for region_name, bounds in config.regions.items():
            cells = _region_cells(lat, region_name, bounds)
            out[region_name] = _pooled(
                diff[cells], w_col[cells], None, (),
                spread[cells] if spread is not None else None)
        return var_name, "0D", out

    if task_name == "1D_lat_bin_profile":
        return var_name, "1D", {
            "lat_bin_profile": _pooled(diff, w_col, col(lat_idx), (nlat,), spread)}

    if task_name == "1D_vert_profile":
        return var_name, "1D", regional_levels(diff, spread, nlev, lev, w_lev, "vert_profile")

    if task_name == "1D_vert_bin_profile":
        return var_name, "1D", regional_levels(diff, spread, nlb, lbin, w_col, "vert_bin_profile")

    if task_name in ("1D_vert_hgt_profile", "1D_vert_pres_profile"):
        hgt = task_name == "1D_vert_hgt_profile"
        field = prepared["diff_hgt"] if hgt else prepared["diff_pres"]
        sprd = prepared["sprd_hgt"] if hgt else prepared["sprd_pres"]
        nk = len(hgt_levels if hgt else pres_levels)
        k = np.broadcast_to(np.arange(nk)[None, :], field.shape)
        w = np.broadcast_to(area[:, None], field.shape)
        prefix = "vert_hgt_profile" if hgt else "vert_pres_profile"
        return var_name, "1D", regional_levels(field, sprd, nk, k, w, prefix)

    if task_name == "2D_lat_lon":
        return var_name, "2D", {
            "lat_lon": _pooled(diff, w_col, col(gid_ll), (nlat, nlon), spread)}

    if task_name == "2D_lat_lev":
        return var_name, "2D", {
            "lat_lev": _pooled(diff, w_col, col(lat_idx) * nlb + lbin, (nlat, nlb), spread)}

    if task_name == "2D_lat_hgt":
        return var_name, "2D", {
            "lat_hgt": interp_product(prepared["diff_hgt"], prepared["sprd_hgt"], hgt_levels, "lat")}

    if task_name == "2D_lat_pres":
        return var_name, "2D", {
            "lat_pres": interp_product(prepared["diff_pres"], prepared["sprd_pres"], pres_levels, "lat")}

    if task_name == "3D_lat_lon_lev":
        return var_name, "3D", {
            "lat_lon_lev": _pooled(diff, w_col, col(gid_ll) * nlb + lbin, (nlat, nlon, nlb), spread)}

    if task_name == "3D_lat_lon_hgt":
        return var_name, "3D", {
            "lat_lon_hgt": interp_product(prepared["diff_hgt"], prepared["sprd_hgt"], hgt_levels, "latlon")}

    if task_name == "3D_lat_lon_pres":
        return var_name, "3D", {
            "lat_lon_pres": interp_product(prepared["diff_pres"], prepared["sprd_pres"], pres_levels, "latlon")}

    raise ValueError(f"Unknown task name: {task_name}")
