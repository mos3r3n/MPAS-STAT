"""Statistics kernels for obs-space innovations: domain/level-binned
(0D/1D) totals and lat-lon / lat-vert gridded (2D) statistics.
"""
from typing import Dict, Optional

import numpy as np

from ..core.grouping import grouped_stats_1d


def calculate_stats(data: np.ndarray, qc: np.ndarray,
                  bin_values: np.ndarray, levels: np.ndarray,
                  bin_method: int, interval: float = None) -> Dict:
    """Calculate statistics for each level.

    For bin_method != 1 (pressure), each `levels[ilev]` is a bin CENTER and
    `interval` is the bin's full width (bot_bin/top_bin = level -+
    interval/2, below) -- so `interval` must match however `levels` itself
    was spaced, or the bins either gap (interval too small -- some
    bin_values fall between bins, silently dropped) or overlap (interval
    too large -- some bin_values double-counted across neighboring bins).
    When not given explicitly, `interval` is derived directly from the
    spacing between consecutive `levels` entries -- get_levels() always
    builds `levels` with np.arange() at a fixed step for bin_method != 1
    (its hardcoded pressure/height/latitude defaults, an explicit
    mpas_stat.obs.sensor_config.ObservationConfig.level_bin_kwargs override, or a
    domain-derived latitude range/width for radiances -- see get_levels()'s
    docstring), so this is always the width that actually produced
    `levels`, rather than a second, independently-hardcoded guess that can
    silently drift out of sync with it (as the old fixed 10.0 (bin_method
    == 3) / 2.0 (bin_method == 2) constants did, e.g. for a sensor whose
    level_bin_kwargs sets bin_width to something else, or radiances' new
    domain-derived width)."""
    if interval is None:
        if bin_method == 1:
            interval = None  # pressure bins use per-level log-midpoint edges instead (below)
        else:
            levels_arr = np.asarray(levels, dtype=np.float64)
            interval = (float(levels_arr[1] - levels_arr[0]) if levels_arr.size >= 2
                       else (10.0 if bin_method == 3 else 2.0))

    results = {
        'rmse': np.zeros(len(levels)+1),
        'mean': np.zeros(len(levels)+1),
        'std': np.zeros(len(levels)+1),
        'count': np.zeros(len(levels)+1),
    }

    # For total
    if qc is not None:
        valid_mask = (qc == 0)
    else:
        # No QC information available for this observation (no
        # EffectiveQC<outerloop> group, not an ensemble ObsError case, and no
        # HofX-missing fallback triggered -- see load_observation_data()) --
        # treat it as "nothing to filter", the same convention
        # compute_stats_2d_latlon()/compute_stats_2d_lat_vert() already use.
        # Before this guard, qc=None fell through to the bare comparison
        # `(None == 0)`, which Python/numpy evaluates as the scalar `False`
        # rather than a boolean array -- `data[False]` silently returns a
        # LENGTH-0 slice (not "keep everything", and no error), so every
        # stat below came out NaN/0 with no crash: this was the exact bug
        # behind an obs type with no EffectiveQC group at all (e.g. sfc)
        # producing an empty diagnosis file.
        valid_mask = np.ones(np.asarray(data).shape, dtype=bool)
    valid_data = data[valid_mask]
    results['rmse'][-1] = np.sqrt(np.nanmean(valid_data**2)) if valid_data.size else np.nan
    results['mean'][-1] = np.nanmean(valid_data) if valid_data.size else np.nan
    results['std'][-1] = np.nanstd(valid_data) if valid_data.size else np.nan
    results['count'][-1] = np.sum(~np.isnan(valid_data))

    for ilev, level in enumerate(levels):
        if bin_method == 1:  # pressure
            log_levels = np.log(levels)
            if ilev == 0:
                top_bin = 200000.
                bot_bin = np.exp((log_levels[ilev] + log_levels[ilev + 1]) / 2.0)
            elif ilev == (len(levels) - 1):
                top_bin = np.exp((log_levels[ilev] + log_levels[ilev - 1]) / 2.0)
                bot_bin = 0.
            else:
                top_bin = np.exp((log_levels[ilev] + log_levels[ilev - 1]) / 2.0)
                bot_bin = np.exp((log_levels[ilev] + log_levels[ilev + 1]) / 2.0)
        else:
            bot_bin = level - interval/2.
            top_bin = level + interval/2.

        mask = (bin_values > bot_bin) & (bin_values <= top_bin) & (valid_mask)
        valid_data = data[mask]

        if np.any(~np.isnan(valid_data)):
            results['rmse'][ilev] = np.sqrt(np.nanmean(valid_data**2))
            results['mean'][ilev] = np.nanmean(valid_data)
            results['std'][ilev] = np.nanstd(valid_data)
            results['count'][ilev] = np.sum(~np.isnan(valid_data))
        else:
            results['rmse'][ilev] = np.nan
            results['mean'][ilev] = np.nan
            results['std'][ilev] = np.nan
            results['count'][ilev] = 0

    return results


def _apply_domain_mask(qc: Optional[np.ndarray], lat: np.ndarray, lon: np.ndarray,
                       domain_name: str, domain_coords) -> Optional[np.ndarray]:
    """Combine one domain's lat/lon selection with an optional QC array.

    Returns a QC-style array (0 == pass, NaN == reject), or None when no
    filtering is needed at all (the global domain AND no QC information).
    The domain selection is applied even when `qc` is None: folding it into
    the QC array must not silently disable it, or a non-global product
    (e.g. NXTro) would be computed from every observation in the file.
    """
    if domain_name == "global":
        return qc

    lat_min, lat_max, lon_min, lon_max = domain_coords
    in_domain = (
        (lat >= lat_min) & (lat <= lat_max) &
        (lon >= lon_min) & (lon <= lon_max)
    )
    if qc is None:
        return np.where(in_domain, 0.0, np.nan)
    return np.where(in_domain, qc, np.nan)


def compute_stats_2d_latlon(values: np.ndarray, qc: Optional[np.ndarray],
                           lat: np.ndarray, lon: np.ndarray,
                           lat_bins: np.ndarray, lon_bins: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Bin scattered obs-space values (OMB or OMA innovations) into a lat-lon
    grid and compute mean/RMS/STD/count per cell.

    Uses MPAS-STAT's own grouped_stats_1d kernel (mpas_stat.model.kernels),
    the same grouped-binning routine MPAS-STAT uses to build its model
    grid-cell "2D lat_lon" statistics -- so obs-space and model-space maps
    are built with identical binning math.

    lat_bins/lon_bins need not cover the whole globe -- see
    mpas_stat.obs.common.resolve_latlon_bounds() for a REGIONAL application.
    Observations outside [lat_bins[0], lat_bins[-1]] x [lon_bins[0],
    lon_bins[-1]] are dropped entirely rather than folded into the edge
    bins, so a regional box's boundary cells aren't contaminated by
    out-of-domain data.
    """
    nlat = len(lat_bins) - 1
    nlon = len(lon_bins) - 1

    in_bounds = (
        (lat >= lat_bins[0]) & (lat <= lat_bins[-1]) &
        (lon >= lon_bins[0]) & (lon <= lon_bins[-1])
    )

    lat_idx = np.clip(np.digitize(lat, lat_bins) - 1, 0, nlat - 1)
    lon_idx = np.clip(np.digitize(lon, lon_bins) - 1, 0, nlon - 1)
    group_ids = (lat_idx * nlon + lon_idx).astype(np.int64)

    if qc is not None:
        masked = np.where(qc == 0, values, np.nan)
    else:
        masked = values
    masked = np.where(in_bounds, masked, np.nan)

    mean, rms, std = grouped_stats_1d(masked, group_ids, nlat * nlon)

    valid = np.isfinite(masked)
    count = np.bincount(group_ids[valid], minlength=nlat * nlon).astype(np.float64)

    shape = (nlat, nlon)
    return {
        "mean": mean.reshape(shape),
        "rmse": rms.reshape(shape),
        "std": std.reshape(shape),
        "count": count.reshape(shape),
    }


def compute_stats_2d_lat_vert(values: np.ndarray, qc: Optional[np.ndarray],
                             bin_values: np.ndarray, lat: np.ndarray,
                             levels: np.ndarray, bin_method: int,
                             lat_bins: np.ndarray, interval: float = None) -> Dict[str, np.ndarray]:
    """
    Bin scattered obs-space values jointly by vertical level (pressure or
    height) AND latitude, producing a (level x latitude) 2-D cross-section --
    the obs-space counterpart of MPAS-STAT's own lat_lev/lat_hgt/lat_pres
    2-D grids.

    Per-level bin edges are computed EXACTLY as in calculate_stats() (log-
    midpoint bins for pressure, +/-interval/2 for height), so a lat_vert
    slice collapsed over all latitude bins reproduces the same per-level
    numbers as the domain="global" 1-D profile -- kept as a cross-check in
    testing. Not meaningful for bin_method == 3 (radiances, whose "level"
    axis already IS latitude) -- callers must skip that case themselves.

    Domain-independent (like compute_stats_2d_latlon): lat_bins spans
    whatever the caller passes in -- see resolve_latlon_bounds() usage in
    process_experiment(), which reuses the SAME latlon_domain-bounded
    latitude range as the lat-lon map for consistency, rather than
    introducing a second, separately-configured latitude box.

    `interval`, when not given explicitly, is derived from `levels`'
    spacing exactly like calculate_stats() -- see that function's
    docstring for why a second, independently-hardcoded guess can
    silently drift out of sync with however `levels` was actually built.
    Never called for bin_method == 3 (radiances) -- see this function's
    "Not meaningful for..." note above -- so there is no domain-derived
    latitude case to handle here.
    """
    if interval is None:
        if bin_method == 2:
            levels_arr = np.asarray(levels, dtype=np.float64)
            interval = float(levels_arr[1] - levels_arr[0]) if levels_arr.size >= 2 else 2.0
        else:
            interval = None

    nlev = len(levels)
    nlat = len(lat_bins) - 1

    valid_mask = (qc == 0) if qc is not None else np.ones_like(values, dtype=bool)

    lat_in_bounds = (lat >= lat_bins[0]) & (lat <= lat_bins[-1])
    lat_idx = np.clip(np.digitize(lat, lat_bins) - 1, 0, nlat - 1)

    mean  = np.full((nlev, nlat), np.nan)
    rmse  = np.full((nlev, nlat), np.nan)
    std   = np.full((nlev, nlat), np.nan)
    count = np.zeros((nlev, nlat))

    log_levels = np.log(levels) if bin_method == 1 else None

    for ilev, level in enumerate(levels):
        if bin_method == 1:  # pressure -- same log-midpoint bins as calculate_stats
            if ilev == 0:
                top_bin = 200000.
                bot_bin = np.exp((log_levels[ilev] + log_levels[ilev + 1]) / 2.0)
            elif ilev == nlev - 1:
                top_bin = np.exp((log_levels[ilev] + log_levels[ilev - 1]) / 2.0)
                bot_bin = 0.
            else:
                top_bin = np.exp((log_levels[ilev] + log_levels[ilev - 1]) / 2.0)
                bot_bin = np.exp((log_levels[ilev] + log_levels[ilev + 1]) / 2.0)
        else:  # height
            bot_bin = level - interval / 2.
            top_bin = level + interval / 2.

        lev_mask = (bin_values > bot_bin) & (bin_values <= top_bin) & valid_mask & lat_in_bounds
        if not np.any(lev_mask):
            continue

        for ilat in range(nlat):
            cell_vals = values[lev_mask & (lat_idx == ilat)]
            if cell_vals.size and np.any(~np.isnan(cell_vals)):
                mean[ilev, ilat]  = np.nanmean(cell_vals)
                rmse[ilev, ilat]  = np.sqrt(np.nanmean(cell_vals**2))
                std[ilev, ilat]   = np.nanstd(cell_vals)
                count[ilev, ilat] = np.sum(~np.isnan(cell_vals))

    return {"mean": mean, "rmse": rmse, "std": std, "count": count}
