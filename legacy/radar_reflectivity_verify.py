#!/usr/bin/env python3
"""
Radar composite-reflectivity forecast verification (FSS / categorical
scores / performance diagram / RMSE vs. forecast lead time).

Optimized version of the original stand-alone script:
  - `calc_composite_reflectivity()`: the per-grid-cell Python loop calling
    `cKDTree.query_ball_point()` one point at a time has been replaced with
    ONE batched radius query (`workers=` lets scipy split that single call
    across threads) plus a fully vectorized `np.maximum.at()` reduction --
    no Python-level loop over grid cells at all.
  - `calc_confidence_bootstrap()`: the 8000-iteration Python `for` loop
    (one `rng.integers()` + `np.mean()` call per iteration) has been
    replaced with a single vectorized resample-and-mean over an
    (n_samples, n) index array. This function is called once per
    (experiment, lead, threshold, radius) combination for every metric, so
    it dominated total runtime in the original script.
  - `calc_categorical_counts()` / `calc_categorical_scores()` shared a
    near-identical convolution body; that's now one internal
    `_categorical_counts()` helper, with `calc_categorical_scores()`
    deriving its scores from it -- same public functions/signatures,
    no duplicated work.
  - The main per-(experiment, lead, cycle) file-processing loop -- the
    actual bulk of the runtime (one HDF5 read + gridding + scoring per
    file, all embarrassingly parallel across files) -- now runs across a
    `ProcessPoolExecutor` instead of serially. Each worker task handles one
    file and returns only the small derived per-cycle quantities (never the
    full obs/hofx arrays or gridded fields), keeping inter-process
    communication cheap. The worker sets the KDTree radius query to a
    single thread (`kdtree_workers=1`) so the outer process pool -- already
    using every core across many independent files -- doesn't get
    oversubscribed by a second, nested layer of threaded parallelism.
  - Aggregation (bootstrap CI over cycles, or accumulated
    hit/miss/false-alarm/correct-negative counts for `aggregated_score`)
    happens in the main process after gathering every worker's result,
    exactly reproducing the original script's math -- only the *order* of
    work changes, not any computation, and bootstrap's mean/percentile
    outputs are invariant to input order.

Plotting is unchanged (it was never the bottleneck) other than being
grouped at the end, after every metric has been computed.
"""
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import partial

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from scipy.signal import convolve2d

# ============================================================
# Plotting helpers (unchanged from the original)
# ============================================================
def plot_time_series(ax, times, data, label, x_label, y_label, main_label, linestyle='-', show_error_bars=True):
    linecolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for iexp in range(len(label)):
        mean_values = data[iexp, :, 0]
        bottom_values = data[iexp, :, 1]
        top_values = data[iexp, :, 2]

        if show_error_bars:
            ax.errorbar(times, mean_values, yerr=[mean_values - bottom_values, top_values - mean_values],
                        label=f'{label[iexp]}', linestyle=linestyle, linewidth=2.5, fmt='o',
                        color=linecolors[iexp % len(linecolors)], capsize=5, capthick=1.5)
        else:
            ax.plot(times, mean_values, linestyle=linestyle, linewidth=2.5, marker='o',
                    label=f'{label[iexp]}', color=linecolors[iexp % len(linecolors)])
            ax.plot(times, mean_values, linestyle=linestyle, linewidth=2.5,
                    color=linecolors[iexp % len(linecolors)])

    ax.set_xlabel(f'{x_label}', fontsize=20)
    ax.set_ylabel(f'{y_label}', fontsize=20)
    ax.set_title(main_label, fontsize=22)

    ax.tick_params(axis='both', which='major', labelsize=20, length=12)

    if not show_error_bars:
        data = data[:, :, 0]

    # Compute max absolute for y-limits in NaN-safe way
    try:
        max_value = np.nanmax(data)
        min_value = np.nanmin(data)
        if np.isnan(max_value) or np.isnan(min_value):
            max_value = 1.0
            min_value = -1.0
        else:
            max_value = float(np.ceil(max_value * 2.0) / 2.0)
            min_value = float(np.floor(min_value * 2.0) / 2.0)
    except Exception:
        max_value = 1.0
        min_value = -1.0

    ax.set_ylim(min_value, max_value)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax.grid(True)

    # Limit x-axis labels (pick evenly spaced ticks, include first tick)
    max_xticks = 8
    num_times = len(times)
    if num_times <= max_xticks:
        xticks = times
    else:
        step = int(np.ceil(num_times / max_xticks))
        xticks = times[::step]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{t}' for t in xticks], fontsize=16, rotation=0)

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(2)


def plot_performance_diagram(ax, pod, far, exp_labels, lead_times, title="Performance Diagram"):
    """
    Plot a Performance Diagram.
    Args:
        pod, far: arrays shaped (n_exp, n_times)
        exp_labels (list): list of experiments
        lead_times (list): list of lead time labels (e.g., ["1h", "3h", "6h"])
        title (str): Plot title
    """
    n_exp = len(exp_labels)

    sr = 1.0 - far

    # vmin/vmax (the eventual axes limits) are computed FIRST, before any
    # decorative artist is drawn, and every decorative artist below (the
    # CSI contour grid, the bias-ratio diagonal lines/labels) is then built
    # to span exactly [vmin, vmax] instead of a fixed [0, 1] range.
    #
    # This matters because matplotlib's tight-bbox measurement (used by
    # savefig(bbox_inches='tight') below) does NOT clip an artist's extent
    # to the axes' visible window -- a Line2D's window extent reflects its
    # full DATA range transformed through the display transform, even for
    # the portion outside [vmin, vmax]. With set_aspect('equal',
    # adjustable='box') and every score clustered within ~1e-4 of a
    # perfect (1, 1) forecast (vmin/vmax collapsing to a razor-thin
    # window), a bias line spanning the old fixed [0, 1] range is ~50-100x
    # wider than the visible window, and that ratio, carried through the
    # display transform, asks the renderer for an enormous pixel buffer --
    # reproducibly a MemoryError/std::bad_alloc, not a cosmetic glitch.
    # Bounding every artist's data range to [vmin, vmax] up front removes
    # the oversized extent at its source, however tightly the scores
    # cluster.
    sr_data = sr.flatten()
    pod_data = pod.flatten()
    valid = np.isfinite(sr_data) & np.isfinite(pod_data)

    if valid.any():
        vmin = max(0.0, min(sr_data[valid].min(), pod_data[valid].min()))
        vmax = min(1.0, max(sr_data[valid].max(), pod_data[valid].max()))
        margin = 0.05 * (vmax - vmin)
        # A minimum margin also keeps the window from collapsing to
        # (numerically) zero width in the first place.
        margin = max(margin, 0.02)
        vmin = max(0.0, vmin - margin)
        vmax = min(1.0, vmax + margin)
    else:
        vmin, vmax = 0.0, 1.0

    sr_grid, pod_grid = np.meshgrid(np.linspace(vmin, vmax, 200),
                                    np.linspace(vmin, vmax, 200))

    with np.errstate(divide='ignore', invalid='ignore'):
        csi = 1.0 / (1.0 / sr_grid + 1.0 / pod_grid - 1.0)
    csi[csi < 0] = np.nan

    csi_levels = np.arange(0.1, 1.0, 0.1)
    csi_contours = ax.contour(sr_grid, pod_grid, csi, levels=csi_levels,
                              colors='purple', linestyles='dotted', linewidths=1.5)
    if csi_contours.allsegs and any(len(seg) for seg in csi_contours.allsegs):
        ax.clabel(csi_contours, inline=True, fmt=lambda x: f"{x:.1f}", fontsize=15)

    bias_values = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5]
    for bias in bias_values:
        sr_line = np.linspace(vmin, vmax, 200)
        pod_line = np.clip(bias * sr_line, vmin, vmax)
        ax.plot(sr_line, pod_line, linestyle='--', color='navy', linewidth=1.5)
        # The label's position is the line's own natural placement (same
        # formula as the un-zoomed [0, 1] full-panel case), NOT clamped
        # into the current [vmin, vmax] window. A zoomed-in view (e.g. once
        # every score is bootstrap-clustered within a fraction of a percent
        # of a perfect forecast) can easily leave a bias ratio's natural
        # label position outside that window -- clamping it to the nearest
        # edge used to pile several rotated labels up right at the axes
        # boundary, where their own rendered footprint (ax.text has
        # clip_on=False by default) spills out into the figure margin as
        # stray floating digits. Only drawing the label when it actually
        # falls inside the visible window (and clip_on=True as a second
        # line of defense for the rare case where it lands right at the
        # edge) keeps every digit inside the panel -- the line itself still
        # draws across the full window either way.
        label_x = min(1.0, 0.8 / bias)
        label_y = min(1.0, bias * label_x)
        if vmin <= label_x <= vmax and vmin <= label_y <= vmax:
            ax.text(label_x, label_y, f"{bias:.2g}", fontsize=15,
                    color='navy', rotation=35, ha='center', va='bottom',
                    clip_on=True)

    markers = ['o', 's', 'D', 'h', '8', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    sizes = [400, 400, 400, 480, 480, 480]

    for iexp, exp in enumerate(exp_labels):
        color = colors[iexp]
        marker = markers[iexp]
        size = sizes[iexp]

        ax.scatter(sr[iexp, :], pod[iexp, :], marker=marker, s=size,
                   color=color, edgecolor='black', label=exp, zorder=3)

        for x, y, label in zip(sr[iexp, :], pod[iexp, :], lead_times):
            ax.text(x, y, label, fontsize=15, ha='center', va='center', color='white',
                    clip_on=True)

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal', adjustable='box')

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel("Success Ratio (1 - FAR)", fontsize=20)
    ax.set_ylabel("Probability of Detection (POD)", fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.grid(True, linestyle=':')

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    return ax

# ============================================================
# Statistical helpers
# ============================================================
def remove_outliers(data, threshold=3):
    data = np.asarray(data)
    clean = data[np.isfinite(data)]
    if clean.size == 0:
        return clean
    mean, std = np.mean(clean), np.std(clean)
    if std == 0:
        return clean
    z = np.abs((clean - mean) / std)
    return clean[z < threshold]


def calc_confidence_bootstrap(data, alpha=0.95, n_samples=8000):
    """(mean, lower, upper) bootstrap CI of the mean.

    Vectorized: draws all `n_samples` resamples as one (n_samples, n) index
    array and reduces with a single `.mean(axis=1)` call, instead of an
    explicit Python loop doing one resample + mean per iteration. This is
    the single hottest function in the original script (called once per
    (experiment, lead, threshold, radius, metric) combination), so this is
    where most of the wall-clock win comes from.
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return (np.nan, np.nan, np.nan)
    data = remove_outliers(data)
    n = data.size
    if n == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng()
    idx = rng.integers(0, n, size=(n_samples, n))
    boot = data[idx].mean(axis=1)
    lower = np.percentile(boot, (1 - alpha) / 2 * 100)
    upper = np.percentile(boot, (alpha + (1 - alpha) / 2) * 100)
    return (np.mean(data), lower, upper)

# ============================================================
# Metric calculations
# ============================================================
def calc_composite_reflectivity(obs_lat, obs_lon, obs_var, hofx_var, sample_bin, kdtree_workers=1):
    """Grid scattered obs/hofx reflectivity onto a regular lat/lon grid,
    taking the max value within `sample_bin` of each cell center (radar
    "composite reflectivity" convention).

    Same result as the original per-cell `tree.query_ball_point()` loop,
    but computed as ONE batched radius query (`cKDTree.query_ball_point`
    accepts an array of query points and an optional `workers=` thread
    count) followed by a vectorized `np.maximum.at()` reduction -- no
    Python-level loop over grid cells.

    `kdtree_workers` defaults to 1 (single-threaded) because this function
    normally runs inside a worker process that is itself one of many
    already saturating the machine's cores (see `_process_one_cycle`) --
    a second, nested layer of threaded parallelism here would oversubscribe
    and typically make things slower, not faster. Pass a larger value (or
    -1 for "all cores") only when calling this function on its own, outside
    that outer parallel pipeline.
    """
    lat_min, lat_max = np.nanmin(obs_lat), np.nanmax(obs_lat)
    lon_min, lon_max = np.nanmin(obs_lon), np.nanmax(obs_lon)
    lat_edges = np.arange(lat_min, lat_max + sample_bin, sample_bin)
    lon_edges = np.arange(lon_min, lon_max + sample_bin, sample_bin)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    ny, nx = len(lat_centers), len(lon_centers)

    # Same flattened ordering as the original's
    # np.meshgrid(..., indexing='ij').reshape(2, -1).T: lat varies slowest.
    grid_lat = np.repeat(lat_centers, nx)
    grid_lon = np.tile(lon_centers, ny)
    grid_points = np.column_stack((grid_lat, grid_lon))

    points = np.column_stack((obs_lat, obs_lon))
    tree = cKDTree(points)
    neighbor_lists = tree.query_ball_point(grid_points, r=sample_bin, workers=kdtree_workers)

    lengths = np.fromiter((len(nb) for nb in neighbor_lists), dtype=np.int64, count=len(neighbor_lists))
    total = int(lengths.sum())

    CY_flat = np.full(ny * nx, np.nan)
    CX_flat = np.full(ny * nx, np.nan)

    if total > 0:
        grid_idx_flat = np.repeat(np.arange(ny * nx), lengths)
        obs_idx_flat = np.fromiter(
            (i for nb in neighbor_lists for i in nb), dtype=np.int64, count=total,
        )
        has_neighbors = lengths > 0

        # np.maximum.at (unlike np.nanmax, which the original used) does NOT
        # skip NaN -- max(x, nan) is nan, so it would propagate a single
        # NaN-valued neighbor into the whole cell's result. Substitute -inf
        # for NaN inputs before reducing (so a NaN neighbor can never win
        # the max unless every neighbor is NaN), then convert any cell whose
        # reduced value is still -inf (no neighbors, or every neighbor was
        # NaN) back to NaN -- reproducing np.nanmax's behavior exactly.
        obs_safe = np.where(np.isfinite(obs_var), obs_var, -np.inf)
        hofx_safe = np.where(np.isfinite(hofx_var), hofx_var, -np.inf)

        CY_acc = np.full(ny * nx, -np.inf)
        CX_acc = np.full(ny * nx, -np.inf)
        np.maximum.at(CY_acc, grid_idx_flat, obs_safe[obs_idx_flat])
        np.maximum.at(CX_acc, grid_idx_flat, hofx_safe[obs_idx_flat])
        CY_flat[has_neighbors] = CY_acc[has_neighbors]
        CX_flat[has_neighbors] = CX_acc[has_neighbors]
        CY_flat[CY_flat == -np.inf] = np.nan
        CX_flat[CX_flat == -np.inf] = np.nan

    return CY_flat.reshape(ny, nx), CX_flat.reshape(ny, nx)


def calc_fss_components(obs, fcst, threshold, radius):
    """
    Returns (fss, num, denom) for one cycle across the whole grid.
    num = sum((fcst_frac - obs_frac)**2)
    denom = sum(fcst_frac**2 + obs_frac**2)
    """
    obs_bin = (obs >= threshold).astype(float)
    fcst_bin = (fcst >= threshold).astype(float)
    mask = np.isfinite(obs_bin) & np.isfinite(fcst_bin)
    obs_bin[~mask] = np.nan
    fcst_bin[~mask] = np.nan
    size = 2 * radius + 1
    obs_filt = uniform_filter(np.nan_to_num(obs_bin), size=size, mode='constant')
    fcst_filt = uniform_filter(np.nan_to_num(fcst_bin), size=size, mode='constant')
    valid = uniform_filter(np.nan_to_num(mask.astype(float)), size=size, mode='constant')
    obs_frac = np.where(valid > 0, obs_filt / valid, np.nan)
    fcst_frac = np.where(valid > 0, fcst_filt / valid, np.nan)
    num = np.nansum((fcst_frac - obs_frac) ** 2)
    denom = np.nansum(fcst_frac ** 2 + obs_frac ** 2)
    fss = 1.0 - num / denom if denom > 0 else np.nan
    return fss, num, denom


def calc_fss(obs, fcst, threshold, radius):
    fss, _, _ = calc_fss_components(obs, fcst, threshold, radius)
    return fss


def _circular_kernel(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2 <= radius ** 2).astype(int)


def _categorical_counts(obs, fcst, threshold, radius, kernel=None):
    """Shared core for calc_categorical_counts()/calc_categorical_scores():
    both used to rebuild the same neighborhood convolution independently.
    `kernel` may be passed in (from _circular_kernel(radius)) to skip
    rebuilding it when the caller already has one for this radius."""
    valid_mask = np.isfinite(obs) & np.isfinite(fcst) & ((obs > 0) | (fcst > 0))
    fcst_mask = (fcst >= threshold) & valid_mask
    obs_mask = (obs >= threshold) & valid_mask

    if kernel is None:
        kernel = _circular_kernel(radius)

    fhits = convolve2d(fcst_mask.astype(int), kernel, mode='same', boundary='fill', fillvalue=0)
    ohits = convolve2d(obs_mask.astype(int), kernel, mode='same', boundary='fill', fillvalue=0)

    hits_mask = ((fcst_mask & (ohits > 0)) | (obs_mask & (fhits > 0)))
    misses_mask = (obs_mask & (fhits == 0) & valid_mask)
    false_alarms_mask = (fcst_mask & (ohits == 0) & valid_mask)
    correct_neg_mask = (~fcst_mask & ~obs_mask & valid_mask)

    na = int(np.sum(hits_mask))
    nb = int(np.sum(misses_mask))
    nc = int(np.sum(false_alarms_mask))
    nd = int(np.sum(correct_neg_mask))
    return na, nb, nc, nd


def calc_categorical_counts(obs, fcst, threshold, radii):
    """Return na, nb, nc, nd counts (integers) for one cycle and a given radius."""
    return _categorical_counts(obs, fcst, threshold, radii)


def _scores_from_counts(na, nb, nc, nd):
    total = na + nb + nc + nd
    ts = na / (na + nb + nc) if (na + nb + nc) > 0 else np.nan
    chance = ((na + nc) * (na + nb)) / total if total > 0 else 0
    denom_ets = (na + nb + nc - chance)
    ets = (na - chance) / denom_ets if denom_ets > 0 else np.nan
    fbias = (na + nc) / (na + nb) if (na + nb) > 0 else np.nan
    pod = na / (na + nb) if (na + nb) > 0 else np.nan
    far = nc / (na + nc) if (na + nc) > 0 else np.nan
    return ts, ets, fbias, pod, far


def calc_categorical_scores(obs, fcst, threshold, radii):
    na, nb, nc, nd = _categorical_counts(obs, fcst, threshold, radii)
    return _scores_from_counts(na, nb, nc, nd)

# ============================================================
# Per-cycle worker (runs in a separate process)
# ============================================================
def _process_one_cycle(fname, thresholds, radii, sample_bin, aggregated_score,
                       diag_var='equivalentReflectivityFactor'):
    """Read one obsout file, grid obs/hofx reflectivity, and compute this
    cycle's verification metrics.

    Designed to run inside a `ProcessPoolExecutor` worker: takes only
    picklable, small arguments and returns only small derived quantities
    (never the raw lat/lon/obs arrays or the gridded fields), so
    inter-process communication stays cheap however many workers run.
    Missing files or read errors return None -- the caller drops them,
    exactly like the original script's `continue` / `warnings.warn`.
    """
    if not os.path.exists(fname):
        return None
    try:
        with h5py.File(fname, 'r') as f:
            lat = f['MetaData']['latitude'][:]
            lon = f['MetaData']['longitude'][:]
            obs = f['ObsValue'][diag_var][:]
            hofx = f['hofx'][diag_var][:]
    except Exception as e:
        warnings.warn(f"Error reading {fname}: {e}")
        return None

    obs_grid, fcst_grid = calc_composite_reflectivity(lat, lon, obs, hofx, sample_bin, kdtree_workers=1)

    omf = np.where((obs_grid > 0.) & (fcst_grid > 0.), obs_grid - fcst_grid, np.nan)
    rmse = float(np.sqrt(np.nanmean(np.square(omf))))

    n_thre, n_rad = len(thresholds), len(radii)
    result = {"rmse": rmse}

    if aggregated_score:
        fss_num = np.zeros((n_thre, n_rad))
        fss_denom = np.zeros((n_thre, n_rad))
        na = np.zeros((n_thre, n_rad), dtype=np.int64)
        nb = np.zeros((n_thre, n_rad), dtype=np.int64)
        nc = np.zeros((n_thre, n_rad), dtype=np.int64)
        nd = np.zeros((n_thre, n_rad), dtype=np.int64)
        for irad, rad in enumerate(radii):
            kernel = _circular_kernel(rad)  # built once per radius, reused across thresholds
            for ithre, thr in enumerate(thresholds):
                _, num, denom = calc_fss_components(obs_grid, fcst_grid, thr, rad)
                fss_num[ithre, irad] = 0.0 if np.isnan(num) else num
                fss_denom[ithre, irad] = 0.0 if np.isnan(denom) else denom
                counts = _categorical_counts(obs_grid, fcst_grid, thr, rad, kernel=kernel)
                na[ithre, irad], nb[ithre, irad], nc[ithre, irad], nd[ithre, irad] = counts
        result.update(fss_num=fss_num, fss_denom=fss_denom, na=na, nb=nb, nc=nc, nd=nd)
    else:
        fss = np.full((n_thre, n_rad), np.nan)
        ts = np.full((n_thre, n_rad), np.nan)
        ets = np.full((n_thre, n_rad), np.nan)
        fbias = np.full((n_thre, n_rad), np.nan)
        pod = np.full((n_thre, n_rad), np.nan)
        far = np.full((n_thre, n_rad), np.nan)
        for irad, rad in enumerate(radii):
            kernel = _circular_kernel(rad)
            for ithre, thr in enumerate(thresholds):
                fss[ithre, irad], _, _ = calc_fss_components(obs_grid, fcst_grid, thr, rad)
                na_, nb_, nc_, nd_ = _categorical_counts(obs_grid, fcst_grid, thr, rad, kernel=kernel)
                (ts[ithre, irad], ets[ithre, irad], fbias[ithre, irad],
                 pod[ithre, irad], far[ithre, irad]) = _scores_from_counts(na_, nb_, nc_, nd_)
        result.update(fss=fss, ts=ts, ets=ets, fbias=fbias, pod=pod, far=far)
    return result

# ============================================================
# Main driver
# ============================================================
def process_data(exp_dir, exps_name, exps_label, thresholds, radii, sample_bin,
                 ini_time, end_time, interval, aggregated_score, fcst_range, fcst_label,
                 n_workers=None):
    """n_workers: max worker processes for the file-processing pool (None ->
    ProcessPoolExecutor's own default, os.cpu_count()). Set to 1 to fall
    back to fully sequential processing (e.g. for debugging)."""

    title_label = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)',
                   '(n)', '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(u)', '(v)', '(w)', '(x)', '(y)', '(z)']
    time_fmt = "%Y%m%d%H"
    dirs = []
    t = ini_time
    while t <= end_time:
        dirs.append(t.strftime(time_fmt))
        t += interval

    n_exp, n_fcst, n_thre, n_rad = len(exps_name), len(fcst_range), len(thresholds), len(radii)
    fss_value = np.full((n_exp, n_fcst, n_thre, n_rad, 3), np.nan)
    ts_value = np.full_like(fss_value, np.nan)
    ets_value = np.full_like(fss_value, np.nan)
    fbias_value = np.full_like(fss_value, np.nan)
    rmse_value = np.full((n_exp, n_fcst, 3), np.nan)

    pod_value = np.full((n_exp, n_fcst, n_thre, n_rad), np.nan)
    far_value = np.full((n_exp, n_fcst, n_thre, n_rad), np.nan)

    # --------------------------------------------------------------
    # Build every (experiment, lead, cycle) file task up front, and run
    # them across a process pool -- each file's read + gridding + scoring
    # is independent of every other, so this is the embarrassingly
    # parallel part of the pipeline and the main source of wall-clock time.
    # --------------------------------------------------------------
    tasks = []  # (iexp, ifcst, fname)
    for ifcst, fcst_hr in enumerate(fcst_range):
        for iexp, exp in enumerate(exps_name):
            for d in dirs:
                fname = f"{exp_dir}/{exp}/HofX/{d}/f{fcst_hr:03d}/dbOut/obsout_hofx_radar_zh.h5"
                tasks.append((iexp, ifcst, fname))

    worker = partial(_process_one_cycle, thresholds=thresholds, radii=radii,
                     sample_bin=sample_bin, aggregated_score=aggregated_score)

    results_by_key = {}  # (iexp, ifcst) -> list of per-cycle result dicts, in completion order
    n_missing = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker, fname): (iexp, ifcst) for iexp, ifcst, fname in tasks}
        for fut in as_completed(futures):
            iexp, ifcst = futures[fut]
            res = fut.result()
            if res is None:
                n_missing += 1
                continue
            results_by_key.setdefault((iexp, ifcst), []).append(res)

    if n_missing:
        print(f"{n_missing} of {len(tasks)} files were missing or unreadable and were skipped.")

    # --------------------------------------------------------------
    # Aggregate each (experiment, lead)'s gathered per-cycle results --
    # identical math to the original sequential script, just fed from the
    # parallel gather above instead of an inline loop. Bootstrap's
    # mean/percentile outputs don't depend on cycle order, so gathering
    # results in whatever order workers finish changes nothing numerically.
    # --------------------------------------------------------------
    for ifcst in range(n_fcst):
        for iexp in range(n_exp):
            cycle_results = results_by_key.get((iexp, ifcst), [])
            if not cycle_results:
                continue

            rmse_cycle = np.array([r["rmse"] for r in cycle_results])
            rmse_value[iexp, ifcst, :] = calc_confidence_bootstrap(rmse_cycle)

            if not aggregated_score:
                fss_cycle_tot = np.stack([r["fss"] for r in cycle_results], axis=0)
                ts_cycle_tot = np.stack([r["ts"] for r in cycle_results], axis=0)
                ets_cycle_tot = np.stack([r["ets"] for r in cycle_results], axis=0)
                fbias_cycle_tot = np.stack([r["fbias"] for r in cycle_results], axis=0)
                pod_cycle_tot = np.stack([r["pod"] for r in cycle_results], axis=0)
                far_cycle_tot = np.stack([r["far"] for r in cycle_results], axis=0)

                for ithre in range(n_thre):
                    for irad in range(n_rad):
                        fss_value[iexp, ifcst, ithre, irad, :] = calc_confidence_bootstrap(
                            fss_cycle_tot[:, ithre, irad])
                        ts_value[iexp, ifcst, ithre, irad, :] = calc_confidence_bootstrap(
                            ts_cycle_tot[:, ithre, irad])
                        ets_value[iexp, ifcst, ithre, irad, :] = calc_confidence_bootstrap(
                            ets_cycle_tot[:, ithre, irad])
                        fbias_value[iexp, ifcst, ithre, irad, :] = calc_confidence_bootstrap(
                            fbias_cycle_tot[:, ithre, irad])
                        pod_value[iexp, ifcst, ithre, irad] = np.nanmean(pod_cycle_tot[:, ithre, irad])
                        far_value[iexp, ifcst, ithre, irad] = np.nanmean(far_cycle_tot[:, ithre, irad])
            else:
                fss_num_acc = np.sum([r["fss_num"] for r in cycle_results], axis=0)
                fss_denom_acc = np.sum([r["fss_denom"] for r in cycle_results], axis=0)
                na_acc = np.sum([r["na"] for r in cycle_results], axis=0)
                nb_acc = np.sum([r["nb"] for r in cycle_results], axis=0)
                nc_acc = np.sum([r["nc"] for r in cycle_results], axis=0)
                nd_acc = np.sum([r["nd"] for r in cycle_results], axis=0)

                for ithre in range(n_thre):
                    for irad in range(n_rad):
                        total_num = fss_num_acc[ithre, irad]
                        total_denom = fss_denom_acc[ithre, irad]
                        agg_fss = 1.0 - total_num / total_denom if total_denom > 0 else np.nan

                        na, nb, nc, nd = (int(na_acc[ithre, irad]), int(nb_acc[ithre, irad]),
                                          int(nc_acc[ithre, irad]), int(nd_acc[ithre, irad]))
                        ts, ets, fbias, pod, far = _scores_from_counts(na, nb, nc, nd)

                        fss_value[iexp, ifcst, ithre, irad, :] = (agg_fss, agg_fss, agg_fss)
                        ts_value[iexp, ifcst, ithre, irad, :] = (ts, ts, ts)
                        ets_value[iexp, ifcst, ithre, irad, :] = (ets, ets, ets)
                        fbias_value[iexp, ifcst, ithre, irad, :] = (fbias, fbias, fbias)
                        pod_value[iexp, ifcst, ithre, irad] = pod
                        far_value[iexp, ifcst, ithre, irad] = far

    # ---------------------- Plotting (unchanged) ----------------------
    if n_thre in (1, 2):
        num_rows, num_cols = 1, n_thre
    elif n_thre in (3, 4):
        num_rows, num_cols = 2, 2
    elif n_thre in (5, 6):
        num_rows, num_cols = 3, 2
    elif n_thre in (7, 8, 9):
        num_rows, num_cols = 3, 3
    else:
        num_rows = int(np.ceil(np.sqrt(n_thre)))
        num_cols = int(np.ceil(n_thre / num_rows))

    metric_labels = ['Fractions skill score', 'Threat score', 'Equitable threat score', 'Frequency bias']
    metric_shortnames = ['fractions_skill_score', 'threat_score', 'equitable_threat_score', 'frequency_bias']
    for metric_name, short_name, metric_data in zip(
        metric_labels, metric_shortnames, [fss_value, ts_value, ets_value, fbias_value]
    ):
        for irad, rad in enumerate(radii):
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 12, num_rows * 8))
            axes = np.atleast_1d(axes).ravel()
            for ithre, thr in enumerate(thresholds):
                x_label = 'Forecast lead time'
                y_label = metric_name
                main_label = f'{title_label[ithre]} Threshold (> {thr} dBZ)'
                ax = axes[ithre]
                plot_time_series(ax, fcst_label, metric_data[:, :, ithre, irad, :], exps_label, x_label, y_label, main_label)
                if ithre == n_thre - 1:
                    ax.legend(fontsize=20)
            plt.savefig(f'refl_fcst_{short_name}_radius{rad}.pdf', bbox_inches='tight')
            plt.close()

    for irad, rad in enumerate(radii):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 12, num_rows * 12))
        axes = np.atleast_1d(axes).ravel()
        for ithre, thr in enumerate(thresholds):
            main_label = f'{title_label[ithre]} Threshold (> {thr} dBZ)' if n_thre > 1 else f'Threshold (> {thr} dBZ)'
            ax = axes[ithre]
            plot_performance_diagram(ax, pod_value[:, :, ithre, irad], far_value[:, :, ithre, irad],
                                     exps_label, fcst_label, title=main_label)
            if ithre == n_thre - 1:
                ax.legend(fontsize=20)
        plt.savefig(f'refl_fcst_performance_diagorm_radius{rad}.pdf', bbox_inches='tight')
        plt.close()

    fig, axes = plt.subplots(1, figsize=(12, 8))
    plot_time_series(axes, fcst_label, rmse_value[:, :, :], exps_label, 'Forecast lead time', 'RMSE (dBZ)', '')
    axes.legend(fontsize=20)
    plt.savefig('refl_fcst_rmse.pdf', bbox_inches='tight')
    plt.close()

# ============================================================
# Run Example
# ============================================================
if __name__ == "__main__":
    exp_dir = '/path/to/regional_mpasjedi'

    exps_name = [
        "exp_control",
        "exp_test",
    ]
    exps_label = ['Control', 'Test']
    exps_tag = 'OLUpdate'

    thresholds = [20, 40]
    radii = [3, 5, 9, 15]
    sample_bin = 0.027  # degrees
    ini_time = datetime(2024, 9, 17, 0)
    end_time = datetime(2024, 9, 27, 0)
    interval = timedelta(hours=6)

    aggregated_score = False

    fcst_range = range(1, 13, 1)
    fcst_label = fcst_range

    # None -> one worker process per CPU core (ProcessPoolExecutor default).
    # Lower this if running alongside other jobs on a shared machine, or set
    # to 1 to fall back to fully sequential processing.
    n_workers = None

    process_data(exp_dir, exps_name, exps_label, thresholds, radii, sample_bin,
                 ini_time, end_time, interval, aggregated_score, fcst_range, fcst_label,
                 n_workers=n_workers)
