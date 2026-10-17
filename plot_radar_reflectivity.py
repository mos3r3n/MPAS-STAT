#!/usr/bin/env python3
"""
Radar composite-reflectivity forecast verification -- graphics stage.

Reads back the per-(experiment, init cycle, lead) diagnosis netCDF files
diag_radar_reflectivity.py wrote, aggregates across init cycles for each
(experiment, lead) pair -- via radar_verify_common.aggregate_radar_cycles(),
either bootstrapping each cycle's own derived score or summing raw
accumulator components first, per config["radar"]["aggregated_score"] -- and
draws the same three figure families as the standalone
radar_reflectivity_verify.py script this product was integrated from:
  - FSS / TS / ETS / FBIAS vs. forecast lead time, one panel per threshold,
    one figure per neighborhood radius.
  - Performance Diagram (POD vs. 1-FAR), one panel per threshold, one
    figure per radius, points labeled by lead time.
  - RMSE vs. forecast lead time (one figure).

Only "hofx_forecast"-typed experiments are plotted here (this product is
OMF-only) -- experiment selection reuses plot_ioda_obsout_stats.py's own
_build_forecast_experiments() so the two OMF products (point-wise and
radar) always agree on which experiments/labels/colors represent "the
forecast experiments" for a given config.

Parallelism is per-SENSOR (same convention as plot_ioda_obsout_stats.py's
main()): each worker collects and renders every figure for one radar
sensor.

Usage:
    python plot_radar_reflectivity.py [config.yaml]
"""
import os
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obsstat_common import (
    load_config, resolve_diag_base_dir, generate_cycles, generate_forecast_leads,
    resolve_time_settings, lead_valid_time_allowed, radar_composite_diag_file,
    radar_diag_sensor_has_files,
)
from plot_ioda_obsout_stats import _build_forecast_experiments
from radar_verify_common import resolve_radar_settings, aggregate_radar_cycles

# ============================================================================
# Plotting helpers (unchanged from the standalone radar_reflectivity_verify.py
# script this product was integrated from -- see that file for the original,
# non-integrated CLI entry point).
# ============================================================================
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
    """pod, far: arrays shaped (n_exp, n_times)."""
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
        color = colors[iexp % len(colors)]
        marker = markers[iexp % len(markers)]
        size = sizes[iexp % len(sizes)]

        ax.scatter(sr[iexp, :], pod[iexp, :], marker=marker, s=size,
                   color=color, edgecolor='black', label=exp, zorder=3)

        for x, y, lbl in zip(sr[iexp, :], pod[iexp, :], lead_times):
            ax.text(x, y, lbl, fontsize=15, ha='center', va='center', color='white',
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

# ============================================================================
# Diagnosis-file reader (statistics-stage counterpart:
# diag_radar_reflectivity._write_radar_diag_file())
# ============================================================================
def _read_radar_diag_file(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with nc.Dataset(path, "r") as ds:
            return dict(
                rmse=float(np.asarray(ds.variables["RMSE"][...])),
                fss=np.array(ds.variables["FSS"][:]),
                ts=np.array(ds.variables["TS"][:]),
                ets=np.array(ds.variables["ETS"][:]),
                fbias=np.array(ds.variables["FBIAS"][:]),
                pod=np.array(ds.variables["POD"][:]),
                far=np.array(ds.variables["FAR"][:]),
                fss_num=np.array(ds.variables["FSS_num"][:]),
                fss_denom=np.array(ds.variables["FSS_denom"][:]),
                na=np.array(ds.variables["NA"][:]),
                nb=np.array(ds.variables["NB"][:]),
                nc=np.array(ds.variables["NC"][:]),
                nd=np.array(ds.variables["ND"][:]),
            )
    except Exception as e:
        warnings.warn(f"Error reading {path}: {e}")
        return None


def _collect_cycle_results(diag_base_dir: str, exp_name: str, diag_obs: str,
                          cycles: List, lead: int) -> List[Dict]:
    out = []
    for init_time in cycles:
        if not lead_valid_time_allowed(diag_obs, init_time, lead):
            continue
        path = radar_composite_diag_file(diag_base_dir, exp_name, diag_obs, init_time, lead)
        result = _read_radar_diag_file(path)
        if result is not None:
            out.append(result)
    return out

# ============================================================================
# Figure generation for one sensor
# ============================================================================
def _plot_radar_figures(out_dir: str, fig_fmt: str, exp_labels: List[str], leads: List[int],
                        thresholds: List[float], radii: List[int],
                        fss_value, ts_value, ets_value, fbias_value, rmse_value,
                        pod_value, far_value) -> None:
    """Draws the same figures as the standalone script's process_data(),
    unchanged plotting logic, just parameterized by output dir/format."""
    n_thre = len(thresholds)
    title_label = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)',
                   '(n)', '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(u)', '(v)', '(w)', '(x)', '(y)', '(z)']

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
                main_label = f'{title_label[ithre]} Threshold (> {thr} dBZ)'
                ax = axes[ithre]
                plot_time_series(ax, leads, metric_data[:, :, ithre, irad, :], exp_labels,
                                 'Forecast lead time', metric_name, main_label)
                if ithre == n_thre - 1:
                    ax.legend(fontsize=20)
            for ax in axes[n_thre:]:
                ax.set_visible(False)
            fig.savefig(os.path.join(out_dir, f'refl_fcst_{short_name}_radius{rad}.{fig_fmt}'),
                       bbox_inches='tight')
            plt.close(fig)

    for irad, rad in enumerate(radii):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 12, num_rows * 12))
        axes = np.atleast_1d(axes).ravel()
        for ithre, thr in enumerate(thresholds):
            main_label = f'{title_label[ithre]} Threshold (> {thr} dBZ)' if n_thre > 1 else f'Threshold (> {thr} dBZ)'
            ax = axes[ithre]
            plot_performance_diagram(ax, pod_value[:, :, ithre, irad], far_value[:, :, ithre, irad],
                                     exp_labels, leads, title=main_label)
            if ithre == n_thre - 1:
                ax.legend(fontsize=20)
        for ax in axes[n_thre:]:
            ax.set_visible(False)
        fig.savefig(os.path.join(out_dir, f'refl_fcst_performance_diagram_radius{rad}.{fig_fmt}'),
                   bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    plot_time_series(ax, leads, rmse_value, exp_labels, 'Forecast lead time', 'RMSE (dBZ)', '')
    ax.legend(fontsize=20)
    fig.savefig(os.path.join(out_dir, f'refl_fcst_rmse.{fig_fmt}'), bbox_inches='tight')
    plt.close(fig)


def generate_radar_plots(config: Dict, diag_obs: str, sensor_graphic_dir: str,
                        diag_base_dir: str, radar_cfg: Dict) -> int:
    """Collect and draw every radar figure for one sensor. Returns the
    number of (experiment, lead) pairs that had data (0 -> nothing was
    drawn, e.g. diag_radar_reflectivity.py hasn't been run yet)."""
    names, labels, colors, _ = _build_forecast_experiments(config)
    if not names:
        return 0

    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs, for_forecast=True)
    cycles = generate_cycles(adjusted_start, end_time, interval)
    leads = generate_forecast_leads(config)
    thresholds = radar_cfg["thresholds"]
    radii = radar_cfg["radii"]
    aggregated_score = radar_cfg.get("aggregated_score", False)

    n_exp, n_fcst, n_thre, n_rad = len(names), len(leads), len(thresholds), len(radii)
    fss_value = np.full((n_exp, n_fcst, n_thre, n_rad, 3), np.nan)
    ts_value = np.full_like(fss_value, np.nan)
    ets_value = np.full_like(fss_value, np.nan)
    fbias_value = np.full_like(fss_value, np.nan)
    rmse_value = np.full((n_exp, n_fcst, 3), np.nan)
    pod_value = np.full((n_exp, n_fcst, n_thre, n_rad), np.nan)
    far_value = np.full((n_exp, n_fcst, n_thre, n_rad), np.nan)

    n_have_data = 0
    for iexp, exp_name in enumerate(names):
        for ifcst, lead in enumerate(leads):
            cycle_results = _collect_cycle_results(diag_base_dir, exp_name, diag_obs, cycles, lead)
            agg = aggregate_radar_cycles(cycle_results, aggregated_score)
            if agg is None:
                continue
            n_have_data += 1
            rmse_value[iexp, ifcst, :] = agg["rmse"]
            fss_value[iexp, ifcst, :, :, :] = agg["fss"]
            ts_value[iexp, ifcst, :, :, :] = agg["ts"]
            ets_value[iexp, ifcst, :, :, :] = agg["ets"]
            fbias_value[iexp, ifcst, :, :, :] = agg["fbias"]
            pod_value[iexp, ifcst, :, :] = agg["pod"]
            far_value[iexp, ifcst, :, :] = agg["far"]

    if not n_have_data:
        return 0

    out_dir = os.path.join(sensor_graphic_dir, "radar_reflectivity")
    os.makedirs(out_dir, exist_ok=True)
    fig_fmt = config["plot_settings"].get("figure_format", "png")
    _plot_radar_figures(out_dir, fig_fmt, labels, leads, thresholds, radii,
                        fss_value, ts_value, ets_value, fbias_value, rmse_value,
                        pod_value, far_value)
    return n_have_data


def _process_radar_sensor(config: Dict, diag_obs: str, sensor_graphic_dir: str,
                         diag_base_dir: str, radar_cfg: Dict):
    print(f"[radar:{diag_obs}] collecting figures from diagnosis files in {diag_base_dir}")
    try:
        n = generate_radar_plots(config, diag_obs, sensor_graphic_dir, diag_base_dir, radar_cfg)
    except Exception:
        print(f"ERROR generating radar figures for {diag_obs}:\n{traceback.format_exc()}")
        return diag_obs, 0
    print(f"[radar:{diag_obs}] finished ({n} experiment/lead combinations with data)")
    return diag_obs, n


def main(config_path: Optional[str] = None) -> None:
    """Read radar composite-reflectivity diagnosis files and generate
    figures for every configured radar sensor -- a no-op if the config has
    no "radar" section. Parallelism is per-SENSOR, mirroring
    plot_ioda_obsout_stats.py's main()."""
    config = load_config(config_path)
    radar_cfg = resolve_radar_settings(config)
    if radar_cfg is None:
        return

    graphics_base_dir = config["paths"].get("graphic_dir", "graphics")
    os.makedirs(graphics_base_dir, exist_ok=True)
    diag_base_dir = resolve_diag_base_dir(config)

    exp_names = config["experiments"]["names"]
    n_workers = max(1, int(config.get("n_workers", config.get("parallel", {}).get("n_workers", 1))))

    sensor_tasks = []
    for diag_obs in radar_cfg["sensors"]:
        if not radar_diag_sensor_has_files(diag_base_dir, exp_names, diag_obs):
            print(f"WARNING: no radar diagnosis files found for {diag_obs} under {diag_base_dir}; "
                  f"run diag_radar_reflectivity.py first. Skipping.")
            continue
        sensor_graphic_dir = os.path.join(graphics_base_dir, diag_obs)
        os.makedirs(sensor_graphic_dir, exist_ok=True)
        sensor_tasks.append((diag_obs, sensor_graphic_dir))

    if not sensor_tasks:
        return

    if n_workers == 1 or len(sensor_tasks) == 1:
        for diag_obs, sensor_graphic_dir in sensor_tasks:
            _process_radar_sensor(config, diag_obs, sensor_graphic_dir, diag_base_dir, radar_cfg)
        return

    n_workers = min(n_workers, len(sensor_tasks))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_radar_sensor, config, diag_obs, sensor_graphic_dir,
                       diag_base_dir, radar_cfg): diag_obs
            for diag_obs, sensor_graphic_dir in sensor_tasks
        }
        for future in as_completed(futures):
            diag_obs = futures[future]
            try:
                future.result()
            except Exception:
                print(f"ERROR processing radar sensor {diag_obs}:\n{traceback.format_exc()}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
