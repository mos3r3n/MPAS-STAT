"""0-D line plots: statistic vs. assimilation cycle or forecast lead time.
"""
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from ..core.bootstrap import get_bootstrap_settings
from ..core.pooling import entry_reduction_ci, entry_value
from .axes import _set_compact_time_axis
from .style import (
    _ensure_panel_ticks,
    _relative_ref_threshold,
    _save_fig,
    _section_out_dir,
    _title_with_panel,
    apply_pub_style,
    auto_layout,
    get_exp_plot_settings,
    is_relative_stat,
    significant_from_ci,
)


# ===========================================================================
# Public plot functions
# ===========================================================================
def plot_time_series_0d(Stats, stat_name, cfg):
    """Time series of scalar (0-D) statistics across assimilation cycles."""
    apply_pub_style(cfg)
    var_names   = cfg["variables"]
    region_name = cfg.get("region", "global")
    out_dir     = _section_out_dir(cfg, "time_series_0d")

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    label_map = dict(zip(exp_names, exp_labels))

    valid_exps = [e for e in exp_names if e in Stats and Stats[e]]
    if not valid_exps:
        print(f"WARNING: no valid experiments for {stat_name}")
        return

    nvars, (nrows, ncols) = len(var_names), auto_layout(len(var_names))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(ncols * 6.0, nrows * 4.0),
                            squeeze=False)
    axs = axs.ravel()

    for ivar, var_name in enumerate(var_names):
        ax = axs[ivar]
        for iexp, exp_name in enumerate(valid_exps):
            yvals, xlabels = [], []
            for cycle in sorted(Stats[exp_name]):
                try:
                    val = Stats[exp_name][cycle][var_name]["0D"][region_name][stat_name]
                    yvals.append(float(val))
                    xlabels.append(cycle)
                except KeyError:
                    continue
            if not yvals:
                continue
            x = np.arange(len(xlabels))
            ax.plot(x, yvals,
                    label=label_map[exp_name],
                    color=colors[iexp % len(colors)],
                    linestyle=linestyles[iexp % len(linestyles)])

        ax.set_title(_title_with_panel(ivar, var_name))
        ax.set_ylabel(stat_name)
        if yvals:
            _set_compact_time_axis(ax, x, xlabels, cfg, xlabel="Assimilation cycle")
        else:
            ax.set_xticks([])

        if ivar == nvars - 1:
            ax.legend()

    for i in range(nvars, nrows * ncols):
        axs[i].axis("off")

    fig.suptitle(f"{stat_name}  –  {region_name}", y=1.01)
    fig.tight_layout()
    figname = f"time_series_{stat_name}_{region_name}"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)


# ---------------------------------------------------------------------------
def plot_time_series_reduction_0d(Stats_ref, Stats_exp, cfg):
    """Percent reduction of a 0-D statistic across assimilation cycles,
    between two PARALLEL per-cycle statistic sets for the SAME set of
    experiments -- e.g. observation-space OMB (`Stats_ref`) vs. OMA
    (`Stats_exp`) -- rather than between different experiments on the same
    statistic (see plot_time_series_0d, whose reference/comparison axis is
    cfg["experiments"], not a second Stats dict).

    At each cycle where both sets have a finite value for an experiment,
    plots

        100 * (ref - exp) / ref

    one line per cfg["experiments"] entry -- positive means `Stats_exp`
    improved on `Stats_ref` (e.g. the analysis improved on the background).
    A reference value smaller than cfg["relative_min_reference"] (default
    1e-12) is skipped rather than producing a meaningless huge percentage,
    mirroring _relative_ref_threshold's small-reference guard used
    elsewhere in this module. No bootstrap significance testing is done
    here (unlike plot_forecast_lead_diff_0d/plot_profile_diff_1d) since a
    0-D time series has only one sample (the already-aggregated statistic)
    per cycle, not the per-observation samples bootstrap needs.

    Always plots the "RMS" stat (`Stats_ref`/`Stats_exp` are expected to
    hold Stats[exp][cycle][var]["0D"][region]["RMS"], the same shape
    plot_time_series_0d reads from) -- RMS reduction is the conventional
    background/analysis-error-reduction verification metric; nothing here
    is RMS-specific mechanically, but a single around-zero percent axis
    mixing units of different statistics would not be meaningful on one
    figure.
    """
    apply_pub_style(cfg)
    var_names   = cfg["variables"]
    region_name = cfg.get("region", "global")
    out_dir     = _section_out_dir(cfg, "time_series_diff_0d")
    ref_thr     = float(cfg.get("relative_min_reference", 1.0e-12))

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    label_map = dict(zip(exp_names, exp_labels))

    valid_exps = [e for e in exp_names
                 if Stats_ref.get(e) and Stats_exp.get(e)]
    if not valid_exps:
        print("WARNING: no valid experiments for RMS reduction time series")
        return

    nvars, (nrows, ncols) = len(var_names), auto_layout(len(var_names))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(ncols * 6.0, nrows * 4.0),
                            squeeze=False)
    axs = axs.ravel()

    any_plotted = False
    for ivar, var_name in enumerate(var_names):
        ax = axs[ivar]
        xlabels = []
        for iexp, exp_name in enumerate(valid_exps):
            yvals, xlabels = [], []
            cycles = sorted(set(Stats_ref[exp_name]) & set(Stats_exp[exp_name]))
            for cycle in cycles:
                try:
                    ref_val = float(Stats_ref[exp_name][cycle][var_name]["0D"][region_name]["RMS"])
                    exp_val = float(Stats_exp[exp_name][cycle][var_name]["0D"][region_name]["RMS"])
                except KeyError:
                    continue
                if not (np.isfinite(ref_val) and np.isfinite(exp_val)) or abs(ref_val) < ref_thr:
                    continue
                yvals.append(100.0 * (ref_val - exp_val) / ref_val)
                xlabels.append(cycle)
            if not yvals:
                continue
            any_plotted = True
            x = np.arange(len(xlabels))
            ax.plot(x, yvals, label=label_map[exp_name],
                    color=colors[iexp % len(colors)],
                    linestyle=linestyles[iexp % len(linestyles)],
                    marker="o", markersize=3)

        ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
        ax.set_title(_title_with_panel(ivar, var_name))
        ax.set_ylabel("RMS: reduction (%)")
        if xlabels:
            _set_compact_time_axis(ax, x, xlabels, cfg, xlabel="Assimilation cycle")
        else:
            ax.set_xticks([])
        _ensure_panel_ticks(ax)
        if ivar == nvars - 1:
            ax.legend()

    for i in range(nvars, nrows * ncols):
        axs[i].axis("off")

    if not any_plotted:
        plt.close(fig)
        print("WARNING: no data for RMS reduction time series")
        return

    fig.suptitle(f"RMS Reduction (%)  –  {region_name}", y=1.01)
    fig.tight_layout()
    figname = f"time_series_RMS_reduction_pct_{region_name}"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)


# ---------------------------------------------------------------------------
def plot_forecast_lead_0d(ForecastLead, stat_name, cfg):
    """Extended-forecast verification: statistic vs. forecast lead time.

    One panel per variable; one line per experiment, averaged (with
    bootstrap CI shading/error bars) over all initialization cycles at each
    forecast lead hour.

    Since every plotted point is itself a mean over many initialization
    cycles, each non-reference experiment (exp_names[1:]) is additionally
    bootstrap-tested against the reference/control experiment (exp_names[0],
    using the same per-cycle samples) at every lead time.  Lead times where
    the difference is statistically significant are marked with a black
    hollow circle, matching the significance convention used elsewhere in
    this module (see add_significance / plot_profile_diff_1d).
    """
    apply_pub_style(cfg)
    var_names   = cfg["variables"]
    region_name = cfg.get("region", "global")
    out_dir     = _section_out_dir(cfg, "forecast_lead_0d")
    ci_style    = str(cfg.get("profile_ci_style",
                              cfg.get("plot_style", {}).get("profile_ci_style", "shade"))).lower()
    boot = get_bootstrap_settings(cfg)

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    label_map = dict(zip(exp_names, exp_labels))

    valid_exps = [e for e in exp_names if e in ForecastLead and ForecastLead[e]]
    if not valid_exps:
        print(f"WARNING: no valid experiments for forecast_lead_0d {stat_name}")
        return

    ref_exp = valid_exps[0]

    nvars, (nrows, ncols) = len(var_names), auto_layout(len(var_names))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(ncols * 6.0, nrows * 4.0),
                            squeeze=False)
    axs = axs.ravel()

    any_significance_marked = False

    for ivar, var_name in enumerate(var_names):
        ax = axs[ivar]
        has_data = False
        x = None
        ref_entry   = ForecastLead.get(ref_exp, {}).get(var_name)
        ref_samples = ref_entry["samples"] if ref_entry is not None else None

        for iexp, exp_name in enumerate(valid_exps):
            entry = ForecastLead[exp_name].get(var_name)
            if entry is None:
                continue
            has_data = True
            x     = entry["leads"]
            y     = entry["data"]
            color = colors[iexp % len(colors)]

            ax.plot(x, y,
                    marker="o",
                    label=label_map[exp_name],
                    color=color,
                    linestyle=linestyles[iexp % len(linestyles)])

            if "ci_low" in entry and "ci_high" in entry:
                if ci_style == "shade":
                    ax.fill_between(x, entry["ci_low"], entry["ci_high"],
                                    color=color, alpha=0.15, linewidth=0)
                elif ci_style == "errorbar":
                    yerr = np.vstack([y - entry["ci_low"], entry["ci_high"] - y])
                    ax.errorbar(x, y, yerr=yerr, color=color, fmt="none",
                               capsize=2, alpha=0.6)

            # --- Significance vs. the reference/control experiment ---
            # Re-uses the same per-cycle samples matrix (n_cycles x n_leads)
            # that feeds the CI shading above, exactly as
            # plot_profile_diff_1d tests each experiment against
            # exp_names[0] using bootstrap_reduction_ci.
            if (boot["plot_significance"] and iexp > 0
                    and ref_samples is not None
                    and entry["samples"].shape == ref_samples.shape):
                _, ci_low, ci_high = entry_reduction_ci(ref_entry, entry, boot)
                threshold = _relative_ref_threshold(var_name, cfg)
                ref_mean  = entry_value(ref_entry)
                small_ref = ~np.isfinite(ref_mean) | (np.abs(ref_mean) < threshold)
                ci_low      = np.where(small_ref, np.nan, ci_low)
                ci_high     = np.where(small_ref, np.nan, ci_high)
                significant = significant_from_ci(ci_low, ci_high)

                if np.any(significant):
                    ax.scatter(x[significant], y[significant],
                               s=cfg.get("significance_marker_size", 70),
                               facecolors="none", edgecolors="black",
                               marker="o", linewidths=1.0, zorder=6)
                    any_significance_marked = True

        ax.set_title(_title_with_panel(ivar, var_name))
        ax.set_xlabel("Forecast lead time (h)")
        ax.set_ylabel(stat_name)
        if has_data:
            ax.set_xticks(x)
        _ensure_panel_ticks(ax)

        if ivar == nvars - 1:
            handles, labels = ax.get_legend_handles_labels()
            if any_significance_marked:
                sig_handle = mlines.Line2D(
                    [0], [0], marker="o", linestyle="none",
                    markerfacecolor="none", markeredgecolor="black",
                    markersize=8,
                    label=f"significant vs. {label_map[ref_exp]} ({boot['ci']:.0f}% CI)",
                )
                handles.append(sig_handle)
                labels.append(sig_handle.get_label())
            ax.legend(handles, labels)

    for i in range(nvars, nrows * ncols):
        axs[i].axis("off")

    fig.suptitle(f"Extended forecast {stat_name}  –  {region_name}", y=1.01)
    fig.tight_layout()
    figname = f"forecast_lead_{stat_name}_{region_name}"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)


# ---------------------------------------------------------------------------
# Extended-forecast percent-difference time series (vs. forecast lead time)
# ---------------------------------------------------------------------------
def plot_forecast_lead_diff_0d(ForecastLead, stat_name, cfg):
    """Extended-forecast verification: percent difference vs. forecast lead
    time, relative to the reference/control experiment (exp_names[0]).

    Mirrors plot_profile_diff_1d, but the independent axis is forecast lead
    time (from collect_forecast_lead_0d's per-cycle samples at each lead)
    instead of a vertical coordinate. Positive values indicate improvement
    over the reference; significant lead times (bootstrap CI vs. the
    reference excludes zero) are marked with a dot, same convention as
    plot_profile_diff_1d.
    """
    apply_pub_style(cfg)
    if not is_relative_stat(stat_name, cfg):
        return

    var_names = cfg["variables"]
    out_dir   = _section_out_dir(cfg, "forecast_lead_diff_0d")

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    boot     = get_bootstrap_settings(cfg)
    ci_style = str(cfg.get("profile_ci_style",
                          cfg.get("plot_style", {}).get("profile_ci_style", "shade"))).lower()
    ref_exp  = exp_names[0]

    nvars, (nrows, ncols) = len(var_names), auto_layout(len(var_names))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(ncols * 6.0, nrows * 4.0),
                            squeeze=False)
    axs = axs.ravel()

    for ivar, var_name in enumerate(var_names):
        ax = axs[ivar]
        ref_entry = (ForecastLead.get(ref_exp) or {}).get(var_name)
        if ref_entry is None:
            ax.axis("off")
            continue

        x         = ref_entry["leads"]
        has_curve = False

        for iexp, exp_name in enumerate(exp_names[1:], start=1):
            exp_entry = (ForecastLead.get(exp_name) or {}).get(var_name)
            if exp_entry is None:
                continue

            red, ci_low, ci_high = entry_reduction_ci(ref_entry, exp_entry, boot)
            threshold = _relative_ref_threshold(var_name, cfg)
            ref_mean  = entry_value(ref_entry)
            small_ref = ~np.isfinite(ref_mean) | (np.abs(ref_mean) < threshold)
            red     = np.where(small_ref, np.nan, red)
            ci_low  = np.where(small_ref, np.nan, ci_low)
            ci_high = np.where(small_ref, np.nan, ci_high)
            significant = significant_from_ci(ci_low, ci_high)

            color = colors[iexp % len(colors)]
            ls    = linestyles[iexp % len(linestyles)]

            ax.plot(x, red, color=color, linestyle=ls, marker="o",
                    label=exp_labels[iexp])
            if ci_style in ("shade", "shading", "fill"):
                ax.fill_between(x, ci_low, ci_high, color=color,
                                alpha=cfg.get("ci_alpha", 0.14), linewidth=0)
            elif ci_style in ("bar", "bars", "errorbar"):
                yerr = np.vstack([red - ci_low, ci_high - red])
                ax.errorbar(x, red, yerr=yerr, fmt="none",
                           ecolor=color, elinewidth=0.8, capsize=2, alpha=0.75)
            if boot["plot_significance"]:
                ax.scatter(x[significant], red[significant],
                          color=color, s=14, marker="o",
                          edgecolors="none", zorder=5)

            has_curve = True

        ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
        ax.set_title(_title_with_panel(ivar, var_name))
        ax.set_xlabel("Forecast lead time (h)")
        ax.set_ylabel(f"{stat_name}: diff. (%) from {exp_labels[0]}")
        if has_curve:
            ax.set_xticks(x)
        _ensure_panel_ticks(ax)

        if not has_curve:
            ax.text(0.5, 0.5, "No comparison data",
                    transform=ax.transAxes, ha="center", va="center")

        if ivar == nvars - 1:
            ax.legend()

    for i in range(nvars, nrows * ncols):
        axs[i].axis("off")

    fig.suptitle(f"Extended forecast {stat_name}: % difference from {exp_labels[0]}", y=1.01)
    fig.tight_layout()
    figname = f"forecast_lead_diff_{stat_name}"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)
