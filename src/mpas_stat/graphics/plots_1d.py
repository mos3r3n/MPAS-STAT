"""1-D vertical/latitude profile plots.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from ..core.bootstrap import get_bootstrap_settings
from ..core.pooling import entry_difference_ci, entry_value
from .axes import (
    _apply_latitude_formatter,
    _categorical_positions,
    _is_vertical_categorical_axis,
    _set_categorical_axis,
)
from .style import (
    _ensure_panel_ticks,
    _relative_ref_threshold,
    _save_fig,
    _section_out_dir,
    _title_with_panel,
    apply_pub_style,
    auto_layout,
    draw_ci,
    get_exp_plot_settings,
    is_relative_stat,
    significant_from_ci,
)


# ---------------------------------------------------------------------------
def plot_profiles_1d(Profiles, profile_key, stat_name, cfg):
    """Time-averaged 1-D profiles with bootstrap CI shading/bar error intervals."""
    apply_pub_style(cfg)
    var_names     = cfg["variables"]
    out_dir       = _section_out_dir(cfg, "profiles_1d")

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    label_map = dict(zip(exp_names, exp_labels))

    nvars, (nrows, ncols) = len(var_names), auto_layout(len(var_names))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(ncols * 4.5, nrows * 5.0),
                            squeeze=False)
    axs = axs.ravel()

    for ivar, var_name in enumerate(var_names):
        ax = axs[ivar]

        for iexp, exp_name in enumerate(exp_names):
            entry = (Profiles.get(exp_name) or {}).get(var_name)
            if entry is None:
                continue

            prof      = entry["mean_profile"]
            ci_low    = entry.get("ci_low")
            ci_high   = entry.get("ci_high")
            coord     = entry["coord"]
            clabel    = entry["coord_label"]
            color     = colors[iexp % len(colors)]
            ls        = linestyles[iexp % len(linestyles)]

            # Use a consistent profile orientation for all 1-D profiles:
            #   x-axis = statistic value
            #   y-axis = coordinate.
            # For vertical profiles, draw the coordinate as categorical/evenly
            # spaced positions, matching the 2-D sections. The original
            # vertical coordinate values remain as tick labels.
            y_label = "Latitude" if profile_key == "lat_bin_profile" else clabel
            use_categorical_y = (profile_key != "lat_bin_profile"
                                 and _is_vertical_categorical_axis(y_label))
            y_plot = _categorical_positions(coord) if use_categorical_y else coord

            ax.plot(prof, y_plot, label=label_map[exp_name],
                    color=color, linestyle=ls, zorder=3)

            draw_ci(ax, y_plot, prof, ci_low, ci_high, color, cfg,
                    horizontal=True, alpha_default=0.16)

            ax.set_xlabel(stat_name)
            ax.set_ylabel(y_label)
            if profile_key == "lat_bin_profile":
                _apply_latitude_formatter(ax, axis="y")
            if use_categorical_y:
                _set_categorical_axis(ax, coord, axis="y",
                                      max_ticks=cfg.get("vertical_tick_max", 13))
            #if invert_y:
            #    ax.invert_yaxis()

        ax.set_title(_title_with_panel(ivar, var_name))
        _ensure_panel_ticks(ax)
        if ivar == nvars - 1:
            ax.legend()

    for i in range(nvars, nrows * ncols):
        axs[i].axis("off")

    fig.tight_layout()
    figname = f"profile_{profile_key}_{stat_name}"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)


# ---------------------------------------------------------------------------
def plot_profile_reduction_1d(Profiles_ref, Profiles_exp, profile_key, cfg):
    """Percent reduction of the RMS profile between two PARALLEL
    per-experiment profile sets for the SAME set of experiments -- e.g.
    observation-space OMB (`Profiles_ref`) vs. OMA (`Profiles_exp`) --
    rather than between different experiments on the same statistic (see
    plot_profile_diff_1d, whose reference/comparison axis is
    cfg["experiments"][0] vs. cfg["experiments"][1:] within a single
    Profiles dict).

    At each level, plots

        100 * (ref - exp) / ref

    one line per cfg["experiments"] entry -- positive means `Profiles_exp`
    improved on `Profiles_ref` (e.g. the analysis improved on the
    background) at that level. A reference value smaller than
    cfg["relative_min_reference"] (default 1e-12) is masked to NaN rather
    than producing a meaningless huge percentage, mirroring
    _relative_ref_threshold's small-reference guard used elsewhere in this
    module. No bootstrap significance testing is done here (unlike
    plot_profile_diff_1d) since collect_obs_profiles_1d-shaped entries
    carry only the already-averaged mean_profile, not the per-observation
    samples bootstrap needs (ci_low/ci_high are always None there).

    Always plots the "RMS" stat (`Profiles_ref`/`Profiles_exp` are expected
    to be the {var_name: entry} dict for stat_disp == "RMS", i.e. what a
    caller would get from Profiles["RMS"] in plot_profiles_1d's own
    Profiles shape) -- see plot_time_series_reduction_0d's docstring for
    why RMS specifically.
    """
    apply_pub_style(cfg)
    var_names = cfg["variables"]
    out_dir   = _section_out_dir(cfg, "profile_reduction_1d")
    ref_thr   = float(cfg.get("relative_min_reference", 1.0e-12))

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    label_map = dict(zip(exp_names, exp_labels))

    nvars, (nrows, ncols) = len(var_names), auto_layout(len(var_names))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(ncols * 4.5, nrows * 5.0),
                            squeeze=False)
    axs = axs.ravel()

    any_plotted = False
    for ivar, var_name in enumerate(var_names):
        ax = axs[ivar]
        y_label = None

        for iexp, exp_name in enumerate(exp_names):
            ref_entry = (Profiles_ref.get(exp_name) or {}).get(var_name)
            exp_entry = (Profiles_exp.get(exp_name) or {}).get(var_name)
            if ref_entry is None or exp_entry is None:
                continue

            ref_prof = np.asarray(ref_entry["mean_profile"], dtype=np.float64)
            exp_prof = np.asarray(exp_entry["mean_profile"], dtype=np.float64)
            if ref_prof.shape != exp_prof.shape:
                continue
            coord    = ref_entry["coord"]
            clabel   = ref_entry["coord_label"]
            color    = colors[iexp % len(colors)]
            ls       = linestyles[iexp % len(linestyles)]

            with np.errstate(divide="ignore", invalid="ignore"):
                reduction = np.where(np.abs(ref_prof) >= ref_thr,
                                     100.0 * (ref_prof - exp_prof) / ref_prof,
                                     np.nan)
            if not np.any(np.isfinite(reduction)):
                continue

            y_label = "Latitude" if profile_key == "lat_bin_profile" else clabel
            use_categorical_y = (profile_key != "lat_bin_profile"
                                 and _is_vertical_categorical_axis(y_label))
            y_plot = _categorical_positions(coord) if use_categorical_y else coord

            ax.plot(reduction, y_plot, label=label_map[exp_name],
                    color=color, linestyle=ls, marker="o", markersize=3, zorder=3)
            any_plotted = True

            if profile_key == "lat_bin_profile":
                _apply_latitude_formatter(ax, axis="y")
            if use_categorical_y:
                _set_categorical_axis(ax, coord, axis="y",
                                      max_ticks=cfg.get("vertical_tick_max", 13))

        ax.axvline(0.0, color="0.4", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_xlabel("RMS: reduction (%)")
        if y_label:
            ax.set_ylabel(y_label)
        ax.set_title(_title_with_panel(ivar, var_name))
        _ensure_panel_ticks(ax)
        if ivar == nvars - 1:
            ax.legend()

    for i in range(nvars, nrows * ncols):
        axs[i].axis("off")

    if not any_plotted:
        plt.close(fig)
        print(f"WARNING: no data for RMS reduction profile {profile_key}")
        return

    fig.tight_layout()
    figname = f"profile_{profile_key}_RMS_reduction_pct"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)


# ---------------------------------------------------------------------------
def plot_profile_diff_1d(Profiles, profile_key, stat_name, cfg):
    """Percent difference profile vs the reference experiment,
    100 * (exp - ref) / ref.

    Negative values = lower than the reference (an improvement for RMS/STD).
    Significant levels are marked with black dots.
    """
    apply_pub_style(cfg)
    if not is_relative_stat(stat_name, cfg):
        return

    var_names = cfg["variables"]
    out_dir   = _section_out_dir(cfg, "profile_reduction_1d")

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    boot    = get_bootstrap_settings(cfg)
    ref_exp = exp_names[0]

    nvars, (nrows, ncols) = len(var_names), auto_layout(len(var_names))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(ncols * 4.5, nrows * 5.0),
                            squeeze=False)
    axs = axs.ravel()

    for ivar, var_name in enumerate(var_names):
        ax = axs[ivar]
        ref_entry = (Profiles.get(ref_exp) or {}).get(var_name)
        if ref_entry is None:
            ax.axis("off")
            continue

        coord       = ref_entry["coord"]
        clabel      = ref_entry["coord_label"]
        y_label     = "Latitude" if profile_key == "lat_bin_profile" else clabel
        use_categorical_y = (profile_key != "lat_bin_profile"
                             and _is_vertical_categorical_axis(y_label))
        y_plot      = _categorical_positions(coord) if use_categorical_y else coord
        has_curve   = False

        for iexp, exp_name in enumerate(exp_names[1:], start=1):
            exp_entry = (Profiles.get(exp_name) or {}).get(var_name)
            if exp_entry is None:
                continue

            red, ci_low, ci_high = entry_difference_ci(ref_entry, exp_entry, boot)
            threshold = _relative_ref_threshold(var_name, cfg)
            ref_mean = entry_value(ref_entry)
            small_ref = ~np.isfinite(ref_mean) | (np.abs(ref_mean) < threshold)
            red = np.where(small_ref, np.nan, red)
            ci_low = np.where(small_ref, np.nan, ci_low)
            ci_high = np.where(small_ref, np.nan, ci_high)
            significant = significant_from_ci(ci_low, ci_high)

            color = colors[iexp % len(colors)]
            ls    = linestyles[iexp % len(linestyles)]

            ax.plot(red, y_plot, color=color, linestyle=ls,
                    label=exp_labels[iexp])
            draw_ci(ax, y_plot, red, ci_low, ci_high, color, cfg,
                    horizontal=True, alpha_default=0.14)
            if boot["plot_significance"]:
                # Use the same colour as the corresponding reduction line so
                # significant levels are visually linked to each experiment.
                ax.scatter(red[significant], y_plot[significant],
                           color=color, s=14, marker="o",
                           edgecolors="none", zorder=5)
            ax.set_xlabel(f"{stat_name}: diff. (%) from {exp_labels[0]}")
            ax.set_ylabel(y_label)
            if profile_key == "lat_bin_profile":
                _apply_latitude_formatter(ax, axis="y")
            if use_categorical_y:
                _set_categorical_axis(ax, coord, axis="y",
                                      max_ticks=cfg.get("vertical_tick_max", 13))

            has_curve = True

        ax.axvline(0.0, color="0.4", linewidth=0.8, linestyle="--")
        ax.set_title(_title_with_panel(ivar, var_name))
        _ensure_panel_ticks(ax)

        #if invert_y:
        #    ax.invert_yaxis()

        if not has_curve:
            ax.text(0.5, 0.5, "No comparison data",
                    transform=ax.transAxes, ha="center", va="center")

        if ivar == nvars - 1:
            ax.legend()

    for i in range(nvars, nrows * ncols):
        axs[i].axis("off")

    fig.tight_layout()
    figname=f"profile_{profile_key}_{stat_name}"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)


# ---------------------------------------------------------------------------
# Observation-error consistency (Desroziers diagnostics, consistency ratio)
# ---------------------------------------------------------------------------
def plot_error_consistency_1d(ErrProfiles, profile_key, cfg):
    """Observation-error consistency profiles, one column per variable.

    Top row (error standard deviations), one colour per experiment:
      solid  -- observation error assumed by the DA (ObsErrorAssumed)
      dashed -- Desroziers estimate of the observation error
      dotted -- Desroziers estimate of the background error in obs space
    Bottom row (ratios; 1 = consistent):
      solid  -- Desroziers / assumed observation error (< 1: the assumed
                error is too large, > 1: too small)
      dashed -- ensemble consistency ratio TotalSpread / RMS(OMB), when an
                ensemble spread is available (< 1: under-dispersive)

    ErrProfiles[exp][var] holds "coord", "coord_label" and any of the keys
    ObsErrorAssumed, ObsErrorDesroziers, BkgErrorDesroziers, ConsistencyRatio.
    """
    apply_pub_style(cfg)
    var_names = cfg["variables"]
    out_dir = _section_out_dir(cfg, "error_consistency_1d")
    exp_names, exp_labels, colors, _ = get_exp_plot_settings(cfg)

    nvars = len(var_names)
    fig, axs = plt.subplots(2, nvars, figsize=(4.3 * nvars, 9.0), squeeze=False)
    styles = {"ObsErrorAssumed": "-", "ObsErrorDesroziers": "--", "BkgErrorDesroziers": ":"}
    has_any = False

    for ivar, var_name in enumerate(var_names):
        ax_s, ax_r = axs[0, ivar], axs[1, ivar]
        y_label = clabel = None
        for iexp, exp_name in enumerate(exp_names):
            entry = (ErrProfiles.get(exp_name) or {}).get(var_name)
            if not entry:
                continue
            color = colors[iexp % len(colors)]
            coord = entry["coord"]
            clabel = entry["coord_label"]
            y_label = "Latitude" if profile_key == "lat_bin_profile" else clabel
            use_categorical_y = (profile_key != "lat_bin_profile"
                                 and _is_vertical_categorical_axis(y_label))
            y_plot = _categorical_positions(coord) if use_categorical_y else coord

            for key, ls in styles.items():
                prof = entry.get(key)
                if prof is not None and np.any(np.isfinite(prof)):
                    ax_s.plot(prof, y_plot, color=color, linestyle=ls, zorder=3)
                    has_any = True

            assumed, desroz = entry.get("ObsErrorAssumed"), entry.get("ObsErrorDesroziers")
            if assumed is not None and desroz is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.where(assumed > 0, desroz / assumed, np.nan)
                if np.any(np.isfinite(ratio)):
                    ax_r.plot(ratio, y_plot, color=color, linestyle="-", zorder=3)
            cr = entry.get("ConsistencyRatio")
            if cr is not None and np.any(np.isfinite(cr)):
                ax_r.plot(cr, y_plot, color=color, linestyle="--", zorder=3)

            for ax in (ax_s, ax_r):
                ax.set_ylabel(y_label)
                if profile_key == "lat_bin_profile":
                    _apply_latitude_formatter(ax, axis="y")
                if use_categorical_y:
                    _set_categorical_axis(ax, coord, axis="y",
                                          max_ticks=cfg.get("vertical_tick_max", 13))

        ax_r.axvline(1.0, color="0.4", linewidth=0.8, linestyle="--", zorder=0)
        ax_s.set_xlabel("error std. dev.")
        ax_r.set_xlabel("ratio")
        ax_s.set_title(_title_with_panel(ivar, var_name))
        ax_r.set_title(_title_with_panel(nvars + ivar, var_name))
        for ax in (ax_s, ax_r):
            _ensure_panel_ticks(ax)

    if not has_any:
        plt.close(fig)
        return

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=lab)
               for i, lab in enumerate(exp_labels)]
    handles += [
        Line2D([0], [0], color="0.3", ls="-", label=r"$\sigma_o$ assumed"),
        Line2D([0], [0], color="0.3", ls="--", label=r"$\sigma_o$ Desroziers"),
        Line2D([0], [0], color="0.3", ls=":", label=r"$\sigma_b$ Desroziers"),
    ]
    axs[0, -1].legend(handles=handles, loc="best", fontsize="small")
    ratio_handles = [
        Line2D([0], [0], color="0.3", ls="-", label=r"$\sigma_o$ Desroziers / assumed"),
        Line2D([0], [0], color="0.3", ls="--", label="TotalSpread / RMS(OMB)"),
    ]
    axs[1, -1].legend(handles=ratio_handles, loc="best", fontsize="small")

    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"error_consistency_{profile_key}"), cfg)
