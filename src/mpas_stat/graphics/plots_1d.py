"""1-D vertical/latitude profile plots.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from ..core.bootstrap import bootstrap_reduction_ci, get_bootstrap_settings, safe_nanmean
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
    ci_style      = str(cfg.get("profile_ci_style", cfg.get("plot_style", {}).get("profile_ci_style", "shade"))).lower()
    plot_errorbar = cfg.get("profile_errorbar", True)  # backward compatible switch

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

            if plot_errorbar and ci_low is not None and ci_high is not None:
                if ci_style in ("shade", "shading", "fill"):
                    # Horizontal CI envelope around the profile.
                    alpha = cfg.get("ci_alpha", 0.16)
                    ax.fill_betweenx(y_plot, ci_low, ci_high,
                                     color=color, alpha=alpha, linewidth=0, zorder=1)
                elif ci_style in ("bar", "bars", "errorbar"):
                    xerr = np.vstack([prof - ci_low, ci_high - prof])
                    ax.errorbar(prof, y_plot, xerr=xerr, fmt="none",
                                ecolor=color, elinewidth=0.8, capsize=2,
                                alpha=0.75, zorder=2)

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
    """Percent reduction profile vs the reference experiment.

    Positive values indicate improvement over the reference.
    Significant levels are marked with black dots.
    """
    apply_pub_style(cfg)
    if not is_relative_stat(stat_name, cfg):
        return

    var_names = cfg["variables"]
    out_dir   = _section_out_dir(cfg, "profile_reduction_1d")

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    boot    = get_bootstrap_settings(cfg)
    ci_style = str(cfg.get("profile_ci_style", cfg.get("plot_style", {}).get("profile_ci_style", "shade"))).lower()
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

        ref_samples = ref_entry["samples"]
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

            red, ci_low, ci_high = bootstrap_reduction_ci(
                ref_samples, exp_entry["samples"],
                n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"],
            )
            threshold = _relative_ref_threshold(var_name, cfg)
            ref_mean = safe_nanmean(ref_samples, axis=0)
            small_ref = ~np.isfinite(ref_mean) | (np.abs(ref_mean) < threshold)
            red = np.where(small_ref, np.nan, red)
            ci_low = np.where(small_ref, np.nan, ci_low)
            ci_high = np.where(small_ref, np.nan, ci_high)
            significant = significant_from_ci(ci_low, ci_high)

            color = colors[iexp % len(colors)]
            ls    = linestyles[iexp % len(linestyles)]

            ax.plot(red, y_plot, color=color, linestyle=ls,
                    label=exp_labels[iexp])
            if ci_style in ("shade", "shading", "fill"):
                ax.fill_betweenx(y_plot, ci_low, ci_high,
                                 color=color, alpha=cfg.get("ci_alpha", 0.14), linewidth=0)
            elif ci_style in ("bar", "bars", "errorbar"):
                xerr = np.vstack([red - ci_low, ci_high - red])
                ax.errorbar(red, y_plot, xerr=xerr, fmt="none",
                            ecolor=color, elinewidth=0.8, capsize=2, alpha=0.75)
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
