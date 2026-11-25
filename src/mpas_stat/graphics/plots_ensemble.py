"""Ensemble reliability figures: rank histograms and spread-skill.

  plot_rank_histogram          one panel per variable, one line per
                               ensemble experiment, frequency relative to
                               a flat histogram (1 = reliable)
  plot_rank_histogram_levels   rank x level map of the same ratio, one
                               figure per experiment
  plot_spread_skill            RMS(OMB) against total spread in groups of
                               equal observation count; on the diagonal =
                               spread consistent with the error
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .labels import cfg_var_label
from .axes import (
    _apply_latitude_formatter,
    _categorical_positions,
    _is_vertical_categorical_axis,
    _set_categorical_axis,
)
from .colour import colorbar_ticks, colour_scale
from .extent import apply_extent, extent_mode, finite_extent
from .style import (
    _save_fig,
    _section_out_dir,
    _title_with_panel,
    apply_pub_style,
    auto_layout,
    get_exp_plot_settings,
)

__all__ = ["plot_rank_histogram", "plot_rank_histogram_levels", "plot_spread_skill",
           "rank_histogram_summary"]


def rank_histogram_summary(hist):
    """Summarize a rank histogram of ``n_member + 1`` counts.

    Returns ``(normalized frequency, outlier ratio, reliability index)``:

    normalized frequency
        ``count / (total / n_rank)``; 1 means flat.
    outlier ratio
        Share of observations in the two extreme ranks divided by its
        expected value ``2 / n_rank``; above 1 means under-dispersive or
        biased.
    reliability index
        Mean ``|frequency - 1|`` (Delle Monache et al. 2006); 0 for a
        perfectly flat histogram.
    """
    hist = np.asarray(hist, dtype=np.float64)
    n = hist.size
    total = hist.sum()
    if total <= 0 or n < 2:
        return np.full(n, np.nan), np.nan, np.nan
    freq = hist / (total / n)
    outlier = (hist[0] + hist[-1]) / total / (2.0 / n)
    return freq, float(outlier), float(np.mean(np.abs(freq - 1.0)))


def _grid(cfg, nvars, layout=None):
    nrows, ncols = auto_layout(nvars)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False,
                            layout=layout)
    for k in range(nvars, nrows * ncols):
        axs.flat[k].set_visible(False)
    return fig, axs


def plot_rank_histogram(RankData, domain, cfg):
    """RankData[exp][var] = {"hist": counts, ...} (see
    mpas_stat.obs.collect.collect_obs_rank_histograms)."""
    apply_pub_style(cfg)
    var_names = cfg["variables"]
    exp_names, exp_labels, colors, _ = get_exp_plot_settings(cfg)
    out_dir = _section_out_dir(cfg, "rank_histogram")
    fig, axs = _grid(cfg, len(var_names))
    sizes = {RankData.get(e, {}).get(v, {}).get("hist", np.zeros(0)).size
             for e in exp_names for v in var_names} - {0}
    same_size = len(sizes) == 1
    single = sum(1 for e in exp_names if RankData.get(e)) == 1
    any_data = False

    for iv, var in enumerate(var_names):
        ax = axs.flat[iv]
        notes = []
        for ie, exp in enumerate(exp_names):
            entry = (RankData.get(exp) or {}).get(var)
            if not entry:
                continue
            freq, outlier, delta = rank_histogram_summary(entry["hist"])
            n = freq.size
            x = np.arange(n) if same_size else (np.arange(n) + 0.5) / n
            color = colors[ie % len(colors)]
            if single and same_size:
                ax.bar(x, freq, width=0.85, color=color, alpha=0.75, zorder=3)
            else:
                ax.step(np.r_[x - (0.5 if same_size else 0.5 / n), x[-1] + (0.5 if same_size else 0.5 / n)],
                        np.r_[freq, freq[-1]], where="post", color=color, lw=1.8, zorder=3)
            notes.append((color, f"outliers ×{outlier:.2f}  Δ={delta:.2f}"))
            any_data = True
        ax.axhline(1.0, color="0.3", lw=0.9, ls="--", zorder=2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("rank of observation" if same_size else "normalized rank")
        ax.set_ylabel("frequency / flat")
        ax.set_title(_title_with_panel(iv, cfg_var_label(var, cfg)))
        for k, (c, t) in enumerate(notes):
            ax.text(0.03, 0.04 + 0.075 * k, t, transform=ax.transAxes, fontsize=7.5,
                    color=c, ha="left", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
    if not any_data:
        plt.close(fig)
        return
    handles = [Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=lab)
               for i, (e, lab) in enumerate(zip(exp_names, exp_labels)) if RankData.get(e)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Rank histogram (OMB, {domain}):  U = under-dispersive, "
                 "dome = over-dispersive, slope = bias", y=1.04, fontsize=11)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"rank_histogram_{domain}"), cfg)


def plot_rank_histogram_levels(RankData, exp_name, profile_key, coord_label, cfg):
    """Frequency relative to flat, as a rank x level (or latitude) map, for
    one experiment: one panel per variable."""
    apply_pub_style(cfg)
    var_names = cfg["variables"]
    exp_names, exp_labels, _, _ = get_exp_plot_settings(cfg)
    label = dict(zip(exp_names, exp_labels)).get(exp_name, exp_name)
    out_dir = _section_out_dir(cfg, "rank_histogram")
    data = RankData.get(exp_name) or {}
    maps = {}
    for var in var_names:
        e = data.get(var)
        if not e or e.get("hist_levels") is None or e.get("coord") is None:
            continue
        h = np.asarray(e["hist_levels"], dtype=np.float64)
        tot = h.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(tot >= 10 * h.shape[1], h / (tot / h.shape[1]), np.nan)
        maps[var] = (ratio, np.asarray(e["coord"]))
    if not maps:
        return
    fig, axs = _grid(cfg, len(var_names), layout="constrained")
    lim = max(0.1, float(np.nanmax([np.nanmax(np.abs(r - 1.0)) for r, _ in maps.values()])))
    lim = min(lim, 1.0)
    cmap, norm, levels = colour_scale(
        (cfg.get("plot_style") or {}).get("diverging_cmap", "RdBu_r"), -lim, lim, cfg,
        diverging=True, extend="both")
    pcm = None
    latitude = profile_key == "lat_bin_profile"
    for iv, var in enumerate(var_names):
        ax = axs.flat[iv]
        ax.grid(False)
        ax.set_title(_title_with_panel(iv, cfg_var_label(var, cfg)))
        if var not in maps:
            continue
        ratio, coord = maps[var]
        n = ratio.shape[1]
        categorical = not latitude and _is_vertical_categorical_axis(coord_label)
        y = _categorical_positions(coord) if categorical else coord
        if categorical or y.size < 2:
            y_edges = np.arange(y.size + 1) - 0.5
        else:
            mid = 0.5 * (y[1:] + y[:-1])
            y_edges = np.r_[y[0] - (mid[0] - y[0]), mid, y[-1] + (y[-1] - mid[-1])]
        pcm = ax.pcolormesh(np.arange(n + 1) - 0.5, y_edges, ratio - 1.0, cmap=cmap, norm=norm)
        ax.set_xlabel("rank of observation")
        ax.set_ylabel("Latitude" if latitude else coord_label)
        if categorical:
            _set_categorical_axis(ax, coord, axis="y", max_ticks=cfg.get("vertical_tick_max", 13))
            ax.set_ylim(y_edges[0], y_edges[-1])   # first bin (surface) at the bottom
        if latitude:
            _apply_latitude_formatter(ax, axis="y")
            ext = finite_extent(np.arange(n), coord, ratio)
            if ext and extent_mode(cfg) != "full":
                apply_extent([ax], lat_span=ext[1])
    if pcm is not None:
        cb = fig.colorbar(pcm, ax=list(axs.flat[:len(var_names)]), shrink=0.85,
                          ticks=colorbar_ticks(levels) if levels is not None else None)
        cb.set_label("frequency / flat − 1")
    fig.suptitle(f"{label}: rank histogram by {'latitude' if latitude else 'level'} (OMB)",
                 fontsize=11)
    _save_fig(fig, os.path.join(out_dir, f"rank_histogram_{profile_key}_{exp_name}"), cfg)


def plot_spread_skill(SSData, domain, cfg):
    """SSData[exp][var] = {"total_spread", "rms", "std", "count", ...} (see
    mpas_stat.obs.collect.collect_obs_spread_skill)."""
    apply_pub_style(cfg)
    var_names = cfg["variables"]
    exp_names, exp_labels, colors, _ = get_exp_plot_settings(cfg)
    out_dir = _section_out_dir(cfg, "spread_skill")
    fig, axs = _grid(cfg, len(var_names))
    any_data = False
    for iv, var in enumerate(var_names):
        ax = axs.flat[iv]
        lo, hi = np.inf, 0.0
        for ie, exp in enumerate(exp_names):
            e = (SSData.get(exp) or {}).get(var)
            if not e:
                continue
            x, y = e["total_spread"], e["rms"]
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any():
                continue
            c = colors[ie % len(colors)]
            ax.plot(x[ok], y[ok], "-o", color=c, ms=4, lw=1.6, zorder=3)
            lo = min(lo, np.min(x[ok]), np.min(y[ok]))
            hi = max(hi, np.max(x[ok]), np.max(y[ok]))
            any_data = True
        if hi > 0:
            pad = 0.05 * (hi - lo)
            lo, hi = max(lo - pad, 0.0), hi + pad
            ax.plot([lo, hi], [lo, hi], color="0.3", ls="--", lw=0.9, zorder=2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("total spread (ensemble + obs error)")
        ax.set_ylabel("rms(OMB)")
        ax.set_title(_title_with_panel(iv, cfg_var_label(var, cfg)))
    if not any_data:
        plt.close(fig)
        return
    handles = [Line2D([0], [0], color=colors[i % len(colors)], marker="o", lw=2, label=lab)
               for i, (e, lab) in enumerate(zip(exp_names, exp_labels)) if SSData.get(e)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Spread-skill (OMB, {domain}): points above the diagonal = spread too small",
                 y=1.04, fontsize=11)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"spread_skill_{domain}"), cfg)
