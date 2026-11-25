"""Extended-forecast scorecard (triangle markers per variable/level/lead).
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from ..core.bootstrap import get_bootstrap_settings
from ..core.pooling import entry_difference_ci, entry_value
from .style import (
    _ensure_panel_ticks,
    _relative_ref_threshold,
    _save_fig,
    _section_out_dir,
    apply_pub_style,
    get_exp_plot_settings,
    significant_from_ci,
)


# ---------------------------------------------------------------------------
# Extended-forecast scorecard
# ---------------------------------------------------------------------------
def plot_forecast_scorecard(ScoreCard, row_labels, stat_name, cfg):
    """Draw the extended-forecast scorecard.

    One panel per experiment compared with the reference (``exp_names[0]``),
    drawn as a triangle scorecard:

    - Rows are the variable/level combinations in ``cfg["scorecard"]["rows"]``
      (e.g. ``"U 200 hPa"``, ``"Ps"``); columns are forecast lead hours.
    - Each cell is a triangle for the percent difference
      ``100 * (exp - ref) / ref``: up is an improvement (lower rms/STD),
      down a degradation.
    - Marker size scales with the absolute percent difference, capped at
      ``cfg["scorecard"]["max_score"]`` (default 20).
    - Significant cells (the bootstrap interval excludes zero) are solid with
      a black outline; the others are faint with no outline, at the same size.
    - A row with no valid data at any lead is shaded grey.
    """
    apply_pub_style(cfg)
    out_dir = _section_out_dir(cfg, "forecast_scorecard")
    boot    = get_bootstrap_settings(cfg)
    sc_cfg  = cfg.get("scorecard", {})

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    if len(exp_names) < 2:
        print("WARNING: forecast_scorecard needs a reference plus at least one other experiment")
        return
    ref_exp, ref_label = exp_names[0], exp_labels[0]
    test_exps = list(zip(exp_names[1:], exp_labels[1:]))

    nrows = len(row_labels)
    if nrows == 0:
        print("WARNING: forecast_scorecard has no rows configured (cfg['scorecard']['rows'])")
        return

    ref_rows = ScoreCard.get(ref_exp)
    if ref_rows is None or len(ref_rows) != nrows:
        print("WARNING: forecast_scorecard: missing/invalid reference experiment data")
        return

    leads  = ref_rows[0]["leads"]
    nleads = len(leads)
    x = np.arange(nleads)
    y = np.arange(nrows)

    improve_color = sc_cfg.get("improve_color", "#3A9A40")   # solid green
    degrade_color = sc_cfg.get("degrade_color", "#E06666")   # light red
    max_score     = float(sc_cfg.get("max_score", 20.0))
    min_size      = float(sc_cfg.get("min_marker_size", 6.0))
    sig_min_size  = float(sc_cfg.get("significant_min_marker_size", 45.0))
    max_size      = float(sc_cfg.get("max_marker_size", 260.0))
    # Vertical size per row / fixed padding for title, colour bar margins,
    # and the rotated lead-time tick labels at the bottom. Lower row_height
    # to pack rows closer together (tighten the y-axis spacing).
    row_height    = float(sc_cfg.get("row_height", 0.24))
    height_pad    = float(sc_cfg.get("height_pad", 1.6))

    ncols_fig = len(test_exps)
    fig, axs = plt.subplots(1, ncols_fig,
                            figsize=(0.42 * nleads + 2.3, row_height * nrows + height_pad),
                            squeeze=False)
    axs = axs[0]

    for ipanel, (exp_name, exp_label) in enumerate(test_exps):
        ax = axs[ipanel]
        exp_rows = ScoreCard.get(exp_name)
        if exp_rows is None or len(exp_rows) != nrows:
            ax.axis("off")
            continue

        # Faint dotted backdrop over every cell (as in a typical scorecard).
        Xg, Yg = np.meshgrid(x, y)
        ax.scatter(Xg, Yg, s=3, color="0.8", marker=".", zorder=1)

        for irow in range(nrows):
            ref_samples = ref_rows[irow]["samples"]
            exp_samples = exp_rows[irow]["samples"]
            if ref_samples.shape[0] < 1 or exp_samples.shape[0] < 1:
                ax.axhspan(irow - 0.5, irow + 0.5, color="0.85", zorder=0)
                continue

            red, ci_low, ci_high = entry_difference_ci(ref_rows[irow], exp_rows[irow], boot)
            var_key   = row_labels[irow].split()[0]
            threshold = _relative_ref_threshold(var_key, cfg)
            ref_mean  = entry_value(ref_rows[irow])
            small_ref = ~np.isfinite(ref_mean) | (np.abs(ref_mean) < threshold)
            red     = np.where(small_ref, np.nan, red)
            ci_low  = np.where(small_ref, np.nan, ci_low)
            ci_high = np.where(small_ref, np.nan, ci_high)

            if not np.any(np.isfinite(red)):
                ax.axhspan(irow - 0.5, irow + 0.5, color="0.85", zorder=0)
                continue

            sig_row = significant_from_ci(ci_low, ci_high)

            for ilead in range(nleads):
                val = red[ilead]
                if not np.isfinite(val):
                    continue

                # NOTE: "spread" has no universally agreed improve/degrade
                # sign (unlike rms/std, where a smaller value is always
                # better); it is still drawn up/down by sign here so every
                # cell gets a triangle, but treat its direction with care.
                # val is the percent difference 100*(exp-ref)/ref: a
                # negative value (lower RMS/STD than the reference) is an
                # improvement and gets the upward triangle.
                improved = bool(val < 0)
                marker   = "^" if improved else "v"
                color    = improve_color if improved else degrade_color

                frac = min(abs(val), max_score) / max_score if max_score > 0 else 0.0
                is_sig = bool(sig_row[ilead])

                if is_sig:
                    size = sig_min_size + frac * (max_size - sig_min_size)
                    ax.scatter(ilead, irow, s=size, marker=marker,
                              color=color, edgecolors="black", linewidths=0.9,
                              alpha=1.0, zorder=3)
                else:
                    # Not-significant cells are distinguished only by the
                    # missing outline (and lower alpha) now -- they scale
                    # over the same full size range as significant cells
                    # rather than being capped below sig_min_size.
                    size = min_size + frac * (max_size - min_size)
                    ax.scatter(ilead, irow, s=size, marker=marker,
                              color=color, edgecolors="none",
                              alpha=0.45, zorder=2)

        ax.set_xlim(-0.5, nleads - 0.5)
        ax.set_ylim(nrows - 0.5, -0.5)  # first row on top
        ax.set_xticks(x)
        ax.set_xticklabels([f"T+{int(l)}" for l in leads], rotation=90)
        ax.set_yticks(y)
        ax.set_yticklabels(row_labels)
        ax.set_title(f"{exp_label} vs. {ref_label}")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

        prev_var = None
        for irow, label in enumerate(row_labels):
            var_key = label.split()[0]
            if prev_var is not None and var_key != prev_var:
                ax.axhline(irow - 0.5, color="0.3", linewidth=0.8, zorder=4)
            prev_var = var_key

        _ensure_panel_ticks(ax)

    fig.suptitle(
        f"Extended-forecast scorecard: {stat_name} % difference vs. {ref_label}\n"
        f"▲ improvement   ▼ degradation   "
        f"bold/outlined = significant ({boot['ci']:.0f}% CI)   "
        f"size ∝ |% diff|, capped at {max_score:g}%",
        y=1.03, fontsize=10,
    )
    fig.tight_layout()
    figname = f"forecast_scorecard_{stat_name}"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)
