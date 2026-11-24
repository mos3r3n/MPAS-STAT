"""Publication style, shared figure utilities and figure saving.

Importing this module selects the non-interactive Agg backend and applies
the publication rcParams once, so every plot function inherits them.
"""
import os

import matplotlib as mpl

mpl.use("AGG")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)
import numpy as np


# ===========================================================================
# Publication style
# ===========================================================================
def apply_pub_style(cfg=None):
    """Apply journal-ready rcParams. Optional cfg["plot_style"] overrides defaults."""
    style = (cfg or {}).get("plot_style", {})
    base = {
        # --- Font ---
        "font.family":        style.get("font_family", "serif"),
        "font.serif":         style.get("font_serif", ["DejaVu Serif", "Times New Roman", "Times"]),
        "font.size":          style.get("font_size", 11),
        "axes.titlesize":     style.get("title_size", 12),
        "axes.labelsize":     style.get("label_size", 11),
        "xtick.labelsize":    style.get("tick_size", 10),
        "ytick.labelsize":    style.get("tick_size", 10),
        "legend.fontsize":    style.get("legend_size", 10),
        "figure.titlesize":   style.get("suptitle_size", 13),
        # --- Axes ---
        "axes.linewidth":     style.get("axes_linewidth", 0.9),
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "axes.grid":          style.get("grid", True),
        "grid.linewidth":     style.get("grid_linewidth", 0.4),
        "grid.linestyle":     style.get("grid_linestyle", "--"),
        "grid.alpha":         style.get("grid_alpha", 0.4),
        # --- Ticks ---
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top":           True,
        "ytick.right":         True,
        # --- Lines ---
        "lines.linewidth":    style.get("line_width", 1.8),
        "lines.markersize":   style.get("marker_size", 4.0),
        # --- Colour/image defaults ---
        "image.cmap":         style.get("absolute_cmap", "YlGnBu"),
        # Okabe-Ito-like cycle: distinct and robust for publication/colour-blind readers.
        "axes.prop_cycle":    mpl.cycler(color=style.get("color_cycle", [
            "#0072B2", "#D55E00", "#009E73", "#CC79A7",
            "#E69F00", "#56B4E9", "#F0E442", "#000000",
        ])),
        # --- Save ---
        "savefig.dpi":        style.get("dpi", 300),
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
    }
    plt.rcParams.update(base)


# ===========================================================================
# Small shared utilities
# ===========================================================================
def auto_layout(npanels):
    ncols = int(np.ceil(np.sqrt(npanels)))
    nrows = int(np.ceil(npanels / ncols))
    return nrows, ncols


def get_exp_plot_settings(cfg):
    exp_names  = cfg["experiments"]["names"]
    exp_labels = cfg["experiments"].get("labels", exp_names)
    colors     = cfg["experiments"].get("colors", None)
    linestyles = cfg["experiments"].get("linestyles", None)

    if len(exp_labels) != len(exp_names):
        raise ValueError("experiments.labels must have the same length as experiments.names")

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if linestyles is None:
        linestyles = ["-"] * len(exp_names)

    return exp_names, exp_labels, colors, linestyles


def is_relative_stat(stat_name, cfg):
    relative_stats = cfg.get(
        "relative_reduction_statistics",
        ["rms", "std", "spread"],
    )
    return stat_name.lower() in [s.lower() for s in relative_stats]


def significant_from_ci(ci_low, ci_high):
    return (
        np.isfinite(ci_low) & np.isfinite(ci_high)
        & ((ci_low > 0.0) | (ci_high < 0.0))
    )


def _save_fig(fig, path, cfg=None):
    """Save figure using configured output format.

    cfg["_lead_suffix"] (e.g. "_f006"), when present, is appended to the
    base filename -- this is how the forecast_* plot types distinguish one
    forecast lead hour's figure from another while still sharing the same
    output folder as their background-mode counterparts (see _lead_cfg in
    src/mpas_stat/graphics/driver.py).
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fmt = "png"
    lead_suffix = ""
    if cfg is not None:
        fmt = str(cfg.get("figure_format", "png")).lower()
        lead_suffix = str(cfg.get("_lead_suffix", "") or "")

    base, _ = os.path.splitext(path)
    base = base.replace("_Global", "")
    outfile = f"{base}{lead_suffix}.{fmt}"

    fig.savefig(outfile)
    plt.close(fig)

    print(f"Saved {outfile}")


def _panel_label(index):
    """Return panel labels: (a), ..., (z), (aa), ..."""
    letters = ""
    i = int(index)
    while True:
        i, r = divmod(i, 26)
        letters = chr(97 + r) + letters
        if i == 0:
            break
        i -= 1
    return f"({letters})"


def _title_with_panel(index, title):
    return f"{_panel_label(index)} {title}"


def _ensure_panel_ticks(ax):
    """Show tick marks and tick labels on every panel."""
    ax.tick_params(axis="both", which="both",
                   bottom=True, top=True, left=True, right=True,
                   labelbottom=True, labelleft=True)


def _relative_ref_threshold(var_name, cfg):
    """Small-reference cutoff for percent differences.

    QV/water-vapour fields often have very small absolute values.  When the
    reference value is below this threshold, percent reduction is set to NaN to
    avoid unrealistically large percentages.
    """
    default_thr = float(cfg.get("relative_min_reference", 1.0e-12))
    qv_thr = float(cfg.get("qv_relative_min_reference", 5.0e-4))
    key = str(var_name).lower()
    qv_names = [str(v).lower() for v in cfg.get(
        "qv_variable_names",
        ["qv", "qvapor", "qvapor_mpas", "specific_humidity", "humidity"]
    )]
    if key in qv_names or key.startswith("qv") or "specific_humidity" in key:
        return qv_thr
    return default_thr


def _section_out_dir(cfg, section_name):
    """Return output subfolder for a figure section only.

    The folder structure is organized by figure type, not by statistic name
    (e.g., figures/time_series_0d/, not figures/rms/time_series_0d/).
    """
    base_dir = cfg.get("paths", {}).get("out_dir", "figures")
    path = os.path.join(base_dir, section_name)
    os.makedirs(path, exist_ok=True)
    return path


# Call once on import so every function inherits the style.
apply_pub_style()


# ---------------------------------------------------------------------------
# Confidence-interval display
# ---------------------------------------------------------------------------
_CI_STYLE_ALIASES = {
    "shade": "shade", "shading": "shade", "fill": "shade",
    "errorbar": "errorbar", "errorbars": "errorbar", "bar": "errorbar", "bars": "errorbar",
    "none": "none", "off": "none", "false": "none",
}


def get_ci_style(cfg):
    """How bootstrap confidence intervals are drawn on line/profile plots:
    "shade" (coloured band, default), "errorbar" (error bars) or "none".

    Read from cfg["ci_style"], else cfg["plot_style"]["ci_style"], else the
    older profile_ci_style keys; profile_errorbar: false also means "none".
    """
    style = cfg.get("plot_style", {}) or {}
    raw = cfg.get("ci_style", style.get("ci_style",
                  cfg.get("profile_ci_style", style.get("profile_ci_style", "shade"))))
    if raw is False or raw is None:
        raw = "none"
    if cfg.get("profile_errorbar", True) is False:
        raw = "none"
    key = str(raw).strip().lower()
    if key not in _CI_STYLE_ALIASES:
        raise ValueError(f"ci_style must be 'shade', 'errorbar' or 'none', got {raw!r}")
    return _CI_STYLE_ALIASES[key]


def draw_ci(ax, coord, value, ci_low, ci_high, color, cfg, horizontal=False,
            alpha_default=0.15, zorder=1):
    """Draw a confidence interval around a line in the configured ci_style.

    ``horizontal=True`` is for vertical profiles, where the statistic is on
    the x axis and ``coord`` on the y axis.
    """
    style = get_ci_style(cfg)
    if style == "none" or ci_low is None or ci_high is None:
        return
    coord = np.asarray(coord)
    value = np.asarray(value, dtype=float)
    ci_low = np.asarray(ci_low, dtype=float)
    ci_high = np.asarray(ci_high, dtype=float)
    if style == "shade":
        fill = ax.fill_betweenx if horizontal else ax.fill_between
        fill(coord, ci_low, ci_high, color=color,
             alpha=cfg.get("ci_alpha", alpha_default), linewidth=0, zorder=zorder)
    else:
        err = np.vstack([value - ci_low, ci_high - value])
        kw = dict(fmt="none", ecolor=color, elinewidth=cfg.get("ci_errorbar_linewidth", 0.8),
                  capsize=cfg.get("ci_errorbar_capsize", 2), alpha=0.75, zorder=zorder + 1)
        if horizontal:
            ax.errorbar(value, coord, xerr=err, **kw)
        else:
            ax.errorbar(coord, value, yerr=err, **kw)
