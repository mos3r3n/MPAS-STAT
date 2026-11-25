"""Figures of MET-style continuous statistics (see mpas_stat.obs.continuous).

  plot_continuous_profiles   one row per variable; columns MAE, PR_CORR,
                             FSTDEV/OSTDEV and, for ensembles, the fair
                             CRPS; one line per experiment, by level
  plot_taylor_diagram        Taylor (2001) diagram: normalized standard
                             deviation FSTDEV/OSTDEV as the radius and the
                             correlation PR_CORR as the angle, one point per
                             level (or channel) and experiment; the distance
                             to the reference point (1, 0) is the
                             normalized centred RMS difference
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .labels import cfg_var_label
from .style import (
    _save_fig,
    _section_out_dir,
    _title_with_panel,
    apply_pub_style,
    auto_layout,
    get_exp_plot_settings,
)

__all__ = ["plot_continuous_profiles", "plot_taylor_diagram", "CNT_PROFILE_COLUMNS"]

# (statistic, axis label, reference line)
CNT_PROFILE_COLUMNS = [
    ("MAE", "mae", None),
    ("PR_CORR", "correlation", 1.0),
    ("STD_RATIO", "stdev ratio (H(x) / obs)", 1.0),
    ("CRPS_EMP_FAIR", "crps (fair)", None),
]


def _stat(entry, name):
    if name == "STD_RATIO":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.asarray(entry["FSTDEV"]) / np.asarray(entry["OSTDEV"])
    return np.asarray(entry[name]) if name in entry else None


def _positions(coord, invert_y):
    """Equally spaced positions for the levels, the package's categorical
    vertical axis; with ``invert_y`` (pressure) larger values go lower."""
    coord = np.asarray(coord, dtype=np.float64)
    order = np.argsort(coord)
    pos = np.empty(coord.size)
    pos[order] = np.arange(coord.size)
    return pos if not invert_y else pos.max() - pos


def plot_continuous_profiles(CntData, diagtype, domain, coord_label, invert_y, cfg):
    """CntData[exp][var] = {"levels": stats, "coord": levels} (see
    collect_continuous). Columns without data (CRPS for deterministic runs)
    are left out."""
    apply_pub_style(cfg)
    var_names = [v for v in cfg["variables"]
                 if any((CntData.get(e) or {}).get(v, {}).get("levels") for e in CntData)]
    if not var_names:
        return
    exp_names, exp_labels, colors, _ = get_exp_plot_settings(cfg)
    columns = [c for c in CNT_PROFILE_COLUMNS
               if any(_stat(((CntData.get(e) or {}).get(v) or {}).get("levels") or {}, c[0])
                      is not None for e in exp_names for v in var_names)]
    fig, axs = plt.subplots(len(var_names), len(columns),
                            figsize=(3.2 * len(columns), 3.4 * len(var_names)),
                            squeeze=False)
    k = 0
    for iv, var in enumerate(var_names):
        for ic, (name, label, ref) in enumerate(columns):
            ax = axs[iv, ic]
            ticks = None
            for ie, exp in enumerate(exp_names):
                e = (CntData.get(exp) or {}).get(var)
                if not e or not e.get("levels"):
                    continue
                y = _stat(e["levels"], name)
                if y is None:
                    continue
                pos = _positions(e["coord"], invert_y)
                ok = np.isfinite(y)
                ax.plot(y[ok], pos[ok], "-o", color=colors[ie % len(colors)], ms=3, lw=1.6)
                ticks = (pos, e["coord"])
            if ref is not None:
                ax.axvline(ref, color="0.4", lw=0.8, ls="--")
            if ticks is not None:
                step = max(1, len(ticks[0]) // 10)
                idx = np.argsort(ticks[0])[::step]
                ax.set_yticks(ticks[0][idx])
                ax.set_yticklabels([f"{c:g}" for c in np.asarray(ticks[1])[idx]])
            ax.set_xlabel(label)
            if ic == 0:
                ax.set_ylabel(coord_label)
            ax.set_title(_title_with_panel(k, cfg_var_label(var, cfg)))
            ax.grid(alpha=0.3)
            k += 1
    handles = [Line2D([0], [0], color=colors[i % len(colors)], marker="o", lw=2, label=lab)
               for i, (e, lab) in enumerate(zip(exp_names, exp_labels)) if CntData.get(e)]
    fig.legend(handles=handles, loc="upper center", ncol=max(1, len(handles)),
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Continuous statistics, H(x) against observations ({diagtype.upper()}, "
                 f"{domain})", y=1.03, fontsize=11)
    fig.tight_layout()
    _save_fig(fig, os.path.join(_section_out_dir(cfg, "continuous"),
                                f"cnt_profile_{diagtype}_{domain}"), cfg)


def _taylor_axes(ax, rmax):
    """Draw the quarter-circle frame of a Taylor diagram on cartesian ``ax``."""
    t = np.linspace(0.0, np.pi / 2.0, 200)
    for r in np.arange(0.25, rmax + 1.0e-9, 0.25):
        ax.plot(r * np.cos(t), r * np.sin(t), color="0.85" if r != 1.0 else "0.5",
                lw=0.6 if r != 1.0 else 0.9, ls="-" if r != 1.0 else "--", zorder=1)
    for c in (0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99):
        a = np.arccos(c)
        ax.plot([0, rmax * np.cos(a)], [0, rmax * np.sin(a)], color="0.85", lw=0.6, zorder=1)
        ax.text(1.02 * rmax * np.cos(a), 1.02 * rmax * np.sin(a), f"{c:g}", fontsize=7,
                ha="left", va="bottom", color="0.35")
    # centred RMS difference arcs around the reference point (1, 0)
    for d in (0.25, 0.5, 0.75, 1.0):
        x, y = 1.0 + d * np.cos(np.linspace(0, np.pi, 200)), d * np.sin(np.linspace(0, np.pi, 200))
        keep = (x * x + y * y) <= rmax * rmax
        ax.plot(x[keep], y[keep], color="#7fb2d5", lw=0.6, ls=":", zorder=1)
    ax.plot([0, rmax], [0, 0], color="k", lw=0.8)
    ax.plot([0, 0], [0, rmax], color="k", lw=0.8)
    ax.plot(rmax * np.cos(t), rmax * np.sin(t), color="k", lw=0.8)
    ax.plot(1.0, 0.0, marker="*", ms=12, color="k", zorder=5, clip_on=False)
    ax.text(rmax * 0.72, rmax * 0.78, "correlation", rotation=-45, fontsize=8, color="0.35")
    ax.set_xlim(0, rmax * 1.08)
    ax.set_ylim(0, rmax * 1.08)
    ax.set_aspect("equal")
    ax.set_xlabel("stdev ratio (H(x) / obs)")
    ax.set_ylabel("stdev ratio (H(x) / obs)")


def plot_taylor_diagram(CntData, diagtype, domain, cfg, by="levels"):
    """Taylor diagram per variable, one point per level and experiment
    (``by="levels"``); with ``by="variables"`` (radiances) one panel with a
    point per channel and experiment from the whole-domain statistics."""
    apply_pub_style(cfg)
    exp_names, exp_labels, colors, _ = get_exp_plot_settings(cfg)
    if by == "variables":
        panels = [("all", cfg["variables"])]
    else:
        panels = [(v, [v]) for v in cfg["variables"]
                  if any((CntData.get(e) or {}).get(v, {}).get("levels") for e in exp_names)]
    if not panels:
        return
    points = []          # (panel, exp index, ratio, corr, label)
    for ip, (_, vars_) in enumerate(panels):
        for ie, exp in enumerate(exp_names):
            for var in vars_:
                e = (CntData.get(exp) or {}).get(var)
                if not e:
                    continue
                if by == "variables":
                    s = e.get("all") or {}
                    if s:
                        points.append((ip, ie, float(s["FSTDEV"]) / float(s["OSTDEV"])
                                       if float(s["OSTDEV"]) > 0 else np.nan,
                                       float(s["PR_CORR"]), str(var).replace("Channel ", "")))
                elif e.get("levels"):
                    s = e["levels"]
                    for i, lev in enumerate(np.asarray(e["coord"])):
                        o = float(s["OSTDEV"][i])
                        points.append((ip, ie, float(s["FSTDEV"][i]) / o if o > 0 else np.nan,
                                       float(s["PR_CORR"][i]), f"{lev:g}"))
    points = [p for p in points if np.isfinite(p[2]) and np.isfinite(p[3]) and p[3] > 0]
    if not points:
        return
    rmax = max(1.5, np.ceil(max(p[2] for p in points) * 4.0) / 4.0 + 0.25)
    nrows, ncols = auto_layout(len(panels))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.6 * nrows), squeeze=False)
    for k in range(len(panels), nrows * ncols):
        axs.flat[k].set_visible(False)
    for ip, (title, _) in enumerate(panels):
        ax = axs.flat[ip]
        _taylor_axes(ax, rmax)
        for _, ie, ratio, corr, label in (p for p in points if p[0] == ip):
            a = np.arccos(np.clip(corr, -1.0, 1.0))
            x, y = ratio * np.cos(a), ratio * np.sin(a)
            c = colors[ie % len(colors)]
            ax.plot(x, y, "o", color=c, ms=5, zorder=4)
            ax.annotate(label, (x, y), xytext=(3, 3), textcoords="offset points",
                        fontsize=6, color=c)
        name = "channels" if by == "variables" else cfg_var_label(title, cfg)
        ax.set_title(_title_with_panel(ip, name))
    handles = [Line2D([0], [0], color=colors[i % len(colors)], marker="o", lw=0, label=lab)
               for i, (e, lab) in enumerate(zip(exp_names, exp_labels)) if CntData.get(e)]
    handles.append(Line2D([0], [0], color="k", marker="*", lw=0, ms=10, label="observations"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Taylor diagram ({diagtype.upper()}, {domain}): labels are "
                 f"{'channels' if by == 'variables' else 'levels'}; dotted arcs = centred "
                 f"RMS difference / obs stdev", y=1.04, fontsize=10)
    fig.tight_layout()
    _save_fig(fig, os.path.join(_section_out_dir(cfg, "continuous"),
                                f"taylor_{diagtype}_{domain}"), cfg)
