"""Normalized-departure figures.

  plot_normdep_histogram   distribution of (O-B)/sigma_o (and, for
                           ensembles, (O-B)/total spread) on a log scale,
                           with a Gaussian of the same RMS for reference:
                           heavier tails than the Gaussian are what QC
                           thresholds act on
  plot_normdep_profile     RMS (top) and mean (bottom) of the normalized
                           departures per level: O-B and O-A by sigma_o,
                           O-B by total spread (should be 1)
"""
import math
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
from .style import (
    _save_fig,
    _section_out_dir,
    _title_with_panel,
    apply_pub_style,
    auto_layout,
    get_exp_plot_settings,
)

__all__ = ["plot_normdep_histogram", "plot_normdep_profile", "gaussian_tail_fraction"]


def gaussian_tail_fraction(k, sigma=1.0):
    """Return ``P(|z| > k)`` for ``z ~ N(0, sigma**2)``."""
    if not sigma or not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    return math.erfc(k / (sigma * math.sqrt(2.0)))


def _tail_fraction(hist, edges, k):
    centers = 0.5 * (edges[1:] + edges[:-1])
    total = hist.sum()
    return float(hist[np.abs(centers) > k].sum() / total) if total > 0 else np.nan


def plot_normdep_histogram(Data, domain, cfg, tail=3.0):
    """Data[exp][var] from mpas_stat.obs.collect.collect_obs_normalized_departures."""
    apply_pub_style(cfg)
    var_names = cfg["variables"]
    exp_names, exp_labels, colors, _ = get_exp_plot_settings(cfg)
    out_dir = _section_out_dir(cfg, "normalized_departure")
    nrows, ncols = auto_layout(len(var_names))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    for k in range(len(var_names), nrows * ncols):
        axs.flat[k].set_visible(False)
    any_data = has_total = False
    for iv, var in enumerate(var_names):
        ax = axs.flat[iv]
        notes = []
        for ie, exp in enumerate(exp_names):
            entry = (Data.get(exp) or {}).get(var)
            if not entry or entry.get("edges") is None:
                continue
            edges = np.asarray(entry["edges"], dtype=np.float64)
            width = np.diff(edges)
            centers = 0.5 * (edges[1:] + edges[:-1])
            color = colors[ie % len(colors)]
            for kind, ls in (("omb", "-"), ("omb_total", "--")):
                e = entry.get(kind)
                if not e or e.get("hist") is None or e["hist"].sum() <= 0:
                    continue
                h = np.asarray(e["hist"], dtype=np.float64)
                dens = h / (h.sum() * width)
                ax.step(np.r_[edges[:-1], edges[-1]], np.r_[dens, dens[-1]], where="post",
                        color=color, ls=ls, lw=1.6, zorder=3)
                sigma = 1.0 if kind == "omb_total" else e["rms"]
                gauss = np.exp(-0.5 * (centers / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
                ax.plot(centers, gauss, color=color, ls=":", lw=0.9, zorder=2)
                frac = _tail_fraction(h, edges, tail)
                ref = gaussian_tail_fraction(tail, sigma)
                tag = "O-B/σo" if kind == "omb" else "O-B/total"
                notes.append((color, f"{tag}: RMS {e['rms']:.2f}, mean {e['mean']:+.2f}, "
                                     f"|z|>{tail:g} {100 * frac:.2f}% (Gauss {100 * ref:.2f}%)"))
                has_total |= kind == "omb_total"
                any_data = True
        ax.set_yscale("log")
        ax.set_ylim(1.0e-5, 50.0)        # headroom above the peak for the notes
        ax.set_xlim(-8, 8)
        ax.set_xlabel("normalized departure")
        ax.set_ylabel("density")
        ax.set_title(_title_with_panel(iv, cfg_var_label(var, cfg)))
        for k, (c, t) in enumerate(notes):
            ax.text(0.02, 0.97 - 0.07 * k, t, transform=ax.transAxes, fontsize=6.5, color=c,
                    ha="left", va="top",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
    if not any_data:
        plt.close(fig)
        return
    handles = [Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=lab)
               for i, (e, lab) in enumerate(zip(exp_names, exp_labels)) if Data.get(e)]
    handles += [Line2D([0], [0], color="0.3", ls="-", label="(O-B) / σo assumed")]
    if has_total:
        handles += [Line2D([0], [0], color="0.3", ls="--", label="(O-B) / total spread")]
    handles += [Line2D([0], [0], color="0.3", ls=":", label="Gaussian, same RMS")]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Normalized departures (OMB, {domain})", y=1.05, fontsize=11)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"normdep_histogram_{domain}"), cfg)


def plot_normdep_profile(Data, profile_key, coord_label, cfg):
    """RMS and mean of the normalized departures per level / latitude bin."""
    apply_pub_style(cfg)
    var_names = cfg["variables"]
    exp_names, exp_labels, colors, _ = get_exp_plot_settings(cfg)
    out_dir = _section_out_dir(cfg, "normalized_departure")
    nvars = len(var_names)
    fig, axs = plt.subplots(2, nvars, figsize=(4.2 * nvars, 8.6), squeeze=False)
    latitude = profile_key == "lat_bin_profile"
    styles = (("omb", "-"), ("oma", ":"), ("omb_total", "--"))
    present = set()
    for iv, var in enumerate(var_names):
        ax_r, ax_m = axs[0, iv], axs[1, iv]
        coord = None
        for ie, exp in enumerate(exp_names):
            entry = (Data.get(exp) or {}).get(var)
            if not entry or entry.get("coord") is None:
                continue
            coord = np.asarray(entry["coord"])
            categorical = not latitude and _is_vertical_categorical_axis(coord_label)
            y = _categorical_positions(coord) if categorical else coord
            color = colors[ie % len(colors)]
            for kind, ls in styles:
                e = entry.get(kind)
                if not e or "rms_profile" not in e:
                    continue
                ax_r.plot(e["rms_profile"], y, color=color, ls=ls, lw=1.6, zorder=3)
                ax_m.plot(e["mean_profile"], y, color=color, ls=ls, lw=1.6, zorder=3)
                present.add(kind)
        ax_r.axvline(1.0, color="0.4", lw=0.8, ls="--", zorder=0)
        ax_m.axvline(0.0, color="0.4", lw=0.8, ls="--", zorder=0)
        ax_r.set_xlabel("rms of normalized departure")
        ax_m.set_xlabel("mean normalized departure")
        ax_r.set_title(_title_with_panel(iv, cfg_var_label(var, cfg)))
        ax_m.set_title(_title_with_panel(nvars + iv, cfg_var_label(var, cfg)))
        for ax in (ax_r, ax_m):
            ax.set_ylabel("Latitude" if latitude else coord_label)
            if coord is not None:
                if latitude:
                    _apply_latitude_formatter(ax, axis="y")
                elif _is_vertical_categorical_axis(coord_label):
                    _set_categorical_axis(ax, coord, axis="y",
                                          max_ticks=cfg.get("vertical_tick_max", 13))
    if not present:
        plt.close(fig)
        return
    handles = [Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=lab)
               for i, (e, lab) in enumerate(zip(exp_names, exp_labels)) if Data.get(e)]
    labels = {"omb": "(O-B) / σo", "oma": "(O-A) / σo   (RMS² = 2 Jo / n)",
              "omb_total": "(O-B) / total spread   (should be 1)"}
    handles += [Line2D([0], [0], color="0.3", ls=ls, label=labels[k])
                for k, ls in styles if k in present]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig, os.path.join(out_dir, f"normdep_{profile_key}"), cfg)
