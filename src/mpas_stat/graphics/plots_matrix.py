"""Channel x channel correlation matrices of radiance sensors."""
import os

import matplotlib.pyplot as plt
import numpy as np

from .colour import colorbar_ticks, colour_scale
from .style import _save_fig, _section_out_dir, apply_pub_style, get_exp_plot_settings

__all__ = ["plot_channel_correlation"]


def plot_channel_correlation(Data, sensor, domain, cfg):
    """One column per experiment: Desroziers observation-error correlation
    (top) and background-error correlation in observation space (bottom).

    Data[exp] = {"channels", "R_corr", "B_corr", "R_sigma", "min_eig", ...}
    (see mpas_stat.obs.figures.generate_channel_correlation_graphics)."""
    apply_pub_style(cfg)
    exp_names, exp_labels, _, _ = get_exp_plot_settings(cfg)
    exps = [(e, l) for e, l in zip(exp_names, exp_labels) if Data.get(e)]
    if not exps:
        return
    out_dir = _section_out_dir(cfg, "channel_correlation")
    nch = max(len(Data[e]["channels"]) for e, _ in exps)
    size = min(max(3.2, 0.32 * nch + 1.6), 9.0)
    fig, axs = plt.subplots(2, len(exps), figsize=(size * len(exps) + 1.2, 2 * size + 0.6),
                            squeeze=False, layout="constrained")
    cmap, norm, levels = colour_scale(
        (cfg.get("plot_style") or {}).get("diverging_cmap", "RdBu_r"), -1.0, 1.0, cfg,
        diverging=True)
    mesh = None
    for j, (exp, label) in enumerate(exps):
        d = Data[exp]
        ch = np.asarray(d["channels"])
        for i, (key, title) in enumerate((("R_corr", "obs-error correlation"),
                                          ("B_corr", "background-error correlation"))):
            ax = axs[i, j]
            ax.grid(False)
            ax.minorticks_off()
            m = np.asarray(d[key], dtype=np.float64)
            mesh = ax.pcolormesh(np.arange(ch.size + 1), np.arange(ch.size + 1),
                                 np.ma.masked_invalid(m), cmap=cmap, norm=norm)
            ax.set_xlim(0, ch.size)
            ax.set_ylim(ch.size, 0)
            ax.set_aspect("equal")
            step = max(1, int(np.ceil(ch.size / 16)))
            ticks = np.arange(0, ch.size, step)
            for setter, lab in ((ax.set_xticks, ax.set_xticklabels),
                                (ax.set_yticks, ax.set_yticklabels)):
                setter(ticks + 0.5)
                lab([str(c) for c in ch[ticks]], fontsize=7)
            if ch.size <= 12:
                for a in range(ch.size):
                    for b in range(ch.size):
                        if np.isfinite(m[a, b]) and a != b:
                            ax.text(b + 0.5, a + 0.5, f"{m[a, b]:.2f}", ha="center",
                                    va="center", fontsize=6.5,
                                    color="white" if abs(m[a, b]) > 0.6 else "black")
            ax.set_xlabel("channel")
            ax.set_ylabel("channel")
            extra = ""
            if key == "R_corr" and np.isfinite(d.get("min_eig", np.nan)):
                extra = f"\n(min eigenvalue {d['min_eig']:.2g}, {d.get('n_cycles', 0)} cycles)"
            ax.set_title(f"{label}: {title}{extra}", fontsize=9.5)
    cb = fig.colorbar(mesh, ax=axs, shrink=0.6,
                      ticks=colorbar_ticks(levels) if levels is not None else None)
    cb.set_label("correlation")
    fig.suptitle(f"{sensor}: Desroziers inter-channel correlations ({domain})", fontsize=11)
    _save_fig(fig, os.path.join(out_dir, f"channel_correlation_{domain}"), cfg)
