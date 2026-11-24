"""Colour scales and line colours shared by every figure.

Approach adapted from NCAR's mpas-jedi-graphics (plot_styles.py,
basic_plot_functions.py; Apache-2.0):

* 2-D fields use DISCRETE colour levels (BoundaryNorm) at "nice" values, so
  each colour stands for a readable value range;
* limits are rounded to clean numbers and, by default, shared by every panel
  of a variable row, so experiments can be compared by colour;
* diverging levels are symmetric with 0.0 on a level boundary;
* experiment line colours default to a colour-vision-deficiency (CVD) safe
  ColorBrewer palette.

Settings (model graphic.yaml top level; obs-space configs under
plot_settings)::

    colorbar:
      discrete: true          # false -> continuous colour scale
      levels_diverging: 20    # approximate number of colour bins
      levels_sequential: 16
      share: row              # row (all panels of a variable) | panel
    plot_style:
      diverging_cmap: RdBu_r  # or RdYlBu_r, ... (any matplotlib name)
      absolute_cmap: YlGnBu
      palette: [...]          # experiment line colours
"""
import copy

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Black for the control/reference experiment, then ColorBrewer "Dark2"
# (qualitative, CVD-friendly; https://colorbrewer2.org).
CVD_SAFE_PALETTE = [
    "black",
    "#d95f02",  # dark orange
    "#7570b3",  # purple
    "#1b9e77",  # teal green
    "#e7298a",  # magenta
    "#66a61e",  # olive green
    "#e6ab02",  # mustard
    "#a6761d",  # brown
    "#666666",  # grey
]


def line_palette(cfg=None):
    """Experiment line colours: plot_style.palette (or palette) if set,
    else CVD_SAFE_PALETTE."""
    cfg = cfg or {}
    style = cfg.get("plot_style", {}) or {}
    pal = cfg.get("palette", style.get("palette"))
    return list(pal) if pal else list(CVD_SAFE_PALETTE)


def _colorbar_cfg(cfg):
    cb = cfg.get("colorbar", {}) or {}
    return {
        "discrete": bool(cb.get("discrete", True)),
        "levels_diverging": int(cb.get("levels_diverging", 20)),
        "levels_sequential": int(cb.get("levels_sequential", 16)),
        "share": str(cb.get("share", "row")).lower(),
    }


def share_limits(cfg):
    """True when panels of one variable row share their colour limits."""
    return _colorbar_cfg(cfg)["share"] in ("row", "variable", "true", "yes")


def data_range(arrays, diverging):
    """(vmin, vmax) covering every finite value in `arrays`; symmetric about
    zero for diverging scales. (None, None) when there is no finite value."""
    chunks = []
    for arr in arrays:
        if arr is None:
            continue
        vals = np.asarray(arr, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            chunks.append(vals)
    if not chunks:
        return None, None
    vals = np.concatenate(chunks)
    if diverging:
        amp = float(np.max(np.abs(vals)))
        return (-amp, amp) if amp > 0 else (None, None)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if lo == hi:
        pad = abs(lo) * 0.05 or 1.0
        lo, hi = lo - pad, hi + pad
    return lo, hi


def nice_levels(vmin, vmax, nbins, diverging):
    """Level boundaries at nice values covering [vmin, vmax]; symmetric with
    0.0 on a boundary for diverging scales."""
    locator = mticker.MaxNLocator(nbins=max(int(nbins), 2), symmetric=bool(diverging),
                                  steps=[1, 2, 2.5, 5, 10])
    levels = np.asarray(locator.tick_values(vmin, vmax), dtype=np.float64)
    # tick_values may overshoot the range by one step on each side; keep
    # only what is needed to cover [vmin, vmax].
    lo = levels[levels <= vmin]
    hi = levels[levels >= vmax]
    first = lo[-1] if lo.size else levels[0]
    last = hi[0] if hi.size else levels[-1]
    levels = levels[(levels >= first) & (levels <= last)]
    if diverging:
        amp = max(abs(levels[0]), abs(levels[-1]))
        step = np.min(np.diff(levels)) if levels.size > 1 else amp
        n = int(round(amp / step))
        levels = np.linspace(-amp, amp, 2 * n + 1)
    return levels


def colour_scale(cmap_name, vmin, vmax, cfg, diverging, extend="neither", data_arrays=None):
    """(cmap, norm, levels) for pcolormesh.

    vmin/vmax: explicit limits (either may be None -> taken from
    `data_arrays`). With colorbar.discrete (default) the limits are rounded
    to nice values and a BoundaryNorm with discrete levels is returned;
    otherwise a continuous Normalize. `levels` is None for continuous scales.
    """
    cb = _colorbar_cfg(cfg)
    if vmin is None or vmax is None:
        dmin, dmax = data_range(data_arrays or [], diverging)
        vmin = dmin if vmin is None else vmin
        vmax = dmax if vmax is None else vmax
    cmap = copy.copy(plt.get_cmap(cmap_name))
    if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return cmap, mcolors.Normalize(vmin=vmin, vmax=vmax), None
    if not cb["discrete"]:
        return cmap, mcolors.Normalize(vmin=vmin, vmax=vmax), None
    nbins = cb["levels_diverging"] if diverging else cb["levels_sequential"]
    levels = nice_levels(vmin, vmax, nbins, diverging)
    if levels.size < 2:
        return cmap, mcolors.Normalize(vmin=vmin, vmax=vmax), None
    norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend=extend)
    return cmap, norm, levels


def colorbar_ticks(levels, max_ticks=9):
    """Every k-th level boundary, keeping 0.0 for symmetric levels."""
    if levels is None:
        return None
    levels = np.asarray(levels)
    step = max(1, int(np.ceil((levels.size - 1) / max(max_ticks - 1, 1))))
    if np.any(np.isclose(levels, 0.0)):
        i0 = int(np.argmin(np.abs(levels)))
        idx = sorted(set(range(i0, -1, -step)) | set(range(i0, levels.size, step)))
        return levels[idx]
    return levels[::step]
