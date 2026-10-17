"""
==================================
Plotting for MPAS diagnostic statistics.
==================================
"""
import os
from datetime import datetime
import numpy as np
import matplotlib as mpl
mpl.use("AGG")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
# Local libarary
from mpas_stats_graphic_io import bootstrap_reduction_ci, get_bootstrap_settings, safe_nanmean

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

# Call once on import so every function inherits the style.
apply_pub_style()

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
    mpas_stats_graphic_driver.py).
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

# ---------------------------------------------------------------------------
# Axis-label helpers
# ---------------------------------------------------------------------------
def _parse_cycle_time(cycle):
    """Parse MPAS cycle strings such as YYYYMMDDHH; return None if invalid."""
    try:
        return datetime.strptime(str(cycle), "%Y%m%d%H")
    except Exception:
        return None

def _compact_cycle_label(cycle, previous_labeled_cycle=None):
    """Compact time label: only 00Z is labeled; other cycles are blank.

    For 00Z cycles, show day only.  Add month at the first labeled tick or a
    new month, and add year at the first labeled tick or a new year.
    """
    dt = _parse_cycle_time(cycle)
    if dt is None:
        return str(cycle)
    if dt.hour != 0:
        return ""

    prev = (_parse_cycle_time(previous_labeled_cycle)
            if previous_labeled_cycle is not None else None)

    label = dt.strftime("%d")
    if prev is None or dt.month != prev.month or dt.year != prev.year:
        label += " " + dt.strftime("%b")
    if prev is None or dt.year != prev.year:
        label += "\n" + dt.strftime("%Y")
    return label

def _time_tick_indices(times, cfg):
    """Return tick positions; labels will be shown only at 00Z cycles."""
    n = len(times)
    if n == 0:
        return []
    max_ticks = int(cfg.get("max_time_ticks", cfg.get("plot_style", {}).get("max_time_ticks", 12)))
    max_ticks = max(2, max_ticks)

    # Prefer 00Z ticks because only those are labeled.
    zeroz = [i for i, t in enumerate(times)
             if (_parse_cycle_time(t) is not None and _parse_cycle_time(t).hour == 0)]
    if len(zeroz) == 0:
        return sorted(set(np.linspace(0, n - 1, min(max_ticks, n), dtype=int).tolist()))
    if len(zeroz) <= max_ticks:
        return zeroz
    pick = np.linspace(0, len(zeroz) - 1, max_ticks, dtype=int)
    return [zeroz[i] for i in pick]

def _set_compact_time_axis(ax, x, times, cfg, xlabel="Assimilation cycle"):
    idx = _time_tick_indices(times, cfg)
    ax.set_xticks(np.asarray(x)[idx])
    labels = []
    previous_labeled = None
    for i in idx:
        label = _compact_cycle_label(times[i], previous_labeled)
        labels.append(label)
        if label:
            previous_labeled = times[i]
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_xlabel(xlabel)
    _ensure_panel_ticks(ax)

def _format_latitude_tick(value, pos=None):
    """Format latitude values using N/S hemispheres instead of signed numbers."""
    try:
        lat = float(value)
    except Exception:
        return str(value)

    if not np.isfinite(lat):
        return ""

    if abs(lat) < 1.0e-10:
        return "0°"

    hemi = "N" if lat > 0.0 else "S"
    aval = abs(lat)

    # Avoid unnecessary trailing zeros.
    if abs(aval - round(aval)) < 1.0e-8:
        text = f"{int(round(aval))}"
    else:
        text = f"{aval:.1f}".rstrip("0").rstrip(".")

    return f"{text}°{hemi}"


def _apply_latitude_formatter(ax, axis="y"):
    """Apply degree-N/S tick labels to a latitude axis."""
    formatter = mticker.FuncFormatter(_format_latitude_tick)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)


def _is_vertical_categorical_axis(label):
    """Return True for vertical coordinates that should be drawn as equal bins.

    Height, pressure, and model-level coordinates are diagnostic bin labels here,
    not physical coordinates for pcolormesh spacing.  Treating them as categorical
    keeps every 2-D pixel the same visual size.
    """
    text = str(label or "").lower()
    keys = ("height", "pressure", "model level", "level bin", "level")
    return any(k in text for k in keys) and "latitude" not in text and "longitude" not in text


def _format_coord_tick(value):
    """Compact numeric tick label for categorical coordinate labels."""
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return ""
    if abs(v - round(v)) < 1.0e-8:
        return f"{int(round(v))}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _categorical_positions(values):
    """Return equally spaced categorical positions for an axis."""
    return np.arange(len(values), dtype=float)


def _set_categorical_axis(ax, values, axis="y", max_ticks=8):
    """Use equally spaced positions but label them with the original values."""
    vals = np.asarray(values)
    n = vals.size
    if n == 0:
        return
    nticks = min(int(max_ticks), n)
    nticks = max(1, nticks)
    if n <= nticks:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, nticks, dtype=int))
    labels = [_format_coord_tick(vals[i]) for i in idx]
    if axis == "x":
        ax.set_xticks(idx)
        ax.set_xticklabels(labels)
    else:
        ax.set_yticks(idx)
        ax.set_yticklabels(labels)


def _adaptive_geo_locator(values):
    """Return a domain-aware latitude/longitude tick locator.

    This avoids showing only 0 degrees for regional domains while still
    keeping sensible spacing for global maps.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return mticker.MaxNLocator(nbins=5)

    span = float(np.nanmax(arr) - np.nanmin(arr))
    if span <= 0.0 or not np.isfinite(span):
        return mticker.MaxNLocator(nbins=5)

    if span <= 2.0:
        step = 0.25
    elif span <= 5.0:
        step = 0.5
    elif span <= 12.0:
        step = 1.0
    elif span <= 30.0:
        step = 5.0
    elif span <= 90.0:
        step = 10.0
    else:
        step = 30.0
    return mticker.MultipleLocator(step)


def _latlon_projection_info():
    """Return Cartopy projection and lat-lon plotting metadata if available."""
    try:
        import cartopy.crs     as ccrs
        import cartopy.feature as cfeature
        projection = ccrs.PlateCarree()
        cartopy_info = {
            "has_cartopy": True,
            "transform":   ccrs.PlateCarree(),
            "cfeature":    cfeature,
            "lon_key":     "lon",
            "lat_key":     "lat",
        }
    except Exception:
        projection = None
        cartopy_info = None
    return projection, cartopy_info


def _nice_colorbar(fig, ax, pcm, label, cfg, relative=False):
    """Consistent, publication-friendly colorbar formatting."""
    cbar_cfg = cfg.get("colorbar", {})
    cbar = fig.colorbar(
        pcm, ax=ax,
        shrink=cbar_cfg.get("shrink", 0.86),
        pad=cbar_cfg.get("pad", 0.02),
        aspect=cbar_cfg.get("aspect", 28),
        extend=cbar_cfg.get("extend_relative" if relative else "extend", "both" if relative else "neither"),
    )
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"], width=0.8, length=3)
    try:
        cbar.ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    except Exception:
        pass
    return cbar


def _pixel_edge_kwargs(cfg):
    """White borders around 2-D pixels, configurable for publication clarity.

    Defaults are conservative so the grid cells are visible without making
    high-resolution plots too heavy. Set ``pixel_edges: false`` in YAML to
    disable, or tune ``pixel_edge_lw`` / ``pixel_edge_color``.
    """
    style = cfg.get("plot_style", {})
    use_edges = cfg.get("pixel_edges", style.get("pixel_edges", True))
    if not use_edges:
        return {}

    return {
        "edgecolors": cfg.get("pixel_edge_color", style.get("pixel_edge_color", "white")),
        "linewidth": float(cfg.get("pixel_edge_lw", style.get("pixel_edge_lw", 0.12))),
        "antialiased": bool(cfg.get("pixel_edge_antialiased", style.get("pixel_edge_antialiased", True))),
    }


# ---------------------------------------------------------------------------
# Colour-map / range helper  (single source of truth for absolute vs relative)
# ---------------------------------------------------------------------------

def _colourmap_settings(use_relative, stat_name, cfg, data=None, refname="cntl"):
    """Return (cmap, vmin, vmax, cbar_label) for a panel.

    For 2-D absolute ``mean`` fields, use the same diverging colour map as
    difference / relative panels, because mean diagnostics are often signed
    and centred near zero. Other absolute statistics keep the sequential
    light-to-deep colour map.
    """
    style = cfg.get("plot_style", {})
    relative_cmap = cfg.get("relative_reduction_cmap",
                            style.get("relative_cmap", "RdBu_r"))

    if use_relative:
        # Diverging and colorblind-friendly enough for signed percent changes.
        # Honour an explicit vmin/vmax if the user configured one; otherwise
        # scale symmetrically to this panel's own data (same approach as the
        # "mean" branch below) instead of a fixed +/-40% range.
        vmin = cfg.get("relative_reduction_vmin", style.get("relative_vmin", None))
        vmax = cfg.get("relative_reduction_vmax", style.get("relative_vmax", None))
        if (vmin is None or vmax is None) and data is not None:
            finite = np.asarray(data)[np.isfinite(data)]
            if finite.size > 0:
                amp = np.nanmax(np.abs(finite))
                if np.isfinite(amp) and amp > 0.0:
                    vmin = -amp if vmin is None else vmin
                    vmax =  amp if vmax is None else vmax
        # Fallback when nothing was configured and no usable data was given.
        if vmin is None:
            vmin = -40.0
        if vmax is None:
            vmax = 40.0
        return (
            relative_cmap,
            vmin,
            vmax,
            f"{stat_name} diff. (%) from {refname}",
        )

    # Special case requested: only absolute 2-D mean fields should use the
    # difference-style diverging colour map. Do not apply this to bias/OMB/OMA.
    if str(stat_name).lower() == "mean":
        vmin = cfg.get("mean_vmin", None)
        vmax = cfg.get("mean_vmax", None)
        if (vmin is None or vmax is None) and data is not None:
            finite = np.asarray(data)[np.isfinite(data)]
            if finite.size > 0:
                amp = np.nanmax(np.abs(finite))
                if np.isfinite(amp) and amp > 0.0:
                    vmin = -amp if vmin is None else vmin
                    vmax =  amp if vmax is None else vmax
        return (
            cfg.get("mean_cmap", relative_cmap),
            vmin,
            vmax,
            stat_name,
        )

    return (
        cfg.get("absolute_cmap", style.get("absolute_cmap", "YlGnBu")),
        cfg.get("absolute_vmin", None),
        cfg.get("absolute_vmax", None),
        stat_name,
    )


# ---------------------------------------------------------------------------
# Significance dot overlay
# ---------------------------------------------------------------------------
def add_significance(ax, x, y, significant, improved=None, transpose=False,
                          transform=None, marker_size=60, edge_width=0.75):
    """Scatter black hollow triangles at significant grid points.

    Upward triangle: significant improvement
    Downward triangle: significant degradation / non-improvement
    """
    sig = np.asarray(significant, dtype=bool)
    if transpose:
        sig = sig.T
    
    X, Y = np.meshgrid(x, y)

    if sig.shape != X.shape:
        print(f"WARNING: significance shape {sig.shape} vs mesh {X.shape}")
        return

    if improved is not None:
        if improved.shape != X.shape:
            print(f"WARNING: improved shape {improved.shape} vs mesh {X.shape}")
            return

    kw = {} if transform is None else {"transform": transform}

    if improved is None:
        jj, ii = np.where(sig)
        if len(jj) > 0:
            ax.scatter(X[jj, ii], Y[jj, ii],
                       s=marker_size, facecolors="none", edgecolors="black",
                       marker="o", linewidths=edge_width, zorder=6, **kw)
    else:
        jj, ii = np.where(sig & improved)
        if len(jj) > 0:
            ax.scatter(X[jj, ii], Y[jj, ii],
                       s=marker_size, facecolors="none", edgecolors="black",
                       marker="^", linewidths=edge_width, zorder=6, **kw)
        jj, ii = np.where(sig & (~improved))
        if len(jj) > 0:
            ax.scatter(X[jj, ii], Y[jj, ii],
                       s=marker_size, facecolors="none", edgecolors="black", 
                       marker="v", linewidths=edge_width, zorder=6, **kw)

# ---------------------------------------------------------------------------
# Shared 2-D pcolormesh panel builder
# ---------------------------------------------------------------------------
def _draw_2d_panels(fig, axs, var_names, exp_names, exp_labels,
                    data_store, stat_name, cfg,
                    x_key, y_key, y_label_key, invert_y_key,
                    x_axis_label, y_axis_label_override=None,
                    transpose_data=True,
                    ref_key="samples",
                    cartopy_info=None,
                    mean_cbar_limits=None):
    """
    Fill a pre-created ``axs[nvars, nexps]`` grid with pcolormesh panels.

    Parameters
    ----------
    data_store       : dict  exp_name -> var_name -> dict
    x_key, y_key     : keys for the x- and y-coordinate arrays in each entry
    y_label_key      : key for the y-axis label string
    invert_y_key     : key for the bool that controls y-axis inversion
    x_axis_label     : fixed x-axis label (e.g. "Latitude"), or None to skip
    y_axis_label_override : override the per-panel y label, or None
    transpose_data   : whether to call ``.T`` on the data before pcolormesh
    ref_key          : key for the sample array used in bootstrap reduction
    cartopy_info     : dict with keys ``transform``, ``has_cartopy``,
                       ``cfeature``, ``lon_key``, ``lat_key``
                       (only for lat-lon maps); None otherwise
    """
    boot       = get_bootstrap_settings(cfg)
    use_rel    = is_relative_stat(stat_name, cfg)
    ref_exp    = exp_names[0]
    panel_id   = 0

    for ivar, var_name in enumerate(var_names):
        ref_entry   = (data_store.get(ref_exp) or {}).get(var_name)
        ref_samples = ref_entry[ref_key] if ref_entry else None

        for iexp, exp_name in enumerate(exp_names):
            ax = axs[ivar, iexp]
            panel_id += 1
            entry = (data_store.get(exp_name) or {}).get(var_name)

            ax.grid(False)
            if entry is None:
                ax.axis("off")
                continue

            plot_data, significant = _compute_panel_data(
                entry, ref_samples, iexp, use_rel, stat_name,
                boot, cfg, ref_key, var_name=var_name,
            )
            if plot_data is None:
                ax.axis("off")
                continue

            cmap, vmin, vmax, cbar_label = _colourmap_settings(
                use_rel and iexp > 0, stat_name, cfg, data=plot_data, refname=exp_labels[0])

            # Unified colourbar limits for mean statistics.
            if (not (use_rel and iexp > 0)
                    and str(stat_name).lower() == "mean"
                    and mean_cbar_limits is not None):
                shared_limits = mean_cbar_limits.get(var_name)
                if shared_limits is not None:
                    shared_vmin, shared_vmax = shared_limits
                    if shared_vmin is not None and shared_vmax is not None:
                        vmin, vmax = shared_vmin, shared_vmax

            x    = entry[x_key]
            y    = entry[y_key]
            data = plot_data.T if transpose_data else plot_data

            raw_y_label = (y_axis_label_override
                           if y_axis_label_override
                           else entry.get(y_label_key, ""))
            use_categorical_y = (not (cartopy_info and cartopy_info["has_cartopy"])
                                 and _is_vertical_categorical_axis(raw_y_label))
            y_plot = _categorical_positions(y) if use_categorical_y else y

            if cartopy_info and cartopy_info["has_cartopy"]:
                transform = cartopy_info["transform"]
                lon       = entry[cartopy_info["lon_key"]]
                lat       = entry[cartopy_info["lat_key"]]
                pcm = ax.pcolormesh(lon, lat, data, shading="auto", rasterized=True,
                                    cmap=cmap, vmin=vmin, vmax=vmax,
                                    transform=transform,
                                    **_pixel_edge_kwargs(cfg))
                if significant is not None and boot["plot_significance"]:
                    add_significance(ax, lon, lat, significant, 
                                         improved=None if stat_name == "spread" else (data > 0),
                                         transpose=False, transform=transform,
                                         marker_size=cfg.get("significance_marker_size", 30))
                ax.coastlines(resolution="10m", linewidth=0.5)
                ax.add_feature(cartopy_info["cfeature"].BORDERS, linewidth=0.3)
                gl = ax.gridlines(draw_labels=True, linewidth=0, color='none')
                gl.top_labels   = False
                gl.right_labels = False
                gl.bottom_labels = True
                gl.left_labels = True
                try:
                    import cartopy.mpl.ticker as cticker
                    gl.xformatter = cticker.LongitudeFormatter()
                    gl.yformatter = cticker.LatitudeFormatter()
                except Exception:
                    pass
                gl.xlocator = _adaptive_geo_locator(lon)
                gl.ylocator = _adaptive_geo_locator(lat)
                
            else:
                pcm = ax.pcolormesh(x, y_plot, data, shading="auto", rasterized=True,
                                    cmap=cmap, vmin=vmin, vmax=vmax,
                                    **_pixel_edge_kwargs(cfg))
                if significant is not None and boot["plot_significance"]:
                    add_significance(ax, x, y_plot, significant,
                                         improved=None if stat_name == "spread" else (data > 0),
                                         # `data` above is plot_data.T'd (or not) per
                                         # transpose_data; `significant` (used here) has
                                         # NOT been transposed, so it must be transposed
                                         # here under the exact same condition `data` was,
                                         # to stay aligned with the (x, y_plot) mesh. A
                                         # hardcoded True here was a latent bug: harmless
                                         # for callers with transpose_data=True (lat_vert_2d,
                                         # time_level_2d, forecast_time_level_2d), but wrong
                                         # for plot_lat_lon_2d (transpose_data=False) whenever
                                         # Cartopy isn't installed and relative-reduction
                                         # significance is active -- shape mismatch, so
                                         # add_significance silently no-ops (a printed
                                         # WARNING, not a crash).
                                         transpose=transpose_data,
                                         marker_size=cfg.get("significance_marker_size", 30))

                if x_axis_label:
                    ax.set_xlabel(x_axis_label)
                    if str(x_axis_label).lower().startswith("latitude"):
                        _apply_latitude_formatter(ax, axis="x")
                    elif "lead" in str(x_axis_label).lower():
                        # Use the actual forecast lead hours (e.g. 6, 12, 18, ...)
                        # as tick positions instead of matplotlib's automatic
                        # locator, which would otherwise pick round numbers
                        # like 5, 10, 15 that don't line up with the data.
                        ax.set_xticks(x)
                y_label = raw_y_label
                ax.set_ylabel(y_label)
                if str(y_label).lower().startswith("latitude"):
                    _apply_latitude_formatter(ax, axis="y")
                if use_categorical_y:
                    _set_categorical_axis(ax, y, axis="y",
                                          max_ticks=cfg.get("vertical_tick_max", 13))
                # Categorical positions already follow the coordinate array's
                # natural order (e.g. PressureLevels listed surface-first),
                # so index 0 already lands at the bottom of the axis. The
                # invert_y flag exists for the old continuous-numeric axes
                # and would flip that back to the wrong order here -- the
                # same reason plot_profiles_1d / plot_profile_diff_1d /
                # plot_time_level_2d skip it for categorical axes too.
                elif entry.get(invert_y_key, False):
                    ax.invert_yaxis()

            _nice_colorbar(fig, ax, pcm, cbar_label, cfg, relative=(use_rel and iexp > 0))
            ax.set_title(_title_with_panel(panel_id - 1, f"{exp_labels[iexp]}\n{var_name}"))
            _ensure_panel_ticks(ax)

def _symmetric_limits_from_arrays(arrays):
    """Return symmetric colour limits covering all finite values in arrays."""
    finite_chunks = []
    for arr in arrays:
        vals = np.asarray(arr, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            finite_chunks.append(vals)

    if not finite_chunks:
        return None, None

    finite = np.concatenate(finite_chunks)
    amp = np.nanmax(np.abs(finite))
    if not np.isfinite(amp) or amp <= 0.0:
        return None, None
    return -amp, amp

def _mean_colorbar_limits(data_store, stat_name, cfg, exp_names, var_names,
                          scope=None, data_key="data"):
    """Shared colour limits for 2-D mean-stat panels.

    Default: one symmetric colour range per variable across all experiments.
    Set ``mean_colorbar_scope: all`` to use one range for the whole figure,
    or ``mean_colorbar_scope: none`` to recover per-panel autoscaling.
    Explicit ``mean_vmin`` and ``mean_vmax`` always take priority.
    """
    if str(stat_name).lower() != "mean":
        return {}

    if cfg.get("mean_vmin", None) is not None and cfg.get("mean_vmax", None) is not None:
        return {
            var_name: (cfg.get("mean_vmin"), cfg.get("mean_vmax"))
            for var_name in var_names
        }

    scope = str(scope if scope is not None else cfg.get("mean_colorbar_scope", "variable")).lower()
    if scope in ("none", "panel", "individual"):
        return {}

    if scope in ("all", "figure", "global"):
        arrays = []
        for exp_name in exp_names:
            for var_name in var_names:
                entry = (data_store.get(exp_name) or {}).get(var_name)
                if entry is not None and data_key in entry:
                    arrays.append(entry[data_key])
        shared = _symmetric_limits_from_arrays(arrays)
        return {var_name: shared for var_name in var_names}

    # Default/recommended option: one colour range per variable across experiments.
    limits = {}
    for var_name in var_names:
        arrays = []
        for exp_name in exp_names:
            entry = (data_store.get(exp_name) or {}).get(var_name)
            if entry is not None and data_key in entry:
                arrays.append(entry[data_key])
        limits[var_name] = _symmetric_limits_from_arrays(arrays)
    return limits

def _compute_panel_data(entry, ref_samples, iexp, use_rel, stat_name,
                        boot, cfg, ref_key="samples", var_name=None):
    """Return (plot_data, significant_mask) for a single panel."""
    data = entry["data"]

    if use_rel and iexp > 0:
        exp_samples = entry[ref_key]
        if ref_samples is None or ref_samples.shape != exp_samples.shape:
            return None, None
        plot_data, ci_low, ci_high = bootstrap_reduction_ci(
            ref_samples, exp_samples,
            n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"],
        )
        # Additional masking for near-zero absolute reference values.
        # This is especially important for QV, where tiny reference values can
        # create meaningless percent reductions.
        threshold = _relative_ref_threshold(var_name, cfg)
        ref_mean = safe_nanmean(ref_samples, axis=0)
        small_ref = ~np.isfinite(ref_mean) | (np.abs(ref_mean) < threshold)
        plot_data = np.where(small_ref, np.nan, plot_data)
        ci_low = np.where(small_ref, np.nan, ci_low)
        ci_high = np.where(small_ref, np.nan, ci_high)
        significant = significant_from_ci(ci_low, ci_high)
        return plot_data, significant
    return data, None

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
                _, ci_low, ci_high = bootstrap_reduction_ci(
                    ref_samples, entry["samples"],
                    n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"],
                )
                threshold = _relative_ref_threshold(var_name, cfg)
                ref_mean  = safe_nanmean(ref_samples, axis=0)
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

# ---------------------------------------------------------------------------
def plot_time_level_2d(TimeLev, profile_key, stat_name, cfg):
    """Time × level cross-section.

    Reference experiment: absolute values.
    Other experiments (for relative stats): percent reduction vs reference.
    Note: no bootstrap significance here; each x-point is a single cycle.
    """
    apply_pub_style(cfg)
    var_names = cfg["variables"]
    out_dir   = _section_out_dir(cfg, "time_level_2d")

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    use_rel = is_relative_stat(stat_name, cfg)
    mean_cbar_limits = _mean_colorbar_limits(
        TimeLev, stat_name, cfg, exp_names, var_names,
        scope=cfg.get("time_level_mean_colorbar_scope", cfg.get("mean_colorbar_scope", "variable")),
    )

    nvars, nexps = len(var_names), len(exp_names)
    fig, axs = plt.subplots(nvars, nexps,
                            figsize=(4.5 * nexps, 3.0 * nvars),
                            squeeze=False)

    panel_id = 0
    ref_exp  = exp_names[0]

    for ivar, var_name in enumerate(var_names):
        ref_data = (TimeLev.get(ref_exp) or {}).get(var_name, {}).get("data")

        for iexp, exp_name in enumerate(exp_names):
            ax = axs[ivar, iexp]
            panel_id += 1
            entry = (TimeLev.get(exp_name) or {}).get(var_name)
            if entry is None:
                ax.axis("off")
                continue

            data   = entry["data"]
            times  = entry["times"]
            levs   = entry["coord"]
            clabel = entry["coord_label"]
            x      = np.arange(len(times))

            show_rel = use_rel and iexp > 0

            if show_rel:
                if ref_data is None or ref_data.shape != data.shape:
                    ax.axis("off")
                    continue
                plot_data = np.full_like(data, np.nan)
                threshold = _relative_ref_threshold(var_name, cfg)
                valid = (np.isfinite(ref_data) & np.isfinite(data)
                         & (np.abs(ref_data) >= threshold))
                plot_data[valid] = (100.0
                                    * (ref_data[valid] - data[valid])
                                    / ref_data[valid])
            else:
                plot_data = data

            cmap, vmin, vmax, cbar_label = _colourmap_settings(
                show_rel, stat_name, cfg, data=plot_data, refname=exp_labels[0])

            # For time-level mean statistics, use unified colourbar limits
            # across experiments for the same variable by default.
            if (not show_rel) and str(stat_name).lower() == "mean":
                shared_limits = mean_cbar_limits.get(var_name)
                if shared_limits is not None:
                    shared_vmin, shared_vmax = shared_limits
                    if shared_vmin is not None and shared_vmax is not None:
                        vmin, vmax = shared_vmin, shared_vmax

            # Treat height/pressure/model levels as categorical bins for
            # time-height/time-pressure cross sections. This keeps all 2-D
            # pixels the same visual size while preserving the original values
            # as tick labels.
            use_categorical_y = _is_vertical_categorical_axis(clabel)
            y_plot = _categorical_positions(levs) if use_categorical_y else levs

            pcm = ax.pcolormesh(x, y_plot, plot_data.T, shading="auto", rasterized=True,
                                cmap=cmap, vmin=vmin, vmax=vmax,
                                **_pixel_edge_kwargs(cfg))
            _nice_colorbar(fig, ax, pcm, cbar_label, cfg, relative=show_rel)
            ax.set_title(_title_with_panel(panel_id - 1, f"{exp_labels[iexp]}\u2002{var_name}"))
            ax.set_ylabel(clabel)

            if use_categorical_y:
                _set_categorical_axis(ax, levs, axis="y",
                                      max_ticks=cfg.get("vertical_tick_max", 13))
            if clabel == "Latitude":
                _apply_latitude_formatter(ax, axis="y")

            _ensure_panel_ticks(ax)

            _set_compact_time_axis(ax, x, times, cfg, xlabel="Time")

    fig.tight_layout()
    figname=f"cross_section_time_{profile_key}_{stat_name}"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)


def plot_forecast_time_level_2d(TimeLev, profile_key, stat_name, cfg):
    """Extended-forecast lead-time x level cross-section.

    Like plot_lat_vert_2d, but the x-axis is forecast lead time instead of
    latitude. Each column already averages over all initialization cycles
    at that lead, so bootstrap significance vs. the reference/control
    experiment (exp_names[0]) is available and drawn automatically by the
    shared _draw_2d_panels / add_significance machinery -- the same
    machinery plot_lat_vert_2d and plot_lat_lon_2d already use.
    """
    apply_pub_style(cfg)
    var_names = cfg["variables"]
    out_dir   = _section_out_dir(cfg, "forecast_time_level_2d")

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    nvars, nexps = len(var_names), len(exp_names)
    fig, axs = plt.subplots(nvars, nexps,
                            figsize=(4.5 * nexps, 3.2 * nvars),
                            squeeze=False)

    mean_cbar_limits = _mean_colorbar_limits(
        TimeLev, stat_name, cfg, exp_names, var_names,
        scope=cfg.get("time_level_mean_colorbar_scope", cfg.get("mean_colorbar_scope", "variable")),
    )

    _draw_2d_panels(
        fig=fig, axs=axs,
        var_names=var_names, exp_names=exp_names, exp_labels=exp_labels,
        data_store=TimeLev, stat_name=stat_name, cfg=cfg,
        x_key="leads", y_key="coord", y_label_key="coord_label",
        invert_y_key="invert_y",
        x_axis_label="Forecast lead time (h)",
        mean_cbar_limits=mean_cbar_limits,
    )

    fig.tight_layout()
    figname = f"forecast_cross_section_lead_{profile_key}_{stat_name}"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)

# ---------------------------------------------------------------------------
def plot_lat_vert_2d(LatVert, section_key, stat_name, cfg):
    """Latitude-vertical cross-section averaged across cycles.

    Reference: absolute. Others (for relative stats): percent reduction.
    Significant regions marked with dots.
    """
    apply_pub_style(cfg)
    var_names = cfg["variables"]
    out_dir   = _section_out_dir(cfg, "lat_vert_2d")

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    nvars, nexps = len(var_names), len(exp_names)
    fig, axs = plt.subplots(nvars, nexps,
                            figsize=(4.5 * nexps, 3.2 * nvars),
                            squeeze=False)

    mean_cbar_limits = _mean_colorbar_limits(
        LatVert, stat_name, cfg, exp_names, var_names,
        scope=cfg.get("lat_vert_mean_colorbar_scope", cfg.get("mean_colorbar_scope", "variable")),
    )

    _draw_2d_panels(
        fig=fig, axs=axs,
        var_names=var_names, exp_names=exp_names, exp_labels=exp_labels,
        data_store=LatVert, stat_name=stat_name, cfg=cfg,
        x_key="lat", y_key="vert", y_label_key="vert_label",
        invert_y_key="invert_y",
        x_axis_label="Latitude",
        mean_cbar_limits=mean_cbar_limits,
    )

    fig.tight_layout()
    figname=f"cross_section_latitude_{section_key}_{stat_name}"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)

# ---------------------------------------------------------------------------
def plot_lat_lon_2d(LatLon, stat_name, cfg):
    """Global lat-lon map averaged across cycles.

    Uses Cartopy if available, otherwise plain pcolormesh.
    """
    apply_pub_style(cfg)
    projection, cartopy_info = _latlon_projection_info()

    var_names = cfg["variables"]
    out_dir   = _section_out_dir(cfg, "lat_lon_2d")

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    nvars, nexps = len(var_names), len(exp_names)

    fig, axs = plt.subplots(
        nvars, nexps,
        figsize=(5.0 * nexps, 3.5 * nvars),
        squeeze=False,
        subplot_kw={"projection": projection} if cartopy_info is not None else None,
    )

    mean_cbar_limits = _mean_colorbar_limits(
        LatLon, stat_name, cfg, exp_names, var_names,
        scope=cfg.get("lat_lon_mean_colorbar_scope", cfg.get("mean_colorbar_scope", "variable")),
    )

    _draw_2d_panels(
        fig=fig, axs=axs,
        var_names=var_names, exp_names=exp_names, exp_labels=exp_labels,
        data_store=LatLon, stat_name=stat_name, cfg=cfg,
        x_key="lon", y_key="lat", y_label_key=None,
        invert_y_key=None,
        x_axis_label="Longitude",
        y_axis_label_override="Latitude",
        transpose_data=False,
        cartopy_info=cartopy_info,
        mean_cbar_limits=mean_cbar_limits,
    )

    fig.tight_layout()
    figname=f"lat_lon_{stat_name}"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)

# ---------------------------------------------------------------------------
def _level_suffix(entry):
    """Compact suffix for selected 3-D level map filenames/titles."""
    vlabel = entry.get("vertical_label", "Level")
    level = entry.get("selected_level", None)
    if level is None:
        return "selected level"
    try:
        lv = float(level)
        if abs(lv - round(lv)) < 1.0e-8:
            lv_text = str(int(round(lv)))
        else:
            lv_text = f"{lv:.2f}".rstrip("0").rstrip(".")
    except Exception:
        lv_text = str(level)
    return f"{lv_text} {vlabel}"

def plot_lat_lon_slice_2d(LatLonLev, stat_name, cfg, section_cfg):
    """Lat-lon map at one selected pressure/height/model level from 3-D diagnostics.

    Reference experiment: absolute values. Other experiments for relative stats:
    percent difference from the reference, with bootstrap significance triangles.
    """
    apply_pub_style(cfg)
    projection, cartopy_info = _latlon_projection_info()

    var_names = cfg["variables"]
    out_dir   = _section_out_dir(cfg, "lat_lon_2d")

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    nvars, nexps = len(var_names), len(exp_names)

    fig, axs = plt.subplots(
        nvars, nexps,
        figsize=(5.0 * nexps, 3.5 * nvars),
        squeeze=False,
        subplot_kw={"projection": projection} if cartopy_info is not None else None,
    )

    mean_cbar_limits = _mean_colorbar_limits(
        LatLonLev, stat_name, cfg, exp_names, var_names,
        scope=cfg.get("lat_lon_3d_mean_colorbar_scope", cfg.get("mean_colorbar_scope", "variable")),
    )

    _draw_2d_panels(
        fig=fig, axs=axs,
        var_names=var_names, exp_names=exp_names, exp_labels=exp_labels,
        data_store=LatLonLev, stat_name=stat_name, cfg=cfg,
        x_key="lon", y_key="lat", y_label_key=None,
        invert_y_key=None,
        x_axis_label="Longitude",
        y_axis_label_override="Latitude",
        transpose_data=False,
        cartopy_info=cartopy_info,
        mean_cbar_limits=mean_cbar_limits,
    )

    # Add selected level in the figure title if available.
    first_entry = None
    for e in exp_names:
        for v in var_names:
            first_entry = (LatLonLev.get(e) or {}).get(v)
            if first_entry is not None:
                break
        if first_entry is not None:
            break
    level_text = _level_suffix(first_entry or {})
    fig.suptitle(f"{stat_name} at {level_text}", y=1.01)

    fig.tight_layout()
    section_name = section_cfg.get("name", "selected_level")
    figname=f"lat_lon_{section_name}_{stat_name}"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)

# ---------------------------------------------------------------------------
# Extended-forecast scorecard
# ---------------------------------------------------------------------------
def plot_forecast_scorecard(ScoreCard, row_labels, stat_name, cfg):
    """Extended-forecast scorecard: percent difference vs. the
    reference/control experiment (exp_names[0]), one panel per other
    experiment -- styled after a classic triangle scorecard rather than a
    colour heatmap:

      - Rows are variable/level combinations from cfg["scorecard"]["rows"]
        (e.g. "U 200 hPa", ..., "Ps"); columns are forecast lead hours.
      - Each cell is a single triangle: pointing up = improvement, pointing
        down = degradation (sign of the bootstrap percent difference).
      - Marker size scales with |percent difference|, capped at
        cfg["scorecard"]["max_score"] (default 20%).
      - Statistically significant cells (bootstrap CI vs. the reference
        excludes zero) are drawn solid/bold with a black outline; cells
        that are not significant are drawn faint with no outline, but use
        the same size-vs-|% diff| scale -- only the outline (and alpha)
        distinguishes significant from not-significant.
      - A row with no valid data at any lead time is shaded grey, matching
        the blanked-out rows in a typical precipitation-threshold
        scorecard.
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

            red, ci_low, ci_high = bootstrap_reduction_ci(
                ref_samples, exp_samples,
                n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"],
            )
            var_key   = row_labels[irow].split()[0]
            threshold = _relative_ref_threshold(var_key, cfg)
            ref_mean  = safe_nanmean(ref_samples, axis=0)
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
                improved = bool(val > 0)
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

        ref_samples = ref_entry["samples"]
        x         = ref_entry["leads"]
        has_curve = False

        for iexp, exp_name in enumerate(exp_names[1:], start=1):
            exp_entry = (ForecastLead.get(exp_name) or {}).get(var_name)
            if exp_entry is None:
                continue

            red, ci_low, ci_high = bootstrap_reduction_ci(
                ref_samples, exp_entry["samples"],
                n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"],
            )
            threshold = _relative_ref_threshold(var_name, cfg)
            ref_mean  = safe_nanmean(ref_samples, axis=0)
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
