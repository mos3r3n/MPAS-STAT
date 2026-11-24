"""Axis helpers: compact cycle/time axes, latitude and coordinate tick
formatting, categorical axes, geographic locators and colour bars.
"""
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .colour import colorbar_ticks
from .style import _ensure_panel_ticks


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
        import cartopy.crs as ccrs
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


def _nice_colorbar(fig, ax, pcm, label, cfg, relative=False, levels=None):
    """Consistent, publication-friendly colorbar formatting.

    With discrete ``levels`` (see graphics.colour), the bar shows one block
    per level and ticks on a subset of the level boundaries."""
    cbar_cfg = cfg.get("colorbar", {}) or {}
    cbar = fig.colorbar(
        pcm, ax=ax,
        shrink=cbar_cfg.get("shrink", 0.86),
        pad=cbar_cfg.get("pad", 0.02),
        aspect=cbar_cfg.get("aspect", 28),
        extend=cbar_cfg.get("extend_relative" if relative else "extend", "both" if relative else "neither"),
    )
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"], width=0.8, length=3)
    if levels is not None:
        cbar.set_ticks(colorbar_ticks(levels, max_ticks=int(cbar_cfg.get("max_ticks", 9))))
        cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
    else:
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
