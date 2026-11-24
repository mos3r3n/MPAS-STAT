"""Axis extents that follow the data.

Lat-lon maps and latitude axes are gridded over a fixed box (the whole
globe unless configured otherwise), so a regional application -- or a
regional sub-domain -- used to be drawn as a small patch in an otherwise
empty frame. These helpers find the region actually covered by finite
values (cell edges of the finite cells, unioned over every panel of a
figure) and apply it.

Configured by `map_extent` (model-space graphics config, or
plot_settings.map_extent in obs space):
    auto  (default)  zoom to the data
    full             keep the full grid (previous behaviour)
    [lat_min, lat_max, lon_min, lon_max]   a fixed box
"""
import numpy as np

__all__ = ["cell_edges", "finite_extent", "store_extent", "extent_mode", "apply_extent",
           "lon_extent_center"]


def cell_edges(centers):
    """Cell edges of a 1-D array of (roughly evenly spaced) cell centres."""
    c = np.asarray(centers, dtype=np.float64).reshape(-1)
    if c.size == 0:
        return c
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    return np.r_[c[0] - (mid[0] - c[0]), mid, c[-1] + (c[-1] - mid[-1])]


def _span(edges, occupied):
    idx = np.flatnonzero(occupied)
    if idx.size == 0:
        return None
    lo, hi = edges[idx[0]], edges[idx[-1] + 1]
    return (min(lo, hi), max(lo, hi))


def _lon_span(edges, occupied):
    """Longitude span of the occupied columns; for a region crossing the
    dateline (occupied at both ends of a -180..180 grid, with a larger gap
    in between) returns lon_min > 180-wrapped form: (lon_min, lon_max + 360)."""
    idx = np.flatnonzero(occupied)
    if idx.size == 0:
        return None
    n = occupied.size
    plain = (edges[idx[0]], edges[idx[-1] + 1])
    if idx.size == n or n < 3:
        return plain
    gaps = np.diff(np.r_[idx, idx[0] + n])          # circular gaps between occupied columns
    k = int(np.argmax(gaps))
    if k == idx.size - 1:                           # largest gap is the plain outside gap
        return plain
    start, end = idx[k + 1], idx[k]                 # occupied run wraps around the grid end
    width = edges[-1] - edges[0]
    if width < 359.0:                               # not a full-circle grid
        return plain
    return (edges[start], edges[end + 1] + 360.0)


def finite_extent(x, y, data, lon_axis=None):
    """((x0, x1), (y0, y1)) cell-edge spans of the finite cells of `data`
    (shape (len(y), len(x))), or None. lon_axis "x"/"y" enables dateline
    handling for that axis."""
    d = np.ma.filled(np.ma.asarray(data, dtype=np.float64), np.nan)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if d.ndim != 2 or d.shape != (y.size, x.size):
        return None
    ok = np.isfinite(d)
    if not ok.any():
        return None
    fx = _lon_span if lon_axis == "x" else _span
    fy = _lon_span if lon_axis == "y" else _span
    return fx(cell_edges(x), ok.any(axis=0)), fy(cell_edges(y), ok.any(axis=1))


def _union(spans):
    spans = [s for s in spans if s is not None]
    if not spans:
        return None
    return (min(s[0] for s in spans), max(s[1] for s in spans))


def store_extent(data_store, x_key, y_key, transpose=False, data_key="data", lon_axis=None):
    """Union of finite_extent over every entry of data_store[exp][var]."""
    xs, ys = [], []
    for per_var in (data_store or {}).values():
        for entry in (per_var or {}).values():
            if not entry or data_key not in entry or x_key not in entry or y_key not in entry:
                continue
            d = np.asarray(entry[data_key])
            if transpose:
                d = d.T
            ext = finite_extent(entry[x_key], entry[y_key], d, lon_axis=lon_axis)
            if ext:
                xs.append(ext[0])
                ys.append(ext[1])
    if not xs:
        return None
    return _union(xs), _union(ys)


def extent_mode(cfg):
    """"auto", "full" or a [lat_min, lat_max, lon_min, lon_max] box."""
    mode = (cfg or {}).get("map_extent", "auto")
    if isinstance(mode, (list, tuple)) and len(mode) == 4:
        return [float(v) for v in mode]
    return "full" if str(mode).lower() == "full" else "auto"


def _pad(span, frac=0.03, minimum=0.5):
    if span is None:
        return None
    p = max(minimum, frac * (span[1] - span[0]))
    return (span[0] - p, span[1] + p)


def apply_extent(axs, lat_span=None, lon_span=None, cartopy_info=None, lat_axis="y",
                 clip_lat=True):
    """Set the latitude (and longitude) limits of every visible axis."""
    lat_span, lon_span = _pad(lat_span), _pad(lon_span)
    if lat_span is not None and clip_lat:
        lat_span = (max(lat_span[0], -90.0), min(lat_span[1], 90.0))
    for ax in np.ravel(axs):
        if not ax.get_visible() or not ax.axison:
            continue
        if cartopy_info and cartopy_info.get("has_cartopy") and hasattr(ax, "set_extent"):
            if lat_span is not None and lon_span is not None:
                ax.set_extent([lon_span[0], lon_span[1], lat_span[0], lat_span[1]],
                              crs=cartopy_info["transform"])
                _inner_gridline_ticks(ax, lon_span, lat_span)
            continue
        if lon_span is not None:
            ax.set_xlim(*lon_span)
        if lat_span is not None:
            (ax.set_ylim if lat_axis == "y" else ax.set_xlim)(*lat_span)


def inner_ticks(span, nbins=4):
    """Round tick values strictly inside `span` (away from the frame, so
    edge labels are not clipped)."""
    from matplotlib.ticker import MaxNLocator
    lo, hi = span
    margin = 0.06 * (hi - lo)
    ticks = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(lo + margin, hi - margin)
    return [t for t in ticks if lo + margin <= t <= hi - margin]


def _inner_gridline_ticks(ax, lon_span, lat_span):
    """Re-place the gridline labels of a zoomed Cartopy map inside it."""
    try:
        from matplotlib.ticker import FixedLocator
        lon_ticks = [((t + 180.0) % 360.0) - 180.0 for t in inner_ticks(lon_span)]
        from cartopy.mpl.gridliner import Gridliner
        gridliners = list(getattr(ax, "_gridliners", []))
        gridliners += [a for a in getattr(ax, "artists", []) if isinstance(a, Gridliner)]
        for gl in gridliners:
            gl.xlocator = FixedLocator(lon_ticks)
            gl.ylocator = FixedLocator(inner_ticks(lat_span))
    except Exception:
        pass


def lon_extent_center(lon_span):
    """Central longitude for a PlateCarree map of `lon_span` (180 when the
    span crosses the dateline, else 0)."""
    if lon_span is None:
        return 0.0
    return 180.0 if lon_span[1] > 180.0 else 0.0
