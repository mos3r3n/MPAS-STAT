"""2-D plots: time-level, latitude-vertical, lat-lon maps and lat-lon slices.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from .labels import cfg_var_label
from .axes import (
    _apply_latitude_formatter,
    _categorical_positions,
    _is_vertical_categorical_axis,
    _latlon_projection_info,
    _nice_colorbar,
    _pixel_edge_kwargs,
    _set_categorical_axis,
    _set_compact_time_axis,
)
from .colour import colour_scale, share_limits
from .extent import apply_extent, extent_mode, finite_extent, lon_extent_center, store_extent
from .panels import (
    _colorbar_extend,
    _colourmap_settings,
    _draw_2d_panels,
    _is_diverging,
    _mean_colorbar_limits,
)
from .style import (
    _ensure_panel_ticks,
    _relative_ref_threshold,
    _save_fig,
    _section_out_dir,
    _title_with_panel,
    apply_pub_style,
    get_exp_plot_settings,
    is_relative_stat,
)


# ---------------------------------------------------------------------------
def _time_level_panel_data(data, ref_data, show_rel, var_name, cfg):
    """Absolute data, or the percent difference 100*(exp-ref)/ref
    (negative = lower than the reference) for comparison panels; None when
    the reference is missing or has a different shape."""
    if not show_rel:
        return data
    if ref_data is None or ref_data.shape != data.shape:
        return None
    plot_data = np.full_like(data, np.nan)
    threshold = _relative_ref_threshold(var_name, cfg)
    valid = (np.isfinite(ref_data) & np.isfinite(data)
             & (np.abs(ref_data) >= threshold))
    plot_data[valid] = 100.0 * (data[valid] - ref_data[valid]) / ref_data[valid]
    return plot_data


def _map_spans(store, cfg):
    """(lat_span, lon_span) to show on a lat-lon map: the configured box,
    the finite data (auto), or (None, None) for the full grid."""
    mode = extent_mode(cfg)
    if mode == "full":
        return None, None
    if isinstance(mode, list):
        return (mode[0], mode[1]), (mode[2], mode[3])
    ext = store_extent(store, "lon", "lat", transpose=False, lon_axis="x")
    if ext is None:
        return None, None
    return ext[1], ext[0]


def _map_figsize(nvars, nexps, lat_span, lon_span, cartopy_info):
    """Figure size for a grid of lat-lon maps, with rows fitted to the map.

    Maps are drawn with equal aspect, so a fixed panel size leaves a wide
    domain (the globe is about 2.5:1) as a short strip in a tall slot, with
    a colour bar far longer than the map. Each panel instead gets
    ``MAP_WIDTH`` inches of map plus room for the colour bar, and a row
    height that fits the map plus its title and tick labels. Tall domains
    are capped at ``MAP_MAX_HEIGHT`` and drawn narrower.
    """
    if not (cartopy_info and cartopy_info.get("has_cartopy")):
        return 5.0 * nexps, 3.5 * nvars
    lat0, lat1 = lat_span if lat_span is not None else (-90.0, 90.0)
    lon0, lon1 = lon_span if lon_span is not None else (-180.0, 180.0)
    aspect = max(lon1 - lon0, 1e-3) / max(lat1 - lat0, 1e-3)
    aspect = min(max(aspect, 0.4), 4.0)
    map_w = MAP_WIDTH
    map_h = map_w / aspect
    if map_h > MAP_MAX_HEIGHT:
        map_h = MAP_MAX_HEIGHT
        map_w = map_h * aspect
    panel_w = map_w + MAP_SIDE_SPACE
    row_h = map_h + MAP_ROW_SPACE
    return panel_w * nexps, row_h * nvars


MAP_WIDTH = 3.9        # inches of map per panel
MAP_MAX_HEIGHT = 4.5   # cap for tall domains
MAP_SIDE_SPACE = 1.4   # latitude labels and colour bar
MAP_ROW_SPACE = 1.0    # two-line panel title and longitude labels


def _latitude_rows_span(spans, coord, data_t):
    """Accumulate the latitude span of the finite rows of data_t
    (shape (len(coord), n))."""
    ext = finite_extent(np.arange(np.asarray(data_t).shape[1]), coord, data_t)
    if ext:
        spans.append(ext[1])


def plot_time_level_2d(TimeLev, profile_key, stat_name, cfg):
    """Time × level cross-section.

    Reference experiment: absolute values.
    Other experiments (for relative stats): percent reduction vs reference.
    Note: no bootstrap significance here; each x-point is a single cycle.
    """
    apply_pub_style(cfg)
    lat_spans = []
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

        # Pass 1: every panel's data, so a row can share one colour scale.
        panels = []
        for iexp, exp_name in enumerate(exp_names):
            entry = (TimeLev.get(exp_name) or {}).get(var_name)
            plot_data = None
            if entry is not None:
                plot_data = _time_level_panel_data(entry["data"], ref_data, use_rel and iexp > 0,
                                                   var_name, cfg)
            panels.append((iexp, entry, plot_data))
        row_data = {rel: [pd for iexp, e, pd in panels
                          if pd is not None and (use_rel and iexp > 0) == rel]
                    for rel in (True, False)}

        for iexp, entry, plot_data in panels:
            ax = axs[ivar, iexp]
            panel_id += 1
            if entry is None or plot_data is None:
                ax.axis("off")
                continue

            times  = entry["times"]
            levs   = entry["coord"]
            clabel = entry["coord_label"]
            x      = np.arange(len(times))

            show_rel = use_rel and iexp > 0

            cmap_name, vmin, vmax, cbar_label = _colourmap_settings(
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

            cmap, norm, levels = colour_scale(
                cmap_name, vmin, vmax, cfg, _is_diverging(show_rel, stat_name),
                extend=_colorbar_extend(cfg, show_rel),
                data_arrays=row_data[show_rel] if share_limits(cfg) else [plot_data],
            )
            if clabel == "Latitude":
                _latitude_rows_span(lat_spans, levs, plot_data.T)
            pcm = ax.pcolormesh(x, y_plot, plot_data.T, shading="auto", rasterized=True,
                                cmap=cmap, norm=norm,
                                **_pixel_edge_kwargs(cfg))
            _nice_colorbar(fig, ax, pcm, cbar_label, cfg, relative=show_rel, levels=levels)
            ax.set_title(_title_with_panel(panel_id - 1, f"{exp_labels[iexp]}\u2002{cfg_var_label(var_name, cfg)}"))
            ax.set_ylabel(clabel)

            if use_categorical_y:
                _set_categorical_axis(ax, levs, axis="y",
                                      max_ticks=cfg.get("vertical_tick_max", 13))
            if clabel == "Latitude":
                _apply_latitude_formatter(ax, axis="y")

            _ensure_panel_ticks(ax)

            _set_compact_time_axis(ax, x, times, cfg, xlabel="Time")

    if lat_spans and extent_mode(cfg) != "full":
        box = extent_mode(cfg)
        apply_extent(axs, lat_span=(box[0], box[1]) if isinstance(box, list) else
                     (min(s[0] for s in lat_spans), max(s[1] for s in lat_spans)))
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
    first = next((e for pv in TimeLev.values() for e in (pv or {}).values() if e), None)
    if first and first.get("coord_label") == "Latitude" and extent_mode(cfg) != "full":
        box = extent_mode(cfg)
        lat_span = (box[0], box[1]) if isinstance(box, list) else (
            (store_extent(TimeLev, "leads", "coord", transpose=True) or (None, None))[1])
        apply_extent(axs, lat_span=lat_span)

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
    if extent_mode(cfg) != "full":
        box = extent_mode(cfg)
        lat_span = (box[0], box[1]) if isinstance(box, list) else (
            (store_extent(LatVert, "lat", "vert", transpose=True) or (None, None))[0])
        apply_extent(axs, lat_span=lat_span, lat_axis="x")

    fig.tight_layout()
    figname=f"cross_section_latitude_{section_key}_{stat_name}"
    _save_fig(fig, os.path.join(out_dir, figname), cfg)


# ---------------------------------------------------------------------------
def plot_lat_lon_2d(LatLon, stat_name, cfg):
    """Global lat-lon map averaged across cycles.

    Uses Cartopy if available, otherwise plain pcolormesh.
    """
    apply_pub_style(cfg)
    lat_span, lon_span = _map_spans(LatLon, cfg)
    projection, cartopy_info = _latlon_projection_info(lon_extent_center(lon_span))

    var_names = cfg["variables"]
    out_dir   = _section_out_dir(cfg, "lat_lon_2d")

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    nvars, nexps = len(var_names), len(exp_names)

    fig, axs = plt.subplots(
        nvars, nexps,
        figsize=_map_figsize(nvars, nexps, lat_span, lon_span, cartopy_info),
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
    apply_extent(axs, lat_span, lon_span, cartopy_info)

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
    lat_span, lon_span = _map_spans(LatLonLev, cfg)
    projection, cartopy_info = _latlon_projection_info(lon_extent_center(lon_span))

    var_names = cfg["variables"]
    out_dir   = _section_out_dir(cfg, "lat_lon_2d")

    exp_names, exp_labels, colors, linestyles = get_exp_plot_settings(cfg)
    nvars, nexps = len(var_names), len(exp_names)

    fig, axs = plt.subplots(
        nvars, nexps,
        figsize=_map_figsize(nvars, nexps, lat_span, lon_span, cartopy_info),
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
    apply_extent(axs, lat_span, lon_span, cartopy_info)

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
