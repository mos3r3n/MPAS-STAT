"""2-D plots: time-level, latitude-vertical, lat-lon maps and lat-lon slices.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

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
from .panels import _colourmap_settings, _draw_2d_panels, _mean_colorbar_limits
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
