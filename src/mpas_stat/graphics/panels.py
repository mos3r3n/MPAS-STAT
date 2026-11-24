"""Colour-map selection, significance overlays and the shared 2-D
pcolormesh panel builder used by every 2-D plot type.
"""
import numpy as np

from ..core.bootstrap import get_bootstrap_settings
from ..core.pooling import entry_difference_ci, entry_value
from .labels import cfg_var_label, stat_label, pct_diff_label
from .axes import (
    _adaptive_geo_locator,
    _apply_latitude_formatter,
    _categorical_positions,
    _is_vertical_categorical_axis,
    _nice_colorbar,
    _pixel_edge_kwargs,
    _set_categorical_axis,
)
from .colour import colour_scale, share_limits
from .style import (
    _ensure_panel_ticks,
    _relative_ref_threshold,
    _title_with_panel,
    is_relative_stat,
    significant_from_ci,
)

# ---------------------------------------------------------------------------
# Colour-map / range helper  (single source of truth for absolute vs relative)
# ---------------------------------------------------------------------------

def _colourmap_settings(use_relative, stat_name, cfg, data=None, refname="cntl"):
    """Return (cmap_name, vmin, vmax, cbar_label) for a panel.

    vmin/vmax are only the EXPLICITLY configured limits (None otherwise);
    data-driven limits are chosen by graphics.colour.colour_scale() from all
    panels that share a colour bar. For 2-D absolute ``mean`` fields the
    diverging colour map is used too, because mean diagnostics are signed and
    centred near zero; other absolute statistics use the sequential map.
    """
    style = cfg.get("plot_style", {})
    diverging_cmap = cfg.get("relative_reduction_cmap",
                             style.get("diverging_cmap", style.get("relative_cmap", "RdBu_r")))

    if use_relative:
        return (
            diverging_cmap,
            cfg.get("relative_reduction_vmin", style.get("relative_vmin", None)),
            cfg.get("relative_reduction_vmax", style.get("relative_vmax", None)),
            pct_diff_label(stat_name, refname),
        )

    if str(stat_name).lower() == "mean":
        return (
            cfg.get("mean_cmap", diverging_cmap),
            cfg.get("mean_vmin", None),
            cfg.get("mean_vmax", None),
            stat_label(stat_name),
        )

    return (
        cfg.get("absolute_cmap", style.get("absolute_cmap", "YlGnBu")),
        cfg.get("absolute_vmin", None),
        cfg.get("absolute_vmax", None),
        stat_label(stat_name),
    )


def _colorbar_extend(cfg, relative):
    """Colour-bar extension arrows, as configured for _nice_colorbar."""
    cbar_cfg = cfg.get("colorbar", {}) or {}
    return cbar_cfg.get("extend_relative" if relative else "extend",
                        "both" if relative else "neither")


def _is_diverging(use_relative, stat_name):
    return bool(use_relative) or str(stat_name).lower() == "mean"


# ---------------------------------------------------------------------------
# Significance dot overlay
# ---------------------------------------------------------------------------
def add_significance(ax, x, y, significant, improved=None, transpose=False,
                          transform=None, marker_size=60, edge_width=0.75):
    """Scatter black hollow triangles at significant grid points.

    Upward triangle: significant improvement (``improved``; for percent
    differences of RMS/STD that is a NEGATIVE difference)
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

        # Pass 1: every panel's data, so panels of one row can share a
        # rounded colour scale (colorbar.share: row).
        panels = []
        for iexp, exp_name in enumerate(exp_names):
            entry = (data_store.get(exp_name) or {}).get(var_name)
            plot_data = significant = None
            if entry is not None:
                plot_data, significant = _compute_panel_data(
                    entry, ref_samples, iexp, use_rel, stat_name,
                    boot, cfg, ref_key, var_name=var_name, ref_entry=ref_entry,
                )
            panels.append((iexp, entry, plot_data, significant))
        row_data = {
            rel: [pd for iexp, e, pd, _ in panels
                  if pd is not None and (use_rel and iexp > 0) == rel]
            for rel in (True, False)
        }

        for iexp, entry, plot_data, significant in panels:
            exp_name = exp_names[iexp]
            ax = axs[ivar, iexp]
            panel_id += 1

            ax.grid(False)
            if entry is None or plot_data is None:
                ax.axis("off")
                continue

            show_rel = use_rel and iexp > 0
            cmap_name, vmin, vmax, cbar_label = _colourmap_settings(
                show_rel, stat_name, cfg, data=plot_data, refname=exp_labels[0])

            # Unified colourbar limits for mean statistics.
            if (not show_rel
                    and str(stat_name).lower() == "mean"
                    and mean_cbar_limits is not None):
                shared_limits = mean_cbar_limits.get(var_name)
                if shared_limits is not None:
                    shared_vmin, shared_vmax = shared_limits
                    if shared_vmin is not None and shared_vmax is not None:
                        vmin, vmax = shared_vmin, shared_vmax

            diverging = _is_diverging(show_rel, stat_name)
            cmap, norm, levels = colour_scale(
                cmap_name, vmin, vmax, cfg, diverging,
                extend=_colorbar_extend(cfg, show_rel),
                data_arrays=row_data[show_rel] if share_limits(cfg) else [plot_data],
            )

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
                                    cmap=cmap, norm=norm,
                                    transform=transform,
                                    **_pixel_edge_kwargs(cfg))
                if significant is not None and boot["plot_significance"]:
                    add_significance(ax, lon, lat, significant, 
                                         improved=None if stat_name == "spread" else (data < 0),
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
                                    cmap=cmap, norm=norm,
                                    **_pixel_edge_kwargs(cfg))
                if significant is not None and boot["plot_significance"]:
                    add_significance(ax, x, y_plot, significant,
                                         improved=None if stat_name == "spread" else (data < 0),
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

            _nice_colorbar(fig, ax, pcm, cbar_label, cfg, relative=show_rel, levels=levels)
            ax.set_title(_title_with_panel(panel_id - 1, f"{exp_labels[iexp]}\n{cfg_var_label(var_name, cfg)}"))
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
                        boot, cfg, ref_key="samples", var_name=None, ref_entry=None):
    """Return (plot_data, significant_mask) for a single panel.

    Comparison panels (iexp > 0 for a relative statistic) show the percent
    difference 100 * (exp - ref) / ref: negative = lower than the reference.

    Percent reductions use pooled cycle aggregation when both `entry` and
    `ref_entry` carry a "pool" (see mpas_stat.core.pooling), else the plain
    mean of `ref_samples` / entry[ref_key]."""
    data = entry["data"]

    if use_rel and iexp > 0:
        exp_samples = entry[ref_key]
        if ref_samples is None or ref_samples.shape != exp_samples.shape:
            return None, None
        if ref_entry is None:
            ref_entry = {ref_key: ref_samples}
        plot_data, ci_low, ci_high = entry_difference_ci(ref_entry, entry, boot, key=ref_key)
        # Additional masking for near-zero absolute reference values.
        # This is especially important for QV, where tiny reference values can
        # create meaningless percent reductions.
        threshold = _relative_ref_threshold(var_name, cfg)
        ref_mean = entry_value(ref_entry, key=ref_key)
        small_ref = ~np.isfinite(ref_mean) | (np.abs(ref_mean) < threshold)
        plot_data = np.where(small_ref, np.nan, plot_data)
        ci_low = np.where(small_ref, np.nan, ci_low)
        ci_high = np.where(small_ref, np.nan, ci_high)
        significant = significant_from_ci(ci_low, ci_high)
        return plot_data, significant
    return data, None
