"""Plotting for MPAS diagnostic statistics.

Backwards-compatible facade: the implementation now lives in focused
modules (style, axes, panels, plots_0d, plots_1d, plots_2d, scorecard).
Import from here or from the specific module; both work.
"""
from .axes import (
    _adaptive_geo_locator,
    _apply_latitude_formatter,
    _categorical_positions,
    _compact_cycle_label,
    _format_coord_tick,
    _format_latitude_tick,
    _is_vertical_categorical_axis,
    _latlon_projection_info,
    _nice_colorbar,
    _parse_cycle_time,
    _pixel_edge_kwargs,
    _set_categorical_axis,
    _set_compact_time_axis,
    _time_tick_indices,
)
from .panels import (
    _colourmap_settings,
    _compute_panel_data,
    _draw_2d_panels,
    _mean_colorbar_limits,
    _symmetric_limits_from_arrays,
    add_significance,
)
from .plots_0d import (
    plot_forecast_lead_0d,
    plot_forecast_lead_diff_0d,
    plot_time_series_0d,
    plot_time_series_reduction_0d,
)
from .plots_1d import (
    plot_error_consistency_1d,
    plot_profile_diff_1d,
    plot_profile_reduction_1d,
    plot_profiles_1d,
)
from .plots_normdep import (
    plot_normdep_histogram,
    plot_normdep_profile,
)
from .plots_ensemble import (
    plot_rank_histogram,
    plot_rank_histogram_levels,
    plot_spread_skill,
)
from .plots_2d import (
    _level_suffix,
    plot_forecast_time_level_2d,
    plot_lat_lon_2d,
    plot_lat_lon_slice_2d,
    plot_lat_vert_2d,
    plot_time_level_2d,
)
from .scorecard import (
    plot_forecast_scorecard,
)
from .style import (
    _ensure_panel_ticks,
    _panel_label,
    _relative_ref_threshold,
    _save_fig,
    _section_out_dir,
    _title_with_panel,
    apply_pub_style,
    auto_layout,
    get_exp_plot_settings,
    is_relative_stat,
    significant_from_ci,
)

__all__ = [
    "plot_error_consistency_1d",
    "plot_normdep_histogram",
    "plot_normdep_profile",
    "plot_rank_histogram",
    "plot_rank_histogram_levels",
    "plot_spread_skill",
    "apply_pub_style",
    "auto_layout",
    "get_exp_plot_settings",
    "is_relative_stat",
    "significant_from_ci",
    "_save_fig",
    "_panel_label",
    "_title_with_panel",
    "_ensure_panel_ticks",
    "_relative_ref_threshold",
    "_section_out_dir",
    "_parse_cycle_time",
    "_compact_cycle_label",
    "_time_tick_indices",
    "_set_compact_time_axis",
    "_format_latitude_tick",
    "_apply_latitude_formatter",
    "_is_vertical_categorical_axis",
    "_format_coord_tick",
    "_categorical_positions",
    "_set_categorical_axis",
    "_adaptive_geo_locator",
    "_latlon_projection_info",
    "_nice_colorbar",
    "_pixel_edge_kwargs",
    "_colourmap_settings",
    "add_significance",
    "_draw_2d_panels",
    "_symmetric_limits_from_arrays",
    "_mean_colorbar_limits",
    "_compute_panel_data",
    "plot_time_series_0d",
    "plot_time_series_reduction_0d",
    "plot_forecast_lead_0d",
    "plot_forecast_lead_diff_0d",
    "plot_profiles_1d",
    "plot_profile_reduction_1d",
    "plot_profile_diff_1d",
    "plot_time_level_2d",
    "plot_forecast_time_level_2d",
    "plot_lat_vert_2d",
    "plot_lat_lon_2d",
    "_level_suffix",
    "plot_lat_lon_slice_2d",
    "plot_forecast_scorecard",
]
