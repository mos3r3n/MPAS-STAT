"""Data access for the model-space graphics stage.

Backwards-compatible facade: date helpers and bootstrap statistics live in
mpas_stat.core, netCDF helpers in graphics.ncread, and the collectors in
graphics.collect (cycling) and graphics.collect_forecast (extended forecast).
"""
from ..core.timeutils import (
    yyyymmddhh_to_datetime,
    datetime_to_cycle,
    datetime_to_datefile,
    generate_cycles,
    generate_forecast_leads,
)
from ..core.bootstrap import (
    safe_nanmean,
    safe_nanpercentile,
    get_bootstrap_settings,
    bootstrap_mean_ci,
    bootstrap_reduction_ci,
)
from .ncread import (
    read_nc_path,
    _has_expected_ndim,
    _maybe_print_missing,
    get_mpasstat_file,
    get_mpasstat_forecast_file,
    get_profile_coord,
    get_lat_vert_coord,
    apply_region_suffix,
)
from .collect import (
    collect_stats_0d,
    collect_profiles_1d,
    collect_time_level_2d,
    collect_lat_vert_2d,
    collect_lat_lon_2d,
    _canonical_3d_lat_lon_section,
    _get_3d_lat_lon_vertical_coord,
    _nearest_level_index,
    _slice_3d_lat_lon_field,
    collect_lat_lon_slice_2d,
)
from .collect_forecast import (
    collect_forecast_lead_0d,
    collect_forecast_profiles_1d,
    collect_forecast_time_level_2d,
    collect_forecast_lat_vert_2d,
    collect_forecast_lat_lon_2d,
    collect_forecast_scorecard,
)

__all__ = [
    "yyyymmddhh_to_datetime",
    "datetime_to_cycle",
    "datetime_to_datefile",
    "generate_cycles",
    "generate_forecast_leads",
    "safe_nanmean",
    "safe_nanpercentile",
    "get_bootstrap_settings",
    "bootstrap_mean_ci",
    "bootstrap_reduction_ci",
    "read_nc_path",
    "_has_expected_ndim",
    "_maybe_print_missing",
    "get_mpasstat_file",
    "get_mpasstat_forecast_file",
    "get_profile_coord",
    "get_lat_vert_coord",
    "apply_region_suffix",
    "collect_stats_0d",
    "collect_profiles_1d",
    "collect_time_level_2d",
    "collect_lat_vert_2d",
    "collect_lat_lon_2d",
    "_canonical_3d_lat_lon_section",
    "_get_3d_lat_lon_vertical_coord",
    "_nearest_level_index",
    "_slice_3d_lat_lon_field",
    "collect_lat_lon_slice_2d",
    "collect_forecast_lead_0d",
    "collect_forecast_profiles_1d",
    "collect_forecast_time_level_2d",
    "collect_forecast_lat_vert_2d",
    "collect_forecast_lat_lon_2d",
    "collect_forecast_scorecard",
]
