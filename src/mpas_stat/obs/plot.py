"""Observation-space verification: graphics entry point.

Kept for backwards compatibility; the implementation is in
:mod:`~mpas_stat.obs.collect` (reads diagnosis files),
:mod:`~mpas_stat.obs.experiments` (experiment lists, colours) and
:mod:`~mpas_stat.obs.figures` (figure jobs, ``main``).

Run with ``python -m mpas_stat.cli.obs_graphics <config.yaml>``.
"""
from .collect import (
    collect_obs_stats_0d,
    collect_obs_profiles_1d,
    collect_obs_lat_lon_2d,
    collect_obs_time_level_2d,
    collect_obs_lat_vert_2d,
    collect_obs_forecast_lead_0d,
    collect_obs_forecast_profiles_1d,
    collect_obs_forecast_lat_lon_2d,
    collect_obs_forecast_lat_vert_2d,
    collect_obs_forecast_time_level_2d,
)
from .experiments import (
    EXPERIMENT_PALETTE,
    _build_pseudo_experiments,
    _build_forecast_experiments,
    _experiments_for_diagtype,
    _comparison_graphic_cfg,
    _profile_axis_meta,
    _base_graphic_cfg,
)
from .figures import (
    _process_sensor,
    generate_time_series_graphics,
    generate_profile_graphics,
    generate_lat_lon_graphics,
    generate_time_level_graphics,
    generate_lat_vert_graphics,
    generate_forecast_lead_graphics,
    generate_forecast_profile_graphics,
    generate_forecast_lat_lon_graphics,
    generate_forecast_lat_vert_graphics,
    generate_forecast_time_level_graphics,
    generate_plots,
    main,
)

__all__ = [
    "collect_obs_stats_0d",
    "collect_obs_profiles_1d",
    "collect_obs_lat_lon_2d",
    "collect_obs_time_level_2d",
    "collect_obs_lat_vert_2d",
    "collect_obs_forecast_lead_0d",
    "collect_obs_forecast_profiles_1d",
    "collect_obs_forecast_lat_lon_2d",
    "collect_obs_forecast_lat_vert_2d",
    "collect_obs_forecast_time_level_2d",
    "EXPERIMENT_PALETTE",
    "_build_pseudo_experiments",
    "_build_forecast_experiments",
    "_experiments_for_diagtype",
    "_comparison_graphic_cfg",
    "_profile_axis_meta",
    "_base_graphic_cfg",
    "_process_sensor",
    "generate_time_series_graphics",
    "generate_profile_graphics",
    "generate_lat_lon_graphics",
    "generate_time_level_graphics",
    "generate_lat_vert_graphics",
    "generate_forecast_lead_graphics",
    "generate_forecast_profile_graphics",
    "generate_forecast_lat_lon_graphics",
    "generate_forecast_lat_vert_graphics",
    "generate_forecast_time_level_graphics",
    "generate_plots",
    "main",
]


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
