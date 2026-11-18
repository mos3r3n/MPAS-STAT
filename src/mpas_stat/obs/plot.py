"""Observation-space verification -- GRAPHICS stage (entry point).

Backwards-compatible facade. The implementation is split into:
  obs.collect      -- read obs-space diagnosis netCDF files back into the
                      dicts the shared plot_* functions expect
  obs.experiments  -- experiment lists / colours / cfg dicts for plotting
  obs.figures      -- generate_*_graphics job builders and main()

Run with:  python -m mpas_stat.cli.obs_graphics <config.yaml>
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
