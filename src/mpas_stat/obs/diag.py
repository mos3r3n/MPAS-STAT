"""Observation-space verification -- STATISTICS stage.

Reads IODA obs-space HDF5 files, computes statistics
  - domain-averaged                       ("0D"),
  - vertical/latitude binned profiles      ("1D"),
  - lat-lon gridded maps                   ("2D/lat_lon", domain-independent,
                                             covers plot_settings.latlon_domain's
                                             box -- global by default),
  - latitude x level cross-sections        ("2D/lat_vert", domain-independent,
                                             same latitude box as lat_lon;
                                             skipped for radiances),
and writes them to observation-space "diagnosis" netCDF files, using the
SAME group convention MPAS-STAT uses for its own mpasstat.<date>.nc files
(see mpas_stat.model.driver.write_stats_results):
    <var>/0D/<region>/<stat>
    <var>/1D/<profile_key>/<stat>
    <var>/2D/lat_lon/<stat>
    <var>/2D/lat_vert/<stat>       (pressure/height-binned sensors only)

Two independent verification modes are supported, matching two distinct
experiment "types" in the config's experiments.types list:

  - OMB/OMA (background/analysis, cycling DA). exp_type "da" (OMA too) or
    "hofx" (OMB only). Reads
        <da_dir>/<exp>/CyclingDA/<cycle>/dbOut/obsout_da_<obs>.h5
        <hofx_dir>/<exp>/<cycle>/dbOut/obsout_hofx_<obs>.h5
    time-resolved per assimilation cycle, writing one diagnosis file per
    (experiment, cycle, obs type) to
        <diag_dir>/<experiment>/VerifyObs/<cycle>/obsstat_<obs>.<date>.nc
    -- see process_experiment() / write_obs_diag_files(). The experiment
    sits at the top of the tree, exactly mirroring MPAS-STAT's own
    model-space <exp>/VerifyModel/<cycle>/mpasstat.<date>.nc convention --
    obs type is not a separate directory level, since the filename itself
    already identifies the sensor; see the path-convention comment above
    obs_diag_file() in mpas_stat.obs.common for the full rationale.

  - OMF (obs-minus-forecast, extended-forecast skill). exp_type
    "hofx_forecast". Reads
        <hofx_dir>/<exp>/HofX/<init_cycle>/f<HHH>/dbOut/obsout_hofx_<obs>.h5
    for every configured forecast lead hour (config["forecast"]: length,
    interval, start -- same schema as the model-space graphic.yaml), writing
    one diagnosis file per (experiment, init cycle, lead hour, obs type) to
        <diag_dir>/<experiment>/VerifyObs/<init_cycle>/f<HHH>/obsstat_<obs>.<valid_date>.nc
    -- mirroring mpas_stat.graphics.io.get_mpasstat_forecast_file exactly.
    See process_forecast_experiment() / write_obs_forecast_diag_files().

An experiments list may freely mix "da"/"hofx" and "hofx_forecast" entries;
each is processed and written by the function above that understands it.

This stage has NO plotting code and does not import matplotlib or any
MPAS-STAT graphics module -- see mpas_stat.obs.plot for the separate
graphics stage, which reads these files back and draws figures via
mpas_stat.graphics.functions.

Implementation modules:
  obs.reader   -- IODA file names and HDF5 reading (load_observation_data)
  obs.binning  -- 0D/1D/2D statistics kernels
  obs.bounds   -- automatic lat-lon bounds detection
  obs.process  -- per-sensor OMB/OMA and OMF processing, main()
  obs.writer   -- diagnosis netCDF writers
"""
from .reader import (
    _resolve_h5_path,
    build_filename,
    build_hofx_forecast_filename,
    _hofx_fill_mask,
    _apply_hofx_qc,
    _read_obs_type,
    _HOFX_IS_INNOVATION,
    load_observation_data,
)
from .binning import (
    calculate_stats,
    _apply_domain_mask,
    compute_stats_2d_latlon,
    compute_stats_2d_lat_vert,
)
from .bounds import (
    _accumulate_valid_latlon_bounds,
    auto_detect_latlon_bounds,
    _resolve_latlon_bounds_for,
)
from .process import (
    _INNOV_STATS,
    _SPREAD_FIELDS,
    _bin_values,
    _resolve_stat_grids,
    _allocate_results,
    _store_2d_stats,
    _store_domain_stats,
    _metadata,
    _grid_metadata,
    process_experiment,
    process_forecast_experiment,
    main,
)
from .writer import (
    write_obs_diag_files,
    write_obs_forecast_diag_files,
)

__all__ = [
    "_resolve_h5_path",
    "build_filename",
    "build_hofx_forecast_filename",
    "_hofx_fill_mask",
    "_apply_hofx_qc",
    "_read_obs_type",
    "_HOFX_IS_INNOVATION",
    "load_observation_data",
    "calculate_stats",
    "_apply_domain_mask",
    "compute_stats_2d_latlon",
    "compute_stats_2d_lat_vert",
    "_accumulate_valid_latlon_bounds",
    "auto_detect_latlon_bounds",
    "_resolve_latlon_bounds_for",
    "_INNOV_STATS",
    "_SPREAD_FIELDS",
    "_bin_values",
    "_resolve_stat_grids",
    "_allocate_results",
    "_store_2d_stats",
    "_store_domain_stats",
    "_metadata",
    "_grid_metadata",
    "process_experiment",
    "process_forecast_experiment",
    "main",
    "write_obs_diag_files",
    "write_obs_forecast_diag_files",
]


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
