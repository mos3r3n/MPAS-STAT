"""Observation-space verification: statistics stage.

Reads IODA observation-space HDF5 files and computes

- domain averages (``0D``);
- vertical or latitude profiles (``1D``);
- lat-lon maps (``2D/lat_lon``) over ``plot_settings.latlon_domain``
  (global by default);
- latitude-level cross-sections (``2D/lat_vert``; not for radiances);

then writes them to diagnosis netCDF files with the same group layout as
the model-space ``mpasstat.<date>.nc`` files::

    <var>/0D/<region>/<stat>
    <var>/1D/<profile_key>/<stat>
    <var>/2D/lat_lon/<stat>
    <var>/2D/lat_vert/<stat>

Two modes, chosen by each experiment's ``types``:

O−B / O−A (``da``, or ``hofx`` for O−B only)
    Reads ``<da_dir>/<exp>/CyclingDA/<cycle>/dbOut/obsout_da_<obs>.h5`` or
    ``<hofx_dir>/<exp>/<cycle>/dbOut/obsout_hofx_<obs>.h5`` and writes one
    file per experiment, cycle and observation type to
    ``<diag_dir>/<exp>/VerifyOBS/<cycle>/obsstat_<obs>.<date>.nc``.

O−F (``hofx_forecast``)
    Reads ``<hofx_dir>/<exp>/HofX/<init>/f<HHH>/dbOut/obsout_hofx_<obs>.h5``
    for every lead in ``forecast`` and writes
    ``<diag_dir>/<exp>/VerifyOBS/<init>/f<HHH>/obsstat_<obs>.<valid>.nc``.

The two kinds can be mixed in one experiments list. This stage does not
import matplotlib; :mod:`mpas_stat.obs.plot` draws the figures.

Implementation modules: :mod:`~mpas_stat.obs.reader` (file names, HDF5
reading), :mod:`~mpas_stat.obs.binning` (statistics kernels),
:mod:`~mpas_stat.obs.bounds` (lat-lon bounds), :mod:`~mpas_stat.obs.process`
(per-sensor processing, ``main``) and :mod:`~mpas_stat.obs.writer`
(diagnosis files).
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
