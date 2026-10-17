import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Any, Optional

import netCDF4 as nc4
import numpy as np

from .variables import extract_mpas_variables, extract_mpas_variables_list
from .config import MPASConfig
from .kernels import (
    prepare_one_variable_for_stats,
    calculate_statistics_one_task,
    build_task_list,
)
from .interp import build_shared_interp_coeffs, configure_numba_threads


def setup_logger(config: MPASConfig) -> logging.Logger:
    """Setup application logger."""
    logger = logging.getLogger("MPAS_STATS")
    logger.setLevel(logging.DEBUG if getattr(config, "verbose", False) else logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _get_nworkers(config: MPASConfig, n_tasks: int) -> int:
    """Use only config.n_processes to control all thread-pool stages."""
    return min(max(1, config.n_processes), max(1, n_tasks))


def prepare_bin_metadata(config: MPASConfig) -> Dict[str, Any]:
    """Read grid information and prepare bin metadata."""
    with nc4.Dataset(config.init_file, 'r') as init:
        lat = np.degrees(init.variables['latCell'][:]).astype(np.float64)
        lon = np.degrees(init.variables['lonCell'][:]).astype(np.float64)
        lon = (lon + 180.0) % 360.0 - 180.0
        nVertLevels = init.dimensions['nVertLevels'].size

    if config.regional:
        lat_min, lat_max = lat.min(), lat.max()
        lon_min, lon_max = lon.min(), lon.max()
        lat_min_rounded = np.floor(lat_min)
        lat_max_rounded = np.ceil(lat_max)
        lon_min_rounded = np.floor(lon_min)
        lon_max_rounded = np.ceil(lon_max)
    else:
        lat_min_rounded = -90.0
        lat_max_rounded = 90.0
        lon_min_rounded = -180.0
        lon_max_rounded = 180.0

    lon_bins = np.arange(
        lon_min_rounded,
        lon_max_rounded + config.lon_bin_width,
        config.lon_bin_width,
        dtype=np.float64,
    )
    lat_bins = np.arange(
        lat_min_rounded,
        lat_max_rounded + config.lat_bin_width,
        config.lat_bin_width,
        dtype=np.float64,
    )

    # Derive centers from the SAME edges used for binning.  Computing them
    # independently with arange(...) drops the last center whenever the
    # regional span is not an exact multiple of the bin width (e.g. lat
    # 20..49 with 2-degree bins: 15 bins but only 14 centers), which later
    # made the netCDF writer create a dimension shorter than the statistic
    # arrays and fail with a shape mismatch.
    lon_centers = 0.5 * (lon_bins[:-1] + lon_bins[1:])
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])

    nlat = len(lat_bins) - 1
    nlon = len(lon_bins) - 1
    if len(lat_centers) != nlat or len(lon_centers) != nlon:
        raise ValueError(
            "Lat/lon bin metadata is inconsistent: "
            f"{nlat} latitude bins but {len(lat_centers)} centers, "
            f"{nlon} longitude bins but {len(lon_centers)} centers"
        )

    lat_idx = np.digitize(lat, lat_bins) - 1
    lon_idx = np.digitize(lon, lon_bins) - 1

    lat_idx = np.clip(lat_idx, 0, nlat - 1)
    lon_idx = np.clip(lon_idx, 0, nlon - 1)

    group_ids_latlon = lat_idx * nlon + lon_idx

    level_bins = np.arange(
        0, nVertLevels + config.level_bin_size, config.level_bin_size, dtype=np.int32
    )
    level_bins[-1] = nVertLevels
    level_centers = 0.5 * (level_bins[:-1] + level_bins[1:])

    return {
        "lat": lat,
        "lon": lon,
        "nVertLevels": nVertLevels,
        "lat_bins": lat_bins,
        "lon_bins": lon_bins,
        "lat_centers": lat_centers,
        "lon_centers": lon_centers,
        "lat_idx": lat_idx,
        "lon_idx": lon_idx,
        "group_ids_latlon": group_ids_latlon,
        "level_bins": level_bins,
        "level_centers": level_centers,
    }


def _initialize_accumulator(arr: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "sum": np.zeros_like(arr, dtype=np.float64),
        "sumsq": np.zeros_like(arr, dtype=np.float64),
        "count": np.zeros_like(arr, dtype=np.int32),
    }


def _update_accumulator(accum: Dict[str, np.ndarray], arr: np.ndarray):
    valid = np.isfinite(arr)
    accum["sum"][valid] += arr[valid]
    accum["sumsq"][valid] += arr[valid] * arr[valid]
    accum["count"][valid] += 1


def _finalize_accumulator(accum: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    sum_arr = accum["sum"]
    sumsq_arr = accum["sumsq"]
    count_arr = accum["count"]

    mean_arr = np.full_like(sum_arr, np.nan, dtype=np.float64)
    sprd_arr = np.full_like(sum_arr, np.nan, dtype=np.float64)

    valid_count = count_arr > 0
    mean_arr[valid_count] = sum_arr[valid_count] / count_arr[valid_count]

    var_arr = np.full_like(sum_arr, np.nan, dtype=np.float64)
    var_arr[valid_count] = sumsq_arr[valid_count] / count_arr[valid_count] - mean_arr[valid_count] ** 2
    var_arr[valid_count] = np.maximum(var_arr[valid_count], 0.0)
    sprd_arr[valid_count] = np.sqrt(var_arr[valid_count])

    return mean_arr, sprd_arr


def prepare_vertical_support_fields(
    config: MPASConfig,
    variable_cache: Optional[Dict[str, Dict[str, Optional[np.ndarray]]]] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Prepare support fields required for vertical interpolation.

    If variable_cache is provided and contains P, reuse it to avoid rereading.
    """
    init_z = np.asarray(extract_mpas_variables(config.init_file, 'Z'), dtype=np.float64)

    if variable_cache is not None and "P" in variable_cache:
        init_p = np.asarray(variable_cache["P"]["init"], dtype=np.float64) / 100.0
        fcst_p_mean = np.asarray(variable_cache["P"]["fcst_mean"], dtype=np.float64) / 100.0
        fcst_p_spread = variable_cache["P"]["fcst_sprd"]
        if fcst_p_spread is not None:
            fcst_p_spread = np.asarray(fcst_p_spread, dtype=np.float64) / 100.0
    else:
        init_p = np.asarray(extract_mpas_variables(config.init_file, 'P'), dtype=np.float64) / 100.0

        if config.n_member == 1:
            fcst_p_mean = np.asarray(
                extract_mpas_variables(config.forecast_file, 'P'),
                dtype=np.float64
            ) / 100.0
            fcst_p_spread = None
        else:
            raise RuntimeError(
                "Ensemble run without cached P is not supported in this optimized driver. "
                "Prepare variable_cache first and pass it into prepare_vertical_support_fields()."
            )

    return {
        "init_z": init_z,
        "init_p": init_p,
        "fcst_p_mean": fcst_p_mean,
        "fcst_p_spread": fcst_p_spread,
    }


def prepare_all_variable_fields_deterministic(
    config: MPASConfig,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    """
    Preload all requested variables for deterministic runs.
    """
    vars_needed = list(config.variables)
    if "P" not in vars_needed:
        vars_needed.append("P")

    logger.info(f"Loading {len(vars_needed)} deterministic variables in one pass")

    init_data = extract_mpas_variables_list(config.init_file, vars_needed)
    fcst_data = extract_mpas_variables_list(config.forecast_file, vars_needed)

    variable_cache: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    for var_name in vars_needed:
        variable_cache[var_name] = {
            "init": np.asarray(init_data[var_name], dtype=np.float64),
            "fcst_mean": np.asarray(fcst_data[var_name], dtype=np.float64),
            "fcst_sprd": None,
        }

    logger.info("Finished deterministic preprocessing")
    return variable_cache


def _read_one_member_fields(imem: int, forecast_template: str, vars_needed: list[str]):
    """
    Read all requested variables from one ensemble member.

    Kept as a top-level helper so it can be used safely by executors.
    """
    fcst_file = forecast_template.format(imem)
    member_data = extract_mpas_variables_list(fcst_file, vars_needed)
    return imem, fcst_file, {
        var_name: np.asarray(member_data[var_name], dtype=np.float64)
        for var_name in vars_needed
    }


def prepare_all_variable_fields_ensemble(
    config: MPASConfig,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    """
    Preload requested variables for ensemble runs.

    Changes from the original version:
    - optionally read ensemble members concurrently;
    - keep one file-open per member;
    - update sum/sumsq/count immediately as each member finishes, so member
      arrays are released quickly.
    """
    variable_cache: Dict[str, Dict[str, Optional[np.ndarray]]] = {}

    vars_needed = list(dict.fromkeys(config.variables))
    if "P" not in vars_needed:
        vars_needed.append("P")

    n_io_workers = min(
        max(1, int(getattr(config, "n_processes", 1))),
        max(1, int(config.n_member)),
    )

    logger.info(
        f"Preparing ensemble mean/spread for {len(vars_needed)} variables "
        f"from {config.n_member} members using {n_io_workers} I/O worker(s)"
    )

    init_data = extract_mpas_variables_list(config.init_file, vars_needed)
    accumulators: Dict[str, Dict[str, np.ndarray]] = {}

    def update_from_member(member_data: Dict[str, np.ndarray]):

        if not accumulators:
            for var_name in vars_needed:
                arr0 = np.asarray(member_data[var_name], dtype=np.float64)
                accumulators[var_name] = {
                    "sum": np.zeros_like(arr0, dtype=np.float64),
                    "sumsq": np.zeros_like(arr0, dtype=np.float64),
                    "count": np.zeros_like(arr0, dtype=np.int32),
                }

        for var_name in vars_needed:
            arr = np.asarray(member_data[var_name], dtype=np.float64)
            accum = accumulators[var_name]

            valid = np.isfinite(arr)
            # These three in-place updates are the main ensemble-mean/spread work.
            # np.add(..., where=...) avoids creating large temporary masked arrays.
            np.add(accum["sum"], arr, out=accum["sum"], where=valid)
            np.add(accum["sumsq"], arr * arr, out=accum["sumsq"], where=valid)
            accum["count"] += valid.astype(np.int32, copy=False)

    if n_io_workers == 1:
        for imem in range(1, config.n_member + 1):
            imem, fcst_file, member_data = _read_one_member_fields(
                imem, config.forecast_file, vars_needed
            )
            logger.info(f"Read ensemble member {imem:03d}: {fcst_file}")
            update_from_member(member_data)
    else:
        with ThreadPoolExecutor(max_workers=n_io_workers) as executor:
            future_map = {
                executor.submit(
                    _read_one_member_fields,
                    imem,
                    config.forecast_file,
                    vars_needed,
                ): imem
                for imem in range(1, config.n_member + 1)
            }

            for future in as_completed(future_map):
                imem = future_map[future]
                try:
                    imem, fcst_file, member_data = future.result()
                except Exception as e:
                    logger.error(f"Failed reading ensemble member {imem:03d}: {e}")
                    raise
                logger.info(f"Read ensemble member {imem:03d}: {fcst_file}")
                update_from_member(member_data)

    for var_name in vars_needed:
        fcst_mean, fcst_sprd = _finalize_accumulator(accumulators[var_name])
        variable_cache[var_name] = {
            "init": np.asarray(init_data[var_name], dtype=np.float64),
            "fcst_mean": fcst_mean,
            "fcst_sprd": fcst_sprd,
        }

    logger.info("Finished ensemble mean/spread preprocessing")
    return variable_cache


def prepare_all_variable_fields(
    config: MPASConfig,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    """
    Preload all requested variables.

    Deterministic run:
        read init/forecast once each, all variables together
    Ensemble run:
        read each member file once, all variables together, then accumulate
    """
    if config.n_member == 1:
        return prepare_all_variable_fields_deterministic(config, logger)
    return prepare_all_variable_fields_ensemble(config, logger)


def prepare_all_variables_for_stats(
    variable_cache: Dict[str, Dict[str, Optional[np.ndarray]]],
    support: Dict[str, Optional[np.ndarray]],
    config: MPASConfig,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    """
    Prepare derived arrays for all variables.

    New design:
    - build shared interpolation coefficients once
    - do NOT parallelize over variables
    - interpolation itself is parallelized over target levels
    """
    prepared_cache: Dict[str, Dict[str, Any]] = {}
    vars_to_prepare = list(config.variables)

    interp_threads = int(getattr(config, "n_processes", 1))
    actual_interp_threads = configure_numba_threads(interp_threads)
    logger.info(f"Using {actual_interp_threads} numba thread(s) for vertical interpolation")

    logger.info("Building shared interpolation coefficients once")
    shared_coeffs = build_shared_interp_coeffs(
        {
            "init_z": support["init_z"],
            "init_p": support["init_p"],
            "fcst_p_mean": support["fcst_p_mean"],
        },
        config,
        use_numba=True,
    )
    logger.info("Shared interpolation coefficients prepared")

    logger.info(f"Preparing derived fields for {len(vars_to_prepare)} variables (serial over variables)")

    for var_name in vars_to_prepare:
        try:
            prepared_cache[var_name] = prepare_one_variable_for_stats(
                var_name=var_name,
                fields=variable_cache[var_name],
                shared_coeffs=shared_coeffs,
                use_numba=True,
            )
            logger.debug(f"Prepared {var_name}")
        except Exception as e:
            logger.error(f"Failed while preparing {var_name}: {e}")
            raise RuntimeError(f"Failed while preparing {var_name}: {e}") from e

    return prepared_cache


def calculate_statistics_all_tasks(
    prepared_cache: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
    config: MPASConfig,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    """
    Run statistics as parallel (variable, statistic-product) tasks.
    """
    final_results: Dict[str, Dict[str, Any]] = {}
    task_timers: Dict[str, float] = {}
    variable_timers: Dict[str, float] = {}

    tasks, expected_dim_groups = build_task_list(prepared_cache, config)
    nworkers = _get_nworkers(config, len(tasks))

    logger.info(f"Running {len(tasks)} statistic-product tasks with {nworkers} workers")

    variable_submit_time = {var_name: None for var_name in config.variables}
    variable_done_groups = {var_name: set() for var_name in config.variables}

    with ThreadPoolExecutor(max_workers=nworkers) as executor:
        future_info = {}

        for var_name, task_name in tasks:
            t0 = time.time()
            if variable_submit_time[var_name] is None:
                variable_submit_time[var_name] = t0

            future = executor.submit(
                calculate_statistics_one_task,
                var_name,
                task_name,
                prepared_cache[var_name],
                meta,
                config,
            )
            future_info[future] = (var_name, task_name, t0)

        for future in as_completed(future_info):
            var_name, task_name, t0 = future_info[future]
            try:
                returned_var, dim_group, result = future.result()

                final_results.setdefault(returned_var, {})
                final_results[returned_var].setdefault(dim_group, {})
                final_results[returned_var][dim_group].update(result)

                elapsed = time.time() - t0
                task_timers[f"{returned_var}:{task_name}"] = elapsed
                logger.info(f"Finished {returned_var}, {task_name} in {elapsed:.2f} s")

                variable_done_groups[returned_var].add(dim_group)
                if variable_done_groups[returned_var] == expected_dim_groups[returned_var]:
                    var_elapsed = time.time() - variable_submit_time[returned_var]
                    variable_timers[returned_var] = var_elapsed
                    logger.info(f"Finished all statistics for {returned_var} in {var_elapsed:.2f} s")

            except Exception as e:
                logger.error(f"Failed task ({var_name}, {task_name}): {e}")
                raise RuntimeError(f"Failed task ({var_name}, {task_name}): {e}") from e

    final_results["_task_timing_seconds"] = task_timers
    final_results["_variable_timing_seconds"] = variable_timers
    return final_results


def calculate_statistics(config: MPASConfig):
    """Main driver for all statistics."""
    start_time = time.time()
    logger = setup_logger(config)

    logger.info("Starting MPAS statistics calculation")

    meta = prepare_bin_metadata(config)
    logger.info("Metadata prepared")

    variable_cache = prepare_all_variable_fields(config, logger)
    logger.info("All variables loaded")

    support = prepare_vertical_support_fields(config, variable_cache)
    logger.info("Vertical support fields prepared")

    prepared_cache = prepare_all_variables_for_stats(
        variable_cache=variable_cache,
        support=support,
        config=config,
        logger=logger,
    )
    logger.info("All variables preprocessed for statistics")

    final_results = calculate_statistics_all_tasks(
        prepared_cache=prepared_cache,
        meta=meta,
        config=config,
        logger=logger,
    )

    write_stats_results(final_results, config, logger)

    total_time = time.time() - start_time
    logger.info(f"Completed all processing in {total_time:.2f} seconds")
    logger.info(f"Throughput: {max(total_time, 1e-12) / len(config.variables):.2f} seconds/variable")


def write_stat_group(
    group,
    stat_dict: Dict[str, np.ndarray],
    dim_map: Dict[int, Tuple[str, ...]]
):
    """Write one stat dictionary into a netCDF group."""
    for stat_name, values in stat_dict.items():
        arr = np.asarray(values)

        if arr.ndim == 0:
            var = group.createVariable(stat_name, 'f4', ())
            var[:] = arr
        else:
            dims = dim_map[arr.ndim]
            var = group.createVariable(stat_name, 'f4', dims)
            var[:] = arr.astype(np.float32)


def write_stats_results(results, config: MPASConfig, logger: logging.Logger):
    """Write all statistics to netCDF output."""
    if os.path.exists(config.output_file) and not config.overwrite:
        raise FileExistsError(
            f"Output file already exists: {config.output_file}. "
            f"Use overwrite=True or --overwrite to replace it."
        )

    logger.info(f"Writing output to {config.output_file}")

    meta = prepare_bin_metadata(config)
    lat_centers = meta["lat_centers"]
    lon_centers = meta["lon_centers"]
    level_centers = meta["level_centers"]
    nVertLevels = meta["nVertLevels"]

    with nc4.Dataset(config.output_file, 'w') as out:
        out.createDimension('nLatBins', len(lat_centers))
        out.createDimension('nLonBins', len(lon_centers))
        out.createDimension('nLevelBins', len(level_centers))
        out.createDimension('nHeightLevels', len(config.hgt_levels))
        out.createDimension('nPressureLevels', len(config.pres_levels))
        out.createDimension('nVertLevels', nVertLevels)

        lat_bin = out.createVariable('LatBins', 'f4', ('nLatBins',))
        lat_bin[:] = lat_centers
        lat_bin.units = 'degrees_north'

        lon_bin = out.createVariable('LonBins', 'f4', ('nLonBins',))
        lon_bin[:] = lon_centers
        lon_bin.units = 'degrees_east'

        lev_bin = out.createVariable('LevelBins', 'f4', ('nLevelBins',))
        lev_bin[:] = level_centers
        lev_bin.units = 'model_level_bin_center'

        levels = out.createVariable('ModelLevels', 'i4', ('nVertLevels',))
        levels[:] = np.arange(nVertLevels)
        levels.units = 'model_level'

        hgt_levels = out.createVariable('HeightLevels', 'f4', ('nHeightLevels',))
        hgt_levels[:] = np.asarray(config.hgt_levels, dtype=np.float32)
        hgt_levels.units = 'm'

        pres_levels = out.createVariable('PressureLevels', 'f4', ('nPressureLevels',))
        pres_levels[:] = np.asarray(config.pres_levels, dtype=np.float32)
        pres_levels.units = 'hPa'

        for var_name, dim_results in results.items():
            if var_name in ["_task_timing_seconds", "_variable_timing_seconds"]:
                continue

            var_grp = out.createGroup(var_name)

            for dim_type, dim_data in dim_results.items():
                dim_grp = var_grp.createGroup(dim_type)

                if dim_type == "0D":
                    for region_name, stat_dict in dim_data.items():
                        region_grp = dim_grp.createGroup(region_name)
                        write_stat_group(region_grp, stat_dict, {})

                elif dim_type == "1D":
                    for key, stat_dict in dim_data.items():
                        grp = dim_grp.createGroup(key)

                        if key == "lat_bin_profile":
                            dim_map = {1: ('nLatBins',)}
                        elif key.startswith("vert_profile_"):
                            dim_map = {1: ('nVertLevels',)}
                        elif key.startswith("vert_bin_profile_"):
                            dim_map = {1: ('nLevelBins',)}
                        elif key.startswith("vert_hgt_profile_"):
                            dim_map = {1: ('nHeightLevels',)}
                        elif key.startswith("vert_pres_profile_"):
                            dim_map = {1: ('nPressureLevels',)}
                        else:
                            raise ValueError(f"Unknown 1D result key: {key}")

                        write_stat_group(grp, stat_dict, dim_map)

                elif dim_type == "2D":
                    for key, stat_dict in dim_data.items():
                        grp = dim_grp.createGroup(key)

                        if key == "lat_lon":
                            dim_map = {2: ('nLatBins', 'nLonBins')}
                        elif key == "lat_lev":
                            dim_map = {2: ('nLatBins', 'nLevelBins')}
                        elif key == "lat_hgt":
                            dim_map = {2: ('nLatBins', 'nHeightLevels')}
                        elif key == "lat_pres":
                            dim_map = {2: ('nLatBins', 'nPressureLevels')}
                        else:
                            raise ValueError(f"Unknown 2D result key: {key}")

                        write_stat_group(grp, stat_dict, dim_map)

                elif dim_type == "3D":
                    for key, stat_dict in dim_data.items():
                        grp = dim_grp.createGroup(key)

                        if key == "lat_lon_lev":
                            dim_map = {3: ('nLatBins', 'nLonBins', 'nLevelBins')}
                        elif key == "lat_lon_hgt":
                            dim_map = {3: ('nLatBins', 'nLonBins', 'nHeightLevels')}
                        elif key == "lat_lon_pres":
                            dim_map = {3: ('nLatBins', 'nLonBins', 'nPressureLevels')}
                        else:
                            raise ValueError(f"Unknown 3D result key: {key}")

                        write_stat_group(grp, stat_dict, dim_map)

                else:
                    raise ValueError(f"Unknown dim_type: {dim_type}")

        if "_task_timing_seconds" in results:
            timing_grp = out.createGroup("task_timing_seconds")
            task_order = list(results["_task_timing_seconds"].keys())
            timing_grp.createDimension("nTasks", len(task_order))
            elapsed_var = timing_grp.createVariable("elapsed_seconds", "f4", ("nTasks",))
            elapsed_var[:] = np.array(
                [results["_task_timing_seconds"][k] for k in task_order],
                dtype=np.float32
            )
            timing_grp.task_order = ", ".join(task_order)

        if "_variable_timing_seconds" in results:
            timing_grp = out.createGroup("variable_timing_seconds")
            timing_grp.createDimension("nVariables", len(config.variables))
            elapsed_var = timing_grp.createVariable("elapsed_seconds", "f4", ("nVariables",))
            elapsed_var[:] = np.array(
                [results["_variable_timing_seconds"].get(v, np.nan) for v in config.variables],
                dtype=np.float32
            )
            timing_grp.variable_order = ", ".join(config.variables)

    logger.info("Output writing completed")
