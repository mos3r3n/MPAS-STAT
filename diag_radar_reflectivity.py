#!/usr/bin/env python3
"""
Radar composite-reflectivity forecast verification -- statistics stage.

Computes gridded, neighborhood-based verification scores (FSS / TS / ETS /
FBIAS / POD / FAR / RMSE) for radar composite reflectivity, comparing
gridded obs vs. HofX composite reflectivity at each (experiment, init
cycle, forecast lead) triple, and writes ONE diagnosis netCDF file per
triple -- mirroring diag_ioda_obsout_stats.py's OMF (obs-minus-forecast)
extended-forecast product (same directory convention, same
resolve_time_settings()/generate_forecast_leads()/lead_valid_time_allowed()
cycle/lead handling, same build_hofx_forecast_filename() HofX file lookup),
but for a fundamentally different kind of observation: radar reflectivity
is verified as a GRIDDED, neighborhood-based composite field (FSS/
categorical scores against a regular lat/lon grid), not as per-observation
binned statistics -- so it gets its own statistics-stage script rather than
being folded into process_forecast_experiment().

Only "hofx_forecast"-typed experiments are processed here, exactly like
diag_ioda_obsout_stats.py's own OMF product -- radar composite reflectivity
has no cycling/OMB-OMA counterpart in this package (see
docs/USER_GUIDE.md, section 2.6).

Computation kernels and the config schema live in radar_verify_common.py,
shared with plot_radar_reflectivity.py. Gridding and RMSE are computed
natively (Python); per-cycle FSS/TS/ETS/FBIAS/POD/FAR neighborhood scores
are computed by MET's grid_stat (invoked through METplus) -- see
radar_verify_common.py's module docstring for the full METplus
integration design. The standalone radar_reflectivity_verify.py script
this product was originally integrated from is untouched by that change
and keeps its own independent, fully-native implementation.

Usage (mirrors diag_ioda_obsout_stats.py):
    python diag_radar_reflectivity.py [config.yaml]
"""
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional

import netCDF4 as nc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obsstat_common import (
    load_config, resolve_diag_base_dir, generate_cycles, generate_forecast_leads,
    resolve_time_settings, lead_valid_time_allowed, radar_composite_diag_file,
)
from diag_ioda_obsout_stats import build_hofx_forecast_filename
from radar_verify_common import resolve_radar_settings, process_one_cycle, default_embed_script_path


def _build_metplus_opts(radar_cfg: Dict, diag_base_dir: str) -> Dict:
    """Resolves radar_cfg's METplus-related settings (see
    radar_verify_common.DEFAULT_RADAR_CONFIG) into the plain dict
    process_one_cycle() expects as its `metplus_opts` argument -- computed
    ONCE per (sensor, experiment) here, not per cycle, since
    `metplus_work_dir`'s default depends on diag_base_dir (which
    radar_verify_common.py itself has no reason to know)."""
    work_dir = radar_cfg.get("metplus_work_dir") or os.path.join(diag_base_dir, "metplus_scratch")
    return dict(
        work_dir=work_dir,
        run_metplus_cmd=radar_cfg.get("run_metplus_cmd", "run_metplus.py"),
        nbrhd_shape=radar_cfg.get("nbrhd_shape", "SQUARE"),
        nbrhd_cov_thresh=radar_cfg.get("nbrhd_cov_thresh", ">=0.5"),
        timeout_sec=radar_cfg.get("metplus_timeout_sec", 600),
        keep_scratch=radar_cfg.get("keep_metplus_scratch", False),
        model_name=radar_cfg.get("metplus_model_name"),
        obtype=radar_cfg.get("metplus_obtype"),
        embed_script=default_embed_script_path(),
    )


def _write_radar_diag_file(out_file: str, thresholds, radii, result: Dict) -> None:
    """Write one (experiment, init cycle, lead)'s radar verification
    result -- both the derived scores and the raw accumulator components
    (see radar_verify_common.process_one_cycle()'s docstring for why both
    are always written) -- to a single diagnosis netCDF file."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with nc.Dataset(out_file, "w") as ds:
        ds.createDimension("nThresholds", len(thresholds))
        ds.createDimension("nRadii", len(radii))

        v = ds.createVariable("Thresholds", "f4", ("nThresholds",))
        v[:] = np.asarray(thresholds, dtype=np.float32)
        v.units = "dBZ"

        v = ds.createVariable("Radii", "f4", ("nRadii",))
        v[:] = np.asarray(radii, dtype=np.float32)
        v.units = "grid cells"

        rv = ds.createVariable("RMSE", "f4", ())
        rv[:] = np.float32(result["rmse"])
        rv.units = "dBZ"

        # Per-cycle derived scores.
        for name, key in [("FSS", "fss"), ("TS", "ts"), ("ETS", "ets"),
                          ("FBIAS", "fbias"), ("POD", "pod"), ("FAR", "far")]:
            var = ds.createVariable(name, "f4", ("nThresholds", "nRadii"))
            var[:] = np.asarray(result[key], dtype=np.float32)

        # Raw accumulator components (let the graphics stage aggregate
        # across cycles either way -- see aggregate_radar_cycles()).
        var = ds.createVariable("FSS_num", "f4", ("nThresholds", "nRadii"))
        var[:] = np.asarray(result["fss_num"], dtype=np.float32)
        var = ds.createVariable("FSS_denom", "f4", ("nThresholds", "nRadii"))
        var[:] = np.asarray(result["fss_denom"], dtype=np.float32)
        for name, key in [("NA", "na"), ("NB", "nb"), ("NC", "nc"), ("ND", "nd")]:
            var = ds.createVariable(name, "i8", ("nThresholds", "nRadii"))
            var[:] = np.asarray(result[key], dtype=np.int64)

        # Provenance only (not read back by plot_radar_reflectivity.py):
        # the combined MET .stat file grid_stat wrote for this cycle, for
        # anyone who wants to inspect or feed it to their own
        # stat_analysis jobs independently of this pipeline.
        ds.metplus_stat_file = result.get("stat_file") or ""


def process_radar_experiment(config: Dict, diag_obs: str, exp_name: str, diag_base_dir: str,
                            radar_cfg: Dict) -> None:
    """Process every (init_cycle, lead) pair for one radar sensor / one
    "hofx_forecast" experiment, writing one diagnosis file per pair.
    Mirrors process_forecast_experiment()'s cycle/lead loop in
    diag_ioda_obsout_stats.py, but the per-file work (HDF5 read + KDTree
    gridding + a run_metplus.py/grid_stat subprocess) is much more
    expensive here than the point-wise pipeline's, so it is parallelized
    across a ProcessPoolExecutor -- see
    radar_verify_common.process_one_cycle()."""
    hofx_dir = config["paths"]["hofx_dir"]
    adjusted_start, end_time, interval = resolve_time_settings(config, diag_obs, for_forecast=True)
    cycles = generate_cycles(adjusted_start, end_time, interval)
    leads = generate_forecast_leads(config)

    thresholds = radar_cfg["thresholds"]
    radii = radar_cfg["radii"]
    sample_bin = radar_cfg["sample_bin"]
    diag_var = radar_cfg["diag_var"]
    n_workers = radar_cfg.get("n_workers") or config.get("n_workers", 1)
    n_workers = max(1, int(n_workers))
    metplus_opts = _build_metplus_opts(radar_cfg, diag_base_dir)
    os.makedirs(metplus_opts["work_dir"], exist_ok=True)

    tasks = []  # (init_time, lead, fname)
    for init_time in cycles:
        for lead in leads:
            # Same restriction as process_forecast_experiment(): the INIT
            # time is never restricted (a forecast can launch whenever the
            # model cycles), only each lead's VALID time, per whatever
            # cycle_valid_hours this radar sensor may have configured.
            if not lead_valid_time_allowed(diag_obs, init_time, lead):
                continue
            fname = build_hofx_forecast_filename(hofx_dir, exp_name, init_time, lead, diag_obs)
            tasks.append((init_time, lead, fname))

    if not tasks:
        return

    print(f"[{diag_obs}] {exp_name}: processing {len(tasks)} (init, lead) radar files "
          f"across {n_workers} worker process(es)")

    results = {}
    if n_workers == 1:
        for init_time, lead, fname in tasks:
            results[(init_time, lead)] = process_one_cycle(
                fname, thresholds, radii, sample_bin, diag_var=diag_var,
                exp_name=exp_name, diag_obs=diag_obs, init_time=init_time, lead_hour=lead,
                metplus_opts=metplus_opts)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(process_one_cycle, fname, thresholds, radii, sample_bin,
                          diag_var=diag_var, exp_name=exp_name, diag_obs=diag_obs,
                          init_time=init_time, lead_hour=lead,
                          metplus_opts=metplus_opts): (init_time, lead)
                for init_time, lead, fname in tasks
            }
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    results[key] = fut.result()
                except Exception:
                    print(f"ERROR processing radar file for {diag_obs}/{exp_name} "
                         f"{key}:\n{traceback.format_exc()}")
                    results[key] = None

    n_written = 0
    n_missing = 0
    for (init_time, lead), result in results.items():
        if result is None:
            n_missing += 1
            continue
        out_file = radar_composite_diag_file(diag_base_dir, exp_name, diag_obs, init_time, lead)
        _write_radar_diag_file(out_file, thresholds, radii, result)
        n_written += 1

    if n_missing:
        print(f"[{diag_obs}] {exp_name}: {n_missing} of {len(tasks)} radar files "
              f"were missing or unreadable and were skipped.")
    print(f"[{diag_obs}] {exp_name}: wrote {n_written} radar diagnosis files")


def main(config_path: Optional[str] = None) -> None:
    """Compute radar composite-reflectivity verification statistics and
    write diagnosis files -- a no-op if the config has no "radar" section
    (this product is entirely opt-in, see radar_verify_common.
    resolve_radar_settings())."""
    config = load_config(config_path)
    radar_cfg = resolve_radar_settings(config)
    if radar_cfg is None:
        return

    diag_base_dir = resolve_diag_base_dir(config)
    os.makedirs(diag_base_dir, exist_ok=True)

    exp_names = config["experiments"]["names"]
    exp_types = config["experiments"].get("types", ["da"] * len(exp_names))

    for diag_obs in radar_cfg["sensors"]:
        for exp_name, etype in zip(exp_names, exp_types):
            # Radar composite reflectivity is OMF-only -- only extended-
            # forecast HofX experiments have the (init, lead) file layout
            # this product reads (see build_hofx_forecast_filename()).
            if etype != "hofx_forecast":
                continue
            process_radar_experiment(config, diag_obs, exp_name, diag_base_dir, radar_cfg)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
