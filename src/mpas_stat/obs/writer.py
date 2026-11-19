"""Observation-space diagnosis netCDF writers
(same <var>/0D/<region>/<stat>, <var>/1D/<profile_key>/<stat>,
<var>/2D/lat_lon/<stat> convention as src/mpas_stat/model/driver.py).
"""
import os
from typing import Dict, Tuple

import netCDF4 as nc4
import numpy as np

from .common import (
    STAT_KEYS,
    STAT_KEYS_2D,
    nc_group_name,
    obs_diag_file,
    obs_diag_forecast_file,
    profile_key_for_domain,
)
from .sensor_config import get_bin_method

# ============================================================================
# Observation-space diagnosis files
# (same <var>/0D/<region>/<stat>, <var>/1D/<profile_key>/<stat>,
#  <var>/2D/lat_lon/<stat> convention as src/mpas_stat/model/driver.py)
# ============================================================================

def _write_diag_dataset(out_file: str, results: Dict, bin_method: int, diagtypes,
                        iexp: int, time_idx: Tuple[int, ...]) -> None:
    """Write ONE diagnosis netCDF file holding every diagtype in `diagtypes`
    for experiment `iexp` at time index `time_idx` -- ``(it,)`` for a cycling
    file, ``(it, il)`` for one forecast (init cycle, lead hour) file.

    Layout (shared by OMB/OMA and OMF files):
        Levels / LatBins2D / LonBins2D coordinate variables
        <diagtype>/<var>/0D/<domain>/<stat>
        <diagtype>/<var>/1D/<profile_key>/<stat>
        <diagtype>/<var>/2D/lat_lon/<stat>
        <diagtype>/<var>/2D/lat_vert/<stat>   (pressure/height sensors only)
    """
    var_names = list(results["variables"])
    domains = results["domain_names"]
    levels = np.asarray(results["levels"], dtype=np.float64)
    lat_centers = results.get("latlon_lat_centers")
    lon_centers = results.get("latlon_lon_centers")
    latvert_lat_centers = results.get("latvert_lat_centers")
    level_units = {1: "hPa", 2: "km", 3: "degrees_north"}[bin_method]
    global_idx = domains.index("global") if "global" in domains else None

    with nc4.Dataset(out_file, "w") as ds:
        ds.createDimension("nLevels", len(levels))
        lev_var = ds.createVariable("Levels", "f4", ("nLevels",))
        lev_var[:] = levels.astype(np.float32)
        lev_var.units = level_units

        have_2d = lat_centers is not None and lon_centers is not None
        have_latvert = latvert_lat_centers is not None
        # lat_vert reuses the SAME latitude centers as the lat-lon
        # map (see resolve_latlon_bounds()), so a single nLatBins2D
        # dimension/LatBins2D variable serves both -- created if
        # either product is enabled, from whichever centers array
        # is available (they are numerically identical).
        if have_2d or have_latvert:
            any_lat_centers = lat_centers if have_2d else latvert_lat_centers
            ds.createDimension("nLatBins2D", len(any_lat_centers))
            lb = ds.createVariable("LatBins2D", "f4", ("nLatBins2D",))
            lb[:] = np.asarray(any_lat_centers, dtype=np.float32)
            lb.units = "degrees_north"
        if have_2d:
            ds.createDimension("nLonBins2D", len(lon_centers))
            lo = ds.createVariable("LonBins2D", "f4", ("nLonBins2D",))
            lo[:] = np.asarray(lon_centers, dtype=np.float32)
            lo.units = "degrees_east"

        if bin_method == 3:
            profile_entries = ([("lat_bin_profile", global_idx)]
                               if global_idx is not None else [])
        else:
            profile_entries = [
                (profile_key_for_domain(bin_method, domain), idom)
                for idom, domain in enumerate(domains)
            ]

        for diagtype in diagtypes:
            diag_grp = ds.createGroup(diagtype)

            for iv, var in enumerate(var_names):
                var_grp = diag_grp.createGroup(nc_group_name(var))

                # ---- 0D: domain-averaged scalar stats ----
                d0 = var_grp.createGroup("0D")
                for idom, domain in enumerate(domains):
                    region_grp = d0.createGroup(domain)
                    for stat_disp, prefix in STAT_KEYS:
                        key = f"{prefix}_{diagtype}"
                        if key not in results:
                            continue
                        val = results[key][(iexp, idom) + time_idx + (iv,)]
                        var_out = region_grp.createVariable(stat_disp, "f4", ())
                        var_out[:] = np.float32(val)

                # ---- 1D: binned profile stats ----
                d1 = var_grp.createGroup("1D")
                for pkey, idom in profile_entries:
                    prof_grp = d1.createGroup(pkey)
                    for stat_disp, prefix in STAT_KEYS:
                        key = f"{prefix}_{diagtype}_bin"
                        if key not in results:
                            continue
                        arr = results[key][(iexp, idom) + time_idx + (slice(None), iv)]
                        var_out = prof_grp.createVariable(stat_disp, "f4", ("nLevels",))
                        var_out[:] = arr.astype(np.float32)

                # ---- 2D: lat-lon gridded stats (global) ----
                if have_2d:
                    d2 = var_grp.createGroup("2D").createGroup("lat_lon")
                    for stat_disp, prefix in STAT_KEYS_2D:
                        key = f"{prefix}_{diagtype}_latlon"
                        if key not in results:
                            continue
                        arr = results[key][(iexp,) + time_idx + (slice(None), slice(None), iv)]
                        var_out = d2.createVariable(
                            stat_disp, "f4", ("nLatBins2D", "nLonBins2D")
                        )
                        var_out[:] = arr.astype(np.float32)

                # ---- 2D: lat-vert (level x latitude) cross-section
                #      -- domain-independent, skipped for radiances ----
                if have_latvert:
                    d2v = var_grp.groups.get("2D") or var_grp.createGroup("2D")
                    d2v = d2v.createGroup("lat_vert")
                    for stat_disp, prefix in STAT_KEYS_2D:
                        key = f"{prefix}_{diagtype}_latvert"
                        if key not in results:
                            continue
                        arr = results[key][(iexp,) + time_idx + (slice(None), slice(None), iv)]
                        var_out = d2v.createVariable(
                            stat_disp, "f4", ("nLevels", "nLatBins2D")
                        )
                        var_out[:] = arr.astype(np.float32)


def _experiments_to_write(config: Dict):
    """Experiments whose diagnosis files are written: experiments.write if
    given, else all. The other configured experiments are still read, so
    they shape the common observation sample (statistics.common_sample) --
    this is how a per-experiment job (scripts/run_verify_obs.ksh) writes
    only its own experiment's files while matching against the others."""
    names = config["experiments"]["names"]
    write = config["experiments"].get("write")
    if write is None:
        return set(names)
    if isinstance(write, str):
        write = [write]
    unknown = [w for w in write if w not in names]
    if unknown:
        raise ValueError(f"experiments.write lists unknown experiment(s): {unknown}")
    return set(write)


def write_obs_diag_files(results: Dict, config: Dict, diag_obs: str, diag_base_dir: str) -> None:
    """
    Write one observation-space diagnosis netCDF file per (experiment, cycle),
    following the same 0D/1D/2D group convention MPAS-STAT uses for its own
    mpasstat.<date>.nc model-space statistics files.
    """
    bin_method = get_bin_method(diag_obs)
    exp_names = config["experiments"]["names"]
    exp_types = config["experiments"].get("types", ["da"] * len(exp_names))
    plot_oma = config["plot_settings"].get("plot_oma", True)

    write = _experiments_to_write(config)

    for iexp, exp_name in enumerate(exp_names):
        if exp_types[iexp] == "hofx_forecast":
            # written separately by write_obs_forecast_diag_files()
            continue
        if exp_name not in write:
            continue

        diagtypes = ["omb"]
        if exp_types[iexp] == "da" and plot_oma:
            diagtypes.append("oma")

        for it, cycle_time in enumerate(results["times"]):
            out_file = obs_diag_file(diag_base_dir, exp_name, diag_obs, cycle_time)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            _write_diag_dataset(out_file, results, bin_method, diagtypes, iexp, (it,))
            print(f"Wrote obs diagnosis file: {out_file}")


def write_obs_forecast_diag_files(results: Dict, config: Dict, diag_obs: str, diag_base_dir: str) -> None:
    """
    Write one OMF diagnosis netCDF file per (experiment, init cycle, forecast
    lead hour), following the same 0D/1D/2D-lat_lon/2D-lat_vert group
    convention as write_obs_diag_files() -- just under a single top-level
    "omf" group and the f<HHH>-nested forecast path
    (obs_diag_forecast_file()).
    """
    bin_method = get_bin_method(diag_obs)
    exp_names = config["experiments"]["names"]
    exp_types = config["experiments"].get("types", [])

    write = _experiments_to_write(config)

    for iexp, exp_name in enumerate(exp_names):
        if exp_types[iexp] != "hofx_forecast":
            continue
        if exp_name not in write:
            continue

        for it, init_time in enumerate(results["times"]):
            for il, lead in enumerate(results["leads"]):
                out_file = obs_diag_forecast_file(diag_base_dir, exp_name, diag_obs, init_time, lead)
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                _write_diag_dataset(out_file, results, bin_method, ["omf"], iexp, (it, il))
                print(f"Wrote obs forecast diagnosis file: {out_file}")
