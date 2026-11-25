"""mpasstat.<date>.nc paths, netCDF access and coordinate helpers.
"""
import os
from datetime import timedelta

import numpy as np

from ..core.pooling import read_components
from ..core.timeutils import datetime_to_cycle, datetime_to_datefile


# ---------------------------------------------------------------------------
# NetCDF helpers
# ---------------------------------------------------------------------------
def read_nc_path(ds, path):
    """Read the variable at a group path such as ``"U/0D/Global/rms"`` as a float64 array."""
    obj = ds
    for p in path.strip("/").split("/"):
        obj = obj[p]
    return np.asarray(obj[...], dtype=np.float64)


def _has_expected_ndim(arr, expected_ndim):
    """Return True when the squeezed array has the expected number of dimensions."""
    return np.squeeze(np.asarray(arr)).ndim == int(expected_ndim)


def _maybe_print_missing(cfg, message):
    """Print optional missing-diagnostic messages only when requested."""
    if bool(cfg.get("verbose_missing", False)):
        print(message)


def get_mpasstat_file(base_dir, exp_name, cycle_dt):
    """Path of a cycling statistics file: ``<base_dir>/<exp>/VerifyModel/<cycle>/mpasstat.<date>.nc``."""
    cycle     = datetime_to_cycle(cycle_dt)
    date_file = datetime_to_datefile(cycle_dt)
    return os.path.join(
        base_dir, exp_name, "VerifyModel", cycle,
        f"mpasstat.{date_file}.nc",
    )


def get_mpasstat_forecast_file(base_dir, exp_name, init_dt, lead_hour):
    """Path to an extended-forecast statistics file for one init cycle and lead.

    Matches the layout written by ``scripts/run_verify_model_fcst.ksh``::

        <base_dir>/<exp_name>/VerifyModel/<init_cycle>/f<HHH>/mpasstat.<valid_date>.nc
    """
    init_cycle = datetime_to_cycle(init_dt)
    valid_dt   = init_dt + timedelta(hours=int(lead_hour))
    date_file  = datetime_to_datefile(valid_dt)
    return os.path.join(
        base_dir, exp_name, "VerifyModel", init_cycle,
        f"f{int(lead_hour):03d}",
        f"mpasstat.{date_file}.nc",
    )


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def get_profile_coord(ds, profile_key):
    """
    Return coordinate, label, and whether to invert y-axis.
    Model levels are treated as categorical indices.
    """

    # --- Latitude ---
    if profile_key == "lat_bin_profile":
        return np.asarray(ds["LatBins"][:]), "Latitude", False

    # --- Model level (categorical) ---
    if profile_key.startswith("vert_profile_"):
        nlev = ds["ModelLevels"].shape[0]
        coord = np.arange(1, nlev + 1)   # categorical index
        return coord, "Model level", False

    # --- Level bins (categorical) ---
    if profile_key.startswith("vert_bin_profile_"):
        return np.asarray(ds["LevelBins"][:]), "Model level bin", False

    # --- Height ---
    if profile_key.startswith("vert_hgt_profile_"):
        return np.asarray(ds["HeightLevels"][:]), "Height (m)", False

    # --- Pressure ---
    if profile_key.startswith("vert_pres_profile_"):
        return np.asarray(ds["PressureLevels"][:]), "Pressure (hPa)", True

    raise ValueError(f"Unknown profile key: {profile_key}")


def get_lat_vert_coord(ds, section_key):
    """Coordinates of a latitude–level section: ``(lat, vertical, vertical label, invert_y)``."""
    lat = np.asarray(ds["LatBins"][:])
    _map = {
        "lat_lev":  (ds["LevelBins"][:],    "Model level bin",  False),
        "lat_hgt":  (ds["HeightLevels"][:], "Height (m)",       False),
        "lat_pres": (ds["PressureLevels"][:], "Pressure (hPa)", False),
    }
    if section_key not in _map:
        raise ValueError(f"Unsupported latitude-vertical section key: {section_key}")
    raw_vert, vert_label, invert_y = _map[section_key]
    return lat, np.asarray(raw_vert), vert_label, invert_y


def apply_region_suffix(name, cfg):
    """Append region suffix if needed."""
    region = str(cfg.get("region", "")).strip()

    if not region:
        return name

    if name.endswith(f"_{region}"):
        return name

    return f"{name}_{region}"


def read_stat(ds, path):
    """(value, w, m1, m2) for the statistic at "<group path>/<stat>", with the
    pooling components read from the same group (see
    mpas_stat.core.pooling.read_components)."""
    group_path, stat_name = path.strip("/").rsplit("/", 1)
    grp = ds
    for p in group_path.split("/"):
        grp = grp[p]
    return read_components(grp, stat_name)
