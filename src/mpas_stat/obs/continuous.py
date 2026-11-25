"""MET-style continuous statistics from pooled partial sums.

The statistics stage stores MET SL1L2 partial sums (and, for ensembles,
ECNT terms) per domain and vertical bin (statistics.met_statistics). Here
they are pooled over cycles -- each mean weighted by its TOTAL, exactly as
``stat_analysis -job aggregate_stat -line_type SL1L2`` does -- and turned
into MET's continuous statistics (CNT line type; F = forecast = H(x),
O = observation):

========  =====================================================
ME        mean(F − O) = FBAR − OBAR (MET sign: opposite of O−B)
MSE       FFBAR − 2 FOBAR + OOBAR
RMSE      √MSE
ESTDEV    √(MSE − ME²), standard deviation of the error
MAE       mean absolute error, mean(abs(F − O))
MBIAS     FBAR / OBAR
FSTDEV    √(FFBAR − FBAR²)
OSTDEV    √(OOBAR − OBAR²)
PR_CORR   (FOBAR − FBAR·OBAR) / (FSTDEV · OSTDEV)
========  =====================================================

Standard deviations use the population form (divide by TOTAL). Ensembles
add CRPS_EMP, CRPS_EMP_FAIR, SPREAD and SPREAD_PLUS_OERR, pooled the same
way (spreads as the root of the pooled mean variance).

:func:`export_continuous_tables` writes everything as one CSV per sensor.
"""
import csv
import os

import netCDF4 as nc
import numpy as np

from ..core.timeutils import datetime_to_cycle
from .common import (
    MET_ECNT_KEYS,
    MET_SL1L2_KEYS,
    generate_cycles,
    generate_forecast_leads,
    nc_group_name,
    obs_diag_file,
    obs_diag_forecast_file,
    profile_key_for_domain,
    resolve_domains,
    resolve_statistics_settings,
    resolve_time_settings,
)
from .sensor_config import display_var_names, get_bin_method, get_obs_config

__all__ = ["pool_partial_sums", "continuous_statistics", "collect_continuous",
           "continuous_rows", "export_continuous_tables", "CNT_FIELDS"]

_SL = [name for name, _ in MET_SL1L2_KEYS]
_EC = [name for name, _ in MET_ECNT_KEYS]

CNT_STATS = ["TOTAL", "FBAR", "OBAR", "ME", "MAE", "RMSE", "ESTDEV", "MBIAS",
             "FSTDEV", "OSTDEV", "PR_CORR", "CRPS_EMP", "CRPS_EMP_FAIR", "SPREAD",
             "SPREAD_PLUS_OERR"]
CNT_FIELDS = ["experiment", "sensor", "diagtype", "lead", "variable", "ioda_variable",
              "domain", "level", "level_units"] + CNT_STATS


def _read_group(ds, path):
    grp = ds
    for p in path:
        grp = grp.groups.get(p) if grp is not None else None
        if grp is None:
            return None
    return grp


def pool_partial_sums(ncfiles, path):
    """Pool the SL1L2 / ECNT variables of group ``path`` over ``ncfiles``.

    Returns {"SL1L2_TOTAL": n, "SL1L2_FBAR": mean, ...,
    "ECNT_TOTAL": n, ...} with each mean weighted by its TOTAL, plus
    "_levels" when the files have a Levels coordinate; {} when no file has
    the sums."""
    acc, levels = {}, None
    for f in ncfiles:
        if not os.path.exists(f):
            continue
        with nc.Dataset(f, "r") as ds:
            grp = _read_group(ds, path)
            if grp is None or "SL1L2_TOTAL" not in grp.variables:
                continue
            if levels is None and "Levels" in ds.variables:
                levels = np.asarray(ds.variables["Levels"][:], dtype=np.float64)
            for names, total_name in ((_SL, "SL1L2_TOTAL"), (_EC, "ECNT_TOTAL")):
                if total_name not in grp.variables:
                    continue
                n = np.nan_to_num(np.asarray(grp.variables[total_name][...], dtype=np.float64))
                acc[total_name] = acc.get(total_name, 0.0) + n
                for name in names:
                    if name == total_name or name not in grp.variables:
                        continue
                    v = np.asarray(grp.variables[name][...], dtype=np.float64)
                    if name in ("ECNT_SPREAD", "ECNT_SPREAD_PLUS_OERR"):
                        v = v * v                      # pool variances
                    acc[name] = acc.get(name, 0.0) + np.where(n > 0, np.nan_to_num(v) * n, 0.0)
    if not acc:
        return {}
    out = {}
    for names, total_name in ((_SL, "SL1L2_TOTAL"), (_EC, "ECNT_TOTAL")):
        if total_name not in acc:
            continue
        n = acc[total_name]
        out[total_name] = n
        with np.errstate(invalid="ignore", divide="ignore"):
            for name in names:
                if name in acc and name != total_name:
                    m = np.where(n > 0, acc[name] / np.where(n > 0, n, 1.0), np.nan)
                    if name in ("ECNT_SPREAD", "ECNT_SPREAD_PLUS_OERR"):
                        m = np.sqrt(m)
                    out[name] = m
    if levels is not None:
        out["_levels"] = levels
    return out


def continuous_statistics(sums):
    """MET CNT statistics (see the module table) from pooled sums."""
    if not sums:
        return {}
    n = np.asarray(sums["SL1L2_TOTAL"], dtype=np.float64)
    f, o = sums["SL1L2_FBAR"], sums["SL1L2_OBAR"]
    fo, ff, oo = sums["SL1L2_FOBAR"], sums["SL1L2_FFBAR"], sums["SL1L2_OOBAR"]
    with np.errstate(invalid="ignore", divide="ignore"):
        me = f - o
        mse = np.maximum(ff - 2.0 * fo + oo, 0.0)
        fvar = np.maximum(ff - f * f, 0.0)
        ovar = np.maximum(oo - o * o, 0.0)
        fsd, osd = np.sqrt(fvar), np.sqrt(ovar)
        out = {
            "TOTAL": n, "FBAR": f, "OBAR": o, "ME": me, "MAE": sums["SL1L2_MAE"],
            "RMSE": np.sqrt(mse), "ESTDEV": np.sqrt(np.maximum(mse - me * me, 0.0)),
            "MBIAS": np.where(o != 0, f / np.where(o != 0, o, 1.0), np.nan),
            "FSTDEV": fsd, "OSTDEV": osd,
            "PR_CORR": np.where((fsd > 0) & (osd > 0),
                                (fo - f * o) / np.where(fsd * osd > 0, fsd * osd, 1.0), np.nan),
        }
    for key in ("CRPS_EMP", "CRPS_EMP_FAIR", "SPREAD", "SPREAD_PLUS_OERR"):
        if f"ECNT_{key}" in sums:
            out[key] = sums[f"ECNT_{key}"]
    empty = ~(n > 0)
    for k, v in out.items():
        if k != "TOTAL":
            out[k] = np.where(empty, np.nan, v)
    return out


def collect_continuous(diag_base_dir, exp_name, diag_obs, cycles, diagtype, domain,
                       profile_key=None, lead=None):
    """{display_var: {"all": stats, "levels": stats per bin or None,
    "coord": levels}} for one experiment, diagtype and domain, pooled over
    ``cycles`` (init cycles at ``lead`` for O−F)."""
    if lead is None:
        files = [obs_diag_file(diag_base_dir, exp_name, diag_obs, c) for c in cycles]
    else:
        files = [obs_diag_forecast_file(diag_base_dir, exp_name, diag_obs, c, lead)
                 for c in cycles]
    out = {}
    for var, raw in zip(display_var_names(diag_obs), get_obs_config(diag_obs).variables):
        g = nc_group_name(raw)
        total = continuous_statistics(pool_partial_sums(files, (diagtype, g, "0D", domain)))
        levels, coord = None, None
        if profile_key:
            sums = pool_partial_sums(files, (diagtype, g, "1D", profile_key))
            coord = sums.pop("_levels", None)
            levels = continuous_statistics(sums) or None
        if total or levels:
            out[var] = {"all": total, "levels": levels, "coord": coord}
    return out


def _profile_key(bin_method, domain, diag_obs):
    if diag_obs.startswith("sfc") or (bin_method == 3 and domain != "global"):
        return None
    return profile_key_for_domain(bin_method, domain)


def _fmt(x):
    x = float(x)
    return "" if not np.isfinite(x) else f"{x:.6g}"


def continuous_rows(config, diag_obs, diag_base_dir):
    """CSV rows (dicts with CNT_FIELDS) for every experiment, diagtype,
    domain, variable and level (plus an "all" row per variable)."""
    bin_method = get_bin_method(diag_obs)
    units = {1: "hPa", 2: "km", 3: "deg_lat"}.get(bin_method, "")
    raw_of = dict(zip(display_var_names(diag_obs), get_obs_config(diag_obs).variables))
    names = config["experiments"]["names"]
    types = config["experiments"].get("types", ["da"] * len(names))
    plot_oma = config["plot_settings"].get("plot_oma", True)
    domains = list(resolve_domains(config).keys())

    tasks = []          # (exp, diagtype, lead, cycles)
    if any(t != "hofx_forecast" for t in types):
        start, end, interval = resolve_time_settings(config, diag_obs)
        cycles = generate_cycles(start, end, interval)
        for e, t in zip(names, types):
            if t == "hofx_forecast":
                continue
            tasks.append((e, "omb", None, cycles))
            if t == "da" and plot_oma:
                tasks.append((e, "oma", None, cycles))
    if "hofx_forecast" in types and config.get("forecast"):
        start, end, interval = resolve_time_settings(config, diag_obs, for_forecast=True)
        inits = generate_cycles(start, end, interval)
        for e, t in zip(names, types):
            if t == "hofx_forecast":
                for lead in generate_forecast_leads(config):
                    tasks.append((e, "omf", lead, inits))

    rows = []
    for exp, diag, lead, cycles in tasks:
        for domain in domains:
            data = collect_continuous(diag_base_dir, exp, diag_obs, cycles, diag, domain,
                                      _profile_key(bin_method, domain, diag_obs), lead)
            for var, d in data.items():
                base = {"experiment": exp, "sensor": diag_obs, "diagtype": diag,
                        "lead": "" if lead is None else int(lead), "variable": var,
                        "ioda_variable": raw_of.get(var, var), "domain": domain,
                        "level_units": units}
                if d["all"] and d["all"]["TOTAL"] > 0:
                    rows.append(dict(base, level="all",
                                     **{k: _fmt(d["all"][k]) if k in d["all"] else ""
                                        for k in CNT_STATS}))
                if d["levels"]:
                    for i, lev in enumerate(np.asarray(d["coord"])):
                        if not d["levels"]["TOTAL"][i] > 0:
                            continue
                        rows.append(dict(base, level=f"{float(lev):g}",
                                         **{k: _fmt(d["levels"][k][i]) if k in d["levels"]
                                            else "" for k in CNT_STATS}))
    return rows


def export_continuous_tables(config, diag_obs, diag_base_dir, out_dir):
    """Write ``cnt_<sensor>_<first>-<last>.csv`` to ``out_dir``; returns the
    paths written (none when statistics.met_statistics is off or no file
    has partial sums)."""
    if not resolve_statistics_settings(config).get("met_statistics"):
        return []
    rows = continuous_rows(config, diag_obs, diag_base_dir)
    if not rows:
        return []
    start, end, interval = resolve_time_settings(config, diag_obs)
    cycles = generate_cycles(start, end, interval)
    tag = f"{datetime_to_cycle(cycles[0])}-{datetime_to_cycle(cycles[-1])}" if cycles else "all"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cnt_{diag_obs}_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CNT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return [path]
