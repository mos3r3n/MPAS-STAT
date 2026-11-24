"""Export Desroziers observation-error estimates as tables.

For every "da" experiment with OMB and OMA, and every sensor, this writes
two files under <diag_dir>/<experiment>/VerifyOBS/ObsErrorEstimates/:

  desroziers_<sensor>_<start>-<end>.csv
      One row per (variable, domain, level), plus one "all"-levels row per
      (variable, domain): observation count, assumed observation error,
      Desroziers observation- and background-error estimates, their ratio
      and (ensembles) the consistency ratio TotalSpread / RMS(OMB).
      Values are in the units of the statistics files (specific humidity
      in g/kg).

  desroziers_<sensor>_<start>-<end>.yaml
      The global-domain estimates per variable, and a ready-to-edit UFO
      "Perform Action / assign error" filter that would assign them
      (pressure-binned conventional obs: ObsErrorModelStepwiseLinear on
      MetaData/pressure in Pa; height-binned obs: on MetaData/height in m;
      radiances: one error per channel). These values are in IODA units
      (specific humidity in kg/kg).

The estimates are cycle-pooled exactly as in the error-consistency figure
(see mpas_stat.obs.collect.collect_obs_error_profiles). Desroziers
estimates are only exact when the analysis used correct error statistics,
so treat them as a starting point to iterate on, not a final answer.
"""
import csv
import os

import numpy as np

from ..core.timeutils import datetime_to_cycle
from .collect import collect_obs_error_profiles, collect_obs_error_totals
from .common import (
    generate_cycles,
    load_config,
    profile_key_for_domain,
    resolve_diag_base_dir,
    resolve_domains,
    resolve_statistics_settings,
    resolve_time_settings,
)
from .experiments import _experiments_for_diagtype, _profile_axis_meta
from .sensor_config import display_var_names, get_bin_method, get_obs_config

__all__ = ["export_error_estimates", "error_estimate_rows", "main"]

CSV_FIELDS = ["experiment", "sensor", "variable", "ioda_variable", "domain",
              "level", "level_units", "count", "assumed_obs_error",
              "desroziers_obs_error", "desroziers_bkg_error",
              "desroziers_to_assumed_ratio", "consistency_ratio"]

# Statistics-file units -> IODA units (see mpas_stat.obs.reader).
_TO_IODA = {"specificHumidity": 1.0e-3}


def _level_units(bin_method):
    return {1: "hPa", 2: "km", 3: "deg_lat"}.get(bin_method, "")


def _f(x):
    x = float(x)
    return x if np.isfinite(x) else None


def _row(exp, sensor, var, raw, domain, level, units, e, i=None):
    def get(key):
        v = e.get(key)
        if v is None:
            return None
        return _f(v if i is None else np.asarray(v)[i])
    assumed, desroz = get("ObsErrorAssumed"), get("ObsErrorDesroziers")
    ratio = desroz / assumed if assumed and desroz is not None else None
    count = get("Count")
    return {
        "experiment": exp, "sensor": sensor, "variable": var,
        "ioda_variable": raw, "domain": domain, "level": level,
        "level_units": units, "count": int(count) if count is not None else None,
        "assumed_obs_error": assumed, "desroziers_obs_error": desroz,
        "desroziers_bkg_error": get("BkgErrorDesroziers"),
        "desroziers_to_assumed_ratio": ratio,
        "consistency_ratio": get("ConsistencyRatio"),
    }


def error_estimate_rows(config, diag_obs, exp_name, diag_base_dir, cycles):
    """Rows (dicts with CSV_FIELDS) for one experiment and sensor."""
    settings = resolve_statistics_settings(config)
    agg = settings["time_aggregation"]
    bin_method = get_bin_method(diag_obs)
    display_vars = display_var_names(diag_obs)
    raw_vars = get_obs_config(diag_obs).variables
    raw_of = dict(zip(display_vars, raw_vars))
    units = _level_units(bin_method)
    coord_label, invert_y = _profile_axis_meta(bin_method)
    domains = list(resolve_domains(config).keys())
    if bin_method == 3:
        domains = [d for d in domains if d == "global"] or domains[:1]

    rows = []
    for domain in domains:
        totals = collect_obs_error_totals(diag_base_dir, exp_name, diag_obs, cycles,
                                          display_vars, raw_vars, domain, agg)
        profiles = {}
        if not diag_obs.startswith("sfc"):
            profiles = collect_obs_error_profiles(
                diag_base_dir, exp_name, diag_obs, cycles, display_vars, raw_vars,
                profile_key_for_domain(bin_method, domain), coord_label, invert_y, agg)
        for var in display_vars:
            raw = str(raw_of[var])
            if var in totals:
                rows.append(_row(exp_name, diag_obs, var, raw, domain, "all", units, totals[var]))
            prof = profiles.get(var)
            if prof:
                for i, lev in enumerate(np.asarray(prof["coord"])):
                    rows.append(_row(exp_name, diag_obs, var, raw, domain,
                                     float(lev), units, prof, i))
    return rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else
                            (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]))
                        for k in CSV_FIELDS})


def _yaml_num(x):
    return "null" if x is None else f"{x:.6g}"


def _yaml_list(xs):
    return "[" + ", ".join(_yaml_num(x) for x in xs) + "]"


def _write_yaml(path, rows, diag_obs, exp_name, cycles, min_count):
    bin_method = get_bin_method(diag_obs)
    glob = [r for r in rows if r["domain"] == "global"]
    lines = [
        "# Desroziers observation-error estimates (global domain), written by MPAS-STAT.",
        f"# experiment: {exp_name}   sensor: {diag_obs}",
        f"# cycles: {datetime_to_cycle(cycles[0])}-{datetime_to_cycle(cycles[-1])} "
        f"({len(cycles)} cycles)",
        "#",
        "# assumed = rms of the error the DA used (EffectiveError, i.e. AFTER QC",
        "#           inflation; ObsError when EffectiveError is absent)",
        "# desroziers = sqrt(mean[(O-A)(O-B)]); ratio = desroziers / assumed",
        "# Values are in IODA units. Levels with fewer than "
        f"{min_count} observations are left out of the filter.",
        "#",
        "# The 'jedi_filter' blocks assign the estimates as the observation error.",
        "# Because 'assumed' already includes QC inflation, an alternative is to",
        "# keep your error model and scale it by 'ratio'. Desroziers estimates are",
        "# exact only for an optimal analysis: re-run and iterate, and review",
        "# before use.",
        f"sensor: {diag_obs}",
        f"experiment: {exp_name}",
        "variables:",
    ]
    by_var = {}
    for r in glob:
        by_var.setdefault(r["variable"], []).append(r)

    channels, chan_err = [], []
    for var, vrows in by_var.items():
        raw = vrows[0]["ioda_variable"]
        scale = _TO_IODA.get(raw, 1.0)
        tot = next((r for r in vrows if r["level"] == "all"), None)
        levs = [r for r in vrows if r["level"] != "all"]
        sc = lambda x: None if x is None else x * scale  # noqa: E731
        lines.append(f"  {var}:")
        lines.append(f"    ioda_variable: {raw}")
        if tot:
            lines.append(f"    all_levels: {{count: {tot['count']}, "
                         f"assumed: {_yaml_num(sc(tot['assumed_obs_error']))}, "
                         f"desroziers: {_yaml_num(sc(tot['desroziers_obs_error']))}, "
                         f"ratio: {_yaml_num(tot['desroziers_to_assumed_ratio'])}}}")
            if bin_method == 3:
                channels.append(raw)
                chan_err.append(sc(tot["desroziers_obs_error"]))
        if levs and bin_method in (1, 2):
            lines.append(f"    level_{levs[0]['level_units']}: "
                         f"{_yaml_list([r['level'] for r in levs])}")
            lines.append(f"    count: [{', '.join(str(r['count']) for r in levs)}]")
            for key, name in (("assumed_obs_error", "assumed"),
                              ("desroziers_obs_error", "desroziers")):
                lines.append(f"    {name}: {_yaml_list([sc(r[key]) for r in levs])}")
            lines.append(f"    ratio: {_yaml_list([r['desroziers_to_assumed_ratio'] for r in levs])}")
            ok = [r for r in levs if r["desroziers_obs_error"] is not None
                  and (r["count"] or 0) >= min_count]
            if len(ok) >= 2:
                xvar, fac = ("MetaData/pressure", 100.0) if bin_method == 1 else ("MetaData/height", 1000.0)
                lines += [
                    "    jedi_filter:",
                    "      filter: Perform Action",
                    f"      filter variables: [{{name: {raw}}}]",
                    "      action:",
                    "        name: assign error",
                    "        error function:",
                    "          name: ObsFunction/ObsErrorModelStepwiseLinear",
                    "          options:",
                    f"            xvar: {{name: {xvar}}}",
                    f"            xvals: {_yaml_list([r['level'] * fac for r in ok])}",
                    f"            errors: {_yaml_list([sc(r['desroziers_obs_error']) for r in ok])}",
                ]
    if bin_method == 3 and channels and all(e is not None for e in chan_err):
        lines += [
            "jedi_filter:",
            "  filter: Perform Action",
            f"  filter variables: [{{name: brightnessTemperature, channels: "
            f"{', '.join(str(c) for c in channels)}}}]",
            "  action:",
            "    name: assign error",
            f"    error parameter vector: {_yaml_list(chan_err)}",
        ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def export_error_estimates(config, diag_obs, diag_base_dir):
    """Write the CSV and YAML tables for one sensor, for every "da"
    experiment that has OMA. Returns the list of files written."""
    settings = resolve_statistics_settings(config)
    if not settings.get("error_diagnostics"):
        return []
    names, _, _ = _experiments_for_diagtype(config, "oma")
    start, end, interval = resolve_time_settings(config, diag_obs)
    cycles = generate_cycles(start, end, interval)
    if not names or not cycles:
        return []
    min_count = int(config.get("plot_settings", {}).get("error_table_min_count", 100))
    tag = f"{datetime_to_cycle(cycles[0])}-{datetime_to_cycle(cycles[-1])}"
    written = []
    for exp in names:
        rows = error_estimate_rows(config, diag_obs, exp, diag_base_dir, cycles)
        if not any(r["desroziers_obs_error"] is not None for r in rows):
            continue
        out_dir = os.path.join(diag_base_dir, exp, "VerifyOBS", "ObsErrorEstimates")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"desroziers_{diag_obs}_{tag}")
        _write_csv(base + ".csv", rows)
        _write_yaml(base + ".yaml", rows, diag_obs, exp, cycles, min_count)
        written += [base + ".csv", base + ".yaml"]
    return written


def main(config_path=None):
    """Write the error-estimate tables for every configured sensor."""
    config = load_config(config_path)
    diag_base_dir = resolve_diag_base_dir(config)
    for diag_obs in config["sensors"]:
        for path in export_error_estimates(config, diag_obs, diag_base_dir):
            print(f"Saved {path}")
