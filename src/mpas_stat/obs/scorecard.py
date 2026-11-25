"""Observation-space scorecard: every sensor, variable and level on one page.

One figure per experiment compared with the reference
(``experiments.names[0]``) and per diagnostic (O−B for cycling
experiments, O−F for extended forecasts):

rows
    Sensor / variable / level: ``all`` (the whole column), then each
    pressure or height bin from the top down; one row per channel for
    radiances.
columns
    Statistic × domain (× forecast lead for O−F).
colour
    Percent difference ``100 * (exp - ref) / ref`` of the cycle-pooled
    statistic; negative (blue) means a smaller error than the reference.
marker
    ▲ significantly better, ▼ significantly worse: the paired bootstrap
    interval over cycles excludes 0 (``bootstrap.*`` settings).

A CSV with every cell (difference, interval, counts) is written next to
the figure. Configured by ``plot_settings.scorecard``.
"""
import csv
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4 as nc  # noqa: E402
import numpy as np  # noqa: E402

from ..core.bootstrap import get_bootstrap_settings  # noqa: E402
from ..core.pooling import (  # noqa: E402
    bootstrap_pool_reduction_ci,
    missing_components,
    read_components,
    stat_kind,
)
from ..core.timeutils import generate_forecast_leads  # noqa: E402
from ..graphics.colour import colorbar_ticks, colour_scale  # noqa: E402
from ..graphics.labels import stat_label  # noqa: E402
from ..graphics.style import apply_pub_style  # noqa: E402
from .common import (  # noqa: E402
    generate_cycles,
    nc_group_name,
    obs_diag_file,
    obs_diag_forecast_file,
    profile_key_for_domain,
    resolve_domains,
    resolve_time_settings,
)
from .experiments import (  # noqa: E402
    _build_forecast_experiments,
    _ci_display_settings,
    _experiments_for_diagtype,
)
from .sensor_config import display_var_names, get_bin_method, get_obs_config  # noqa: E402

__all__ = ["scorecard_settings", "scorecard_data", "plot_scorecard", "generate_scorecards"]

_DEFAULTS = {
    "enabled": True,
    "statistics": ["RMS"],
    "diagtypes": ["omb"],        # cycling: omb and/or oma; omf is added for forecasts
    "levels": None,              # None = every bin; or a list of levels (hPa / km)
    "domains": None,             # None = plot_settings.apply_domains
    "leads": None,               # OMF: None = every lead; or a list of lead hours
    "min_count": 10,             # fewer pooled observations in either experiment -> blank
    "min_cycles": 5,             # fewer cycles with data -> no significance marker
    "show_values": True,
    "limit": None,               # colour-scale half range in %, None = automatic
}


def scorecard_settings(config):
    """plot_settings.scorecard merged over the defaults (``scorecard: false``
    turns it off)."""
    raw = config.get("plot_settings", {}).get("scorecard", {})
    if raw is False or raw is None:
        return dict(_DEFAULTS, enabled=False)
    if raw is True:
        raw = {}
    out = dict(_DEFAULTS)
    out.update(raw)
    bad = [s for s in out["statistics"] if stat_kind(s) not in ("rms", "std")]
    if bad:
        raise ValueError(f"plot_settings.scorecard.statistics supports RMS and STD, got {bad}")
    return out


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------
def _read_file(path, diagtype, raw_vars, domains, stats, bin_method, sensor):
    """{(raw, domain, stat): (w, m1, m2)} from one statistics file, each a
    1-D array: [whole column, level 1, level 2, ...] (radiances / surface:
    whole column only). Missing entries are absent. Also returns the level
    coordinate, or None."""
    out, levels = {}, None
    if not os.path.exists(path):
        return out, levels
    with nc.Dataset(path, "r") as ds:
        if bin_method in (1, 2) and "Levels" in ds.variables:
            levels = np.asarray(ds.variables["Levels"][:], dtype=np.float64)
        if diagtype not in ds.groups:
            return out, levels
        dg = ds.groups[diagtype]
        for raw in raw_vars:
            g = dg.groups.get(nc_group_name(raw))
            if g is None:
                continue
            for dom in domains:
                parts = []
                g0 = g.groups.get("0D", None)
                g0 = g0.groups.get(dom) if g0 is not None else None
                g1 = None
                if bin_method in (1, 2) and not sensor.startswith("sfc") and "1D" in g.groups:
                    g1 = g.groups["1D"].groups.get(profile_key_for_domain(bin_method, dom))
                for stat in stats:
                    try:
                        _, w0, a0, b0 = read_components(g0, stat) if g0 is not None else (None,) * 4
                    except KeyError:
                        w0 = None
                    if w0 is None:
                        continue
                    parts = [(np.atleast_1d(w0), np.atleast_1d(a0), np.atleast_1d(b0))]
                    if g1 is not None:
                        try:
                            _, w1, a1, b1 = read_components(g1, stat)
                            parts.append((w1, a1, b1))
                        except KeyError:
                            pass
                    out[(raw, dom, stat)] = tuple(
                        np.concatenate([p[i] for p in parts]) for i in range(3))
    return out, levels


def _stack(per_file, key, ncol):
    """Pool dict over files (axis 0) for one key; missing files -> weight 0."""
    w, m1, m2 = [], [], []
    for d in per_file:
        if key in d and d[key][0].size == ncol:
            a, b, c = d[key]
        else:
            _, a, b, c = missing_components((ncol,))
        w.append(a)
        m1.append(b)
        m2.append(c)
    return {"kind": stat_kind(key[2]), "w": np.asarray(w), "m1": np.asarray(m1),
            "m2": np.asarray(m2)}


def _level_rows(levels, wanted):
    """Indices (into the level coordinate) of the rows to show, top first."""
    if levels is None:
        return []
    idx = list(range(levels.size))
    if wanted:
        idx = sorted({int(np.argmin(np.abs(levels - float(v)))) for v in wanted})
    # top of the atmosphere first: low pressure / large height first
    return sorted(idx, key=lambda i: levels[i] if levels[0] > levels[-1] else -levels[i])


def scorecard_data(config, diag_base_dir, ref, exp, diagtype, settings, sensors=None):
    """Rows, columns and cell values for one (reference, experiment, diagtype).

    Returns dict with "rows" [(sensor, variable, level_label)], "columns"
    [(stat, domain, lead or None)], and arrays "diff", "lo", "hi",
    "count_ref", "count_exp" of shape (nrow, ncol)."""
    boot = get_bootstrap_settings(config)
    domains = settings["domains"] or config["plot_settings"].get("apply_domains") \
        or list(resolve_domains(config).keys())
    domains = [d for d in domains if d in resolve_domains(config)]
    stats = list(settings["statistics"])
    forecast = diagtype == "omf"
    leads = [None]
    if forecast:
        leads = generate_forecast_leads(config)
        if settings["leads"]:
            leads = [lead for lead in leads if lead in set(int(x) for x in settings["leads"])]
    columns = [(s, d, lead) for s in stats for d in domains for lead in leads]

    rows, cells = [], []
    for sensor in sensors or config["sensors"]:
        ocfg = get_obs_config(sensor)
        bin_method = get_bin_method(sensor)
        raw_vars, disp = ocfg.variables, display_var_names(sensor)
        start, end, interval = resolve_time_settings(config, sensor, for_forecast=forecast)
        cycles = generate_cycles(start, end, interval)
        if not cycles:
            continue
        per_lead = {}
        levels = None
        for lead in leads:
            files = {}
            for name in (ref, exp):
                paths = [obs_diag_forecast_file(diag_base_dir, name, sensor, c, lead) if forecast
                         else obs_diag_file(diag_base_dir, name, sensor, c) for c in cycles]
                read = [_read_file(p, diagtype, raw_vars, domains, stats, bin_method, sensor)
                        for p in paths]
                files[name] = [r[0] for r in read]
                levels = next((r[1] for r in read if r[1] is not None), levels)
            per_lead[lead] = files
        lev_idx = _level_rows(levels, settings["levels"])
        ncol_data = 1 + (levels.size if levels is not None and bin_method in (1, 2)
                         and not sensor.startswith("sfc") else 0)
        row_index = [0] + [1 + i for i in lev_idx] if ncol_data > 1 else [0]
        unit = {1: "hPa", 2: "km"}.get(bin_method, "")

        for raw, var in zip(raw_vars, disp):
            block = np.full((len(row_index), len(columns), 6), np.nan)
            any_data = False
            for icol, (stat, dom, lead) in enumerate(columns):
                key = (raw, dom, stat)
                pr = _stack(per_lead[lead][ref], key, ncol_data)
                pe = _stack(per_lead[lead][exp], key, ncol_data)
                if not (pr["w"].any() and pe["w"].any()):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    red, rlo, rhi = bootstrap_pool_reduction_ci(
                        pr, pe, n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"])
                cref, cexp = pr["w"].sum(0), pe["w"].sum(0)
                ncyc = ((pr["w"] > 0) & (pe["w"] > 0)).sum(0)
                for irow, j in enumerate(row_index):
                    block[irow, icol] = (-red[j], -rhi[j], -rlo[j], cref[j], cexp[j], ncyc[j])
                any_data = True
            if not any_data:
                continue
            for irow, j in enumerate(row_index):
                label = "all" if j == 0 else f"{levels[j - 1]:g} {unit}".strip()
                rows.append((sensor, var, label))
                cells.append(block[irow])

    cells = np.asarray(cells) if cells else np.full((0, len(columns), 6), np.nan)
    mc = float(settings["min_count"])
    few = (cells[..., 3] < mc) | (cells[..., 4] < mc) if cells.size else np.zeros((0, len(columns)), bool)
    for k in range(3):
        cells[..., k][few] = np.nan
    if cells.size:
        few_cycles = ~(cells[..., 5] >= int(settings["min_cycles"]))
        cells[..., 1][few_cycles] = np.nan
        cells[..., 2][few_cycles] = np.nan
        keep = np.any(np.isfinite(cells[..., 0]), axis=1)
        rows = [r for r, k in zip(rows, keep) if k]
        cells = cells[keep]
        kcol = np.any(np.isfinite(cells[..., 0]), axis=0)
        columns = [c for c, k in zip(columns, kcol) if k]
        cells = cells[:, kcol]
    return {"rows": rows, "columns": columns, "diff": cells[..., 0], "lo": cells[..., 1],
            "hi": cells[..., 2], "count_ref": cells[..., 3], "count_exp": cells[..., 4],
            "n_cycles": cells[..., 5]}


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def _write_csv(path, data, ref_label, exp_label):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sensor", "variable", "level", "statistic", "domain", "lead_hour",
                    "reference", "experiment", "diff_pct", "ci_low", "ci_high",
                    "significant", "count_reference", "count_experiment", "n_cycles"])
        for i, (sensor, var, lev) in enumerate(data["rows"]):
            for j, (stat, dom, lead) in enumerate(data["columns"]):
                d, lo, hi = data["diff"][i, j], data["lo"][i, j], data["hi"][i, j]
                sig = ("better" if hi < 0 else "worse" if lo > 0 else "") if np.isfinite(lo) else ""
                fmt = (lambda x: f"{x:.4g}" if np.isfinite(x) else "")  # noqa: E731
                w.writerow([sensor, var, lev, stat, dom, "" if lead is None else lead,
                            ref_label, exp_label, fmt(d), fmt(lo), fmt(hi), sig,
                            fmt(data["count_ref"][i, j]), fmt(data["count_exp"][i, j]),
                            fmt(data["n_cycles"][i, j])])


def plot_scorecard(data, path, ref_label, exp_label, title_kind, settings, config):
    """Draw one scorecard (see the module docstring). Returns the saved path."""
    rows, columns = data["rows"], data["columns"]
    if not rows:
        return None
    config = dict(config, **_ci_display_settings(config))
    apply_pub_style(config)
    diff = data["diff"]
    nrow, ncol = diff.shape
    style = config.get("plot_style", {})
    cmap_name = style.get("diverging_cmap", "RdBu_r")
    lim = settings["limit"]
    if lim is None:
        finite = np.abs(diff[np.isfinite(diff)])
        lim = float(np.nanpercentile(finite, 95)) if finite.size else 1.0
        lim = max(lim, 0.5)
    cmap, norm, levels = colour_scale(cmap_name, -lim, lim, config, diverging=True, extend="both")
    cmap.set_bad("0.92")

    has_leads = any(c[2] is not None for c in columns)
    cell_w = 0.42 if has_leads else 0.75
    fig_w = 3.2 + cell_w * ncol + 1.1
    fig_h = 1.6 + 0.24 * nrow
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.grid(False)
    ax.minorticks_off()
    mesh = ax.pcolormesh(np.arange(ncol + 1), np.arange(nrow + 1), np.ma.masked_invalid(diff),
                         cmap=cmap, norm=norm, edgecolors="white", linewidth=0.6)
    ax.set_xlim(0, ncol)
    ax.set_ylim(nrow, 0)

    sig_better = np.isfinite(data["hi"]) & (data["hi"] < 0)
    sig_worse = np.isfinite(data["lo"]) & (data["lo"] > 0)
    fs = 6.5 if has_leads else 7.5
    for i in range(nrow):
        for j in range(ncol):
            d = diff[i, j]
            if not np.isfinite(d):
                continue
            dark = abs(d) > 0.6 * lim
            colour = "white" if dark else "black"
            mark = "▲" if sig_better[i, j] else "▼" if sig_worse[i, j] else ""
            if settings["show_values"]:
                text = f"{mark}{d:+.1f}" if mark else f"{d:+.1f}"
            else:
                text = mark
            if text:
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=fs,
                        color=colour, fontweight="bold" if mark else "normal")

    # y labels: level labels; variable / sensor group labels to the left
    single = {r[:2] for r in rows}
    single = {k for k in single if sum(1 for r in rows if r[:2] == k) == 1}
    ax.set_yticks(np.arange(nrow) + 0.5)
    ax.set_yticklabels([r[1] if r[:2] in single else r[2] for r in rows], fontsize=7.5)
    ax.tick_params(axis="y", length=0, pad=2)
    groups, start = [], 0
    for i in range(1, nrow + 1):
        if i == nrow or rows[i][:2] != rows[start][:2]:
            groups.append((start, i, rows[start][0], rows[start][1]))
            start = i
    prev_sensor = None
    for g0, g1, sensor, var in groups:
        ax.axhline(g0, color="black", linewidth=1.2 if sensor != prev_sensor
                   else (0.6 if g1 - g0 > 1 or (g0 > 0 and rows[g0 - 1][:2] not in single) else 0.0))
        if (sensor, var) not in single:
            ax.text(-0.12, (g0 + g1) / 2, var, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=8.5, fontweight="bold")
        if sensor != prev_sensor:
            ax.text(-0.42, g0 + 0.1, sensor, transform=ax.get_yaxis_transform(),
                    ha="left", va="top", fontsize=9, fontstyle="italic")
        prev_sensor = sensor
    ax.axhline(nrow, color="black", linewidth=1.2)

    # x labels: domain (and lead) per column, statistic / domain spans on top
    if has_leads:
        ax.set_xticks(np.arange(ncol) + 0.5)
        ax.set_xticklabels([f"{c[2]}" for c in columns], fontsize=7)
        ax.set_xlabel("forecast lead (h)")
    else:
        ax.set_xticks(np.arange(ncol) + 0.5)
        ax.set_xticklabels([c[1] for c in columns], fontsize=8, rotation=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("bottom")
    ax.tick_params(axis="x", length=0, pad=2)
    spans, s0 = [], 0
    key = (lambda c: (c[0], c[1])) if has_leads else (lambda c: c[0])  # noqa: E731
    for j in range(1, ncol + 1):
        if j == ncol or key(columns[j]) != key(columns[s0]):
            spans.append((s0, j, columns[s0]))
            s0 = j
    nstats = len({c[0] for c in columns})
    for s0, s1, c in spans:
        ax.axvline(s0, color="black", linewidth=1.2 if s0 == 0 or not has_leads else 0.9)
        label = (f"{c[0]} · {c[1]}" if nstats > 1 else c[1]) if has_leads else c[0]
        if has_leads or nstats > 1:
            ax.text((s0 + s1) / 2, -1.25 if not has_leads else -1.3, label, ha="center",
                    va="bottom", fontsize=9, fontweight="bold", transform=ax.transData,
                    clip_on=False)
    ax.axvline(ncol, color="black", linewidth=1.2)

    stat_txt = "/".join(dict.fromkeys(stat_label(c[0]) for c in columns))
    ax.set_title(f"{exp_label} vs {ref_label}: {title_kind} {stat_txt}, % diff.\n"
                 "▲ significantly better   ▼ significantly worse   "
                 f"({get_bootstrap_settings(config)['ci']:g}% bootstrap CI over cycles)",
                 fontsize=10, pad=34 if (has_leads or nstats > 1) else 22)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02, aspect=25,
                      shrink=min(1.0, 14.0 / max(nrow, 1)) if nrow > 14 else 1.0,
                      ticks=colorbar_ticks(levels) if levels is not None else None)
    cb.set_label(f"{stat_txt}: % diff. from {ref_label}  (negative = better)")
    cb.ax.minorticks_off()

    fmt = str(config["plot_settings"].get("figure_format", "png")).lower()
    out = f"{path}.{fmt}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def generate_scorecards(config, diag_base_dir, sensors=None):
    """Draw every scorecard (and its CSV) under <graphic_dir>/scorecard/.
    Returns the list of files written."""
    settings = scorecard_settings(config)
    if not settings["enabled"]:
        return []
    out_dir = os.path.join(config["paths"].get("graphic_dir", "graphics"), "scorecard")
    labels_of = dict(zip(config["experiments"]["names"],
                         config["experiments"].get("labels", config["experiments"]["names"])))
    jobs = []
    for diagtype in settings["diagtypes"]:
        if diagtype not in ("omb", "oma"):
            continue
        names, _, _ = _experiments_for_diagtype(config, diagtype)
        jobs += [(names[0], e, diagtype, diagtype.upper()) for e in names[1:]]
    fnames = _build_forecast_experiments(config)[0]
    if "forecast" in config and fnames:
        jobs += [(fnames[0], e, "omf", "OMF") for e in fnames[1:]]

    written = []
    for ref, exp, diagtype, kind in jobs:
        data = scorecard_data(config, diag_base_dir, ref, exp, diagtype, settings, sensors)
        if not data["rows"]:
            continue
        base = os.path.join(out_dir, f"scorecard_{diagtype}_{exp}_vs_{ref}")
        os.makedirs(out_dir, exist_ok=True)
        _write_csv(base + ".csv", data, labels_of.get(ref, ref), labels_of.get(exp, exp))
        print(f"Saved {base}.csv")
        written.append(base + ".csv")
        fig = plot_scorecard(data, base, labels_of.get(ref, ref), labels_of.get(exp, exp),
                             kind, settings, config)
        if fig:
            written.append(fig)
    return written
