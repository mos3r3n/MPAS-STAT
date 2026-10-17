#!/usr/bin/env python3
"""
Shared computation kernels and configuration for radar composite-
reflectivity forecast verification (FSS / categorical scores / performance
diagram / RMSE vs. forecast lead time), used by BOTH:
  - src/mpas_stat/radar/diag.py  (statistics stage: reads HofX obsout HDF5,
                                  grids obs/hofx composite reflectivity, and
                                  writes one diagnosis netCDF file per
                                  (experiment, init cycle, forecast lead))
  - src/mpas_stat/radar/plot.py  (graphics stage: reads those diagnosis
                                  files back, aggregates across init cycles,
                                  and draws figures)

This module is the radar product's counterpart of src/mpas_stat/obs/common.py's role
for the point-wise obs-space pipeline: one place for the math and config
conventions both pipeline stages must agree on, with no plotting or HDF5-
obs-reading orchestration of its own (src/mpas_stat/radar/diag.py owns the
HDF5 read + diagnosis-file write; src/mpas_stat/radar/plot.py owns the
diagnosis-file read + figures).

============================================================================
METplus / grid_stat integration (replaces the native FSS/categorical
scoring kernels this module used to carry -- see "REMOVED" note below)
============================================================================
Per an explicit decision to integrate with METplus: per-cycle FSS and
categorical (TS/ETS/FBIAS/POD/FAR) neighborhood scores are now computed by
MET's grid_stat tool (invoked through the METplus GridStat wrapper,
`run_metplus.py`), not by this module's own convolution/uniform_filter
kernels. This is a FULL replacement for this (integrated, config-driven)
product -- the standalone radar_reflectivity_verify.py script this product
was originally lifted from is explicitly NOT touched by this change and
keeps its own independent, original native-kernel implementation.

REMOVED in this revision (previously lived in this module, now gone):
calc_fss_components(), calc_fss(), _circular_kernel(), _categorical_
counts(), calc_categorical_counts(), _scores_from_counts(),
calc_categorical_scores(). calc_composite_reflectivity() (the scattered-
obs-to-grid step), remove_outliers() and calc_confidence_bootstrap() are
UNCHANGED and still used exactly as before -- gridding scattered obs/hofx
reflectivity onto a regular lat/lon grid is still done natively in Python
(once, per cycle), and the cross-cycle bootstrap/aggregation in
aggregate_radar_cycles() below is ALSO unchanged (see why below).

Design, in one paragraph: process_one_cycle() still grids obs/hofx
reflectivity natively (unchanged) and still computes RMSE natively
(unchanged -- RMSE is a simple, unambiguous quantity and was judged out of
scope for "replace the FSS/categorical kernels"; it is not delegated to
MET's CNT line type). It then writes that one cycle's gridded obs/fcst
fields to a small scratch .npz cache file (_write_metplus_npz_cache()),
generates a per-cycle METplus GridStat conf (_build_gridstat_conf()) that
points FCST_VAR1_NAME/OBS_VAR1_NAME at src/mpas_stat/radar/metplus_embed.py (a MET
Python-embedding script that just loads that one cache file back), and
invokes `run_metplus.py` on it via subprocess -- one invocation per cycle,
scoped to exactly that (experiment, init, lead) triple (INIT_BEG=INIT_END,
LEAD_SEQ=<one lead>), running inside the SAME ProcessPoolExecutor-based
per-cycle parallelism src/mpas_stat/radar/diag.py already used. This
deliberately avoids depending on METplus's own filename-into-script-
argument templating (never confirmed with certainty against documentation
during integration) -- the cache file's path is always a literal, fully-
resolved path this module built itself.

grid_stat is configured to emit NBRCTC (neighborhood categorical counts:
FY_OY/FY_ON/FN_OY/FN_ON, mapping directly to this module's existing na/nb/
nc/nd), NBRCTS (neighborhood categorical scores: CSI/GSS/FBIAS/PODY/FAR,
mapping to this module's existing ts/ets/fbias/pod/far) and NBRCNT
(neighborhood continuous scores: FBS/FSS/TOTAL) line types, each with
GRID_STAT_OUTPUT_FLAG_*=BOTH so MET writes BOTH the combined ASCII `.stat`
file (kept on disk per cycle for provenance / the user's own independent
stat_analysis use) AND a dedicated, self-describing per-line-type `.txt`
file with its own correctly-named header row -- this module parses ONLY
those dedicated files, by column NAME, never by position (MET's combined
`.stat` file is NOT self-describing past its common columns -- columns
after LINE_TYPE are positional per line type -- which is why the dedicated
files, not the combined one, are this module's parsing target).

IMPORTANT cross-cycle aggregation note (why aggregate_radar_cycles() did
NOT need to change): MET's `stat_analysis -job aggregate_stat -line_type
NBRCTC -out_line_type NBRCTS` is a documented, correct way to combine
categorical counts across cases -- but NBRCNT (FSS) has NO documented
aggregate_stat conversion; it is only listed under the separate, generic
"-job aggregate" job, whose exact combination semantics for NBRCNT were
NOT confirmable with confidence against the MET User's Guide during this
integration. Rather than depend on that undocumented behavior, cross-cycle
FSS pooling keeps happening in THIS module's own code, exactly as before
(sum fss_num/fss_denom accumulator components across cycles, then take one
ratio -- see aggregate_radar_cycles()) -- just fed by fss_num/fss_denom
values _parse_nbrcnt() below back-derives from MET's own per-cycle FBS/
FSS/TOTAL numbers instead of the old native kernel's. This is also why
process_one_cycle()'s return contract did not need to change shape at all
(see that function's docstring) -- MET only replaced what feeds it.

ASSUMPTION flagged for on-system validation: this integration was
developed and unit-tested in a sandbox with NO MET/METplus installation
available, so none of the grid_stat invocation, conf-file keys, or MET
ASCII output parsing could be exercised against a real MET build -- only
against fabricated/synthetic stand-ins for MET's output (see
test_radar_metplus_integration.py). Everything here was written against
the documented METplus GridStat conf keys and MET Python-embedding/ASCII-
output conventions, but please run a single-cycle dry run on your own
system (with your real run_metplus.py / MET install) before trusting
results, and check src/mpas_stat/radar/metplus_embed.py's own docstring for the one
MET Python-embedding behavior (NaN == missing data) this integration
relies on most and could not verify directly.
"""
import glob
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import h5py
import numpy as np
from scipy.spatial import cKDTree

# ============================================================================
# Configuration
# ============================================================================
DEFAULT_RADAR_CONFIG = {
    # obs-type name(s) for this product. Resolved to an HDF5 filename via
    # mpas_stat.obs.sensor_config.source_obs_name() -- the SAME sensor registry (e.g.
    # the "radar_zh*"/"radar_vr*" wildcard entries) the point-wise pipeline
    # uses, but this product's diagnosis files never collide with that
    # pipeline's, even for an identical diag_obs name -- see
    # mpas_stat.obs.common.radar_composite_diag_file()'s docstring.
    "sensors": ["radar_zh"],
    # HDF5 ObsValue/<diag_var> and hofx/<diag_var> group variable to verify.
    "diag_var": "equivalentReflectivityFactor",
    # dBZ thresholds for the categorical (TS/ETS/FBIAS/POD/FAR) and FSS
    # scores -- passed to MET as ">="-style thresholds (ge<value>), matching
    # this product's original (obs >= threshold) convention.
    "thresholds": [20, 40],
    # Neighborhood radii, in GRID CELLS (not degrees/km) of the
    # sample_bin-spaced composite-reflectivity grid -- matches the
    # standalone script's own `radii` convention. Converted to MET's own
    # neighborhood WIDTH convention (width = 2*radius + 1, an odd number of
    # grid cells across) when building the GridStat conf -- see
    # _nbrhd_width().
    "radii": [3, 5, 9, 15],
    # Composite-reflectivity gridding radius, in DEGREES (KDTree search
    # radius around each regular grid-cell center -- see
    # calc_composite_reflectivity()).
    "sample_bin": 0.027,
    # Max worker processes for the (expensive: HDF5 read + KDTree gridding
    # + one run_metplus.py subprocess, one file per (experiment, init
    # cycle, lead)) file-processing pool in src/mpas_stat/radar/diag.py.
    # None -> falls back to the top-level config["n_workers"].
    "n_workers": None,
    # Aggregation convention at the GRAPHICS stage (both are always
    # available in every diagnosis file -- see aggregate_radar_cycles()):
    #   False (default) -- bootstrap CI over each init cycle's own derived
    #                       FSS/TS/ETS/FBIAS/RMSE; POD/FAR are plain
    #                       cycle-means (never bootstrapped, matching the
    #                       standalone script).
    #   True            -- sum each cycle's raw FSS_num/FSS_denom/
    #                       NA/NB/NC/ND across cycles first, then compute
    #                       ONE score from the totals (no CI -- the
    #                       standalone script's own aggregated_score=True
    #                       convention). See _parse_nbrcnt()'s docstring
    #                       for how FSS_num/FSS_denom are now back-derived
    #                       from MET's own NBRCNT output for this mode.
    "aggregated_score": False,
    # --- METplus / grid_stat settings (new) -------------------------------
    # MET neighborhood-coverage SHAPE/threshold -- the user's chosen
    # defaults ("MET defaults: square window, 50% coverage"), NOT the old
    # native kernel's "any point in radius" circular-dilation convention.
    "nbrhd_shape": "SQUARE",
    "nbrhd_cov_thresh": ">=0.5",
    # Command used to invoke METplus -- assumed to be on PATH (standard
    # METplus module/conda-activate convention); override with a full path
    # if your site needs one. A list is also accepted (e.g.
    # ["conda", "run", "-n", "metplus", "run_metplus.py"]).
    "run_metplus_cmd": "run_metplus.py",
    # Scratch directory for per-cycle METplus conf files, logs, the .npz
    # cache src/mpas_stat/radar/metplus_embed.py reads, and MET's own output files.
    # Defaults to <diag_base_dir>/metplus_scratch when unset (resolved by
    # src/mpas_stat/radar/diag.py, which is the only place that already has
    # diag_base_dir in hand).
    "metplus_work_dir": None,
    # Per-cycle run_metplus.py subprocess timeout, in seconds.
    "metplus_timeout_sec": 600,
    # Keep each cycle's full METplus scratch tree (conf/log/MET outputs)
    # after a SUCCESSFUL run, for debugging -- the conf/log/ASCII outputs
    # are always small text and kept regardless; this only controls the
    # (larger, binary) .npz cache, which is deleted after grid_stat reads
    # it unless this is set.
    "keep_metplus_scratch": False,
    # MET's MODEL/OBTYPE header fields for the grid_stat conf -- default to
    # the experiment name / diag_obs.upper() when unset.
    "metplus_model_name": None,
    "metplus_obtype": None,
}


def resolve_radar_settings(config: Dict) -> Optional[Dict]:
    """Merge config["radar"] over DEFAULT_RADAR_CONFIG, or None if the
    config has no "radar" section at all -- both src/mpas_stat/radar/diag.py
    and src/mpas_stat/radar/plot.py use this to no-op cleanly (this product
    is entirely opt-in) for a config that doesn't use it."""
    if not config.get("radar"):
        return None
    merged = dict(DEFAULT_RADAR_CONFIG)
    merged.update(config["radar"])
    return merged


def default_embed_script_path() -> str:
    """Path to metplus_embed.py, the MET Python-embedding script this
    module's GridStat confs point FCST_VAR1_NAME/OBS_VAR1_NAME at --
    resolved relative to THIS file so it works regardless of the caller's
    own working directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "metplus_embed.py")

# ============================================================================
# Statistical helpers
# ============================================================================
def remove_outliers(data, threshold=3):
    data = np.asarray(data)
    clean = data[np.isfinite(data)]
    if clean.size == 0:
        return clean
    mean, std = np.mean(clean), np.std(clean)
    if std == 0:
        return clean
    z = np.abs((clean - mean) / std)
    return clean[z < threshold]


def calc_confidence_bootstrap(data, alpha=0.95, n_samples=8000):
    """(mean, lower, upper) bootstrap CI of the mean -- vectorized: draws
    all `n_samples` resamples as one (n_samples, n) index array and reduces
    with a single `.mean(axis=1)` call."""
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return (np.nan, np.nan, np.nan)
    data = remove_outliers(data)
    n = data.size
    if n == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng()
    idx = rng.integers(0, n, size=(n_samples, n))
    boot = data[idx].mean(axis=1)
    lower = np.percentile(boot, (1 - alpha) / 2 * 100)
    upper = np.percentile(boot, (alpha + (1 - alpha) / 2) * 100)
    return (np.mean(data), lower, upper)

# ============================================================================
# Gridding (unchanged)
# ============================================================================
def calc_composite_reflectivity(obs_lat, obs_lon, obs_var, hofx_var, sample_bin, kdtree_workers=1):
    """Grid scattered obs/hofx reflectivity onto a regular lat/lon grid,
    taking the max value within `sample_bin` of each cell center (radar
    "composite reflectivity" convention). `kdtree_workers` defaults to 1
    because this normally runs inside a worker process that is itself one
    of many already saturating the machine's cores (see process_one_cycle())
    -- a second, nested layer of threaded parallelism here would
    oversubscribe. Pass a larger value (or -1 for "all cores") only when
    calling this on its own, outside that outer parallel pipeline."""
    lat_min, lat_max = np.nanmin(obs_lat), np.nanmax(obs_lat)
    lon_min, lon_max = np.nanmin(obs_lon), np.nanmax(obs_lon)
    lat_edges = np.arange(lat_min, lat_max + sample_bin, sample_bin)
    lon_edges = np.arange(lon_min, lon_max + sample_bin, sample_bin)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    ny, nx = len(lat_centers), len(lon_centers)

    # lat varies slowest, matching np.meshgrid(..., indexing='ij').
    grid_lat = np.repeat(lat_centers, nx)
    grid_lon = np.tile(lon_centers, ny)
    grid_points = np.column_stack((grid_lat, grid_lon))

    points = np.column_stack((obs_lat, obs_lon))
    tree = cKDTree(points)
    neighbor_lists = tree.query_ball_point(grid_points, r=sample_bin, workers=kdtree_workers)

    lengths = np.fromiter((len(nb) for nb in neighbor_lists), dtype=np.int64, count=len(neighbor_lists))
    total = int(lengths.sum())

    CY_flat = np.full(ny * nx, np.nan)
    CX_flat = np.full(ny * nx, np.nan)

    if total > 0:
        grid_idx_flat = np.repeat(np.arange(ny * nx), lengths)
        obs_idx_flat = np.fromiter(
            (i for nb in neighbor_lists for i in nb), dtype=np.int64, count=total,
        )
        has_neighbors = lengths > 0

        # np.maximum.at does NOT skip NaN like np.nanmax does -- substitute
        # -inf for NaN inputs before reducing, then convert any cell whose
        # reduced value is still -inf back to NaN (see module docstring).
        obs_safe = np.where(np.isfinite(obs_var), obs_var, -np.inf)
        hofx_safe = np.where(np.isfinite(hofx_var), hofx_var, -np.inf)

        CY_acc = np.full(ny * nx, -np.inf)
        CX_acc = np.full(ny * nx, -np.inf)
        np.maximum.at(CY_acc, grid_idx_flat, obs_safe[obs_idx_flat])
        np.maximum.at(CX_acc, grid_idx_flat, hofx_safe[obs_idx_flat])
        CY_flat[has_neighbors] = CY_acc[has_neighbors]
        CX_flat[has_neighbors] = CX_acc[has_neighbors]
        CY_flat[CY_flat == -np.inf] = np.nan
        CX_flat[CX_flat == -np.inf] = np.nan

    return CY_flat.reshape(ny, nx), CX_flat.reshape(ny, nx)


def _grid_centers(lat, lon, sample_bin):
    """Rebuilds the SAME 1-D lat/lon cell-center arrays
    calc_composite_reflectivity() computes internally -- needed again here
    (only) to describe the grid to MET (see _write_metplus_npz_cache()).
    Cheap (1-D) so recomputing is simpler than threading an extra return
    value through calc_composite_reflectivity()."""
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    lat_edges = np.arange(lat_min, lat_max + sample_bin, sample_bin)
    lon_edges = np.arange(lon_min, lon_max + sample_bin, sample_bin)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    return lat_centers, lon_centers

# ============================================================================
# METplus / grid_stat helpers
# ============================================================================
def _met_threshold_str(threshold) -> str:
    """MET threshold string for this product's (obs/fcst >= threshold)
    convention -- e.g. 20 -> "ge20"."""
    return f"ge{threshold:g}"


def _nbrhd_width(radius) -> int:
    """MET neighborhood WIDTH (an odd number of grid cells across a square
    window) from this product's own `radii` convention (a radius in grid
    cells) -- e.g. radius=3 -> width=7."""
    return 2 * int(radius) + 1


def _write_metplus_npz_cache(cache_path: str, obs_grid, fcst_grid, lat_centers, lon_centers,
                             valid_dt: datetime, init_dt: datetime, lead_hour,
                             diag_var: str) -> None:
    """Writes the one scratch file src/mpas_stat/radar/metplus_embed.py reads back (see
    that script's own docstring for the exact contract) -- everything it
    needs, computed ONCE here where the real datetimes/grid spacing are
    already in hand, so the embedding script never has to re-derive
    anything itself."""
    delta_lat = float(lat_centers[1] - lat_centers[0]) if len(lat_centers) > 1 else float(lat_centers[0]) or 1e-3
    delta_lon = float(lon_centers[1] - lon_centers[0]) if len(lon_centers) > 1 else float(lon_centers[0]) or 1e-3
    np.savez(
        cache_path,
        obs_grid=np.asarray(obs_grid, dtype=np.float64),
        fcst_grid=np.asarray(fcst_grid, dtype=np.float64),
        grid_name="RadarCompositeGrid",
        lat_ll=float(lat_centers[0]),
        lon_ll=float(lon_centers[0]),
        delta_lat=delta_lat,
        delta_lon=delta_lon,
        nlat=len(lat_centers),
        nlon=len(lon_centers),
        valid=valid_dt.strftime("%Y%m%d_%H%M%S"),
        init=init_dt.strftime("%Y%m%d_%H"),
        lead=f"{int(lead_hour):02d}",
        accum="000000",
        name=diag_var,
        long_name="Composite reflectivity",
        level="L0",
        units="dBZ",
    )


_GRIDSTAT_CONF_TEMPLATE = """\
[config]
PROCESS_LIST = GridStat
LOOP_BY = INIT
INIT_TIME_FMT = %Y%m%d%H
INIT_BEG = {init}
INIT_END = {init}
INIT_INCREMENT = 3600
LEAD_SEQ = {lead}

MODEL = {model}
OBTYPE = {obtype}

FCST_IS_PROB = false
OBS_IS_PROB = false

FCST_GRID_STAT_INPUT_DIR =
FCST_GRID_STAT_INPUT_TEMPLATE = PYTHON_NUMPY
OBS_GRID_STAT_INPUT_DIR =
OBS_GRID_STAT_INPUT_TEMPLATE = PYTHON_NUMPY

FCST_VAR1_NAME = {fcst_cmd}
FCST_VAR1_LEVELS = (*,*)
FCST_VAR1_THRESH = {thresh_list}
OBS_VAR1_NAME = {obs_cmd}
OBS_VAR1_LEVELS = (*,*)
OBS_VAR1_THRESH = {thresh_list}

GRID_STAT_NEIGHBORHOOD_WIDTH = {width_list}
GRID_STAT_NEIGHBORHOOD_SHAPE = {shape}
GRID_STAT_NEIGHBORHOOD_COV_THRESH = {cov_thresh}

GRID_STAT_OUTPUT_FLAG_FHO = NONE
GRID_STAT_OUTPUT_FLAG_CTC = NONE
GRID_STAT_OUTPUT_FLAG_CTS = NONE
GRID_STAT_OUTPUT_FLAG_CNT = NONE
GRID_STAT_OUTPUT_FLAG_MCTC = NONE
GRID_STAT_OUTPUT_FLAG_MCTS = NONE
GRID_STAT_OUTPUT_FLAG_SL1L2 = NONE
GRID_STAT_OUTPUT_FLAG_NBRCTC = BOTH
GRID_STAT_OUTPUT_FLAG_NBRCTS = BOTH
GRID_STAT_OUTPUT_FLAG_NBRCNT = BOTH

GRID_STAT_OUTPUT_DIR = {output_dir}
GRID_STAT_OUTPUT_PREFIX = {prefix}

OUTPUT_BASE = {output_base}
LOG_DIR = {log_dir}
TMP_DIR = {tmp_dir}
LOG_MET_OUTPUT_TO_METPLUS = no
"""


def _build_gridstat_conf(*, init_dt: datetime, lead_hour, model: str, obtype: str,
                        embed_script: str, cache_path: str, thresholds: List[float],
                        radii: List[int], nbrhd_shape: str, nbrhd_cov_thresh: str,
                        output_dir: str, prefix: str, work_dir: str) -> str:
    """Builds one cycle's METplus GridStat conf text -- scoped to exactly
    this (init, lead) pair, with every threshold and every neighborhood
    width (converted from `radii`) requested in ONE grid_stat invocation
    (MET computes NBRCTC/NBRCTS/NBRCNT for every variable/threshold/width
    combination in a single run)."""
    thresh_list = ", ".join(_met_threshold_str(t) for t in thresholds)
    width_list = ", ".join(str(_nbrhd_width(r)) for r in radii)
    py = sys.executable or "python3"
    fcst_cmd = f"{py} {embed_script} {cache_path} FCST"
    obs_cmd = f"{py} {embed_script} {cache_path} OBS"
    return _GRIDSTAT_CONF_TEMPLATE.format(
        init=init_dt.strftime("%Y%m%d%H"),
        lead=f"{int(lead_hour)}",
        model=model, obtype=obtype,
        fcst_cmd=fcst_cmd, obs_cmd=obs_cmd,
        thresh_list=thresh_list, width_list=width_list,
        shape=nbrhd_shape, cov_thresh=nbrhd_cov_thresh,
        output_dir=output_dir, prefix=prefix,
        output_base=work_dir, log_dir=os.path.join(work_dir, "logs"),
        tmp_dir=os.path.join(work_dir, "tmp"),
    )


def _run_metplus(conf_path: str, run_metplus_cmd, log_path: str, timeout_sec) -> None:
    """Runs `run_metplus.py <conf_path>` (or whatever `run_metplus_cmd`
    resolves to), capturing all of its stdout/stderr to `log_path` -- the
    first thing to check if a cycle fails. Raises RuntimeError (caught by
    process_one_cycle(), which drops that cycle like any other failure) on
    a non-zero exit or timeout."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = list(run_metplus_cmd) if isinstance(run_metplus_cmd, (list, tuple)) else str(run_metplus_cmd).split()
    cmd = cmd + [conf_path]
    with open(log_path, "w") as logf:
        try:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=timeout_sec)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"run_metplus.py timed out after {timeout_sec}s for {conf_path}; "
                f"see {log_path} for its output so far.") from e
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Could not launch {cmd[0]!r} -- is run_metplus.py on PATH? "
                f"(set radar.run_metplus_cmd in the config to override)") from e
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_metplus.py exited {proc.returncode} for {conf_path}; "
            f"see {log_path} for its output.")


def _find_metplus_outputs(output_dir: str) -> Dict[str, str]:
    """Locates the combined .stat file and the three dedicated,
    self-describing per-line-type ASCII files grid_stat wrote for one
    cycle (GRID_STAT_OUTPUT_FLAG_{{NBRCTC,NBRCTS,NBRCNT}}=BOTH -- see
    module docstring). Each run_metplus.py invocation here is scoped to
    exactly one cycle, so exactly one of each is expected; if MET's own
    directory-nesting-by-date ever produces more than one, the most
    recently modified is used."""
    patterns = {
        "stat_file": "*.stat",
        "nbrctc_file": "*_nbrctc.txt",
        "nbrcts_file": "*_nbrcts.txt",
        "nbrcnt_file": "*_nbrcnt.txt",
    }
    found = {
        key: glob.glob(os.path.join(output_dir, "**", pat), recursive=True)
        for key, pat in patterns.items()
    }
    missing = [patterns[key] for key, paths in found.items() if not paths]
    if missing:
        raise RuntimeError(
            f"grid_stat did not produce expected output file(s) {missing} under "
            f"{output_dir} -- check this cycle's run_metplus.py log.")
    return {key: max(paths, key=os.path.getmtime) for key, paths in found.items()}


def _read_met_ascii_table(path: str) -> List[Dict[str, str]]:
    """Parses a MET dedicated per-line-type ASCII output file (header row
    of column names, one data row per (variable, threshold, neighborhood
    width) combination) into a list of {column_name: raw_string_value}
    dicts -- deliberately by column NAME, never by position (see module
    docstring for why the COMBINED .stat file can't be parsed this way)."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return []
    header = lines[0].split()
    rows = []
    for ln in lines[1:]:
        tokens = ln.split()
        if len(tokens) != len(header):
            # Tolerate a row that doesn't tokenize to the header's column
            # count (an unanticipated MET ASCII quirk) by skipping just
            # that row rather than failing the whole cycle on it.
            continue
        rows.append(dict(zip(header, tokens)))
    return rows


def _thresh_value(raw: Optional[str]) -> Optional[float]:
    """Pulls the numeric value out of a MET threshold string regardless of
    its exact formatting (e.g. "ge20.000000", ">=20.000000") -- used to
    match an output row back to one of this product's configured
    `thresholds`, since MET's own canonical formatting was not confirmed
    with certainty against documentation during integration."""
    if raw is None:
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    return float(m.group()) if m else None


def _nan_from(row: Dict[str, str], key: str) -> float:
    v = row.get(key)
    if v is None or v.strip().upper() in ("NA", "NAN", ""):
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def _match_rows(rows: List[Dict[str, str]], thresholds: List[float], radii: List[int]
               ) -> Dict:
    """Indexes a dedicated-file's rows by (threshold_index, radius_index),
    matching each row's own FCST_THRESH (numeric value, tolerant of MET's
    exact string formatting -- see _thresh_value()) and INTERP_PNTS (==
    width**2 for a SQUARE neighborhood, independent of this module's own
    `radii` convention) -- rather than assuming grid_stat emits rows in
    the same order they were requested in."""
    out = {}
    for row in rows:
        thr_val = _thresh_value(row.get("FCST_THRESH"))
        pts_raw = row.get("INTERP_PNTS")
        if thr_val is None or pts_raw is None:
            continue
        try:
            pts = int(float(pts_raw))
        except ValueError:
            continue
        ithre = next((i for i, t in enumerate(thresholds) if np.isclose(t, thr_val)), None)
        irad = next((i for i, r in enumerate(radii) if pts == _nbrhd_width(r) ** 2), None)
        if ithre is not None and irad is not None:
            out[(ithre, irad)] = row
    return out


def _parse_nbrctc(rows: List[Dict[str, str]], thresholds: List[float], radii: List[int]):
    """NBRCTC counts (FY_OY/FY_ON/FN_OY/FN_ON) -> this module's existing
    na/nb/nc/nd convention (na=hits, nb=misses, nc=false alarms,
    nd=correct negatives)."""
    matched = _match_rows(rows, thresholds, radii)
    n_thre, n_rad = len(thresholds), len(radii)
    na = np.zeros((n_thre, n_rad), dtype=np.int64)
    nb = np.zeros((n_thre, n_rad), dtype=np.int64)
    nc = np.zeros((n_thre, n_rad), dtype=np.int64)
    nd = np.zeros((n_thre, n_rad), dtype=np.int64)
    for (ithre, irad), row in matched.items():
        na[ithre, irad] = int(_nan_from(row, "FY_OY") or 0)
        nb[ithre, irad] = int(_nan_from(row, "FN_OY") or 0)
        nc[ithre, irad] = int(_nan_from(row, "FY_ON") or 0)
        nd[ithre, irad] = int(_nan_from(row, "FN_ON") or 0)
    return na, nb, nc, nd


def _parse_nbrcts(rows: List[Dict[str, str]], thresholds: List[float], radii: List[int]):
    """NBRCTS scores -> this module's existing ts/ets/fbias/pod/far names
    (CSI->ts, GSS->ets, FBIAS->fbias, PODY->pod, FAR->far)."""
    matched = _match_rows(rows, thresholds, radii)
    n_thre, n_rad = len(thresholds), len(radii)
    ts = np.full((n_thre, n_rad), np.nan)
    ets = np.full((n_thre, n_rad), np.nan)
    fbias = np.full((n_thre, n_rad), np.nan)
    pod = np.full((n_thre, n_rad), np.nan)
    far = np.full((n_thre, n_rad), np.nan)
    for (ithre, irad), row in matched.items():
        ts[ithre, irad] = _nan_from(row, "CSI")
        ets[ithre, irad] = _nan_from(row, "GSS")
        fbias[ithre, irad] = _nan_from(row, "FBIAS")
        pod[ithre, irad] = _nan_from(row, "PODY")
        far[ithre, irad] = _nan_from(row, "FAR")
    return ts, ets, fbias, pod, far


def _parse_nbrcnt(rows: List[Dict[str, str]], thresholds: List[float], radii: List[int]):
    """Returns (fss, fss_num, fss_denom). fss is NBRCNT's own FSS column,
    used as-is for the aggregated_score=False (bootstrap each cycle's own
    derived score) path, unchanged from before.

    fss_num/fss_denom are the SAME accumulator-style components
    aggregate_radar_cycles() has always summed across cycles before taking
    ONE ratio for the aggregated_score=True path (see that function's
    docstring) -- back-derived here from NBRCNT's own FBS/FSS/TOTAL
    columns:
        FBS  = (1/TOTAL) * sum((fcst_frac - obs_frac)**2)      [a MEAN]
        FSS  = 1 - FBS / FBS_worst
        =>    FBS_worst = FBS / (1 - FSS)
    so num = FBS*TOTAL (= sum((fcst_frac-obs_frac)**2) for this cycle) and
    denom = FBS_worst*TOTAL (= sum(fcst_frac**2 + obs_frac**2) for this
    cycle) -- exactly the fss_num/fss_denom this module's aggregation has
    always summed. This sidesteps depending on MET's own, NOT clearly
    documented, cross-case NBRCNT combination behavior (see module
    docstring's "IMPORTANT cross-cycle aggregation note").

    KNOWN LIMITATION: a cycle with FSS numerically indistinguishable from
    1.0 (effectively a perfect match for that cycle) cannot be divided
    back into a finite FBS_worst (it's a 0/0) -- such a cycle's
    fss_num/fss_denom are both set to 0, so it stops contributing to the
    pooled ratio entirely (consistent with aggregate_radar_cycles()'s
    existing `if total_denom > 0` guard), which very slightly understates
    its weight in the pooled FSS rather than mis-weighting it. This is
    expected to be rare for a continuous field like reflectivity."""
    matched = _match_rows(rows, thresholds, radii)
    n_thre, n_rad = len(thresholds), len(radii)
    fss = np.full((n_thre, n_rad), np.nan)
    fss_num = np.zeros((n_thre, n_rad))
    fss_denom = np.zeros((n_thre, n_rad))
    for (ithre, irad), row in matched.items():
        fbs = _nan_from(row, "FBS")
        fss_val = _nan_from(row, "FSS")
        total = _nan_from(row, "TOTAL")
        fss[ithre, irad] = fss_val
        if not (np.isfinite(fbs) and np.isfinite(fss_val) and np.isfinite(total)):
            continue
        if fss_val < 0.999999:
            fbs_worst = fbs / (1.0 - fss_val)
            fss_num[ithre, irad] = fbs * total
            fss_denom[ithre, irad] = fbs_worst * total
        # else: leave at 0 (see KNOWN LIMITATION above).
    return fss, fss_num, fss_denom

# ============================================================================
# Per-cycle worker (runs in a separate process at the statistics stage)
# ============================================================================
def process_one_cycle(fname: Optional[str], thresholds: List[float], radii: List[int],
                      sample_bin: float, diag_var: str = "equivalentReflectivityFactor",
                      *, exp_name: str, diag_obs: str, init_time: datetime, lead_hour,
                      metplus_opts: Dict) -> Optional[Dict[str, "np.ndarray"]]:
    """Read one HofX obsout file, grid obs/hofx composite reflectivity
    natively (unchanged), compute RMSE natively (unchanged), then hand the
    neighborhood FSS/categorical scoring off to MET's grid_stat (via
    METplus) for this one cycle -- see module docstring.

    Designed to run inside a ProcessPoolExecutor worker (see
    mpas_stat.radar.diag.process_radar_experiment()): picklable
    arguments in (metplus_opts is a plain dict of strings/numbers/bools),
    only small derived quantities out -- never the raw lat/lon/obs/hofx
    arrays or the gridded fields.

    Return-dict shape is UNCHANGED from before this integration (same
    keys: rmse/fss/ts/ets/fbias/pod/far/fss_num/fss_denom/na/nb/nc/nd,
    same (nThresholds, nRadii) array shapes) plus one new key, "stat_file"
    (the path to the combined MET .stat file this cycle's grid_stat run
    wrote, kept for provenance/debugging -- not read by
    aggregate_radar_cycles()). This is deliberate: the statistics- and
    graphics-stage code around this function (_write_radar_diag_file(),
    aggregate_radar_cycles(), everything in src/mpas_stat/radar/plot.py)
    did not need to change, because MET only replaced what COMPUTES these
    numbers, not their shape.

    `fname` may be None (mpas_stat.obs.common.build_hofx_forecast_filename-style
    lookup found no matching file) or a path that doesn't exist -- both
    return None, same as a read failure or a failed grid_stat run, and the
    caller drops them."""
    if not fname or not os.path.exists(fname):
        return None
    try:
        with h5py.File(fname, "r") as f:
            lat = f["MetaData"]["latitude"][:]
            lon = f["MetaData"]["longitude"][:]
            obs = f["ObsValue"][diag_var][:]
            hofx = f["hofx"][diag_var][:]
    except Exception as e:
        warnings.warn(f"Error reading {fname}: {e}")
        return None

    obs_grid, fcst_grid = calc_composite_reflectivity(lat, lon, obs, hofx, sample_bin, kdtree_workers=1)
    lat_centers, lon_centers = _grid_centers(lat, lon, sample_bin)

    omf = np.where((obs_grid > 0.) & (fcst_grid > 0.), obs_grid - fcst_grid, np.nan)
    rmse = float(np.sqrt(np.nanmean(np.square(omf))))

    valid_time = init_time + timedelta(hours=int(lead_hour))
    cycle_tag = f"{exp_name}_{diag_obs}_{init_time.strftime('%Y%m%d%H')}_f{int(lead_hour):03d}"
    work_dir = os.path.join(metplus_opts["work_dir"], cycle_tag)
    os.makedirs(work_dir, exist_ok=True)

    cache_path = os.path.join(work_dir, "data.npz")
    _write_metplus_npz_cache(cache_path, obs_grid, fcst_grid, lat_centers, lon_centers,
                             valid_time, init_time, lead_hour, diag_var)

    output_dir = os.path.join(work_dir, "metplus_out")
    conf_text = _build_gridstat_conf(
        init_dt=init_time, lead_hour=lead_hour,
        model=metplus_opts.get("model_name") or exp_name,
        obtype=metplus_opts.get("obtype") or diag_obs.upper(),
        embed_script=metplus_opts["embed_script"], cache_path=cache_path,
        thresholds=thresholds, radii=radii,
        nbrhd_shape=metplus_opts.get("nbrhd_shape", "SQUARE"),
        nbrhd_cov_thresh=metplus_opts.get("nbrhd_cov_thresh", ">=0.5"),
        output_dir=output_dir, prefix=cycle_tag, work_dir=work_dir,
    )
    conf_path = os.path.join(work_dir, "GridStat.conf")
    with open(conf_path, "w") as f:
        f.write(conf_text)

    log_path = os.path.join(work_dir, "logs", "run_metplus.log")
    try:
        _run_metplus(conf_path, metplus_opts.get("run_metplus_cmd", "run_metplus.py"),
                    log_path, metplus_opts.get("timeout_sec", 600))
        outputs = _find_metplus_outputs(output_dir)
        nbrctc_rows = _read_met_ascii_table(outputs["nbrctc_file"])
        nbrcts_rows = _read_met_ascii_table(outputs["nbrcts_file"])
        nbrcnt_rows = _read_met_ascii_table(outputs["nbrcnt_file"])
        na, nb, nc, nd = _parse_nbrctc(nbrctc_rows, thresholds, radii)
        ts, ets, fbias, pod, far = _parse_nbrcts(nbrcts_rows, thresholds, radii)
        fss, fss_num, fss_denom = _parse_nbrcnt(nbrcnt_rows, thresholds, radii)
        stat_file = outputs["stat_file"]
    except Exception as e:
        warnings.warn(f"METplus grid_stat failed for {cycle_tag}: {e}")
        return None
    finally:
        # The .npz cache can go as soon as grid_stat has read it -- keep
        # the conf/log/MET ASCII outputs regardless (small text, and the
        # useful provenance/debugging trail); the cache is the only
        # sizeable/binary scratch file here.
        if not metplus_opts.get("keep_scratch", False) and os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except OSError:
                pass

    return dict(rmse=rmse, fss=fss, ts=ts, ets=ets, fbias=fbias, pod=pod, far=far,
                fss_num=fss_num, fss_denom=fss_denom, na=na, nb=nb, nc=nc, nd=nd,
                stat_file=stat_file)

# ============================================================================
# Cross-cycle aggregation (statistics stage writes per-cycle files; the
# graphics stage gathers them back and calls this) -- UNCHANGED by the
# METplus integration, see module docstring's "IMPORTANT cross-cycle
# aggregation note".
# ============================================================================
def aggregate_radar_cycles(cycle_results: List[Dict], aggregated_score: bool
                           ) -> Optional[Dict[str, "np.ndarray"]]:
    """Aggregate one (experiment, lead)'s per-cycle radar verification
    results -- gathered by reading back the diagnosis files
    src/mpas_stat/radar/diag.py wrote for every init cycle at this lead --
    into the (mean, lower, upper) triples src/mpas_stat/radar/plot.py's
    figures need.

    aggregated_score=False: bootstrap CI over each cycle's own derived
    score -- fss/ts/ets/fbias/rmse each get their own (mean, lo, hi). pod/
    far are plain cycle-means (never bootstrapped, matching the original).

    aggregated_score=True: sum each cycle's raw accumulator components
    (fss_num/fss_denom, na/nb/nc/nd) across cycles first, then compute ONE
    score from the totals -- returned as a degenerate (val, val, val)
    triple so callers can treat both modes identically. See
    _parse_nbrcnt()'s docstring for how fss_num/fss_denom are now
    populated (back-derived from MET's NBRCNT output) in this mode.

    cycle_results: list of dicts shaped like process_one_cycle()'s return
    (as read back from diagnosis files), already filtered to drop any
    missing/unreadable entries. Returns None if the list is empty."""
    if not cycle_results:
        return None

    n_thre, n_rad = np.asarray(cycle_results[0]["fss"]).shape

    rmse_cycle = np.array([r["rmse"] for r in cycle_results])
    out = {"rmse": np.array(calc_confidence_bootstrap(rmse_cycle))}

    fss = np.full((n_thre, n_rad, 3), np.nan)
    ts = np.full((n_thre, n_rad, 3), np.nan)
    ets = np.full((n_thre, n_rad, 3), np.nan)
    fbias = np.full((n_thre, n_rad, 3), np.nan)
    pod = np.full((n_thre, n_rad), np.nan)
    far = np.full((n_thre, n_rad), np.nan)

    if not aggregated_score:
        fss_tot = np.stack([r["fss"] for r in cycle_results], axis=0)
        ts_tot = np.stack([r["ts"] for r in cycle_results], axis=0)
        ets_tot = np.stack([r["ets"] for r in cycle_results], axis=0)
        fbias_tot = np.stack([r["fbias"] for r in cycle_results], axis=0)
        pod_tot = np.stack([r["pod"] for r in cycle_results], axis=0)
        far_tot = np.stack([r["far"] for r in cycle_results], axis=0)
        for ithre in range(n_thre):
            for irad in range(n_rad):
                fss[ithre, irad, :] = calc_confidence_bootstrap(fss_tot[:, ithre, irad])
                ts[ithre, irad, :] = calc_confidence_bootstrap(ts_tot[:, ithre, irad])
                ets[ithre, irad, :] = calc_confidence_bootstrap(ets_tot[:, ithre, irad])
                fbias[ithre, irad, :] = calc_confidence_bootstrap(fbias_tot[:, ithre, irad])
                pod[ithre, irad] = np.nanmean(pod_tot[:, ithre, irad])
                far[ithre, irad] = np.nanmean(far_tot[:, ithre, irad])
    else:
        fss_num_acc = np.sum([r["fss_num"] for r in cycle_results], axis=0)
        fss_denom_acc = np.sum([r["fss_denom"] for r in cycle_results], axis=0)
        na_acc = np.sum([r["na"] for r in cycle_results], axis=0)
        nb_acc = np.sum([r["nb"] for r in cycle_results], axis=0)
        nc_acc = np.sum([r["nc"] for r in cycle_results], axis=0)
        nd_acc = np.sum([r["nd"] for r in cycle_results], axis=0)
        for ithre in range(n_thre):
            for irad in range(n_rad):
                total_num = fss_num_acc[ithre, irad]
                total_denom = fss_denom_acc[ithre, irad]
                agg_fss = 1.0 - total_num / total_denom if total_denom > 0 else np.nan

                na_, nb_, nc_, nd_ = (int(na_acc[ithre, irad]), int(nb_acc[ithre, irad]),
                                      int(nc_acc[ithre, irad]), int(nd_acc[ithre, irad]))
                total = na_ + nb_ + nc_ + nd_
                ts_ = na_ / (na_ + nb_ + nc_) if (na_ + nb_ + nc_) > 0 else np.nan
                chance = ((na_ + nc_) * (na_ + nb_)) / total if total > 0 else 0
                denom_ets = (na_ + nb_ + nc_ - chance)
                ets_ = (na_ - chance) / denom_ets if denom_ets > 0 else np.nan
                fbias_ = (na_ + nc_) / (na_ + nb_) if (na_ + nb_) > 0 else np.nan
                pod_ = na_ / (na_ + nb_) if (na_ + nb_) > 0 else np.nan
                far_ = nc_ / (na_ + nc_) if (na_ + nc_) > 0 else np.nan

                fss[ithre, irad, :] = (agg_fss, agg_fss, agg_fss)
                ts[ithre, irad, :] = (ts_, ts_, ts_)
                ets[ithre, irad, :] = (ets_, ets_, ets_)
                fbias[ithre, irad, :] = (fbias_, fbias_, fbias_)
                pod[ithre, irad] = pod_
                far[ithre, irad] = far_

    out.update(fss=fss, ts=ts, ets=ets, fbias=fbias, pod=pod, far=far)
    return out
