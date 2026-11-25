# 2.6 Radar composite reflectivity

An optional product that scores reflectivity forecasts the way they are
usually judged: as a field, with neighbourhood and categorical scores,
rather than point by point.

For every `hofx_forecast` experiment, init cycle and lead:

1. Observed and forecast reflectivity are gridded onto a regular lat-lon
   grid as a composite: the maximum within `sample_bin` of each cell
   centre.
2. **MET `grid_stat`** (run through METplus) computes the Fractions Skill
   Score (FSS) and the categorical scores — threat score (TS), equitable
   threat score (ETS), frequency bias (FBIAS), probability of detection
   (POD) and false-alarm ratio (FAR) — for every threshold and
   neighbourhood size.
3. The RMSE of the gridded field is computed in Python.

It is **O−F only** (no cycling counterpart) and **opt-in**: nothing
happens unless the configuration has a top-level `radar:` block.

```{admonition} Check one cycle first
:class: warning
The MET integration was written against the documented GridStat settings
and output format and tested with synthetic MET output, not with a real
MET installation. Run one cycle and check
`metplus_scratch/.../logs/run_metplus.log` before a full run.
```

## Running it

Requires MET and METplus, with `run_metplus.py` on `PATH` (or set
`radar.run_metplus_cmd`).

```bash
python -m mpas_stat.cli.radar_stats    my_radar_config.yaml
python -m mpas_stat.cli.radar_graphics my_radar_config.yaml
```

Statistics are written to
`<diag_dir>/<exp>/VerifyOBS/<init>/f<HHH>/radarstat_<obs>.<valid>.nc`.
The `radarstat_` prefix keeps them apart from the point-wise `obsstat_`
files, so a radar sensor can be in both products. See
`examples/obs/11_radar_reflectivity.yaml`.

## Configuration

The `radar:` block; `paths`, `experiments`, `time_settings`, `forecast`
and `n_workers` are shared with the rest of the configuration.

```yaml
radar:
  sensors: [radar_zh]                    # IODA observation types
  diag_var: equivalentReflectivityFactor
  thresholds: [20, 40]                   # dBZ
  radii: [3, 5, 9, 15]                   # neighbourhood radius in grid cells
  sample_bin: 0.027                      # composite grid spacing and search radius (degrees)
  aggregated_score: false                # how cycles are pooled, see below
  n_workers: null                        # statistics processes (default: top-level n_workers)

  # MET / METplus
  nbrhd_shape: SQUARE
  nbrhd_cov_thresh: ">=0.5"
  run_metplus_cmd: run_metplus.py        # or [conda, run, -n, metplus, run_metplus.py]
  metplus_work_dir: null                 # default: <diag_dir>/metplus_scratch
  metplus_timeout_sec: 600
  keep_metplus_scratch: false            # keep each cycle's .npz cache
  metplus_model_name: null               # default: the experiment name
  metplus_obtype: null                   # default: the sensor name, upper case
```

## Pooling over cycles

Each statistics file holds both the per-cycle scores and the counts they
come from, so `aggregated_score` can be changed without recomputing:

`false` (default)
: Average each cycle's score, with a bootstrap confidence interval (POD
  and FAR: plain mean).

`true`
: Sum the contingency counts and the FSS numerator and denominator over
  cycles, then compute one score. This is the usual way to score a
  period, and it does not give a cycle with little reflectivity the same
  weight as an active one. No confidence interval.

## Figures

Written to `<graphic_dir>/<sensor>/radar_reflectivity/`:

| Figure | File |
|---|---|
| FSS, TS, ETS, FBIAS against lead; one panel per threshold, one figure per radius | `refl_fcst_<metric>_radius<r>.<fmt>` |
| Performance diagram (POD against 1 − FAR), points labelled by lead | `refl_fcst_performance_diagram_radius<r>.<fmt>` |
| RMSE against lead | `refl_fcst_rmse.<fmt>` |

These figures have their own simpler styling, not the shared publication
style.

## How MET is used

For each cycle:
- The gridded fields are cached as a `.npz` file.
- A GridStat configuration is written for that one init and lead, with
  every threshold and every neighbourhood width (`2·radius + 1`).
- Its forecast and observation inputs are calls to
  `mpas_stat/radar/metplus_embed.py`, a MET Python-embedding script that
  loads the cache.

Cycles run in parallel.

GridStat writes the NBRCTC (counts), NBRCTS (CSI, GSS, FBIAS, PODY, FAR)
and NBRCNT (FBS, FSS) line types. Each is also written to its own `.txt`
file, which is read by column name. The combined `.stat` file is kept;
its path is stored in the statistics file's `metplus_stat_file`
attribute.

Pooling over cycles is done in Python, not with `stat_analysis`, because
MET documents no aggregation from NBRCNT to FSS.

Caveats:
- Cells without observations are passed to MET as NaN, assuming MET
  treats NaN as missing.
- A cycle with FSS of exactly 1 has no recoverable numerator and
  denominator, so it adds nothing to the pooled FSS.

The original stand-alone script, `legacy/radar_reflectivity_verify.py`,
computes the scores in Python without MET and is kept for reference.
