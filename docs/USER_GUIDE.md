# MPAS-STAT User Guide

This repository contains two verification pipelines that share one graphics
engine:

See [`README.md`](../README.md) for installation, quick-start commands, and
the repository layout; this guide is the detailed reference.

| | Model-space | Observation-space |
|---|---|---|
| Verifies | MPAS forecast vs. init, on the model grid | IODA obs-space OMB/OMA/OMF, at observation locations |
| Input | MPAS `forecast`/`init` netCDF files | IODA `obsout_da_*.h5` / `obsout_hofx_*.h5` files |
| Statistics entry point | `src/mpas_stat/cli/model_stats.py` | `src/mpas_stat/obs/diag.py` |
| Graphics entry point | `src/mpas_stat/cli/model_graphics.py` | `src/mpas_stat/obs/plot.py` |
| PBS job script(s) | `scripts/run_verify_model.ksh`, `scripts/run_verify_model_fcst.ksh` | `scripts/run_verify_obs.ksh`, `scripts/run_verify_obs_fcst.ksh` |
| Diagnosis file | `mpasstat.<date>.nc` | `obsstat_<obs>.<date>.nc` |
| Extended-forecast skill | ✓ (`forecast_lead_0d`, `ExtendedFC/`) | ✓ (`hofx_forecast` experiments, `HofX/<cycle>/f<HHH>/`) |

Both pipelines follow the same two-stage design, and the observation-space
side draws its figures by calling the model-space side's own plotting
functions directly (not a reimplementation) — see
[Shared conventions](#shared-conventions--how-the-two-sides-connect).

```
 STATISTICS STAGE                        GRAPHICS STAGE
 (reads raw model/obs files,       ->     (reads diagnosis netCDF files,
  writes one netCDF file per              draws one PNG/PDF per figure via
  experiment x cycle)                     src/mpas_stat/graphics/functions.py)
```

---

## 1. Model-space verification

Compares an MPAS forecast field against its own initial/analysis field, on
the native model grid, binned by lat/lon/level.

### 1.1 Files

| File | Role |
|---|---|
| `src/mpas_stat/cli/model_stats.py` | Statistics entry point: `python -m mpas_stat.cli.model_stats -c configs/model_input.yaml` |
| `src/mpas_stat/model/args.py` | CLI argument parsing / config overrides (`-f/-i/-o/-n/-v`, `--lat-bin`, ...) |
| `src/mpas_stat/model/config.py` | `MPASConfig` dataclass — the schema `configs/model_input.yaml` is loaded into |
| `src/mpas_stat/model/variables.py` | Extracts named physical variables (`U`, `V`, `T`, `Qv`, `RH`, `Z`, `U10`, `Ps`, ...) from raw MPAS netCDF fields, with units |
| `src/mpas_stat/model/interp.py` | Builds shared vertical-interpolation coefficients (model level → height/pressure), numba-accelerated |
| `src/mpas_stat/model/kernels.py` | The actual statistic kernels: 0-D region means, 1-D profiles, 2-D and 3-D binned products |
| `src/mpas_stat/model/driver.py` | `calculate_statistics(config)` — orchestrates loading, interpolation, threaded statistic tasks, and netCDF output |
| `configs/model_input.yaml` | Statistics-stage configuration template (see below) |
| `scripts/run_verify_model.ksh` | PBS job script: loops over assimilation cycles, fills in the input config, runs the statistics stage, retries on failure |
| `scripts/run_verify_model_fcst.ksh` | Same, for extended-forecast verification (writes to `f<HHH>/` subfolders) |
| `src/mpas_stat/cli/model_graphics.py` | Graphics entry point: `python -m mpas_stat.cli.model_graphics -c configs/model_graphic.yaml` |
| `src/mpas_stat/graphics/io.py` | Cycle/date helpers, bootstrap CI helpers, and the `collect_*` functions that read `mpasstat.<date>.nc` files back into plotting-ready Python dicts |
| `src/mpas_stat/graphics/functions.py` | All plotting functions (`plot_time_series_0d`, `plot_profiles_1d`, `plot_lat_lon_2d`, `plot_lat_vert_2d`, `plot_time_level_2d`, the `forecast_*` and scorecard variants, plus `plot_time_series_reduction_0d`/`plot_profile_reduction_1d` — added for the obs-space OMB→OMA RMS-reduction-percent figures, §2.4) plus shared styling/colorbar/significance helpers |
| `src/mpas_stat/graphics/driver.py` | `run_graphics(cfg)` — dispatches over `cfg["plot_types"]`, optionally in parallel (`n_workers`) |
| `configs/model_graphic.yaml` | Graphics-stage configuration template (see below) |

### 1.2 Running it

```bash
# Statistics: one call per experiment x cycle (scripts/run_verify_model.ksh
# loops this over a cycle range and fills in the config placeholders via sed
# before each call)
python -m mpas_stat.cli.model_stats -c configs/model_input.yaml

# Graphics: one call reads back every cycle in the configured time range and
# draws every figure listed in configs/model_graphic.yaml's plot_types
python -m mpas_stat.cli.model_graphics -c configs/model_graphic.yaml
```

`scripts/run_verify_model.ksh` is a PBS batch script; it templates the input
config per cycle (`forecastFile`, `initFile`, `outputFile`, `nMember`,
`nProcesses` placeholders) and writes output to:

```
<main_dir>/<exp_name>/VerifyModel/<cycle>/mpasstat.<date>.nc
```

`scripts/run_verify_model_fcst.ksh` does the same for extended-forecast runs,
writing to `.../VerifyModel/<init_cycle>/f<HHH>/mpasstat.<valid_date>.nc`.

### 1.3 `input.yaml` (statistics stage) — key fields

| Key | Meaning |
|---|---|
| `forecast_file`, `init_file`, `output_file` | Paths (filled in per-cycle by the ksh driver) |
| `n_member` | 1 = deterministic; >1 = ensemble mean/spread |
| `variables` | Physical variables to verify (`U`, `V`, `T`, `Qv`, `U10`, `Ps`, ...) |
| `lat_bin_width`, `lon_bin_width`, `level_bin_size` | Binning resolution for the 1-D/2-D/3-D gridded products |
| `pres_levels`, `hgt_levels` | Target pressure/height levels for vertical interpolation |
| `regions` | Named latitude-band regions (`Global`, `Tropics`, `NHem`, ...) statistics are averaged over |
| `regional` | `true` limits the lat/lon binning box to the mesh's own extent instead of the full globe |
| `n_processes` | Thread-pool width for the statistic tasks |

### 1.4 `graphic.yaml` (graphics stage) — key fields

| Key | Meaning |
|---|---|
| `paths.base_dir` / `paths.out_dir` | Where to read `mpasstat.<date>.nc` from / write figures to |
| `experiments.names/labels/colors/linestyles` | One entry per experiment being compared |
| `time.start/end/interval_hours` | Assimilation-cycle range to read |
| `variables`, `statistics` | Which variables and which stats (`rms`, `mean`, `std`, `spread`) to plot |
| `region` | Which of `input.yaml`'s `regions` to plot (one region per run) |
| `plot_types` | Which figures to generate — see §1.5 |
| `forecast.*`, `scorecard.*`, `level_slices` | Extended-forecast / scorecard / 2-D-slice-specific settings |

### 1.5 Available `plot_types` (model-space)

| `plot_types` entry | Function | What it draws |
|---|---|---|
| `time_series_0d` | `plot_time_series_0d` | Region-averaged statistic vs. assimilation cycle |
| `profiles_1d` | `plot_profiles_1d` | Time-averaged vertical/latitude profile |
| `profile_diff_1d` | `plot_profile_diff_1d` | Profile, plus bootstrap-significance vs. the reference experiment |
| `time_level_2d` | `plot_time_level_2d` | Cycle × level Hovmöller |
| `lat_vert_2d` | `plot_lat_vert_2d` | Latitude × vertical cross-section |
| `lat_lon_2d` | `plot_lat_lon_2d` | Lat-lon map |
| `lat_lon_slice_2d` | `plot_lat_lon_slice_2d` | Lat-lon map at one selected 3-D level |
| `forecast_lead_0d`, `forecast_*` | various | Extended-forecast-lead counterparts of the above (need `forecast:` block) |
| `forecast_scorecard` | `plot_forecast_scorecard` | Improve/degrade triangle scorecard vs. lead time |

### 1.6 `mpasstat.<date>.nc` schema

One file per experiment × cycle:

```
<top level>
  LatBins, LonBins, LevelBins, ModelLevels, HeightLevels, PressureLevels   (coordinate variables)
  task_timing_seconds/, variable_timing_seconds/                          (perf diagnostics)

  <variable>/                     e.g. U, V, T, Qv, Ps, ...
    0D/<region>/                  e.g. Global, Tropics, NHem, ...
        mean, rms, std, spread            (scalars)
    1D/
        lat_bin_profile/                          mean, rms, std, spread   (vs. LatBins)
        vert_profile_<region>/                    mean, rms, std, spread   (vs. ModelLevels)
        vert_bin_profile_<region>/                mean, rms, std, spread   (vs. LevelBins)
        vert_hgt_profile_<region>/                mean, rms, std, spread   (vs. HeightLevels)
        vert_pres_profile_<region>/               mean, rms, std, spread   (vs. PressureLevels)
    2D/
        lat_lon/, lat_lev/, lat_hgt/, lat_pres/    mean, rms, std, spread   (2-D grids)
    3D/                            (3-D variables only, e.g. U/V/T/Qv, not U10/T2/Ps/...)
        lat_lon_lev/, lat_lon_hgt/, lat_lon_pres/  mean, rms, std, spread   (3-D grids)
```

`spread` only appears when `n_member > 1`. `region` names come from
`input.yaml`'s `regions` map.

---

## 2. Observation-space verification

Compares IODA obs-space background/analysis equivalents (OMB/OMA) against
observations, at the scattered observation locations, for conventional and
radiance obs types. **Built to plug directly into the model-space graphics
engine above** — see §3.

### 2.1 Files

| File | Role |
|---|---|
| `src/mpas_stat/obs/sensor_config.py` | **Single source of truth for sensor/observation metadata.** The `OBSERVATION_CONFIGS` registry: per-sensor variables, units, panel labels, bin method, and quirks (`file_alias`, `obs_type_filter`, `level_bin_kwargs`, cycle-cadence overrides). Zero dependencies — importable standalone from either stage, a notebook, or a unit test. Add/adjust a sensor here only. |
| `src/mpas_stat/obs/common.py` | Shared layer: config schema, bin-level helpers, netCDF group naming, and the file-path convention. Imports `obs/sensor_config.py` for sensor lookups. Imported by both stages below so they can never disagree on where files live. |
| `src/mpas_stat/obs/diag.py` | **Statistics stage.** Reads IODA obs-space HDF5, computes OMB/OMA/OMF stats, writes diagnosis netCDF files. No `matplotlib` import. |
| `src/mpas_stat/obs/plot.py` | **Graphics stage.** Reads the diagnosis netCDF files back and draws figures by calling `mpas_stat.graphics.functions` directly. No `h5py` import. |
| `scripts/run_verify_obs.ksh` | PBS job script for **OMB/OMA** (cycling DA) verification: writes a config YAML from its own exported settings, then runs both stages. |
| `scripts/run_verify_obs_fcst.ksh` | PBS job script for **OMF** (extended-forecast skill) verification: same idea, `experiments.types: [hofx_forecast, ...]` and a `forecast:` block. |
| `src/mpas_stat/radar/common.py` | **Radar composite-reflectivity product** (§2.6, separate opt-in add-on): shared config schema (`radar:` block), composite-reflectivity gridding, RMSE, bootstrap/aggregation, and the METplus/`grid_stat` integration that computes per-cycle FSS/categorical neighborhood scores — this pipeline's counterpart of `obs/common.py`, imported by both stages below. No plotting, no HDF5-obs-reading orchestration of its own. |
| `src/mpas_stat/radar/metplus_embed.py` | MET **Python-embedding** script `grid_stat` (via METplus) calls to read back the gridded obs/HofX fields `radar/common.py` writes to a scratch `.npz` cache — see §2.6's METplus subsection. Not meant to be run by hand. |
| `src/mpas_stat/radar/diag.py` | Radar product **statistics stage.** Reads HofX obsout HDF5 for `hofx_forecast` experiments, grids obs/hofx composite reflectivity, writes `radarstat_<obs>.<date>.nc` diagnosis files. |
| `src/mpas_stat/radar/plot.py` | Radar product **graphics stage.** Reads those diagnosis files back, aggregates across init cycles, draws FSS/TS/ETS/FBIAS/RMSE-vs-lead-time and Performance Diagram figures. Has its own small plotting functions (not `mpas_stat.graphics.functions`) — see the callout in §2.6. |
| `legacy/radar_reflectivity_verify.py` | The **original standalone script** this product was integrated from (optimized: vectorized composite-reflectivity gridding, vectorized bootstrap, `ProcessPoolExecutor` file-level parallelism). Kept as a self-contained, independently runnable reference/example — NOT imported by the pipeline files above, which have their own (near-identical) copies of its computation kernels in `radar/common.py`. |

### 2.2 Running it

Two independent verification modes, matching two `experiments.types` values
(an experiments list may freely mix both types in one config, but the two
ksh job scripts below each set up one mode for clarity):

- **OMB/OMA** (background/analysis, cycling DA) — `exp_type: da` (OMB+OMA)
  or `hofx` (OMB only). Reads
  `<da_dir>/<exp>/CyclingDA/<cycle>/dbOut/obsout_da_<obs>.h5` /
  `<hofx_dir>/<exp>/<cycle>/dbOut/obsout_hofx_<obs>.h5`, time-resolved per
  assimilation cycle.
- **OMF** (obs-minus-forecast, extended-forecast skill) — `exp_type:
  hofx_forecast`. Reads
  `<hofx_dir>/<exp>/HofX/<init_cycle>/f<HHH>/dbOut/obsout_hofx_<obs>.h5` for
  every configured forecast lead hour.

```bash
# Statistics: reads obsout_da/hofx_<obs>.h5 (or, for hofx_forecast
# experiments, HofX/<cycle>/f<HHH>/dbOut/obsout_hofx_<obs>.h5) for every
# configured experiment x cycle [x lead] x sensor, writes obsstat_<obs>.<date>.nc
python -m mpas_stat.cli.obs_stats my_config.yaml

# Graphics: reads those diagnosis files back and draws every figure
python -m mpas_stat.cli.obs_graphics my_config.yaml
```

Both stages read the **same config file** (unlike the model side, which
splits `configs/model_input.yaml` / `configs/model_graphic.yaml`) — the
obs-space schema is smaller and sensor list / experiments / domains / time
range are needed by both stages to agree on which cycles (and lead hours) and
sensors to look for.

`scripts/run_verify_obs.ksh` / `scripts/run_verify_obs_fcst.ksh` are PBS
job-script wrappers around the two commands above, mirroring
`scripts/run_verify_model.ksh` in spirit (same PBS directives,
retry-on-failure loop). They are simpler than their model-space
counterpart, though: **neither loops per cycle in the shell.** Both python
stages already sweep the *entire* configured cycle range (and, for OMF,
every forecast lead hour) internally from a single config file, so each job
script templates one YAML config from its own exported shell variables and
runs the two stages exactly once each — there is no per-cycle
`sed`/re-invoke loop like `scripts/run_verify_model.ksh`'s. Edit the `export ...`
lines near the top of either script to point it at real
experiments/sensors/directories, then `qsub scripts/run_verify_obs[_fcst].ksh`.

### 2.3 Config file — key fields

```yaml
# Render figures in parallel across this many worker processes in
# src/mpas_stat/obs/plot.py. The unit of parallel work is one SENSOR: each
# worker reads that sensor's own diagnosis files and renders all of its
# figures, so no process ever needs more than one sensor's statistics in
# memory at a time -- see _process_sensor(). 1 = sequential (default,
# unchanged behaviour, sensors processed one at a time). With many sensors
# configured this can be set close to the number of CPUs available to the
# job -- the same top-level key as the model-space package's own
# configs/model_graphic.yaml / mpas_stat.graphics.driver, just applied at
# sensor granularity here since obs-space has a natural per-sensor split
# that model-space doesn't.
n_workers: 1

paths:
  da_dir: ...       # CyclingDA root for exp_type: da
  hofx_dir: ...      # root for exp_type: hofx (no OMA)
  diag_dir: ...       # obs-space diagnosis netCDF files (defaults to <graphic_dir>/diag_files)
  graphic_dir: ...    # figures

experiments:
  names: [...]
  labels: [...]
  types: [da, hofx_forecast]   # "da" -> OMB+OMA (CyclingDA); "hofx" -> OMB only;
                                # "hofx_forecast" -> OMF (extended-forecast skill, HofX/<cycle>/f<HHH>/)
  outerloop: [2, 0]             # which analysis outer-loop's OMA to read ("da", n_member==1 only)
  n_member: [1, 1]              # >1 enables ensemble spread; also changes WHERE OMA is read from --
                                  # see note below

sensors: [sondes, aircraft, amsua_n18, ...]   # keys into OBSERVATION_CONFIGS (src/mpas_stat/obs/sensor_config.py)

domains:
  global: [null, null, null, null]
  NXTro: [30.0, 90.0, -180.0, 180.0]          # [lat_min, lat_max, lon_min, lon_max]

plot_settings:
  plot_oma: true
  apply_domains: [global, NXTro]               # which `domains` entries to compute/plot
  figure_format: png
  max_time_ticks: 12
  generate_lat_lon_2d: true
  latlon_domain: global                        # "global" (default, full globe), a `domains` key
                                                  # (fixed hand-specified box), or "auto" (detect the
                                                  # box from valid-statistics extent) -- see note below
  latlon_lat_bin_width: 10.0                   # 2-D map grid resolution (degrees)
  latlon_lon_bin_width: 10.0
  generate_time_level_2d: true                 # cycle x level Hovmoller -- see the comparison-
                                                  # convention callout below
  generate_lat_vert_2d: true                   # latitude x level cross-section (skipped for radiances)
  comparison_relative_reduction_statistics: [RMS, STD]   # which stats use the reference/reduction
                                                            # convention on the 2-D comparison figures

time_settings:
  start_time: "2024-09-13 06:00:00"        # initialization-cycle range for BOTH modes --
  end_time: "2024-09-21 12:00:00"          # for hofx_forecast, one HofX forecast per cycle
  interval: 6                                   # hours; overridden per-sensor when the sensor
                                                  # sets cycle_interval_hours (e.g. sondes -> 12h)

# Required only when experiments.types includes "hofx_forecast" -- same
# schema as the model-space graphic.yaml's "forecast:" block / the
# forecast_length/forecast_interval exports in scripts/run_verify_model_fcst.ksh.
forecast:
  length: 24     # maximum forecast lead hour
  interval: 6    # lead-time spacing in hours
  # start: 6     # first lead hour (default = interval; 0 includes the init time itself)
```

> **`n_member` also changes where OMA is read from (`exp_type: da` only).**
> Deterministic DA (`n_member: 1`, e.g. a single-outer-loop 3DEnVar/hybrid
> run) writes background AND analysis innovations as separate groups
> (`ombg`/`oman`, or `hofx0`/`hofx<outerloop>`) inside the **same**
> `obsout_da_<obs>.h5` file. Ensemble DA (`n_member > 1`, e.g. LETKF/GETKF)
> writes OMA to a **separate** file instead:
> `obsout_da_<obs>_ana.h5`, alongside the background's `obsout_da_<obs>.h5`.
> `build_filename()` in `src/mpas_stat/obs/diag.py` handles both cases
> automatically from `n_member` — nothing else to configure, but worth
> knowing if you're checking where a given cycle's OMA actually lives on
> disk.

> **Sondes only count 00Z/12Z — but what that means differs between
> OMB/OMA (cycling) and OMF (extended forecast).** Radiosondes only launch
> at 00Z/12Z, so `src/mpas_stat/obs/sensor_config.py`'s `"sondes"` entry sets
> `cycle_valid_hours: [0, 12]` (alongside `cycle_interval_hours: 12`). How
> that restriction gets applied depends on what the generated time list
> actually represents:
>
> - **OMB/OMA (cycling)** — the generated times ARE the observation/cycle
>   times, so the restriction applies directly to them.
>   `resolve_time_settings()` in `src/mpas_stat/obs/common.py` (called with its
>   default `for_forecast=False`) snaps the configured
>   `time_settings.start_time` forward to the next 00Z or 12Z
>   (`_next_valid_cycle_start()` — rolling into the next day if both hours
>   today have already passed) before generating sondes' cycle list, so
>   it's impossible to end up with a sondes cycle at any other hour no
>   matter what `start_time`/`interval` the rest of the config uses — this
>   works for ANY configured start hour, not just specific ones. Since 24
>   hours divides evenly by the 12-hour cadence, every cycle after the
>   first stays on 00Z/12Z too, with no per-cycle re-checking needed.
> - **OMF (extended forecast, `hofx_forecast` experiments)** — the
>   generated times are INIT (forecast launch) times, a property of the
>   MODEL/DA system, not of any one sensor's observation network: a
>   forecast can be launched whenever the model actually cycles (e.g.
>   00/06/12/18Z), regardless of which sensors have observations at that
>   hour, and a launch at an "invalid" hour for some sensor can still
>   validate against it later — a sondes-verifying forecast launched at
>   06Z has no sonde to compare against at its own init time, but its +6h
>   lead validates at 12Z (a real sonde launch) and its +18h lead
>   validates at the next day's 00Z. So `process_forecast_experiment()`
>   calls `resolve_time_settings(config, diag_obs, for_forecast=True)`,
>   which ALWAYS uses the plain config-wide `time_settings` cadence for
>   the init-time list, ignoring `cycle_valid_hours` entirely — restricting
>   the init time the same way cycling restricts observation times would
>   silently drop every one of those still-valid lead comparisons. The
>   restriction instead applies per `(init_time, lead_hour)` pair, checked
>   against the **lead's valid time** (`init_time + lead_hour`), via
>   `lead_valid_time_allowed()`: a lead whose valid time doesn't land on
>   00Z/12Z is left as NaN in the diagnosis file (same convention as any
>   other kind of missing data) rather than getting a real statistic. The
>   graphics stage's forecast collectors also pass `for_forecast=True` so
>   their own init-cycle list matches exactly what the statistics stage
>   wrote files for; they need no separate lead-valid-time filtering of
>   their own, since the NaNs already propagate correctly through the
>   existing NaN-aware aggregation.
>
> `profiler` shares the same obsout file as sondes (`file_alias: sondes`)
> but has no `cycle_valid_hours` of its own, so it keeps following the
> config-wide cadence in both modes, and its OMF leads are never skipped —
> the 00Z/12Z restriction is specific to the radiosonde launch schedule,
> not a property of the shared file.

> **A missing HofX value is excluded from the diagnosis, not just a
> failed/missing QC flag.** This matters most for OMF (`hofx_forecast`
> experiments, `process_forecast_experiment()`): those files are plain
> HofX runs with no DA quality control applied, so there is often no
> `EffectiveQC0` group at all to catch anything, and a HofX value can
> easily be missing for some observations without being explicitly
> flagged -- most commonly, the observation falls outside a **REGIONAL
> MPAS-JEDI domain's extent** (see the `latlon_domain` callout below) and
> the forecast simply has nothing to evaluate there, so IODA writes the
> default netCDF fill value (`~9.9692e+36` for float32/float64) or `NaN`
> in its place. Left unfiltered, `ObsValue - HofX` at a missing HofX
> silently becomes a huge, meaningless "innovation" that dominates every
> RMS/mean it touches -- easy to miss since it produces a number, not a
> crash. `load_observation_data()` in `src/mpas_stat/obs/diag.py` now
> checks every HofX field actually used to compute an innovation
> (`_hofx_fill_mask()`, `_apply_hofx_qc()`: `NaN`, `+/-Inf`, or a value
> `>= 1e30` all count as missing) and folds that into the QC mask --
> initializing it from scratch when no `EffectiveQC`/`ObsError`-derived QC
> existed at all, or AND-ing it into whatever QC already applied
> otherwise. Ensemble HofX members (`spread`/`totSpread`) get the same
> per-member treatment before `nanstd`. This applies anywhere a HofX field
> feeds an innovation, not only OMF -- e.g. an `exp_type: hofx` OMB
> experiment, or the `hofx<outerloop>` fallback when an `oman` group is
> absent -- since it is the identical risk; precomputed `ombg`/`oman`
> innovations (already the DA system's own QC'd output) are untouched.
>
> **For that same HofX-derived path, `EffectiveQC` flag 19 counts as a
> pass, alongside 0.** Every other QC check in this module is a plain
> `qc == 0`, but a HofX-derived innovation (the same `hofx_used is not
> None` branch as above) treats flag values 0 AND 19 as passing.
> `_apply_hofx_qc()` remaps 19 -> 0 right where the HofX-missing exclusion
> already happens, so every downstream consumer (`calculate_stats()`,
> `compute_stats_2d_latlon()`, `compute_stats_2d_lat_vert()`) keeps its
> plain `qc == 0` check and doesn't need to know about this exception.
> Precomputed `ombg`/`oman` innovations are untouched here too -- a flag
> value of 19 there still means "rejected", exactly like every other
> nonzero code.

> **`plot_settings.latlon_domain` — 2-D maps for a REGIONAL MPAS-JEDI
> application.** By default (`latlon_domain: global`) the 2-D lat-lon
> gridded statistics (§2.4/§2.5) bin observations over the whole globe,
> mirroring the model-space package's own default. For a regional
> (limited-area) MPAS-JEDI application, most of that global grid is empty
> and wasted resolution — point `latlon_domain` at a tighter box instead,
> either:
>   - **`latlon_domain: auto`** (recommended) — let
>     `mpas_stat.obs.diag.auto_detect_latlon_bounds()` determine the box
>     itself, as the tightest `(lat_min, lat_max, lon_min, lon_max)`
>     containing every observation with a finite, QC-passed innovation —
>     i.e. exactly the locations that would otherwise earn a real
>     (non-NaN/Inf) statistic. It scans each configured cycle's obsout file
>     directly (only the sensor's first configured variable, since an
>     out-of-domain observation is invalid because the mesh doesn't cover
>     that location — a property of the interpolation, not of the physical
>     field — so one variable's valid/invalid footprint stands in for every
>     other's), for both OMB/OMA (`file_kind="cycling"`) and OMF
>     (`file_kind="forecast"`) separately. No hand-guessed box, and no
>     wasted grid cells: every cell the resulting `LatBins2D`/`LonBins2D`
>     covers has at least one real observation in it.
>   - **`latlon_domain: <domains key>`** — a fixed, hand-specified box, the
>     key of a `domains` entry covering the mesh's own extent (add one to
>     the `domains:` map if none of the built-in ones fit), the same way the
>     model-space package's `MPASConfig.regional` flag limits *its* lat/lon
>     binning box. Still useful when the mesh's extent is known exactly and
>     you'd rather not pay the extra file scan, or want a box that doesn't
>     depend on which observations happen to have been assimilated.
>
> `resolve_latlon_bounds()` in `src/mpas_stat/obs/common.py` resolves a `domains`-key
> box (it does *not* handle `"auto"` itself — no file access from there —
> and raises rather than silently falling back to full-globe bounds if
> called with it directly); `auto_detect_latlon_bounds()` in
> `src/mpas_stat/obs/diag.py` resolves `"auto"`; `_resolve_latlon_bounds_for()`
> picks between the two so `process_experiment()`/`process_forecast_experiment()`
> don't need to branch themselves. Either way, `src/mpas_stat/obs/diag.py`
> then builds `LatBins2D`/`LonBins2D` from the resulting box at
> `latlon_lat/lon_bin_width` resolution. This box is **fixed across every
> cycle** a config processes — including for `"auto"`, which scans every
> configured cycle up front before gridding any of them — because the
> graphics stage (`collect_obs_lat_lon_2d`) stacks each cycle's 2-D map into
> one `(n_cycles, nlat, nlon)` array and averages across cycles, which
> requires every cycle's grid to have the identical shape. Observations
> falling outside the resulting box are dropped (not folded into the
> nearest edge bin) by `compute_stats_2d_latlon`, so a regional box's
> boundary cells reflect only in-domain data. No changes are needed to
> `src/mpas_stat/graphics/functions.py` or `src/mpas_stat/obs/plot.py` — both
> already size and extent the map entirely from whatever `LatBins2D`/
> `LonBins2D` the diagnosis file contains.

> **Radiances' 1D latitude profile now shares the SAME domain box/bin width
> as the 2D products above, instead of a hardcoded global default.**
> `get_levels()` (`src/mpas_stat/obs/common.py`) previously always returned a fixed
> `np.arange(-80, 80+10, 10)` for `bin_method == LATITUDE` (radiances)
> regardless of `plot_settings.latlon_domain` — so a regional or
> `"auto"`-detected application's 2D lat-lon map would correctly grid only
> the mesh's actual latitude extent, while that SAME sensor's 1D
> `lat_bin_profile` still spanned the whole globe at a fixed 10° width,
> unrelated to `latlon_lat_bin_width` or the resolved domain box. Both
> `process_experiment()` and `process_forecast_experiment()`
> (`src/mpas_stat/obs/diag.py`) now resolve `lat_min_2d`/`lat_max_2d`/
> `lat_bin_width` (the SAME `_resolve_latlon_bounds_for()` call already used
> for the 2D lat-lon/lat-vert products) *before* computing `levels`, and for
> `bin_method == LATITUDE` pass that box straight into `get_levels()`'s new
> `domain_range=(lat_min_2d, lat_max_2d)` parameter (alongside
> `bin_width=lat_bin_width`) — unless the sensor has its own explicit
> `level_bin_kwargs` override (`src/mpas_stat/obs/sensor_config.py`), which always takes
> priority. Pressure/height sensors (`bin_method` `PRESSURE`/`HEIGHT`) are
> unaffected — they never take this branch.
>
> This also exposed (and fixes) a latent, previously-inert bug in
> `calculate_stats()`/`compute_stats_2d_lat_vert()`: the half-width used to
> decide which observations fall in which level bin (`bot_bin`/`top_bin` =
> `level -+ interval/2`) was a SECOND, independently-hardcoded constant
> (`10.0` for latitude, `2.0` for height) rather than being derived from
> `levels`' own spacing — so any sensor whose `levels` were built at a
> *different* width than that hardcoded constant (radiances' new
> domain-derived width above, or an existing `level_bin_kwargs` override
> such as `radar_vr*`/`radar_zh*`'s `bin_width: 0.5`, previously binned
> against a stale `interval: 2.0`) got bins that gapped or overlapped
> relative to `levels`' actual spacing, without any error — e.g. `radar_vr*`
> silently double- and triple-counted observations across overlapping 2 km
> bins spaced only 0.5 km apart. Both functions now derive `interval` from
> `levels[1] - levels[0]` (the actual spacing `get_levels()` built) whenever
> it isn't passed explicitly, which by construction always matches — see
> `calculate_stats()`'s docstring for the full explanation. Pressure-binned
> sensors are unaffected (they use per-level log-midpoint edges computed
> directly from `levels`, not a fixed `interval`, both before and after this
> fix).

> **2-D products (lat-lon map, lev-time, lev-lat) compare REAL experiments,
> not OMB vs. OMA.** Every other obs-space product (0D time series, 1D
> profiles) uses MPAS-STAT's "pseudo-experiment" trick (§3): OMB and OMA
> share one figure per (domain, statistic), drawn as solid/dashed lines for
> each real experiment. The three 2-D products instead put the REAL
> configured experiment on the panel axis, mirroring the model-space
> package's own `plot_lat_lon_2d`/`plot_time_level_2d`/`plot_lat_vert_2d`
> convention exactly: `experiments.names[0]` draws as the absolute/
> full-state reference, and every other configured experiment draws as a
> bootstrap `%` reduction vs. it (for whichever statistics are listed in
> `comparison_relative_reduction_statistics` -- default `[RMS, STD]`),
> with significant grid cells marked (bootstrap CI excludes zero, same
> `add_significance` convention used throughout). This is the standard
> "does experiment B verify better than experiment A, and where" spatial
> comparison, and needs at least two configured experiments to show
> anything beyond the reference panel -- with only one experiment
> configured, each figure is just that experiment's absolute panel, same
> as before.
>
> Because OMB and OMA can't both be on this real-experiment axis at the
> same time as the reference/reduction convention, each diagtype gets its
> own separate set of figures, under `<sensor_graphic_dir>/omb/` and
> `<sensor_graphic_dir>/oma/` respectively (e.g.
> `sondes/omb/lat_lon_2d/lat_lon_RMS.png` and
> `sondes/oma/lat_lon_2d/lat_lon_RMS.png` are two different figures, each
> comparing experiments spatially for OMB and OMA respectively). This is a
> new subfolder layout for these three products only -- `time_series_0d/`
> and `profiles_1d/` keep their original, un-nested paths.
>
> **lev-time** (`generate_time_level_2d`, default on) is a cycle x level
> Hovmoller, via `plot_time_level_2d`. It reuses the SAME per-cycle 1D
> profile groups (`vert_pres_profile_<domain>` / `vert_hgt_profile_<domain>`
> / `lat_bin_profile`) the 1D profile plots already read -- no new
> statistics-stage computation, purely a graphics-stage reshaping (kept
> cycle-resolved instead of time-averaged). One figure per (diagtype,
> domain, statistic); for radiances this becomes a latitude x time
> Hovmoller (their only profile is `lat_bin_profile`). Each column is a
> single cycle's value, so -- like the model-space package's own
> `plot_time_level_2d` -- there is no bootstrap significance here.
>
> **lev-lat** (`generate_lat_vert_2d`, default on) is a latitude x level
> cross-section, time-averaged across cycles with bootstrap CI, via
> `plot_lat_vert_2d` -- the obs-space counterpart of the model-space
> package's own `lat_lev`/`lat_hgt`/`lat_pres` 2-D grid. This IS a new
> statistics-stage product: `mpas_stat.obs.diag.compute_stats_2d_lat_vert()`
> bins observations jointly by vertical level (using the exact same
> per-level bin edges as the 1D profiles -- log-midpoint bins for pressure,
> `+/-interval/2` for height) and latitude (reusing the SAME
> `latlon_domain`-bounded latitude range as the lat-lon map, so no separate
> latitude-box config is needed), writing a new `2D/lat_vert/<stat>` group
> per variable (see §2.5). Domain-independent, like the lat-lon map.
> **Skipped entirely for radiances** (bin_method LATITUDE) -- their "level"
> axis already IS latitude, which would make this a degenerate lat-vs-lat
> plot.
>
> OMF (`hofx_forecast` experiments) has its own forecast-lead counterpart of
> every product on this page -- lat-lon map, lev-time, and lev-lat alike --
> generated at each configured forecast lead hour; see §2.4's forecast rows
> and the "Same OMF forecast-lead products" bullet in §3.

Per-sensor variables/units/panel-labels (e.g. sondes = U/V/T/Q, `amsua_n18` =
channels 5–9), bin method, and quirks (shared IODA files, custom vertical
bin ranges, non-standard cycle cadence) all live in `OBSERVATION_CONFIGS` in
`src/mpas_stat/obs/sensor_config.py`, not in the YAML — add a sensor there to support a
new obs type; nothing else in the pipeline needs to change. See that file's
`ObservationConfig` dataclass for the full list of fields:

| Field | Purpose |
|---|---|
| `variables`, `units`, `labels` | Physical variable names (or radiance channel numbers), for extraction and panel titles |
| `bin_method` | `PRESSURE` \| `HEIGHT` \| `LATITUDE` — which 1-D profile/domain-masking convention applies |
| `file_alias` | Read another sensor's IODA file instead of this sensor's own name (e.g. `profiler` reads the `sondes` obsout file) |
| `obs_type_filter` | `('lt'\|'eq', threshold)` — keep only rows whose `ObsType` satisfies this, when two sensors share one file |
| `level_bin_kwargs` | Override the default vertical/latitude bin levels (`bin_width`, `bin_bottom`, `bin_top`). Radiances (`LATITUDE`) skip this by default in favor of a domain-derived range — see the callout below — but an explicit override here always wins. |
| `cycle_interval_hours` | Override the config-wide `time_settings.interval` for sensors with their own cadence (e.g. sondes: 12-hourly regardless of the model DA cycle length) |
| `cycle_valid_hours` | Restrict this sensor's cycles to ONLY these UTC hours, e.g. `[0, 12]` for sondes (radiosondes only launch at 00Z/12Z) — the configured start time is snapped forward to the next listed hour, so the generated cycle list can never include any other hour regardless of what `time_settings.start_time` happens to be. See the callout below. |
| `cycle_start_trigger_hours`, `cycle_start_offset_hours` | Older, narrower alternative to `cycle_valid_hours`: shift the start time forward by a fixed offset, but only when the configured start hour is one of a specific listed set — kept for a sensor that wants a plain shift rather than "snap to nearest valid hour" |

### 2.4 Available statistics (observation-space)

| Statistic product | Function used to plot it | Notes |
|---|---|---|
| Domain-averaged time series | `plot_time_series_0d` | OMB/OMA only, one figure per (domain, statistic); OMB solid, OMA dashed |
| **RMS reduction (%) time series** | `plot_time_series_reduction_0d` | OMB/OMA only ("da"-typed experiments with an OMA). The SAME per-cycle RMS numbers as the time series above, plotted as a single derived `100*(RMS_omb-RMS_oma)/RMS_omb` line per experiment instead of two absolute ones — positive means the analysis improved on the background. One figure per domain, in its own `time_series_diff_0d/` folder (`time_series_RMS_reduction_pct_<region>.png`). No bootstrap significance (a 0-D time series has only the already-aggregated statistic per cycle, not per-observation samples) — see the function's own docstring. |
| Vertical/latitude binned profile | `plot_profiles_1d` | OMB/OMA only. Pressure bins (conventional), height bins (GNSS-RO/radar), or latitude bins (radiances) — whichever applies to that sensor |
| **RMS reduction (%) profile** | `plot_profile_reduction_1d` | OMB/OMA only. The profile counterpart of the RMS reduction time series above — `100*(RMS_omb-RMS_oma)/RMS_omb` vs. level, one line per experiment, in its own `profile_reduction_1d/` folder (`profile_<profile_key>_RMS_reduction_pct.png`). Also no bootstrap significance, for the same reason. |
| Lat-lon gridded map | `plot_lat_lon_2d` | OMB/OMA only, as two separate sets of figures. Computed over `plot_settings.latlon_domain`'s box (default: globally; not per `apply_domains`). Real-experiment comparison convention (reference + % reduction) — see the callout in §2.3 |
| Lev-time (cycle x level) Hovmoller | `plot_time_level_2d` | OMB/OMA only, as two separate sets of figures, one per (domain, statistic). Reuses the existing 1D profile data, cycle-resolved. Real-experiment comparison convention, no bootstrap (each column is one cycle) — see the callout in §2.3 |
| Lev-lat (latitude x level) cross-section | `plot_lat_vert_2d` | OMB/OMA only, as two separate sets of figures. Domain-independent, like the lat-lon map; skipped for radiances. Real-experiment comparison convention with bootstrap significance — see the callout in §2.3 |
| **Forecast skill** (statistic vs. forecast lead time) | `plot_forecast_lead_0d` | **OMF only** (`hofx_forecast` experiments). Domain-averaged, bootstrap-aggregated over initialization cycles at each lead hour — the exact function the model-space package uses for its own extended-forecast skill plots. One figure per (domain, statistic); non-reference experiments are bootstrap-tested against `experiments.names[0]` and significant lead times get a hollow-circle marker, same convention as `plot_profile_diff_1d`. |
| **Forecast profile** (vertical/latitude, one figure per lead hour) | `plot_profiles_1d` (reused unmodified) | **OMF only.** The forecast-lead counterpart of the vertical/latitude profile above, mirroring the model-space package's own `forecast_profiles_1d` plot type: one profile figure per (domain, forecast lead hour, statistic), bootstrap-aggregated over initialization cycles at that lead. Lands in the SAME `profiles_1d/` folder as the OMB/OMA profiles, distinguished by an `_f<HHH>` filename suffix (e.g. `profile_vert_pres_profile_global_RMS_f006.png`) — see `generate_forecast_profile_graphics`. |
| **Forecast lat-lon map** (one figure per lead hour) | `plot_lat_lon_2d` (reused unmodified) | **OMF only.** Mirrors the model-space package's own `forecast_lat_lon_2d` plot type: real OMF experiments on the panel axis with the same reference/bootstrap-%-reduction comparison convention as the OMB/OMA map above, one set of figures per lead hour (`_f<HHH>` filename suffix, same `lat_lon_2d/` folder) — see `generate_forecast_lat_lon_graphics`. |
| **Forecast lev-lat cross-section** (one figure per lead hour) | `plot_lat_vert_2d` (reused unmodified) | **OMF only.** Mirrors the model-space package's own `forecast_lat_vert_2d` plot type; same real-experiment comparison convention and `_f<HHH>` filename suffix as the forecast lat-lon map above. Skipped for radiances, like the OMB/OMA lev-lat cross-section — see `generate_forecast_lat_vert_graphics`. |
| **Forecast lev-time (lead x level) Hovmoller** | `plot_forecast_time_level_2d` | **OMF only.** The genuinely different one: forecast lead hour replaces cycle on the x-axis, and — unlike the plain-mean OMB/OMA lev-time Hovmoller — each lead-hour column already averages over every initialization cycle, so bootstrap significance vs. `experiments.names[0]` IS drawn here, via the shared `_draw_2d_panels` machinery. One figure per (domain, statistic), in its own `forecast_time_level_2d/` folder. Mirrors the model-space package's own `forecast_time_level_2d` plot type exactly — see `generate_forecast_time_level_graphics`. |

3-D binned products and scorecards (available on the model side) are
intentionally **not** implemented for obs-space.

### 2.5 `obsstat_<obs>.<date>.nc` schema

One file per experiment × cycle × sensor (OMB/OMA), or per experiment ×
init cycle × forecast lead hour × sensor (OMF, one directory level deeper
— see `obs_diag_forecast_file()`):

```
<top level>
  Levels               (nLevels; units = hPa / km / degrees_north depending on the sensor's bin method)
  LatBins2D, LonBins2D  (2-D map grid coordinates)

  omb/                              # background stats -- OMB/OMA files only, always present
    <variable>/                     # e.g. windNorthward, airTemperature, channel_5, ...
      0D/<domain>/                  # e.g. global, NXTro, ...
          RMS, Bias, STD, Count, Spread, TotalSpread     (scalars)
      1D/
          vert_pres_profile_<domain>/    RMS, Bias, STD, Count, Spread, TotalSpread   (vs. Levels, conventional obs)
          vert_hgt_profile_<domain>/     RMS, Bias, STD, Count, Spread, TotalSpread   (vs. Levels, GNSS-RO/radar)
          lat_bin_profile/               RMS, Bias, STD, Count, Spread, TotalSpread   (vs. Levels, radiances -- region-independent)
      2D/
          lat_lon/                       RMS, Bias, STD, Count                        (vs. LatBins2D x LonBins2D)
          lat_vert/                      RMS, Bias, STD, Count                        (vs. Levels x LatBins2D;
                                                                                         omitted for radiances)
  oma/                              # analysis stats -- OMB/OMA files only, present when exp type is "da" and plot_oma: true
      ... identical layout to omb/ ...
  omf/                              # OMF (obs-minus-forecast) stats -- forecast files only (hofx_forecast experiments)
      ... identical layout to omb/ ...
```

`Spread`/`TotalSpread` only appear when `n_member > 1` for that experiment.
Radiance channels are stored as `channel_<N>` groups (e.g. `channel_5`). A
file never has more than one of `omb+oma` vs. `omf` — which one depends on
whether it came from `write_obs_diag_files()` (OMB/OMA) or
`write_obs_forecast_diag_files()` (OMF). `omf/`'s layout is now literally
identical to `omb/`'s, `2D/lat_vert` included (`process_forecast_experiment()`
computes it exactly like `process_experiment()` does for OMB/OMA, gated by
the same `generate_lat_vert_2d` flag and skipped for radiances) — this is
what lets `generate_forecast_lat_vert_graphics` reuse `plot_lat_vert_2d`
unmodified (§2.4).

### 2.6 Radar composite-reflectivity verification (separate opt-in product)

Radar reflectivity is verified fundamentally differently from every other
sensor in this package: instead of per-observation binned statistics
(RMS/Bias/STD vs. pressure/height/latitude/lat-lon), scattered obs and HofX
reflectivity are first **gridded** onto a regular lat/lon "composite
reflectivity" field (max value within a search radius of each grid-cell
center), then verified with neighborhood/categorical scores — Fractions
Skill Score (FSS), Threat Score (TS), Equitable Threat Score (ETS),
Frequency Bias (FBIAS), Probability of Detection (POD), False Alarm Ratio
(FAR) — plus RMSE of the gridded field itself. This does not fit the
`get_levels()`/`bin_method` framework the rest of §2 uses at all, so it
gets its own pair of scripts (`src/mpas_stat/radar/diag.py` /
`src/mpas_stat/radar/plot.py`) and its own shared module
(`src/mpas_stat/radar/common.py`) rather than being folded into
`src/mpas_stat/obs/diag.py`/`src/mpas_stat/obs/plot.py`.

**The FSS/categorical scores are computed by MET's `grid_stat` (via
METplus), not by a native Python kernel.** Gridding (scattered obs/HofX →
regular lat/lon composite reflectivity) and RMSE are still computed
natively, exactly as before; see "METplus / `grid_stat` integration"
below for the full architecture, required external dependencies, and
important caveats — read it before turning this product on. (This
replacement applies ONLY to this integrated, config-driven product. The
original standalone `legacy/radar_reflectivity_verify.py` script this product
was lifted from is untouched and keeps its own independent, fully-native
FSS/categorical kernels — use it if you want radar verification with no
MET/METplus dependency at all.)

It is **OMF-only** — only `hofx_forecast`-typed experiments are processed
(same `<hofx_dir>/<exp>/HofX/<init_cycle>/f<HHH>/dbOut/obsout_hofx_<obs>.h5`
file convention as the rest of §2's OMF product, via
`build_hofx_forecast_filename()`) — there is no cycling/OMB-OMA
counterpart. It is entirely **opt-in**: both stages no-op immediately if
the config has no top-level `radar:` section (`resolve_radar_settings()`
returns `None`), so adding it never affects a config that doesn't use it.

```bash
# Statistics: for every radar.sensors entry and every hofx_forecast
# experiment, grids obs/hofx composite reflectivity and writes ONE
# radarstat_<obs>.<date>.nc file per (experiment, init cycle, lead)
python -m mpas_stat.cli.radar_stats my_radar_config.yaml

# Graphics: reads those files back, aggregates across init cycles, draws
# FSS/TS/ETS/FBIAS/RMSE vs. lead time and a Performance Diagram
python -m mpas_stat.cli.radar_graphics my_radar_config.yaml
```

Config (a new top-level `radar:` block; everything else — `paths`,
`experiments`, `time_settings`, `forecast`, `n_workers` — is the SAME
config section the rest of §2 already uses, so a radar sensor can share one
config file with the point-wise pipeline):

```yaml
radar:
  sensors: ["radar_zh"]              # obs-type name(s); resolved to an HDF5
                                      # filename via the SAME src/mpas_stat/obs/sensor_config.py
                                      # registry the point-wise pipeline uses
                                      # (e.g. the "radar_zh*" wildcard entry) --
                                      # but see the filename-collision note below
  diag_var: equivalentReflectivityFactor   # HDF5 ObsValue/hofx group variable
  thresholds: [20, 40]               # dBZ thresholds for FSS/TS/ETS/FBIAS/POD/FAR
  radii: [3, 5, 9, 15]               # neighborhood radii, in GRID CELLS of the
                                      # sample_bin-spaced composite-reflectivity grid
  sample_bin: 0.027                  # composite-reflectivity gridding radius, degrees
  n_workers: null                    # file-processing pool for the (expensive)
                                      # statistics stage; null -> falls back to the
                                      # top-level n_workers
  aggregated_score: false            # graphics-stage aggregation convention -- see below

  # --- METplus / grid_stat settings (see the subsection below) -----------
  nbrhd_shape: SQUARE                 # MET neighborhood shape (grid_stat default)
  nbrhd_cov_thresh: ">=0.5"           # MET neighborhood coverage threshold (grid_stat default)
  run_metplus_cmd: run_metplus.py     # assumed on PATH; override with a full path,
                                      # or a list like [conda, run, -n, metplus, run_metplus.py]
  metplus_work_dir: null              # scratch dir for conf/log/npz-cache/MET-output files;
                                      # null -> <diag_dir>/metplus_scratch
  metplus_timeout_sec: 600            # per-cycle run_metplus.py subprocess timeout
  keep_metplus_scratch: false         # keep each cycle's .npz cache after a successful run
                                      # (for debugging; conf/log/MET ASCII outputs are always kept)
  metplus_model_name: null            # MET MODEL field; null -> the experiment name
  metplus_obtype: null                # MET OBTYPE field; null -> radar.sensors entry, upper-cased
```

**No filename collision, even for the same sensor name.** A radar sensor
may be listed in BOTH the top-level `sensors:` list (point-wise pipeline;
e.g. `radar_zh*`'s existing HEIGHT-binned products) and `radar.sensors`
(this product) for the identical `diag_obs` name — the two products' files
never collide because this one uses a distinct `radarstat_` filename
prefix instead of `obsstat_`, in the same
`<diag_base_dir>/<exp>/VerifyObs/<init_cycle>/f<HHH>/` directory (see
`radar_composite_diag_file()` in `src/mpas_stat/obs/common.py` — the path-convention
function lives there, alongside `obs_diag_forecast_file()`, for
consistency with every other path helper in the package, even though the
rest of this product's code lives in its own `src/mpas_stat/radar/common.py`).

**Both aggregation conventions are always available, from one statistics
run.** Every diagnosis file `src/mpas_stat/radar/diag.py` writes carries
BOTH the per-cycle derived scores (`FSS`/`TS`/`ETS`/`FBIAS`/`POD`/`FAR`)
AND the raw accumulator components they're built from
(`FSS_num`/`FSS_denom`/`NA`/`NB`/`NC`/`ND`) unconditionally. `radar.
aggregated_score` only controls how `src/mpas_stat/radar/plot.py`
*aggregates across init cycles* at the graphics stage:
  - `false` (default): bootstrap CI over each cycle's own derived score
    (matches every other bootstrap-CI product in this package). POD/FAR are
    plain cycle-means (never bootstrapped).
  - `true`: sum each cycle's raw counts/FSS numerator+denominator across
    cycles FIRST, then compute one score from the totals (no CI) — the
    conventional way to combine FSS/categorical scores over a verification
    period without giving equal weight to a cycle that happened to have
    very little reflectivity coverage.

Changing `aggregated_score` never requires re-running the statistics stage
— only `src/mpas_stat/radar/plot.py` reads it.

**Figures** (mirrors the standalone `legacy/radar_reflectivity_verify.py` script
this product was integrated from), written to
`<graphic_dir>/<sensor>/radar_reflectivity/`:

| Figure | Filename |
|---|---|
| FSS / TS / ETS / FBIAS vs. forecast lead time (one panel per threshold, one figure per radius) | `refl_fcst_<metric>_radius<r>.<fmt>` |
| Performance Diagram (POD vs. 1-FAR, one panel per threshold, one figure per radius, points labeled by lead) | `refl_fcst_performance_diagram_radius<r>.<fmt>` |
| RMSE vs. forecast lead time | `refl_fcst_rmse.<fmt>` |

**Known limitation:** unlike every other product in §2, these figures are
NOT drawn with `src/mpas_stat/graphics/functions.py` (`apply_pub_style`,
`auto_layout`, etc.) — `src/mpas_stat/radar/plot.py` carries its own small
`plot_time_series`/`plot_performance_diagram` functions (lifted unchanged
from the standalone script), so radar figures won't visually match the
publication styling of the rest of the package. Folding them onto the
shared styling helpers is a reasonable follow-on, not done here.

#### METplus / `grid_stat` integration

Per-cycle FSS and categorical (TS/ETS/FBIAS/POD/FAR) neighborhood scores
are computed by MET's `grid_stat` tool, invoked through the METplus
`GridStat` wrapper (`run_metplus.py`) — **not** by a native Python kernel.
Composite-reflectivity gridding and RMSE are still fully native Python,
unchanged.

**Requires MET and METplus already installed** on the system that runs
`src/mpas_stat/radar/diag.py`, with `run_metplus.py` on `PATH` (or
`radar.run_metplus_cmd` pointed at it). This package does not install or
configure MET/METplus itself.

**How one cycle is processed**, for every (experiment, init cycle, lead):
  1. Obs/HofX composite reflectivity is gridded natively (unchanged), and
     RMSE is computed natively (unchanged — RMSE is NOT delegated to MET's
     `CNT` line type).
  2. The gridded obs/forecast fields are written to a small scratch
     `.npz` cache file under `radar.metplus_work_dir` (default
     `<diag_dir>/metplus_scratch/<exp>_<obs>_<init>_f<lead>/data.npz`).
  3. A per-cycle METplus `GridStat` conf is generated, scoped to exactly
     that one (init, lead) pair, requesting EVERY configured threshold
     and EVERY neighborhood width (`radii` converted to MET's own
     `width = 2*radius + 1` convention) in one `grid_stat` call. The
     forecast/obs inputs are MET **Python-embedding** script invocations
     (`src/mpas_stat/radar/metplus_embed.py <cache_path> FCST|OBS`) that just load the
     cache file back — this deliberately avoids depending on METplus's
     own filename-into-script-argument templating behavior.
  4. `run_metplus.py` is invoked on that conf as a subprocess, inside the
     same `ProcessPoolExecutor` that already parallelizes across cycles.
  5. `grid_stat` is configured to write BOTH the combined ASCII `.stat`
     file (kept per cycle for your own independent inspection/
     `stat_analysis` use — its path is recorded as the diagnosis file's
     `metplus_stat_file` global attribute) AND dedicated, self-describing
     per-line-type `NBRCTC`/`NBRCTS`/`NBRCNT` `.txt` files
     (`GRID_STAT_OUTPUT_FLAG_*=BOTH`). Only the dedicated `.txt` files are
     parsed — by column NAME, matched back to a (threshold, radius) by
     each row's own `FCST_THRESH`/`INTERP_PNTS` values — never by
     position in the combined `.stat` file (which is not self-describing
     past its common columns).
  6. `NBRCTC` counts (`FY_OY`/`FY_ON`/`FN_OY`/`FN_ON`) and `NBRCTS` scores
     (`CSI`/`GSS`/`FBIAS`/`PODY`/`FAR`) map directly onto this product's
     existing `na`/`nb`/`nc`/`nd` and `ts`/`ets`/`fbias`/`pod`/`far`.
     `NBRCNT`'s `FBS`/`FSS`/`TOTAL` are used directly for the per-cycle
     `FSS`, and algebraically back-derived into the SAME
     `fss_num`/`fss_denom` accumulator components this product has always
     summed across cycles for `aggregated_score: true` (see below).

**Cross-cycle aggregation is UNCHANGED** — `src/mpas_stat/radar/plot.py`'s
aggregation (bootstrap CI, or sum-then-ratio) still happens entirely in
this package's own Python code, exactly as before METplus was
introduced. This was a deliberate choice, not an oversight: MET's
`stat_analysis -job aggregate_stat` has a documented `NBRCTC`→`NBRCTS`
conversion for combining categorical counts across cases, but **no**
documented `aggregate_stat` conversion for `NBRCNT` (FSS) — it is only
listed under the separate, generic `-job aggregate` job, whose exact
combination semantics for `NBRCNT` could not be confirmed with
confidence against the MET User's Guide. Rather than depend on that, this
package keeps pooling FSS itself from `fss_num`/`fss_denom` (sum first,
then one ratio), just fed by MET's own per-cycle `FBS`/`FSS`/`TOTAL`
instead of a native kernel's.

**Caveats — please read before trusting results:**
  - **Not tested against a real MET/METplus install.** This integration
    was developed and unit-tested in a sandbox with no MET/METplus
    available — only against fabricated, MET-shaped stand-ins for its
    ASCII output (conf-text generation, ASCII-table parsing, the FBS/FSS
    back-derivation algebra, and `src/mpas_stat/radar/metplus_embed.py`'s `met_data`/
    `attrs` construction were all exercised directly; the actual
    `run_metplus.py`/`grid_stat` invocation was not). **Run a single
    cycle on your own system first** and check its
    `metplus_scratch/.../logs/run_metplus.log` and output files before
    trusting a full run.
  - `src/mpas_stat/radar/metplus_embed.py` relies on MET's Python-embedding support
    treating a non-finite (NaN) `met_data` value as missing/bad data —
    documented behavior, but not exercised against a real MET build here
    either; see that script's own docstring.
  - RMSE is still computed natively (not via MET's `CNT` line type) — by
    design, see above, not an oversight.
  - A cycle whose FSS is numerically indistinguishable from 1.0 (a
    perfect match for that one cycle) cannot be exactly back-divided into
    the `fss_num`/`fss_denom` this package pools across cycles (it's a
    0/0) and is treated as contributing 0 to both — a very slight
    understatement of that cycle's weight, expected to be rare for a
    continuous field like reflectivity. See `_parse_nbrcnt()`'s docstring
    in `src/mpas_stat/radar/common.py`.

---

## 3. Shared conventions & how the two sides connect

The observation-space package was built to *reuse* the model-space package
rather than duplicate it:

- **Same diagnosis-file shape.** Both `mpasstat.<date>.nc` and
  `obsstat_<obs>.<date>.nc` use the identical nested-group convention:
  `<variable>/<0D|1D|2D>/<key>/<statistic>`. Anyone who knows how to read
  one file family with `ncdump`/Panoply immediately recognizes the other.
- **Same binning kernel.** The obs-space 2-D lat-lon gridding calls
  `grouped_stats_1d` straight out of `src/mpas_stat/model/kernels.py` — the
  exact routine the model side uses for its own `2D/lat_lon` product.
- **Same plotting functions.** `src/mpas_stat/obs/plot.py` imports
  `plot_time_series_0d`, `plot_profiles_1d`, `plot_lat_lon_2d`,
  `plot_time_level_2d`, `plot_lat_vert_2d`, `plot_forecast_lead_0d`, and
  `plot_forecast_time_level_2d` directly from `src/mpas_stat/graphics/functions.py`
  — obs-space contributes no plotting logic of its own. Figure styling, colorbars, panel layout,
  and filenames all come from the same code path as the model-space
  figures. One small bug fix WAS made to this shared file while wiring up
  the lat-lon map's real-experiment comparison convention (see the callout
  box in §2.3): `_draw_2d_panels`'s non-Cartopy fallback path (used when
  the `cartopy` package isn't installed) hardcoded `transpose=True` in its
  call to `add_significance`, which is correct for `plot_time_level_2d` /
  `plot_lat_vert_2d` (`transpose_data=True`) but wrong for `plot_lat_lon_2d`
  (`transpose_data=False`) — a shape mismatch that silently dropped
  significance markers (a printed warning, not a crash). Fixed by passing
  `transpose=transpose_data` instead of a hardcoded `True`; behavior is
  unchanged for every existing caller (all already had `transpose_data`
  match the hardcoded value) and for any environment with `cartopy`
  installed (that code path isn't taken). This is the first time obs-space
  exercises `plot_lat_lon_2d`'s relative-reduction/significance path at
  all, which is how the bug surfaced.
- **Same 2-D comparison convention.** The lat-lon/lev-time/lev-lat
  real-experiment comparison figures (§2.3) work by feeding
  `src/mpas_stat/graphics/functions.py` a `cfg["experiments"]` built from the
  REAL configured experiments (not OMB/OMA pseudo-experiments) and setting
  `cfg["relative_reduction_statistics"]` — the exact same mechanism
  (`is_relative_stat`, `_compute_panel_data`, `_draw_2d_panels`) the
  model-space package uses for its own cross-experiment 2-D comparisons.
  Obs-space did not need to add any new comparison/bootstrap logic of its
  own for this.
- **Same forecast-lead machinery.** The OMF "forecast skill" reader
  (`collect_obs_forecast_lead_0d`) calls `mpas_stat.graphics.io`'s own
  `generate_forecast_leads` and `bootstrap_mean_ci` — the same lead-hour
  list and bootstrap-CI aggregation the model side's
  `collect_forecast_lead_0d` uses for its `ExtendedFC` files — so `config`'s
  `forecast:` block means exactly the same thing on both sides.
- **Same OMF forecast-lead products, one level deeper.** The four
  `generate_forecast_*_graphics` functions (§2.4: forecast profile,
  lat-lon map, lev-lat cross-section, lev-time Hovmoller) are the obs-space
  counterparts of the model-space package's own `forecast_profiles_1d` /
  `forecast_lat_lon_2d` / `forecast_lat_vert_2d` / `forecast_time_level_2d`
  plot types (`src/mpas_stat/graphics/driver.py`), built the same way: three of
  the four (profile, lat-lon, lat-vert) just call their non-forecast
  `plot_*` function once per lead hour with a `_lead_suffix`-carrying cfg
  (`_lead_cfg`, ported from `mpas_stat.graphics.driver._lead_cfg` verbatim),
  landing in the SAME output folder as the non-forecast figures; the fourth
  (lev-time) is genuinely different, since averaging every initialization
  cycle into each lead-hour column makes bootstrap significance available
  where the plain per-cycle OMB/OMA Hovmoller has none — so it gets its own
  `plot_forecast_time_level_2d` call and output folder, exactly mirroring
  the model-space package's own `forecast_time_level_2d`/`time_level_2d`
  split.
- **Parallel plotting, at sensor granularity.** `src/mpas_stat/obs/plot.py`'s
  `_process_sensor()` collects (reads that sensor's diagnosis netCDF files)
  AND renders every figure for ONE sensor, and `main()` runs one
  `_process_sensor()` call per sensor — sequentially by default, or spread
  across `config["n_workers"]` worker processes via `ProcessPoolExecutor`
  (the same top-level `n_workers` config key the model-space package uses in
  `run_graphics()`, applied at sensor rather than figure granularity here).
  Each `generate_*_graphics` function still collects its data (netCDF reads)
  and returns a flat list of `(plot_func, args, kwargs)` jobs instead of
  calling `plot_*` directly (run by `_process_sensor()`'s ported
  `_run_job()` helper), but that collection happens per-sensor inside
  whichever process is handling that sensor, not up front for every sensor
  in the main process — so with `n_workers > 1`, different sensors'
  statistics are never all resident in memory at once, and one sensor's
  figures can start rendering before another sensor's diagnosis files have
  even been read. Sequential and parallel runs produce byte-identical
  figures (verified: `n_workers: 1` vs. `n_workers: 4` over the same
  2-sensor synthetic configs produced 84/84 and 132/132 pixel-identical
  PNGs, with wall time roughly halving to match the 2-worker/2-sensor
  split).
- **The "pseudo-experiment" trick.** `src/mpas_stat/graphics/functions.py`'s
  plotting functions draw one line/column per "experiment" — they have no
  built-in idea of "background vs. analysis". To reuse them unmodified,
  `src/mpas_stat/obs/plot.py` exposes each (real experiment, OMB|OMA) pair
  as its own pseudo-experiment — `"<label> - OMB"` (solid) and
  `"<label> - OMA"` (dashed), sharing the real experiment's colour — so the
  familiar OMB-solid/OMA-dashed convention falls out of the shared plotting
  code for free. OMF needs no such trick (`hofx_forecast` experiments map
  one-to-one onto `plot_forecast_lead_0d`'s notion of "experiment"), so
  `_build_forecast_experiments()` skips the doubling entirely.
- **Stat-name convention.** Obs-space stats are labelled `RMS`, `Bias`,
  `STD`, `Count`, `Spread`, `TotalSpread` to read consistently alongside the
  model side's `rms`, `mean`, `std`, `spread`.
- **One module for sensor metadata.** All per-sensor knowledge (variables,
  units, labels, bin method, and quirks like shared IODA files or
  non-standard cycle cadence) lives only in `src/mpas_stat/obs/sensor_config.py`. Neither
  pipeline stage hard-codes a sensor name anywhere else — adding a new obs
  type never requires touching `src/mpas_stat/obs/diag.py`,
  `src/mpas_stat/obs/plot.py`, or `src/mpas_stat/obs/common.py`.

### Known limitations (inherited from the shared graphic functions)

- Axis/legend labels show only the **statistic name**, not physical units —
  true on both the model and obs sides, since `plot_time_series_0d` /
  `plot_profiles_1d` don't carry a units argument.
- `relative_reduction_statistics` is disabled for the 0-D/1-D OMB/OMA
  products, which draw OMB and OMA as two absolute-valued lines/panels on
  the same figure. The 2-D comparison products (lat-lon map, lev-lat,
  lev-time) instead use `comparison_relative_reduction_statistics` and draw
  non-reference experiments as a bootstrap percent reduction vs. the first
  experiment. All percent-reduction/difference figures use the same sign
  convention: **positive = improvement, negative = degradation**.
- Obs-space profiles (1D) and the lev-time Hovmoller are plain time
  averages / cycle-resolved values — no bootstrap confidence-interval
  shading (the model side's `profile_diff_1d` adds that; obs-space
  intentionally does not use it there). The lat-lon map and lev-lat
  cross-section DO use bootstrap CI/significance, via the real-experiment
  comparison convention (§2.3).
- OMF (`hofx_forecast` experiments) now has its own forecast-lead
  counterpart of every OMB/OMA product — profile, lat-lon map, lev-lat
  cross-section, and lev-time Hovmoller — in addition to the domain-averaged
  `forecast_lead_0d` "skill vs. lead time" product (§2.4). Nothing is
  missing from obs-space's own OMB/OMA/OMF figure set any more; only the
  model-space-only products noted above (3-D binned fields, scorecards)
  remain unimplemented for obs-space.

---

## 4. Directory layout at a glance

```
MPAS-STAT/
├── README.md, LICENSE, pyproject.toml, MANIFEST.in
├── requirements.txt, environment.yml
├── docs/USER_GUIDE.md                        # this guide
├── examples/                                 # annotated example obs-space configs
├── tests/test_smoke.py                       # focused unit tests
├── configs/
│   ├── model_input.yaml                      # model-space statistics config
│   └── model_graphic.yaml                    # model-space graphics config
├── scripts/
│   ├── run_verify_model.ksh                  # model-space PBS job script
│   ├── run_verify_model_fcst.ksh             # extended-forecast PBS job script
│   ├── run_verify_obs.ksh                    # OMB/OMA PBS job script
│   └── run_verify_obs_fcst.ksh               # OMF PBS job script
│
├── src/mpas_stat/
│   ├── model/                                # model-space statistics stage
│   │   ├── args.py, config.py, driver.py
│   │   ├── interp.py, kernels.py, variables.py
│   ├── graphics/                             # shared graphics engine
│   │   ├── driver.py, functions.py, io.py
│   ├── obs/                                  # obs-space verification
│   │   ├── sensor_config.py                  # single sensor metadata registry
│   │   ├── common.py                         # config/path conventions
│   │   ├── diag.py                           # statistics stage
│   │   └── plot.py                           # graphics stage
│   ├── radar/                                # optional radar product
│   │   ├── common.py, diag.py, plot.py
│   │   └── metplus_embed.py                  # MET Python-embedding helper
│   └── cli/                                  # console/module entry points
│       ├── model_stats.py, model_graphics.py
│       ├── obs_stats.py, obs_graphics.py
│       └── radar_stats.py, radar_graphics.py
│
├── legacy/radar_reflectivity_verify.py       # original standalone radar script
│
└── <exp_name>/VerifyModel/<cycle>/mpasstat.<date>.nc              # model-space diagnosis files
    <diag_dir>/<exp_name>/VerifyObs/<cycle>/obsstat_<sensor>.<date>.nc   # obs-space OMB/OMA diagnosis files
    <diag_dir>/<exp_name>/VerifyObs/<init_cycle>/f<HHH>/obsstat_<sensor>.<date>.nc   # obs-space OMF diagnosis files
    <diag_dir>/<exp_name>/VerifyObs/<init_cycle>/f<HHH>/radarstat_<sensor>.<date>.nc # radar composite-reflectivity diagnosis files
```

The experiment sits at the top of the obs-space tree too, exactly mirroring
the model-space `<exp_name>/VerifyModel/...` convention (`obs_diag_file()` /
`obs_diag_forecast_file()` in `src/mpas_stat/obs/common.py`) -- there is no separate
`<sensor>` directory level, since the filename itself
(`obsstat_<sensor>.<date>.nc`) already identifies the sensor. Every sensor's
files for one (experiment, cycle) pair land side by side in the same
`VerifyObs/<cycle>/` directory, distinguished only by filename, and every
diagnosis file for one experiment lives under a single
`<diag_dir>/<exp_name>/` subtree, which is what lets that whole experiment's
obs-space output be archived, rsynced, or deleted as one unit.

> **Note for anyone with an existing `diag_dir` from before this
> convention.** Diagnosis files have gone through two now-obsolete older
> layouts: `<sensor>` first
> (`<diag_dir>/<sensor>/<exp_name>/VerifyObs/...`), and later `<sensor>` as
> its own directory level under each experiment
> (`<diag_dir>/<exp_name>/VerifyObs/<sensor>/<cycle>/...`). None of these
> three layouts are compatible with each other on disk
> (`src/mpas_stat/obs/plot.py` looks for files under the current layout
> only). Either re-run `src/mpas_stat/obs/diag.py` to regenerate files under
> the current layout (cheap -- it's a read-IODA/write-netCDF pass, not a DA
> re-run), or move the existing files by hand.
