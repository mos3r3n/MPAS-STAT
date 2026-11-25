# 2. Observation-space verification

Compares IODA obs-space background/analysis equivalents (OMB/OMA) against
observations, at the scattered observation locations, for conventional and
radiance obs types. **Built to plug directly into the model-space graphics
engine above** — see §3.

## 2.1 Files

| File | Role |
|---|---|
| `src/mpas_stat/obs/sensor_config.py` | **Single source of truth for sensor/observation metadata.** The `OBSERVATION_CONFIGS` registry: per-sensor variables, units, panel labels, bin method, and quirks (`file_alias`, `obs_type_filter`, `level_bin_kwargs`, cycle-cadence overrides). Zero dependencies — importable standalone from either stage, a notebook, or a unit test. Add/adjust a sensor here only. |
| `src/mpas_stat/obs/common.py` | Shared layer: config schema, bin-level helpers, netCDF group naming, and the file-path convention. Imports `obs/sensor_config.py` for sensor lookups. Imported by both stages below so they can never disagree on where files live. |
| `src/mpas_stat/obs/diag.py` | **Statistics stage** entry point (facade). Reads IODA obs-space HDF5, computes OMB/OMA/OMF stats, writes diagnosis netCDF files. No `matplotlib` import. Implemented in `obs/reader.py` (file names, `load_observation_data`), `obs/binning.py` (0D/1D/2D statistics kernels), `obs/bounds.py` (`latlon_domain: auto`), `obs/process.py` (`process_experiment`, `process_forecast_experiment`, `main`) and `obs/writer.py` (diagnosis-file writers). |
| `src/mpas_stat/obs/plot.py` | **Graphics stage** entry point (facade). Reads the diagnosis netCDF files back and draws figures by calling `mpas_stat.graphics.functions` directly. No `h5py` import. Implemented in `obs/collect.py` (`collect_obs_*` readers), `obs/experiments.py` (experiment lists/colours/graphic cfg) and `obs/figures.py` (`generate_*_graphics`, `main`). |
| `scripts/run_verify_obs.ksh` | PBS job script for **OMB/OMA** (cycling DA) statistics: one experiment, one `obs_stats` run per cycle from `configs/obs_input.yaml`, output in `<main_dir>/<exp>/VerifyOBS/<cycle>/` (like `run_verify_model.ksh`). |
| `scripts/run_verify_obs_fcst.ksh` | PBS job script for **OMF** (extended-forecast skill) statistics: per initialization cycle and forecast hour, output in `<main_dir>/<exp>/VerifyOBS/<init>/f<HHH>/` (like `run_verify_model_fcst.ksh`). |
| `configs/obs_input.yaml`, `configs/obs_graphic.yaml` | Templates for the per-cycle statistics runs and for the graphics stage. |
| `src/mpas_stat/radar/common.py` | **Radar composite-reflectivity product** (§2.6, separate opt-in add-on): shared config schema (`radar:` block), composite-reflectivity gridding, RMSE, bootstrap/aggregation, and the METplus/`grid_stat` integration that computes per-cycle FSS/categorical neighborhood scores — this pipeline's counterpart of `obs/common.py`, imported by both stages below. No plotting, no HDF5-obs-reading orchestration of its own. |
| `src/mpas_stat/radar/metplus_embed.py` | MET **Python-embedding** script `grid_stat` (via METplus) calls to read back the gridded obs/HofX fields `radar/common.py` writes to a scratch `.npz` cache — see §2.6's METplus subsection. Not meant to be run by hand. |
| `src/mpas_stat/radar/diag.py` | Radar product **statistics stage.** Reads HofX obsout HDF5 for `hofx_forecast` experiments, grids obs/hofx composite reflectivity, writes `radarstat_<obs>.<date>.nc` diagnosis files. |
| `src/mpas_stat/radar/plot.py` | Radar product **graphics stage.** Reads those diagnosis files back, aggregates across init cycles, draws FSS/TS/ETS/FBIAS/RMSE-vs-lead-time and Performance Diagram figures. Has its own small plotting functions (not `mpas_stat.graphics.functions`) — see the callout in §2.6. |
| `legacy/radar_reflectivity_verify.py` | The **original standalone script** this product was integrated from (optimized: vectorized composite-reflectivity gridding, vectorized bootstrap, `ProcessPoolExecutor` file-level parallelism). Kept as a self-contained, independently runnable reference/example — NOT imported by the pipeline files above, which have their own (near-identical) copies of its computation kernels in `radar/common.py`. |

## 2.2 Running it

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

A single config can drive both stages for a whole date range (as above,
handy for quick looks). Production runs follow the model-space layout
instead:

- **Statistics, per experiment and cycle.** `scripts/run_verify_obs.ksh`
  (OMB/OMA) and `scripts/run_verify_obs_fcst.ksh` (OMF, also looping over
  forecast hours) copy the template `configs/obs_input.yaml` into
  `<main_dir>/<exp>/VerifyOBS/<cycle>[/f<HHH>]/input.yaml`, fill its
  placeholders with `sed`, and run `obs_stats` there with retries — exactly
  like `run_verify_model*.ksh` with `configs/model_input.yaml`. The
  statistics files and the run log (`verify_obs.log`) stay in that folder.
- **Common sample across experiments.** Each job lists the experiments to
  compare against in `compare_exps`. They are read (to find the
  observations valid in all experiments, §5.4) but only the job's own
  experiment is written (`experiments.write`). Run one job per experiment
  with the same set of experiments.
- **Graphics, all experiments together.** `python -m
  mpas_stat.cli.obs_graphics -c configs/obs_graphic.yaml`, with `diag_dir`
  set to the same `main_dir`, like `model_graphics` with
  `configs/model_graphic.yaml`.

Keep `domains` and the 2-D grid settings (`latlon_domain`,
`latlon_*_bin_width`) identical in `obs_input.yaml` and `obs_graphic.yaml`,
and use a named `latlon_domain` rather than `auto`: per-cycle runs would
otherwise each pick a different grid.

If the job activates a conda environment, do it in the scripts' "Python
environment" section, which relaxes `set -u` around the activation (conda
hooks often read unset variables, e.g. `GEOMETRIC_DATA_DIR`).

## 2.3 Config file — key fields

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
  diag_dir: ...       # statistics root: files go to <diag_dir>/<exp>/VerifyOBS/ (default <graphic_dir>/diag_files;
                      # the PBS scripts set it to main_dir, so files sit in each experiment's folder)
  graphic_dir: ...    # figures

experiments:
  names: [...]
  labels: [...]
  types: [da, hofx_forecast]   # "da" -> OMB+OMA (CyclingDA); "hofx" -> OMB only;
                                # "hofx_forecast" -> OMF (extended-forecast skill, HofX/<cycle>/f<HHH>/)
  outerloop: [2, 0]             # which analysis outer-loop's OMA to read ("da", n_member==1 only)
  n_member: [1, 1]              # >1 enables ensemble spread; also changes WHERE OMA is read from --
                                  # see note below
  write: [exp_a]                # optional: write statistics files only for these experiments; the
                                  # others are read only, to define the common sample (default: all)

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
  map_extent: auto                             # area drawn: auto (zoom to data) | full |
                                                  # [lat_min, lat_max, lon_min, lon_max]
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
> bootstrap `%` difference `100*(exp-ref)/ref` from it (for whichever statistics are listed in
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

### `statistics:` block (both stages)

```yaml
statistics:
  method: weighted          # or "legacy" (pre-2026-11 behaviour, all settings below ignored)
  qc_pass_flags: [0, 19]    # EffectiveQC values counted as passed (19 = JEDI background check)
  common_sample: true       # one matched observation sample for OMB/OMA/OMF and all experiments
  spread_ddof: 1
  spread_finite_ensemble_correction: true
  time_aggregation: pooled  # or "mean"
  error_diagnostics: true   # Desroziers error estimates (§5.6)
  ensemble_diagnostics: true    # rank histograms + spread-skill, n_member > 1 (§5.7)
  rank_histogram_perturb: true  # perturb members with the obs error before ranking
  normalized_departures: true   # departures / assumed obs error (and / total spread) (§5.8)
  channel_correlations: true    # radiance channel x channel Desroziers covariances (§5.9)
  correlation_channels: {}      # e.g. {amsua_n19: all, iasi_metop-b: [16, 38, 49]}
```

See §5 for what each setting does. `plot_settings.ci_style` (`shade`,
`errorbar` or `none`) chooses how confidence intervals are drawn on the
obs-space line and profile plots, as `ci_style` does for model space.
The top-level `plot_style` (fonts, `diverging_cmap`, …) and `colorbar`
(discrete levels, row sharing) blocks work as in the model-space graphics
config; see `examples/obs/08_publication_figures.yaml`.
`plot_settings.generate_error_consistency` (default `true`) turns the
error-consistency profile figures of §5.6 on or off, and
`plot_settings.write_error_tables` (default `true`) the error tables that go
with them (`error_table_min_count`, default 100, is the fewest observations a
level needs to enter the suggested JEDI filter).

### Scorecard

After the per-sensor figures, one **scorecard** per compared experiment
summarises everything on a single page:
`<graphic_dir>/scorecard/scorecard_<omb|oma|omf>_<exp>_vs_<reference>.png`,
with a CSV of every cell beside it.

- **Rows:** sensor, variable and level. "all" is the whole column, followed
  by each pressure or height bin from the top down. Radiances get one row per
  channel.
- **Columns:** statistic × domain. OMF scorecards also split by forecast lead.
- **Colour:** % difference 100·(exp − ref)/ref of the cycle-pooled
  statistic. Negative (blue) means a smaller error than the reference
  (`experiments.names[0]`).
- **Markers:** ▲/▼ mark a significant improvement or degradation, using the
  same paired bootstrap over cycles as the other comparison figures.

```yaml
plot_settings:
  scorecard:                # or "scorecard: false" to turn it off
    statistics: [RMS]       # RMS and/or STD
    diagtypes: [omb]        # cycling: omb and/or oma (OMF is added for hofx_forecast runs)
    levels: null            # e.g. [850, 500, 250] to show only those bins
    domains: null           # default: plot_settings.apply_domains
    leads: null             # OMF: e.g. [24, 72, 120]
    min_count: 10           # blank cells with fewer pooled observations
    min_cycles: 5           # no ▲/▼ with fewer cycles than this
    show_values: true       # print the % difference in each cell
    limit: null             # colour half-range in %, default automatic
```
Domain boxes (`domains`, and
`plot_settings.latlon_domain`) may cross the dateline: write them as
`[lat_min, lat_max, 170, -170]` or `[lat_min, lat_max, 170, 190]`.

## 2.4 Available statistics (observation-space)

| Statistic product | Function used to plot it | Notes |
|---|---|---|
| Domain-averaged time series | `plot_time_series_0d` | OMB/OMA only, one figure per (domain, statistic); OMB solid, OMA dashed |
| Vertical/latitude binned profile | `plot_profiles_1d` | OMB/OMA only. Pressure bins (conventional), height bins (GNSS-RO/radar), or latitude bins (radiances) — whichever applies to that sensor |
| **Experiment vs. control profile** | `plot_profile_diff_1d` | % difference `100*(exp-ref)/ref` of the pooled RMS/STD (`plot_settings.comparison_relative_reduction_statistics`) from `experiments.names[0]`, per level, with the paired-bootstrap interval (`ci_style`) and ▲/▼ where it is significant. **O−B is compared only with O−B and O−A only with O−A**, in separate figures: `omb/profile_diff_1d/` and `oma/profile_diff_1d/` (`profile_<profile_key>_<stat>.png`). A smaller O−A only means a closer fit to the observations, not necessarily a better analysis; judge experiments mainly on O−B (and OMF). |
| Lat-lon gridded map | `plot_lat_lon_2d` | OMB/OMA only, as two separate sets of figures. Computed over `plot_settings.latlon_domain`'s box (default: globally; not per `apply_domains`). Real-experiment comparison convention (reference + % difference) — see the callout in §2.3 |
| Lev-time (cycle x level) Hovmoller | `plot_time_level_2d` | OMB/OMA only, as two separate sets of figures, one per (domain, statistic). Reuses the existing 1D profile data, cycle-resolved. Real-experiment comparison convention, no bootstrap (each column is one cycle) — see the callout in §2.3 |
| Lev-lat (latitude x level) cross-section | `plot_lat_vert_2d` | OMB/OMA only, as two separate sets of figures. Domain-independent, like the lat-lon map; skipped for radiances. Real-experiment comparison convention with bootstrap significance — see the callout in §2.3 |
| **Forecast skill** (statistic vs. forecast lead time) | `plot_forecast_lead_0d` | **OMF only** (`hofx_forecast` experiments). Domain-averaged, bootstrap-aggregated over initialization cycles at each lead hour — the exact function the model-space package uses for its own extended-forecast skill plots. One figure per (domain, statistic); non-reference experiments are bootstrap-tested against `experiments.names[0]` and significant lead times get a hollow-circle marker, same convention as `plot_profile_diff_1d`. |
| **Forecast profile** (vertical/latitude, one figure per lead hour) | `plot_profiles_1d` (reused unmodified) | **OMF only.** The forecast-lead counterpart of the vertical/latitude profile above, mirroring the model-space package's own `forecast_profiles_1d` plot type: one profile figure per (domain, forecast lead hour, statistic), bootstrap-aggregated over initialization cycles at that lead. Lands in the SAME `profiles_1d/` folder as the OMB/OMA profiles, distinguished by an `_f<HHH>` filename suffix (e.g. `profile_vert_pres_profile_global_RMS_f006.png`) — see `generate_forecast_profile_graphics`. |
| **Forecast lat-lon map** (one figure per lead hour) | `plot_lat_lon_2d` (reused unmodified) | **OMF only.** Mirrors the model-space package's own `forecast_lat_lon_2d` plot type: real OMF experiments on the panel axis with the same reference/bootstrap-%-reduction comparison convention as the OMB/OMA map above, one set of figures per lead hour (`_f<HHH>` filename suffix, same `lat_lon_2d/` folder) — see `generate_forecast_lat_lon_graphics`. |
| **Forecast lev-lat cross-section** (one figure per lead hour) | `plot_lat_vert_2d` (reused unmodified) | **OMF only.** Mirrors the model-space package's own `forecast_lat_vert_2d` plot type; same real-experiment comparison convention and `_f<HHH>` filename suffix as the forecast lat-lon map above. Skipped for radiances, like the OMB/OMA lev-lat cross-section — see `generate_forecast_lat_vert_graphics`. |
| **Forecast lev-time (lead x level) Hovmoller** | `plot_forecast_time_level_2d` | **OMF only.** The genuinely different one: forecast lead hour replaces cycle on the x-axis, and — unlike the plain-mean OMB/OMA lev-time Hovmoller — each lead-hour column already averages over every initialization cycle, so bootstrap significance vs. `experiments.names[0]` IS drawn here, via the shared `_draw_2d_panels` machinery. One figure per (domain, statistic), in its own `forecast_time_level_2d/` folder. Mirrors the model-space package's own `forecast_time_level_2d` plot type exactly — see `generate_forecast_time_level_graphics`. |

3-D binned products and scorecards (available on the model side) are
intentionally **not** implemented for obs-space.

## 2.5 `obsstat_<obs>.<date>.nc` schema

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
