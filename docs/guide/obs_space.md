# 2. Observation-space verification

Observation space compares the background, analysis and forecast with the
observations, at the observation locations, using the IODA files written
by MPAS–JEDI. It works for conventional observations (radiosondes,
aircraft, profilers, surface, GNSS-RO, …) and radiances.

## 2.1 What is verified

Each experiment has a type, set in `experiments.types`:

| Type | Diagnostics | Reads |
|---|---|---|
| `da` | O−B and O−A, cycling | `<da_dir>/<exp>/CyclingDA/<cycle>/dbOut/obsout_da_<obs>.h5` |
| `hofx` | O−B only | `<hofx_dir>/<exp>/<cycle>/dbOut/obsout_hofx_<obs>.h5` |
| `hofx_forecast` | O−F at each forecast lead | `<hofx_dir>/<exp>/HofX/<init>/f<HHH>/dbOut/obsout_hofx_<obs>.h5` |

Types can be mixed in one configuration.

To keep comparisons fair, every O−B, O−A and O−F being compared is
computed on **one common sample**: the observations that pass QC in all of
them, matched by location and time (§5.4).

```{admonition} Judge experiments on O−B and O−F
:class: tip
A smaller O−A only means the analysis fits the observations more closely,
not that it is better. Use O−B (and O−F) to compare experiments, and O−A
for the error diagnostics of §5.6.
```

## 2.2 Workflow

For a quick look, one configuration drives both stages over a period:

```bash
python -m mpas_stat.cli.obs_stats    my_config.yaml   # statistics files
python -m mpas_stat.cli.obs_graphics my_config.yaml   # figures
```

In production the statistics stage runs per experiment and cycle, like
model space:

| Script | Loops over | Writes |
|---|---|---|
| `scripts/run_verify_obs.ksh` | cycles | `<main_dir>/<exp>/VerifyOBS/<cycle>/obsstat_<obs>.<date>.nc` |
| `scripts/run_verify_obs_fcst.ksh` | init cycles × leads | `<main_dir>/<exp>/VerifyOBS/<init>/f<HHH>/obsstat_<obs>.<valid>.nc` |

Each job fills in a copy of `configs/obs_input.yaml`, runs `obs_stats` in
that folder with retries and keeps its log there. Then draw all
experiments together:

```bash
python -m mpas_stat.cli.obs_graphics -c configs/obs_graphic.yaml
```

Things to keep consistent:

- **Common sample.** Each job lists the experiments it is compared with in
  `compare_exps`. They are read to build the common sample, but only the
  job's own experiment is written (`experiments.write`). Give every job the
  same set.
- **Grids.** Use the same `domains`, `latlon_domain` and bin widths in the
  statistics and graphics configurations, and give `latlon_domain` a named
  box rather than `auto`, so every cycle uses the same grid.
- **Conda.** Activate environments in the scripts' "Python environment"
  section, which relaxes `set -u` for activation hooks.

## 2.3 Configuration

`configs/obs_graphic.yaml` (full template: {doc}`/configuration`). The
main blocks:

```yaml
paths:
  da_dir: /path/to/da          # "da" experiments
  hofx_dir: /path/to/hofx      # "hofx" and "hofx_forecast" experiments
  diag_dir: /path/to/main      # statistics files: <diag_dir>/<exp>/VerifyOBS/...
  graphic_dir: /path/to/figs

experiments:
  names: [control, test]       # the first is the reference for comparisons
  labels: [Control, Test]
  types: [da, da]
  outerloop: [2, 2]            # outer loop whose O−A is read
  n_member: [1, 10]            # >1: ensemble spread and reliability diagnostics

sensors: [sondes, aircraft, amsua_n18]

domains:                       # [lat_min, lat_max, lon_min, lon_max]
  global: [null, null, null, null]
  NXTro: [30.0, 90.0, -180.0, 180.0]

time_settings:
  start_time: "2024-09-13 00:00:00"
  end_time: "2024-09-15 18:00:00"
  interval: 6                  # hours

forecast:                      # only for hofx_forecast experiments
  length: 24
  interval: 6

n_workers: 4                   # sensors drawn in parallel
```

### `plot_settings`

| Key | Meaning |
|---|---|
| `apply_domains` | Which `domains` to compute and plot |
| `plot_oma` | Include O−A (default `true`) |
| `latlon_domain` | Box for maps and latitude bins: `global` (default), a `domains` key, or `auto` (§2.7) |
| `latlon_lat_bin_width`, `latlon_lon_bin_width` | Map grid spacing in degrees |
| `map_extent` | Area drawn: `auto` (zoom to the data, default), `full` or a box |
| `generate_lat_lon_2d`, `generate_lat_vert_2d`, `generate_time_level_2d` | Maps and cross-sections (§2.4) |
| `comparison_relative_reduction_statistics` | Statistics shown as percent differences in comparison figures (default `[RMS, STD]`) |
| `generate_error_consistency`, `write_error_tables`, `error_table_min_count` | Desroziers figures and observation-error tables (§5.6) |
| `generate_normalized_departures` | §5.8 |
| `generate_ensemble_diagnostics`, `spread_skill` | §5.7 |
| `generate_channel_correlations` | §5.9 |
| `scorecard` | One-page summary (§2.4), or `false` |
| `ci_style` | Confidence intervals: `shade`, `errorbar` or `none` |
| `figure_format`, `max_time_ticks`, `palette` | Output format, date ticks, line colours |

The top-level `plot_style` and `colorbar` blocks work as in model space.

### `statistics`

Read by both stages; §5 explains each setting.

```yaml
statistics:
  method: weighted              # "legacy" reproduces pre-2026-11 results
  qc_pass_flags: [0, 19]        # EffectiveQC values counted as passed
  common_sample: true
  spread_ddof: 1
  spread_finite_ensemble_correction: true
  time_aggregation: pooled      # or "mean"
  error_diagnostics: true       # §5.6
  ensemble_diagnostics: true    # §5.7
  rank_histogram_perturb: true
  normalized_departures: true   # §5.8
  channel_correlations: true    # §5.9
  correlation_channels: {}      # e.g. {amsua_n19: all, iasi_metop-b: [16, 38, 49]}
```

### Sensors

What each sensor holds is defined once, in
`mpas_stat/obs/sensor_config.py` (`OBSERVATION_CONFIGS`), not in the YAML.
To add an observation type, add an entry there:

| Field | Purpose |
|---|---|
| `variables`, `units`, `labels` | IODA variables (or radiance channels), units and panel titles |
| `bin_method` | `PRESSURE`, `HEIGHT` or `LATITUDE` (radiances): the profile coordinate |
| `file_alias` | Read another sensor's file, e.g. `profiler` reads the `sondes` file |
| `obs_type_filter` | Keep only some `ObsType` values when two sensors share a file |
| `level_bin_kwargs` | Custom bins (`bin_width`, `bin_bottom`, `bin_top`) |
| `cycle_interval_hours` | The sensor's own cycle spacing, e.g. 12 h for radiosondes |
| `cycle_valid_hours` | UTC hours the sensor reports at, e.g. `[0, 12]` for radiosondes (§2.7) |

## 2.4 Figures

Figures go to `<graphic_dir>/<sensor>/<type>/`, one per domain and
statistic (`RMS`, `Bias`, `STD`, `Count`, plus `Spread` and `TotalSpread`
for ensembles).

### Time series and profiles

O−B is drawn solid and O−A dashed, one colour per experiment.

```{figure} ../gallery/obs_time_series.png
:width: 85%

`time_series_0d/`: domain-averaged O−B and O−A rms by cycle (radiosondes).
```

```{figure} ../gallery/obs_profile.png
:width: 65%

`profiles_1d/`: O−B and O−A rms by pressure bin, pooled over cycles.
Height bins are used for GNSS-RO and radar, latitude bins for radiances.
```

Radiances get one panel per channel:

```{figure} ../gallery/obs_radiance_ts.png
:width: 95%

AMSU-A (NOAA-18) O−B and O−A rms by cycle, channels 5–9.
```

### Comparison with the reference

These figures compare each experiment with the first one, separately for
O−B (`omb/`) and O−A (`oma/`). The difference is
`100·(exp − ref)/ref` of the pooled statistic; ▲/▼ mark differences whose
95 % paired-bootstrap interval over cycles excludes zero (§3.2).

```{figure} ../gallery/obs_profile_diff.png
:width: 65%

`omb/profile_diff_1d/`: O−B rms of Test as a percent difference from
Control, with its confidence interval.
```

```{figure} ../gallery/obs_lat_lon.png
:width: 60%

`omb/lat_lon_2d/`: O−B rms in 10° boxes. Left: the reference; right: the
percent difference. Computed over `latlon_domain`, not per domain.
```

```{figure} ../gallery/obs_lat_vert.png
:width: 60%

`omb/lat_vert_2d/`: latitude–pressure cross-section, pooled over cycles.
Not drawn for radiances, whose profiles are already by latitude.
```

```{figure} ../gallery/obs_time_level.png
:width: 60%

`omb/time_level_2d/`: cycle–pressure section. Each column is a single
cycle, so there are no significance markers.
```

### Scorecard

After the per-sensor figures, one scorecard per compared experiment puts
every sensor, variable and level on one page:
`<graphic_dir>/scorecard/scorecard_<omb|oma|omf>_<exp>_vs_<ref>.png`, with
a CSV of every cell next to it.

```{figure} ../gallery/obs_scorecard.png
:width: 50%

Rows are sensor, variable and level (`all` is the whole column, then each
bin from the top; one row per channel for radiances). Columns are
statistic × domain (× lead for O−F). Blue is a smaller error than the
reference, and ▲/▼ mark significant cells.
```

```yaml
plot_settings:
  scorecard:                # or "scorecard: false"
    statistics: [RMS]       # RMS and/or STD
    diagtypes: [omb]        # omb and/or oma; omf is added for hofx_forecast runs
    levels: null            # e.g. [850, 500, 250]
    domains: null           # default: apply_domains
    leads: null             # O−F, e.g. [24, 72, 120]
    min_count: 10           # blank cells with fewer observations
    min_cycles: 5           # no ▲/▼ with fewer cycles
    show_values: true
    limit: null             # colour half-range in %
```

### Error and ensemble diagnostics

Drawn for `da` experiments with O−A (error diagnostics) and for ensembles
(reliability). §5 describes how to read each one.

| Folder | Shows | Section |
|---|---|---|
| `error_consistency_1d/` | Assumed σo against Desroziers σo and σb | §5.6 |
| `rank_histogram/`, `spread_skill/` | Ensemble reliability | §5.7 |
| `normalized_departure/` | Departures divided by σo and by total spread | §5.8 |
| `channel_correlation/` | Inter-channel error correlations (radiances) | §5.9 |

### Forecasts (O−F)

For `hofx_forecast` experiments, statistics are pooled over init cycles at
each lead:

| Figure | Folder |
|---|---|
| Statistic against lead, with significance against the reference | `forecast_lead_0d/` |
| Lead–level section, with significance | `forecast_time_level_2d/` |
| Profiles, maps and latitude–level sections at each lead (`_f<HHH>` suffix) | `profiles_1d/`, `lat_lon_2d/`, `lat_vert_2d/` |
| O−F scorecard, by lead | `scorecard/` |

```{figure} ../gallery/obs_omf_lead.png
:width: 85%

`forecast_lead_0d/`: radiosonde O−F rms against lead. Circles mark leads
where Test differs significantly from Control.
```

```{figure} ../gallery/obs_omf_lead_level.png
:width: 60%

`forecast_time_level_2d/`: O−F rms by lead and pressure; the right column
is the percent difference from Control.
```

## 2.5 Statistics file

`obsstat_<obs>.<date>.nc`, one per experiment, cycle (and lead) and
sensor:

```
Levels                         bin centres: hPa, km or degrees north
LatBins2D, LonBins2D           map grid
omb/ | oma/ | omf/             O−B and O−A (cycling files) or O−F (forecast files)
  <variable>/                  airTemperature, …, or channel_<N> for radiances
    0D/<domain>/               RMS, Bias, STD, Count [, Spread, TotalSpread]
    1D/vert_pres_profile_<domain>/   by pressure (conventional)
       vert_hgt_profile_<domain>/    by height (GNSS-RO, radar)
       lat_bin_profile/              by latitude (radiances)
    2D/lat_lon/                RMS, Bias, STD, Count by LatBins2D × LonBins2D
       lat_vert/               by Levels × LatBins2D (not for radiances)
channel_covariance/            radiance channel covariances (§5.9)
```

`Spread` and `TotalSpread` are written only for ensembles. The error,
reliability and normalized-departure variables of §5.6–5.8 are added to
the same groups.

## 2.6 Radar reflectivity

A separate, opt-in product: see {doc}`radar`.

## 2.7 Details

**Radiosonde times.** Radiosondes report at 00 and 12 UTC
(`cycle_valid_hours: [0, 12]`).
- *Cycling:* their cycles start at the next 00 or 12 UTC after
  `start_time` and run every 12 h.
- *Forecasts:* init times follow `time_settings` as usual, and a lead whose
  valid time is not 00 or 12 UTC is left empty.
- *Profilers* share the radiosonde file but keep the normal cadence.

**Missing H(x).** A HofX value that is NaN, infinite or at least 1e30 (the
netCDF fill value) is treated as missing and the observation is dropped.
This matters most for O−F in a regional domain, where observations outside
the mesh have no model equivalent and often no QC flag.

**Regional maps.** By default maps and radiance latitude bins cover the
globe. For a regional application set `latlon_domain`:
- `auto` finds the smallest box holding every valid observation (it scans
  the files once);
- a `domains` key uses that fixed box.

The box is the same for every cycle, and observations outside it are
dropped from the maps. Radiance latitude bins use the same box and
`latlon_lat_bin_width`, unless the sensor sets `level_bin_kwargs`.

**Where O−A comes from.** For deterministic `da` runs (`n_member: 1`),
O−B and O−A are groups of the same `obsout_da_<obs>.h5` file
(`ombg`/`oman`, or `hofx0`/`hofx<outerloop>`). For ensemble runs O−A is in
a separate `obsout_da_<obs>_ana.h5`. The file is chosen from `n_member`.

**Domains across the dateline** are written as
`[lat_min, lat_max, 170, -170]` or `[lat_min, lat_max, 170, 190]`.

## 2.8 Code map

| Module | Role |
|---|---|
| `mpas_stat.cli.obs_stats`, `obs_graphics`, `obs_errors` | Command-line entry points |
| `mpas_stat.obs.sensor_config` | Sensor definitions |
| `mpas_stat.obs.common` | Configuration defaults, bins, group names, file paths |
| `mpas_stat.obs.reader` | IODA file names and reading, QC |
| `mpas_stat.obs.binning`, `bounds` | Statistics kernels; `latlon_domain: auto` |
| `mpas_stat.obs.process`, `writer` | Per-sensor processing; statistics files |
| `mpas_stat.obs.collect`, `experiments`, `figures`, `scorecard` | Graphics stage |
