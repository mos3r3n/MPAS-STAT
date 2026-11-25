# 1. Model-space verification

Model space compares each MPAS forecast with its verifying analysis on the
native mesh. The forecast minus analysis difference is pooled over grid
cells by region, latitude band, lat-lon box and vertical level, with area
and mass weights (§5.1).

## 1.1 Workflow

```bash
# 1. statistics: one run per experiment and cycle
python -m mpas_stat.cli.model_stats -c configs/model_input.yaml

# 2. figures: every experiment and cycle in the configured period
python -m mpas_stat.cli.model_graphics -c configs/model_graphic.yaml
```

In production the statistics run is a PBS job:

| Script | Loops over | Writes |
|---|---|---|
| `scripts/run_verify_model.ksh` | assimilation cycles | `<main_dir>/<exp>/VerifyModel/<cycle>/mpasstat.<date>.nc` |
| `scripts/run_verify_model_fcst.ksh` | init cycles × forecast leads | `<main_dir>/<exp>/VerifyModel/<init>/f<HHH>/mpasstat.<valid>.nc` |

Both copy `configs/model_input.yaml` for each run and fill in its
placeholders (`forecastFile`, `initFile`, `outputFile`, `nMember`,
`nProcesses`) with `sed`. Edit the `export` lines at the top of the script
for your experiment, directories and cycle range.

## 1.2 Statistics configuration

`configs/model_input.yaml` (full template: {doc}`/configuration`):

| Key | Meaning |
|---|---|
| `forecast_file`, `init_file`, `output_file` | Input forecast, verifying analysis and output file (filled in by the PBS script) |
| `n_member` | 1 for a deterministic run; above 1, the members' mean is verified and their spread written |
| `variables` | Variables to verify: `U`, `V`, `T`, `TH`, `Tv`, `Qv`, `RH`, `Z`, `U10`, `Ps`, … |
| `regions` | Named latitude bands for domain averages, e.g. `Global: [-90, 90]` |
| `regional` | `true` limits the lat-lon grid to the mesh's own extent |
| `lat_bin_width`, `lon_bin_width`, `level_bin_size` | Resolution of the latitude, lat-lon and model-level bins |
| `pres_levels`, `hgt_levels` | Pressure and height levels to interpolate to |
| `height_reference` | `msl` (default) or `agl`: heights above sea level or above the terrain |
| `statistics_method` | `weighted` (default) or `legacy` (§5.1) |
| `area_weighting`, `mass_weighting` | Weight cells by area and levels by layer mass (default `true`) |
| `spread_ddof`, `spread_finite_ensemble_correction` | Ensemble spread definition (§5.3) |
| `n_processes` | Threads for the statistics tasks |

## 1.3 Graphics configuration

`configs/model_graphic.yaml`:

| Key | Meaning |
|---|---|
| `paths.base_dir`, `paths.out_dir` | Where the statistics files are, and where figures go |
| `experiments.names`, `labels`, `colors`, `linestyles` | One entry per experiment. The **first is the reference** for every comparison |
| `time.start`, `time.end`, `time.interval_hours` | Cycles to read |
| `variables`, `statistics` | What to plot: `rms`, `mean`, `std`, `spread` |
| `region` | Which of the statistics run's `regions` to plot |
| `plot_types` | Which figures to draw (§1.4) |
| `level_slices` | Levels for `lat_lon_slice_2d`, by model level, height or pressure |
| `forecast` | `length`, `interval`, `start` of the forecast leads (for `forecast_*` types) |
| `scorecard` | Rows and limits of `forecast_scorecard` |
| `time_aggregation` | `pooled` (default) or `mean` (§5.2) |
| `ci_style`, `ci_alpha` | Confidence intervals as `shade` (default), `errorbar` or `none` |
| `map_extent` | `auto` (default) zooms maps and latitude axes to the data; `full`; or `[lat_min, lat_max, lon_min, lon_max]` |
| `colorbar.*` | Discrete colour levels (default), number of levels, shared per row or per panel; explicit `*_vmin`/`*_vmax` limits |
| `plot_style.*` | Fonts, colour maps (`diverging_cmap`, `absolute_cmap`) and line `palette` |
| `n_workers` | Processes used to draw figures in parallel |

## 1.4 Figures

Each type is drawn per variable and statistic. Comparison figures show the
percent difference from the first experiment, `100·(exp − ref)/ref`, so
**negative is better** (§3.2).

### Cycling

**`time_series_0d`**: the region average by cycle.

```{figure} ../gallery/model_time_series.png
:width: 85%

Region-averaged rms of the forecast against the verifying analysis, one
panel per variable.
```

**`profiles_1d`**: the profile pooled over cycles, by pressure, height,
model level or latitude, with bootstrap intervals.
**`profile_diff_1d`**: the same as a percent difference from the reference,
with ▲/▼ on significant levels.

```{figure} ../gallery/model_profile.png
:width: 85%

`profiles_1d`: rms by pressure level.
```

```{figure} ../gallery/model_profile_reduction.png
:width: 85%

`profile_diff_1d`: rms percent difference from Control by pressure level.
```

**`lat_lon_2d`** (column statistics) and **`lat_lon_slice_2d`** (at the
levels in `level_slices`): maps with the reference in full and the other
experiments as percent differences, ▲/▼ on significant cells.

```{figure} ../gallery/model_lat_lon.png
:width: 60%

`lat_lon_slice_2d`: rms at 500 hPa.
```

**`lat_vert_2d`**: latitude–level cross-sections (model level, height and
pressure). **`time_level_2d`**: cycle–level sections, one value per cycle,
so without significance markers.

### Extended forecasts

These need the `forecast` block and the `f<HHH>/` statistics files.

| Type | Draws |
|---|---|
| `forecast_lead_0d` | Region average against lead, pooled over init cycles |
| `forecast_lead_diff_0d` | The same as a percent difference from the reference |
| `forecast_profiles_1d`, `forecast_profile_diff_1d` | Profiles at each lead |
| `forecast_lat_lon_2d`, `forecast_lat_vert_2d` | Maps and cross-sections at each lead |
| `forecast_time_level_2d` | Lead–level section, with significance |
| `forecast_scorecard` | Triangle scorecard by variable, level and lead |

```{figure} ../gallery/model_forecast_lead.png
:width: 85%

`forecast_lead_0d`: rms against lead, with bootstrap intervals; circles
mark leads where the difference from Control is significant.
```

```{figure} ../gallery/model_forecast_lead_diff.png
:width: 85%

`forecast_lead_diff_0d`: the same as a percent difference from Control.
```

```{figure} ../gallery/model_forecast_scorecard.png
:width: 55%

`forecast_scorecard`: ▲ improvement, ▼ degradation; outlined triangles
are significant, and size scales with the difference. Rows come from
`scorecard.rows`.
```

## 1.5 Statistics file

`mpasstat.<date>.nc`, one per experiment and cycle (and lead):

```
LatBins, LonBins, LevelBins, ModelLevels, HeightLevels, PressureLevels    coordinates
<variable>/                                       U, V, T, Qv, Ps, …
    0D/<region>/                                  mean, rms, std, spread, weight
    1D/lat_bin_profile/                           … by LatBins
       vert_profile_<region>/                     … by ModelLevels
       vert_bin_profile_<region>/                 … by LevelBins
       vert_hgt_profile_<region>/                 … by HeightLevels
       vert_pres_profile_<region>/                … by PressureLevels
    2D/lat_lon/, lat_lev/, lat_hgt/, lat_pres/    2-D grids
    3D/lat_lon_lev/, lat_lon_hgt/, lat_lon_pres/  3-D grids (3-D variables only)
task_timing_seconds/, variable_timing_seconds/    run times
```

`spread` is present only when `n_member > 1`. `weight` (the summed area ×
layer-mass weight of the valid samples) is written by the `weighted`
method and used to pool cycles.

## 1.6 Code map

| Module | Role |
|---|---|
| `mpas_stat.cli.model_stats`, `model_graphics` | Command-line entry points |
| `mpas_stat.model.config`, `args` | Configuration schema and command-line overrides |
| `mpas_stat.model.variables` | Derives `U`, `T`, `RH`, … with units from raw MPAS fields |
| `mpas_stat.model.interp` | Vertical interpolation to height and pressure (numba-accelerated when available) |
| `mpas_stat.model.kernels` | Statistics kernels: region means, profiles, 2-D and 3-D bins |
| `mpas_stat.model.driver` | Loads, interpolates, runs the statistics tasks, writes the file |
| `mpas_stat.graphics.collect`, `collect_forecast`, `ncread` | Read statistics files back and pool cycles |
| `mpas_stat.graphics.plots_0d`, `plots_1d`, `plots_2d`, `scorecard` | The figures |
| `mpas_stat.graphics.driver` | Turns `plot_types` into figure jobs, optionally in parallel |
