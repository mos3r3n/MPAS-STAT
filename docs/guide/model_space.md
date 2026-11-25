# 1. Model-space verification

Compares an MPAS forecast field against its own initial/analysis field, on
the native model grid, binned by lat/lon/level.

## 1.1 Files

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
| `src/mpas_stat/graphics/io.py` | Facade re-exporting cycle/date helpers (`core/timeutils.py`), bootstrap CI helpers (`core/bootstrap.py`), netCDF/path helpers (`graphics/ncread.py`) and the `collect_*` functions (`graphics/collect.py` for cycling, `graphics/collect_forecast.py` for extended forecasts) that read `mpasstat.<date>.nc` files back into plotting-ready Python dicts |
| `src/mpas_stat/graphics/functions.py` | Facade re-exporting every plotting function and helper, now implemented in `style.py` (publication style, `_save_fig`), `axes.py`, `panels.py` (shared 2-D panel builder, colour maps, significance), `plots_0d.py`, `plots_1d.py`, `plots_2d.py` and `scorecard.py`. Contents: | All plotting functions (`plot_time_series_0d`, `plot_profiles_1d`, `plot_lat_lon_2d`, `plot_lat_vert_2d`, `plot_time_level_2d`, the `forecast_*` and scorecard variants) plus shared styling/colorbar/significance helpers |
| `src/mpas_stat/graphics/driver.py` | `run_graphics(cfg)` — dispatches over `cfg["plot_types"]`, optionally in parallel (`n_workers`) |
| `configs/model_graphic.yaml` | Graphics-stage configuration template (see below) |

## 1.2 Running it

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

## 1.3 `input.yaml` (statistics stage) — key fields

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
| `statistics_method` | `weighted` (default) or `legacy` — see §5 |
| `area_weighting`, `mass_weighting` | Weight cells by `areaCell` and model levels by layer pressure thickness (default `true`; `weighted` method only) — §5.1 |
| `spread_ddof`, `spread_finite_ensemble_correction` | Ensemble spread definition (default `1`, `true`) — §5.3 |
| `height_reference` | `msl` (default) or `agl`: whether `hgt_levels` are above sea level or above the model terrain |

Variable names: `T` is temperature, `TH` potential temperature, `Tv` virtual
temperature (before 2026-11, `Tv` mistakenly returned potential temperature).

## 1.4 `graphic.yaml` (graphics stage) — key fields

| Key | Meaning |
|---|---|
| `paths.base_dir` / `paths.out_dir` | Where to read `mpasstat.<date>.nc` from / write figures to |
| `experiments.names/labels/colors/linestyles` | One entry per experiment being compared |
| `time.start/end/interval_hours` | Assimilation-cycle range to read |
| `variables`, `statistics` | Which variables and which stats (`rms`, `mean`, `std`, `spread`) to plot |
| `region` | Which of `input.yaml`'s `regions` to plot (one region per run) |
| `plot_types` | Which figures to generate — see §1.5 |
| `forecast.*`, `scorecard.*`, `level_slices` | Extended-forecast / scorecard / 2-D-slice-specific settings |
| `time_aggregation` | How cycles are combined: `pooled` (default) or `mean` — §5.2 |
| `colorbar.discrete`, `colorbar.levels_diverging`, `colorbar.levels_sequential`, `colorbar.share` | 2-D colour scales: discrete colour levels at rounded values (default), roughly 20 (diverging) / 16 (sequential) bins, limits shared by every panel of a variable row (`share: row`, default) or per panel (`panel`). Diverging scales are symmetric with 0 on a level boundary. Explicit `relative_reduction_vmin/vmax`, `mean_vmin/vmax`, `absolute_vmin/vmax` still fix the limits. |
| `plot_style.diverging_cmap`, `plot_style.absolute_cmap` | Colour maps for % differences / mean fields (default `RdBu_r`; `RdYlBu_r` is a good alternative) and for absolute statistics (default `YlGnBu`) |
| `plot_style.palette` | Experiment line colours when `experiments.colors` is not given. Default: black, then ColorBrewer Dark2 (colour-vision-deficiency safe). Obs space: `plot_settings.palette`. |
| `ci_style` | Bootstrap confidence intervals on line/profile plots: `shade` (coloured band, default), `errorbar`, or `none`. `ci_alpha` sets the band opacity. (Older `profile_ci_style` / `profile_errorbar: false` still work.) |
| `map_extent` | Area shown by lat-lon maps and latitude axes (lat-vert cross-sections, latitude-binned Hovmöllers): `auto` (default) zooms to the cells that hold data in any panel of the figure, so a regional mesh or sub-domain fills the frame; `full` shows the whole grid; `[lat_min, lat_max, lon_min, lon_max]` fixes a box. A region crossing the dateline is drawn in one piece (map centred on 180°). Obs space: `plot_settings.map_extent`. |

## 1.5 Available `plot_types` (model-space)

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

## 1.6 `mpasstat.<date>.nc` schema

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
`input.yaml`'s `regions` map. With `statistics_method: weighted` (the
default) every group also holds `weight`, the total sample weight behind
its statistics (area × layer mass, summed over valid samples), which the
graphics stage uses to pool cycles (§5.2).


