# Example configurations

Ready-to-adapt YAML files for common tasks. Each file starts with a header
that says what it is for, how to run it, and which output to look at first.
Replace the `/path/to/...` values and the experiment names, then run.
`tests/test_examples.py` loads every file here, so they stay in step with
the code.

The commands below are the installed entry points. Without installing, use
`python -m mpas_stat.cli.<module>` with the module shown in brackets.

## Observation space (`examples/obs/`)

Each file is a **complete config for both stages**: run the statistics
(`mpas-stat-obs` [obs_stats]) and then the figures (`mpas-stat-obs-plot`
[obs_graphics]) with the same file. For long periods on a cluster, run the
statistics with `scripts/run_verify_obs*.ksh` (template `configs/obs_input.yaml`)
and use these files for the figures.

| File | Use it to… |
|---|---|
| `01_compare_two_experiments.yaml` | Compare a test DA experiment with a control: every standard figure plus a one-page scorecard. **Start here.** |
| `02_quick_look_single_experiment.yaml` | Check a running experiment quickly: few sensors, global only, no maps. |
| `03_enkf_ensemble.yaml` | Check ensemble DA reliability with spread, rank histograms, spread–skill and consistency ratio. |
| `04_obs_error_tuning.yaml` | Estimate observation errors (Desroziers) and export suggested values with UFO filters (`mpas-stat-obs-errors`). |
| `05_extended_forecast_omf.yaml` | Verify forecasts against observations by lead time (OMF), with a lead-time scorecard. |
| `06_radiance_monitoring.yaml` | Monitor satellite radiances per channel: maps, latitude profiles, per-channel errors. |
| `07_regional_domain.yaml` | Set up a regional mesh: custom sub-domains (one crosses the dateline) and finer 2-D grids. |
| `08_publication_figures.yaml` | Make paper figures: PDF, error bars, fonts, fixed colour ranges, compact scorecard. |
| `09_legacy_reproduce.yaml` | Reproduce pre-2026-11 results exactly (`statistics.method: legacy`). |
| `10_single_cycle_manual.yaml` | Compute statistics for one cycle by hand, as each batch job does; shows `experiments.write`. |
| `11_radar_reflectivity.yaml` | Get radar composite-reflectivity neighbourhood scores (FSS, ETS, …) with METplus. |
| `12_multi_experiment_sensitivity.yaml` | Compare several experiments against one control (e.g. a localization sweep). |

## Model space (`examples/model/`)

Statistics and figures use **separate** files. `*_stats_input.yaml` files
are for `mpas-stat-model` [model_stats], one forecast file per run (the
`scripts/run_verify_model*.ksh` jobs fill in the file names for every
cycle). `*_graphic_*.yaml` files are for `mpas-stat-graphics` [model_graphics].

| File | Use it to… |
|---|---|
| `01_stats_input_global.yaml` | Compute statistics on a global mesh (deterministic) with latitude-band regions. |
| `02_stats_input_regional.yaml` | Compute statistics on a regional mesh: `regional: true`, fine bins, heights above ground. |
| `03_stats_input_ensemble.yaml` | Compute statistics for an ensemble: member file template, ensemble-mean error and spread. |
| `04_graphic_cycling_compare.yaml` | Compare cycling experiments against analyses: profiles, % differences, maps, cross-sections. |
| `05_graphic_extended_forecast.yaml` | Plot forecast skill vs. lead time, with the model-space forecast scorecard. |
| `06_graphic_ensemble_spread.yaml` | Plot ensemble-mean RMSE next to spread. |
| `07_graphic_publication.yaml` | Make paper figures for model space. |
| `08_graphic_legacy.yaml` | Reproduce pre-2026-11 model-space figures. |

## Where the settings are explained

- All statistics methods and diagnostics: [`docs/guide/methods.md`](../docs/guide/methods.md)
  (§5.6 Desroziers, §5.7 ensemble reliability).
- Obs-space options, scorecard, error tables: [`docs/guide/obs_space.md`](../docs/guide/obs_space.md).
- Radar product: [`docs/guide/radar.md`](../docs/guide/radar.md).
- Full annotated templates with every option: `configs/`.
