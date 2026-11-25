# Examples

Ready-to-adapt configurations for common tasks. Each starts with a header
that says what it is for, how to run it and which output to look at first.
Replace the `/path/to/...` values and experiment names, then run.
`tests/test_examples.py` loads every file, so they stay in step with the code.
The {doc}`gallery` shows the figures they produce.

## Observation space

Each file is one configuration for **both** stages (statistics, then figures).

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

## Model space

Statistics inputs (`*_stats_input.yaml`, one forecast file per run) and
graphics configurations (`*_graphic_*.yaml`) are separate files.

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

## Files

### `obs/01_compare_two_experiments.yaml`

```{literalinclude} ../examples/obs/01_compare_two_experiments.yaml
:language: yaml
```

### `obs/02_quick_look_single_experiment.yaml`

```{literalinclude} ../examples/obs/02_quick_look_single_experiment.yaml
:language: yaml
```

### `obs/03_enkf_ensemble.yaml`

```{literalinclude} ../examples/obs/03_enkf_ensemble.yaml
:language: yaml
```

### `obs/04_obs_error_tuning.yaml`

```{literalinclude} ../examples/obs/04_obs_error_tuning.yaml
:language: yaml
```

### `obs/05_extended_forecast_omf.yaml`

```{literalinclude} ../examples/obs/05_extended_forecast_omf.yaml
:language: yaml
```

### `obs/06_radiance_monitoring.yaml`

```{literalinclude} ../examples/obs/06_radiance_monitoring.yaml
:language: yaml
```

### `obs/07_regional_domain.yaml`

```{literalinclude} ../examples/obs/07_regional_domain.yaml
:language: yaml
```

### `obs/08_publication_figures.yaml`

```{literalinclude} ../examples/obs/08_publication_figures.yaml
:language: yaml
```

### `obs/09_legacy_reproduce.yaml`

```{literalinclude} ../examples/obs/09_legacy_reproduce.yaml
:language: yaml
```

### `obs/10_single_cycle_manual.yaml`

```{literalinclude} ../examples/obs/10_single_cycle_manual.yaml
:language: yaml
```

### `obs/11_radar_reflectivity.yaml`

```{literalinclude} ../examples/obs/11_radar_reflectivity.yaml
:language: yaml
```

### `obs/12_multi_experiment_sensitivity.yaml`

```{literalinclude} ../examples/obs/12_multi_experiment_sensitivity.yaml
:language: yaml
```

### `model/01_stats_input_global.yaml`

```{literalinclude} ../examples/model/01_stats_input_global.yaml
:language: yaml
```

### `model/02_stats_input_regional.yaml`

```{literalinclude} ../examples/model/02_stats_input_regional.yaml
:language: yaml
```

### `model/03_stats_input_ensemble.yaml`

```{literalinclude} ../examples/model/03_stats_input_ensemble.yaml
:language: yaml
```

### `model/04_graphic_cycling_compare.yaml`

```{literalinclude} ../examples/model/04_graphic_cycling_compare.yaml
:language: yaml
```

### `model/05_graphic_extended_forecast.yaml`

```{literalinclude} ../examples/model/05_graphic_extended_forecast.yaml
:language: yaml
```

### `model/06_graphic_ensemble_spread.yaml`

```{literalinclude} ../examples/model/06_graphic_ensemble_spread.yaml
:language: yaml
```

### `model/07_graphic_publication.yaml`

```{literalinclude} ../examples/model/07_graphic_publication.yaml
:language: yaml
```

### `model/08_graphic_legacy.yaml`

```{literalinclude} ../examples/model/08_graphic_legacy.yaml
:language: yaml
```
