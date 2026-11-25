# Quick start

Every task follows the same two steps: compute statistics, then draw the
figures. Ready-made configurations for common tasks are in {doc}`examples`.

## Model space

```bash
cp configs/model_input.yaml my_input.yaml
cp configs/model_graphic.yaml my_graphic.yaml

# statistics for one experiment x cycle
python -m mpas_stat.cli.model_stats -c my_input.yaml

# figures for the configured experiments and cycle range
python -m mpas_stat.cli.model_graphics -c my_graphic.yaml
```

`my_input.yaml` contains placeholders (`forecastFile`, `initFile`,
`outputFile`, `nMember`, `nProcesses`) that the PBS driver fills in with
`sed`; replace them by hand for a one-off run.
`scripts/run_verify_model.ksh` loops the statistics stage over a cycle range;
`scripts/run_verify_model_fcst.ksh` does the same for extended forecasts,
writing to `f<HHH>/` sub-directories.

## Observation space (O−B / O−A)

One YAML file drives both stages:

```bash
python -m mpas_stat.cli.obs_stats    examples/obs/01_compare_two_experiments.yaml
python -m mpas_stat.cli.obs_graphics examples/obs/01_compare_two_experiments.yaml
```

For production runs, `scripts/run_verify_obs.ksh` runs the statistics per
experiment and per cycle from the template `configs/obs_input.yaml`, writing
`<main_dir>/<exp>/VerifyOBS/<cycle>/obsstat_<obs>.<date>.nc`; then draw the
figures for all experiments together:

```bash
python -m mpas_stat.cli.obs_graphics -c configs/obs_graphic.yaml
```

Each per-experiment job also reads the experiments listed in
`compare_exps`, so that all experiments are verified on one common
observation sample, but writes only its own files.

## Observation-space forecast skill (O−F)

Use experiments with `types: [hofx_forecast]` and a `forecast:` block, e.g.
`examples/obs/05_extended_forecast_omf.yaml`.
`scripts/run_verify_obs_fcst.ksh` loops over initialization cycles and
forecast hours, writing to `<main_dir>/<exp>/VerifyOBS/<init>/f<HHH>/`.

## Radar composite reflectivity (optional)

```bash
python -m mpas_stat.cli.radar_stats    my_radar_config.yaml
python -m mpas_stat.cli.radar_graphics my_radar_config.yaml
```

Opt-in: nothing happens unless the configuration has a top-level `radar:`
block. Requires a working METplus installation; see {doc}`guide/radar`.

## PBS jobs

```bash
qsub scripts/run_verify_model.ksh        # cycling model-space statistics
qsub scripts/run_verify_model_fcst.ksh   # extended-forecast model-space statistics
qsub scripts/run_verify_obs.ksh          # O-B/O-A statistics, per experiment and cycle
qsub scripts/run_verify_obs_fcst.ksh     # O-F statistics, per experiment, cycle and lead
python -m mpas_stat.cli.model_graphics -c configs/model_graphic.yaml
python -m mpas_stat.cli.obs_graphics   -c configs/obs_graphic.yaml
```

Edit the `export ...` lines at the top of each script for your experiments,
directories, sensors and cycle range, and set `#PBS -P` to your allocation.

## Where the output goes

```
<base_dir>/<exp>/VerifyModel/<cycle>/mpasstat.<date>.nc
<diag_dir>/<exp>/VerifyOBS/<cycle>/obsstat_<obs>.<date>.nc
<diag_dir>/<exp>/VerifyOBS/<init_cycle>/f<HHH>/obsstat_<obs>.<date>.nc
<diag_dir>/<exp>/VerifyOBS/ObsErrorEstimates/desroziers_*.{csv,yaml}
<graphic_dir>/<sensor>/...           observation-space figures
<graphic_dir>/scorecard/...          observation-space scorecards
```
