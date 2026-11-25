# Figure gallery

A selection of the figures MPAS-STAT draws, with the setting that turns
each one on. Every family is drawn per domain (global, tropics, …), per
statistic (rms, mean, STD, …) and, for forecasts, per lead.

```{note}
These figures were made from **synthetic** data with the current code, so
the numbers mean nothing. The observation-space set compares a
deterministic *Control* with a 10-member ensemble *Test* over twelve
6-hourly cycles. The model-space set uses the small regional test case
from the regression suite.
```

## Observation space (O−B / O−A)

All of these come from one configuration such as
`examples/obs/01_compare_two_experiments.yaml` or, for the ensemble
diagnostics, `examples/obs/03_enkf_ensemble.yaml`. O−B is drawn solid and
O−A dashed, with one colour per experiment.

### Time series

```{figure} gallery/obs_time_series.png
:width: 80%

Domain-averaged rms of O−B and O−A for radiosondes, one panel per
variable, by cycle. Always drawn.
```

```{figure} gallery/obs_radiance_ts.png
:width: 90%

The same for a microwave radiometer (AMSU-A on NOAA-18), one panel per
channel.
```

### Vertical profiles

```{figure} gallery/obs_profile.png
:width: 60%

rms profiles of O−B (solid) and O−A (dashed) by pressure bin, pooled over
all cycles. Always drawn.
```

```{figure} gallery/obs_profile_diff.png
:width: 60%

O−B rms of *Test* as a percent difference from *Control*
(`100·(exp − ref)/ref`), with the 95 % paired-bootstrap interval over
cycles; ▲/▼ mark significant improvements and degradations. O−A is drawn
in a separate figure. Set by
`plot_settings.comparison_relative_reduction_statistics`.
```

### Maps and cross-sections

The first experiment is drawn in full and the others as percent
differences from it, with ▲/▼ on significant cells. O−B and O−A go to
separate `omb/` and `oma/` folders.

```{figure} gallery/obs_lat_lon.png
:width: 70%

Lat-lon map of O−B rms in 10° boxes. Always drawn; the extent follows the
data (`plot_settings.map_extent`).
```

```{figure} gallery/obs_lat_vert.png
:width: 70%

Latitude–pressure cross-section. `plot_settings.generate_lat_vert_2d`.
```

```{figure} gallery/obs_time_level.png
:width: 70%

Cycle–pressure cross-section. `plot_settings.generate_time_level_2d`.
```

### Scorecard

```{figure} gallery/obs_scorecard.png
:width: 60%

Every sensor, variable and level on one page: percent difference of the
cycle-pooled O−B rms from *Control*, by domain. Blue is better, and ▲/▼
mark significant cells. A CSV with every cell is written alongside.
`plot_settings.scorecard`.
```

### Error diagnostics

```{figure} gallery/obs_error_consistency.png
:width: 100%

Error consistency (Desroziers). Top: the observation-error standard
deviation assumed by the DA, compared with the Desroziers estimates
of σo and σb. Bottom: the ratio of estimated to assumed σo, and the
ratio of total spread to rms(O−B); both should be 1. The matching
observation-error tables are written as CSV/YAML.
`plot_settings.generate_error_consistency`.
```

```{figure} gallery/obs_normdep_hist.png
:width: 80%

Normalized departures: histograms of (O−B)/σo, and of (O−B)/total spread
for ensembles, on a log scale against a Gaussian with the same rms. A
heavy tail or an rms far from 1 points to a mis-specified error.
`plot_settings.generate_normalized_departures`.
```

```{figure} gallery/obs_normdep_profile.png
:width: 100%

The same as profiles: the rms (top) and mean (bottom) of the normalized
departures by pressure bin.
```

```{figure} gallery/obs_channel_corr.png
:width: 70%

Radiances: inter-channel correlations of the observation and background
errors (Desroziers), with the smallest eigenvalue of the observation-error
matrix. `plot_settings.generate_channel_correlations`.
```

### Ensemble reliability

Drawn only for experiments with more than one member.
`plot_settings.generate_ensemble_diagnostics`.

```{figure} gallery/obs_rank_histogram.png
:width: 80%

Rank histograms, with observations perturbed by their error. Flat is
reliable, a U-shape is under-dispersive, a dome over-dispersive, and a
slope means a bias. The outlier ratio and reliability index are printed
in each panel.
```

```{figure} gallery/obs_spread_skill.png
:width: 70%

Spread–skill: rms(O−B) in bins of total spread (ensemble plus observation
error). Points above the diagonal mean the spread is too small.
```

## Model space

Drawn by `mpas-stat-graphics` from the `plot_types` list in the graphics
configuration (see `examples/model/`).

```{figure} gallery/model_time_series.png
:width: 80%

`time_series_0d`: domain-averaged rms of the forecast against the
verifying analysis, by cycle.
```

```{figure} gallery/model_profile.png
:width: 80%

`profiles_1d`: rms by pressure level, pooled over cycles, with bootstrap
intervals.
```

```{figure} gallery/model_profile_reduction.png
:width: 80%

`profile_diff_1d`: the same as a percent difference from the first
experiment, with ▼/▲ marking significant levels.
```

```{figure} gallery/model_lat_lon.png
:width: 55%

`lat_lon_2d`: rms at 500 hPa (any pressure, height or model level can be
requested), with the reference in full and the
other experiment as a percent difference. Cartopy coastlines are drawn
when cartopy is installed.
```

### Extended forecasts

```{figure} gallery/model_forecast_lead.png
:width: 80%

`forecast_lead_0d`: rms against forecast lead, pooled over
initialization cycles.
```

```{figure} gallery/model_forecast_lead_diff.png
:width: 80%

`forecast_lead_diff_0d`: the same as a percent difference from the
reference.
```

```{figure} gallery/model_forecast_scorecard.png
:width: 50%

`forecast_scorecard`: a triangle scorecard by variable, level and lead.
▲ is an improvement and ▼ a degradation; outlined cells are significant,
and marker size scales with the difference. The rows come from
`scorecard.rows`.
```
