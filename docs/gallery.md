# Figure gallery

A visual index of what MPAS-STAT draws. Each figure links to the section
that explains it and the setting that turns it on.

```{note}
Drawn by the current code from **synthetic** data, so the numbers mean
nothing. Observation space compares a deterministic *Control* with a
10-member ensemble *Test* over twelve 6-hourly cycles. Model space uses
the small regional case from the test suite.
```

## Observation space

| | | |
|:---:|:---:|:---:|
| [![](gallery/obs_time_series.png)](guide/obs_space.md#time-series-and-profiles) | [![](gallery/obs_profile.png)](guide/obs_space.md#time-series-and-profiles) | [![](gallery/obs_radiance_ts.png)](guide/obs_space.md#time-series-and-profiles) |
| O−B / O−A time series | O−B / O−A profiles | Radiance channels |
| [![](gallery/obs_profile_diff.png)](guide/obs_space.md#comparison-with-the-reference) | [![](gallery/obs_lat_lon.png)](guide/obs_space.md#comparison-with-the-reference) | [![](gallery/obs_lat_vert.png)](guide/obs_space.md#comparison-with-the-reference) |
| % difference profile | Map | Latitude–pressure section |
| [![](gallery/obs_time_level.png)](guide/obs_space.md#comparison-with-the-reference) | [![](gallery/obs_scorecard.png)](guide/obs_space.md#scorecard) | [![](gallery/obs_omf_lead.png)](guide/obs_space.md#forecasts-of) |
| Cycle–pressure section | Scorecard | O−F against lead |

## Error and ensemble diagnostics

| | | |
|:---:|:---:|:---:|
| [![](gallery/obs_error_consistency.png)](guide/methods.md#56-observation-and-background-error-consistency) | [![](gallery/obs_normdep_hist.png)](guide/methods.md#58-normalized-departures) | [![](gallery/obs_channel_corr.png)](guide/methods.md#59-inter-channel-observation-error-correlations-radiances) |
| Desroziers error consistency | Normalized departures | Channel error correlations |
| [![](gallery/obs_rank_histogram.png)](guide/methods.md#57-ensemble-reliability-rank-histograms-and-spread-skill) | [![](gallery/obs_spread_skill.png)](guide/methods.md#57-ensemble-reliability-rank-histograms-and-spread-skill) | [![](gallery/obs_normdep_profile.png)](guide/methods.md#58-normalized-departures) |
| Rank histograms | Spread–skill | Normalized-departure profiles |

## Model space

| | | |
|:---:|:---:|:---:|
| [![](gallery/model_time_series.png)](guide/model_space.md#cycling) | [![](gallery/model_profile.png)](guide/model_space.md#cycling) | [![](gallery/model_profile_reduction.png)](guide/model_space.md#cycling) |
| `time_series_0d` | `profiles_1d` | `profile_diff_1d` |
| [![](gallery/model_lat_lon.png)](guide/model_space.md#cycling) | [![](gallery/model_forecast_lead.png)](guide/model_space.md#extended-forecasts) | [![](gallery/model_forecast_scorecard.png)](guide/model_space.md#extended-forecasts) |
| `lat_lon_slice_2d` | `forecast_lead_0d` | `forecast_scorecard` |
