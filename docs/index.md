# MPAS-STAT

Verification toolkit for **MPAS–JEDI** forecasts and analyses: model-space
statistics against analyses, observation-space statistics (O−B, O−A,
O−F) against IODA observations, and an optional radar-reflectivity
product — with one shared graphics engine.

| | Model space | Observation space |
|---|---|---|
| Verifies | MPAS forecast vs. its verifying analysis, on the model grid | IODA O−B / O−A / O−F, at observation locations |
| Statistics | area/mass-weighted RMS, mean, STD, spread by region, level and lead | the same on one common observation sample, plus error-consistency and ensemble diagnostics |
| Figures | time series, profiles, cross-sections, maps, forecast scorecard | the same plus scorecards, rank histograms, normalized departures, channel correlations |

Both pipelines have two stages: a **statistics stage** writes one netCDF
file per experiment and cycle, and a **graphics stage** reads them back,
pools the cycles and draws the figures.

**Where to start**

- New here: {doc}`installation`, then {doc}`quickstart`.
- See what it draws: the {doc}`gallery`.
- Set up a run: pick a configuration from {doc}`examples`.
- Read a figure: {doc}`guide/conventions` (§3.2) and {doc}`guide/methods`.
- Something looks wrong: {doc}`troubleshooting`.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
gallery
examples
```

```{toctree}
:maxdepth: 2
:caption: User guide

guide/overview
guide/model_space
guide/obs_space
guide/radar
guide/conventions
guide/layout
guide/methods
```

```{toctree}
:maxdepth: 1
:caption: Reference

configuration
troubleshooting
glossary
api/index
```

```{toctree}
:maxdepth: 1
:caption: Development

development
```
