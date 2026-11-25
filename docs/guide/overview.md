# Overview

MPAS-STAT verifies MPAS–JEDI forecasts and analyses in two ways.

| | Model space (§1) | Observation space (§2) |
|---|---|---|
| Question | How far is the forecast from the verifying analysis? | How far are the background, analysis and forecast from the observations? |
| Input | MPAS `forecast` and `init` netCDF files | IODA `obsout_da_*.h5` / `obsout_hofx_*.h5` files |
| Samples | model grid cells, by region, latitude and level | observations, by domain, pressure/height/latitude bin and lat-lon box |
| Statistics file | `mpasstat.<date>.nc` | `obsstat_<obs>.<date>.nc` |
| Extended forecasts | ✓ | ✓ (O−F) |

An optional radar composite-reflectivity product (§2.6) scores
reflectivity forecasts with neighbourhood and categorical scores from MET.

## Two stages

Every pipeline runs in two stages, so the expensive part is done once per
cycle and figures can be redrawn as often as needed:

```
 statistics stage                             graphics stage
 raw model / IODA files      ──────────►      statistics files     ──────────►   figures
 (one run per experiment                      (one per experiment                (all experiments
  and cycle; PBS scripts)                      and cycle)                         and cycles together)
```

1. **Statistics.** Read one experiment's files for one cycle (and lead),
   compute every statistic, write one netCDF file.
2. **Graphics.** Read the files of all experiments over a period, pool the
   cycles (§5.2), compare the experiments with bootstrap confidence
   intervals and draw the figures.

Both sides write statistics files with the same group layout and draw them
with the same plotting code, so figures look and read the same way (§3).

## Where to go next

- Run it: {doc}`/installation`, {doc}`/quickstart` and the ready-made
  {doc}`/examples`.
- See what it draws: the {doc}`/gallery`.
- Configure and read the figures: §1 {doc}`model_space`, §2
  {doc}`obs_space`, §3 {doc}`conventions`.
- Understand the numbers: §5 {doc}`methods`.
