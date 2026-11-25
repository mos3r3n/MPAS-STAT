# Overview

This repository contains two verification pipelines that share one graphics
engine:

See [`README.md`](../installation.md) for installation, quick-start commands, and
the repository layout; this guide is the detailed reference.

| | Model-space | Observation-space |
|---|---|---|
| Verifies | MPAS forecast vs. init, on the model grid | IODA obs-space OMB/OMA/OMF, at observation locations |
| Input | MPAS `forecast`/`init` netCDF files | IODA `obsout_da_*.h5` / `obsout_hofx_*.h5` files |
| Statistics entry point | `src/mpas_stat/cli/model_stats.py` | `src/mpas_stat/obs/diag.py` |
| Graphics entry point | `src/mpas_stat/cli/model_graphics.py` | `src/mpas_stat/obs/plot.py` |
| PBS job script(s) | `scripts/run_verify_model.ksh`, `scripts/run_verify_model_fcst.ksh` | `scripts/run_verify_obs.ksh`, `scripts/run_verify_obs_fcst.ksh` |
| Diagnosis file | `mpasstat.<date>.nc` | `obsstat_<obs>.<date>.nc` |
| Extended-forecast skill | ✓ (`forecast_lead_0d`, `ExtendedFC/`) | ✓ (`hofx_forecast` experiments, `HofX/<cycle>/f<HHH>/`) |

Both pipelines follow the same two-stage design, and the observation-space
side draws its figures by calling the model-space side's own plotting
functions directly (not a reimplementation) — see
[Shared conventions](conventions.md).

```
 STATISTICS STAGE                        GRAPHICS STAGE
 (reads raw model/obs files,       ->     (reads diagnosis netCDF files,
  writes one netCDF file per              draws one PNG/PDF per figure via
  experiment x cycle)                     src/mpas_stat/graphics/functions.py)
```


