# 4. Directory layout at a glance

```
MPAS-STAT/
├── README.md, LICENSE, pyproject.toml, MANIFEST.in
├── requirements.txt, environment.yml
├── docs/                                     # this documentation (Sphinx)
├── examples/                                 # example configs by use case: obs/, model/ (examples/README.md)
├── tests/                                    # unit tests (pytest)
├── configs/
│   ├── model_input.yaml                      # model-space statistics config
│   └── model_graphic.yaml                    # model-space graphics config
├── scripts/
│   ├── run_verify_model.ksh                  # model-space PBS job script
│   ├── run_verify_model_fcst.ksh             # extended-forecast PBS job script
│   ├── run_verify_obs.ksh                    # OMB/OMA PBS job script
│   └── run_verify_obs_fcst.ksh               # OMF PBS job script
│
├── src/mpas_stat/
│   ├── model/                                # model-space statistics stage
│   │   ├── args.py, config.py, driver.py
│   │   ├── interp.py, kernels.py, variables.py
│   ├── graphics/                             # shared graphics engine
│   │   ├── driver.py, functions.py, io.py
│   ├── obs/                                  # obs-space verification
│   │   ├── sensor_config.py                  # single sensor metadata registry
│   │   ├── common.py                         # config/path conventions
│   │   ├── diag.py                           # statistics stage
│   │   └── plot.py                           # graphics stage
│   ├── radar/                                # optional radar product
│   │   ├── common.py, diag.py, plot.py
│   │   └── metplus_embed.py                  # MET Python-embedding helper
│   └── cli/                                  # console/module entry points
│       ├── model_stats.py, model_graphics.py
│       ├── obs_stats.py, obs_graphics.py
│       └── radar_stats.py, radar_graphics.py
│
├── legacy/radar_reflectivity_verify.py       # original standalone radar script
│
└── <exp_name>/VerifyModel/<cycle>/mpasstat.<date>.nc              # model-space diagnosis files
    <diag_dir>/<exp_name>/VerifyOBS/<cycle>/obsstat_<sensor>.<date>.nc   # obs-space OMB/OMA diagnosis files
    <diag_dir>/<exp_name>/VerifyOBS/<init_cycle>/f<HHH>/obsstat_<sensor>.<date>.nc   # obs-space OMF diagnosis files
    <diag_dir>/<exp_name>/VerifyOBS/<init_cycle>/f<HHH>/radarstat_<sensor>.<date>.nc # radar composite-reflectivity diagnosis files
```

The experiment sits at the top of the obs-space tree too, exactly mirroring
the model-space `<exp_name>/VerifyModel/...` convention (`obs_diag_file()` /
`obs_diag_forecast_file()` in `src/mpas_stat/obs/common.py`) -- there is no separate
`<sensor>` directory level, since the filename itself
(`obsstat_<sensor>.<date>.nc`) already identifies the sensor. Every sensor's
files for one (experiment, cycle) pair land side by side in the same
`VerifyOBS/<cycle>/` directory, distinguished only by filename, and every
diagnosis file for one experiment lives under a single
`<diag_dir>/<exp_name>/` subtree, which is what lets that whole experiment's
obs-space output be archived, rsynced, or deleted as one unit.

> **Note for anyone with an existing `diag_dir` from before this
> convention.** Diagnosis files have gone through two now-obsolete older
> layouts: `<sensor>` first
> (`<diag_dir>/<sensor>/<exp_name>/VerifyOBS/...`), and later `<sensor>` as
> its own directory level under each experiment
> (`<diag_dir>/<exp_name>/VerifyOBS/<sensor>/<cycle>/...`). Until 2026-11 the
> folder was also spelled `VerifyObs`; rename it to `VerifyOBS` (or re-run the
> statistics) to use files written before then. None of these
> three layouts are compatible with each other on disk
> (`src/mpas_stat/obs/plot.py` looks for files under the current layout
> only). Either re-run `src/mpas_stat/obs/diag.py` to regenerate files under
> the current layout (cheap -- it's a read-IODA/write-netCDF pass, not a DA
> re-run), or move the existing files by hand.
