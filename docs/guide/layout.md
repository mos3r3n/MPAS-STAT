# 4. Repository layout

```
MPAS-STAT/
├── configs/                    annotated configuration templates
│   ├── model_input.yaml        model space: statistics
│   ├── model_graphic.yaml      model space: figures
│   ├── obs_input.yaml          observation space: statistics (per cycle, PBS)
│   └── obs_graphic.yaml        observation space: figures and error tables
├── examples/                   ready-made configurations by task (obs/, model/)
├── scripts/                    PBS jobs for the statistics stage
│   ├── run_verify_model.ksh          cycling, model space
│   ├── run_verify_model_fcst.ksh     extended forecasts, model space
│   ├── run_verify_obs.ksh            cycling (O−B, O−A)
│   └── run_verify_obs_fcst.ksh       extended forecasts (O−F)
├── src/mpas_stat/
│   ├── cli/                    command-line entry points
│   ├── core/                   shared: pooling, bootstrap, grouping, time, job runner
│   ├── model/                  model-space statistics
│   ├── obs/                    observation-space statistics and figures
│   ├── graphics/               plotting engine shared by both sides
│   └── radar/                  optional radar reflectivity product
├── tests/                      unit tests (pytest), synthetic data only
├── docs/                       this documentation (Sphinx)
└── legacy/                     stand-alone radar script, for reference
```

Each subpackage is described in the {doc}`/api/index`; the code maps at
the end of §1 and §2 list the main modules of each pipeline.
