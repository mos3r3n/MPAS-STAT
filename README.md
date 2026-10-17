# MPAS-STAT

Verification toolkit for regional **MPAS–JEDI** forecasts and analyses.

MPAS-STAT provides two verification pipelines that share one graphics
engine:

| | Model-space | Observation-space |
|---|---|---|
| Verifies | MPAS forecast vs. its initial/analysis field, on the model grid | IODA obs-space OMB/OMA/OMF, at observation locations |
| Input | MPAS `forecast` / `init` netCDF files | IODA `obsout_da_*.h5` / `obsout_hofx_*.h5` |
| Statistics | `mpas_stat.cli.model_stats` | `mpas_stat.cli.obs_stats` |
| Graphics | `mpas_stat.cli.model_graphics` | `mpas_stat.cli.obs_graphics` |
| PBS driver | `scripts/run_verify_model*.ksh` | `scripts/run_verify_obs*.ksh` |
| Diagnosis file | `mpasstat.<date>.nc` | `obsstat_<obs>.<date>.nc` |
| Extended-forecast skill | ✓ | ✓ |

Both pipelines follow a two-stage design: a statistics stage writes one
netCDF diagnosis file per experiment/cycle (plus one per forecast lead for
extended forecasts), and a graphics stage reads those files back to draw
publication-style figures.

An optional radar composite-reflectivity product lives in
`src/mpas_stat/radar/`. It uses METplus/`grid_stat` for neighborhood scores
and is documented in the user guide; the original all-Python standalone
script is kept under `legacy/` for reference.

## Requirements

- Python 3.9 or newer
- Core: `numpy`, `netCDF4`, `matplotlib`, `PyYAML`
- Observation-space: `h5py`; radar product: `scipy`
- Optional: `numba` (accelerated vertical interpolation), `cartopy`
  (map projections), MET/METplus (radar neighborhood scores)

## Installation

```bash
git clone https://github.com/<your-user>/MPAS-STAT.git
cd MPAS-STAT

# Option A: conda
conda env create -f environment.yml
conda activate mpas-stat

# Option B: pip/venv
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

The package uses a `src/` layout. After installation, the commands
`mpas-stat-model`, `mpas-stat-graphics`, `mpas-stat-obs`,
`mpas-stat-obs-plot`, `mpas-stat-radar`, and `mpas-stat-radar-plot` are
available. You can also run any stage without installing:

```bash
PYTHONPATH=src python -m mpas_stat.cli.model_stats --help
```

## Quick start

### 1. Model-space verification

Copy and edit the model configuration:

```bash
cp configs/model_input.yaml my_input.yaml
cp configs/model_graphic.yaml my_graphic.yaml
```

`my_input.yaml` contains placeholders (`forecastFile`, `initFile`,
`outputFile`, `nMember`, `nProcesses`) that the PBS driver fills in with
`sed`; replace them manually for a one-off run.

```bash
# Statistics for one experiment x cycle
python -m mpas_stat.cli.model_stats -c my_input.yaml

# Figures for the configured experiment/cycle range
python -m mpas_stat.cli.model_graphics -c my_graphic.yaml
```

`scripts/run_verify_model.ksh` loops the statistics stage over a cycle range
and fills in the placeholders before each call.
`scripts/run_verify_model_fcst.ksh` does the same for extended forecasts,
writing to `f<HHH>/` subdirectories.

### 2. Observation-space verification (OMB/OMA)

The observation-space stages read one shared YAML config:

```bash
python -m mpas_stat.cli.obs_stats examples/obs_verify_example.yaml
python -m mpas_stat.cli.obs_graphics examples/obs_verify_example.yaml
```

`scripts/run_verify_obs.ksh` generates that config from its exported
settings and runs both stages once for the whole configured cycle range.

### 3. Observation-space forecast skill (OMF)

Use an experiment with `types: [hofx_forecast]` and a `forecast:` block; see
`examples/obs_forecast_example.yaml`. Then run the same two stages (or
`scripts/run_verify_obs_fcst.ksh`).

### 4. Radar composite reflectivity (optional)

```bash
python -m mpas_stat.cli.radar_stats my_radar_config.yaml
python -m mpas_stat.cli.radar_graphics my_radar_config.yaml
```

This product is opt-in: it is a no-op unless the config contains a top-level
`radar:` block. It requires a working METplus installation and is marked
experimental until validated on your system. See the user guide, §2.6.

### PBS jobs

```bash
qsub scripts/run_verify_model.ksh        # cycling model-space stats
qsub scripts/run_verify_model_fcst.ksh   # extended-forecast model-space stats
qsub scripts/run_verify_obs.ksh          # OMB/OMA stats + figures
qsub scripts/run_verify_obs_fcst.ksh     # OMF stats + figures
```

Edit the `export ...` lines at the top of each script for your experiments,
directories, sensors, and cycle range, and replace the
`#PBS -P your_project_id` directive with your own PBS allocation ID. The
scripts set `PYTHONPATH` to the repository's `src/` directory, so no install
step is required on the compute node.

## Repository layout

```
MPAS-STAT/
├── README.md
├── LICENSE
├── pyproject.toml, MANIFEST.in
├── requirements.txt / environment.yml
├── configs/
│   ├── model_input.yaml              # model-space statistics config
│   └── model_graphic.yaml            # model-space graphics config
├── scripts/
│   └── run_verify_*.ksh              # PBS drivers
├── src/mpas_stat/
│   ├── model/                        # model-space stats, interp, kernels
│   ├── graphics/                     # shared plotting engine
│   ├── obs/                          # IODA obs-space stats + figures
│   ├── radar/                        # optional radar product
│   └── cli/                          # command-line entry points
├── docs/USER_GUIDE.md                # full documentation
├── examples/                         # annotated obs-space configs
├── tests/test_smoke.py               # unit tests
└── legacy/radar_reflectivity_verify.py
```

Diagnosis-file conventions:

```
<base_dir>/<exp>/VerifyModel/<cycle>/mpasstat.<date>.nc
<diag_dir>/<exp>/VerifyObs/<cycle>/obsstat_<obs>.<date>.nc
<diag_dir>/<exp>/VerifyObs/<init_cycle>/f<HHH>/obsstat_<obs>.<date>.nc
```

## Documentation

The full guide — config schema, statistics definitions, netCDF schema,
plot types, shared conventions, radar/METplus details, and known
limitations — is in [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

## Testing

```bash
python -m unittest discover -s tests -v
```

The tests insert `src/` on `sys.path`, so they run from the repository root
without installing. They cover bin-center/edge consistency, domain masking
with missing QC, the percent-reduction sign convention, pressure-level
binning, config merging, HDF5 file resolution, `ObsType` layouts, and the
Python 3.9 import path. They need `numpy`, `netCDF4`, `h5py`, and
`matplotlib`; they do not require input data or METplus.

## Status and known limitations

- Percent-reduction/difference figures use **positive = improvement**.
- Percent differences are suppressed when the reference value is below the
  configured `relative_min_reference` threshold (and a separate, larger
  threshold for water vapour).
- Vertical coordinates (pressure, height, model level) are drawn as
  equally spaced categorical axes; the original values are kept as tick
  labels.
- Axis/legend labels show statistic names, not physical units.
- The radar METplus/`grid_stat` integration has not been validated against
  every MET version; run one cycle and inspect the logs before production
  use.
- The `legacy/` standalone radar script is reference code and is not
  imported by the pipeline.

Bug reports and pull requests are welcome. Please include the smallest
config and the exact command that reproduces an issue.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
