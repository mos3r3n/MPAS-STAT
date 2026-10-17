# MPAS-STAT

Verification toolkit for regional **MPAS–JEDI** forecasts and analyses.

MPAS-STAT provides two verification pipelines that share one graphics
engine:

| | Model-space | Observation-space |
|---|---|---|
| Verifies | MPAS forecast vs. its initial/analysis field, on the model grid | IODA obs-space OMB/OMA/OMF, at observation locations |
| Input | MPAS `forecast` / `init` netCDF files | IODA `obsout_da_*.h5` / `obsout_hofx_*.h5` |
| Statistics entry point | `mpas_stats_diag_model.py` | `diag_ioda_obsout_stats.py` |
| Graphics entry point | `mpas_stats_graphic.py` | `plot_ioda_obsout_stats.py` |
| PBS driver | `run_verify_model.ksh`, `run_verify_model_fcst.ksh` | `run_verify_obs.ksh`, `run_verify_obs_fcst.ksh` |
| Diagnosis file | `mpasstat.<date>.nc` | `obsstat_<obs>.<date>.nc` |
| Extended-forecast skill | ✓ | ✓ |

Both pipelines follow a two-stage design: a Python statistics stage writes
one netCDF diagnosis file per experiment/cycle (plus one per forecast lead
for extended forecasts), and a graphics stage reads those files back to draw
publication-style figures.

An optional radar composite-reflectivity product lives in
`radar_verify_common.py`, `diag_radar_reflectivity.py`, and
`plot_radar_reflectivity.py`. It uses METplus/`grid_stat` for neighborhood
scores and is documented in the user guide; the original all-Python
standalone script is kept under `legacy/` for reference.

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
pip install -r requirements.txt
```

The runtime modules are intentionally kept flat at the repository root so
the PBS scripts can symlink `*.py` into a per-cycle working directory. The
historical `mpas_stats_` module prefix is retained for compatibility.

## Quick start

### 1. Model-space verification

Edit `input.yaml` (paths, variables, bin sizes, levels) and `graphic.yaml`
(experiments, time range, plot types), then:

```bash
# Statistics for one experiment x cycle
python mpas_stats_diag_model.py -c input.yaml

# Figures for the configured experiment/cycle range
python mpas_stats_graphic.py -c graphic.yaml
```

`run_verify_model.ksh` loops the statistics stage over a cycle range and
fills in `input.yaml`'s `forecastFile` / `initFile` / `outputFile`
placeholders before each call. `run_verify_model_fcst.ksh` does the same for
extended forecasts, writing to `f<HHH>/` subdirectories.

### 2. Observation-space verification (OMB/OMA)

The observation-space stages read one shared YAML config:

```bash
python diag_ioda_obsout_stats.py examples/obs_verify_example.yaml
python plot_ioda_obsout_stats.py examples/obs_verify_example.yaml
```

`run_verify_obs.ksh` generates that config from its exported settings and
runs both stages once for the whole configured cycle range.

### 3. Observation-space forecast skill (OMF)

Use an experiment with `type: hofx_forecast` and a `forecast:` block; see
`examples/obs_forecast_example.yaml`. Then run the same two Python stages
(or `run_verify_obs_fcst.ksh`).

### 4. Radar composite reflectivity (optional)

```bash
python diag_radar_reflectivity.py my_radar_config.yaml
python plot_radar_reflectivity.py my_radar_config.yaml
```

This product is opt-in: it is a no-op unless the config contains a top-level
`radar:` block. It requires a working METplus installation and is marked
experimental until validated on your system. See the user guide, §2.6.

### PBS jobs

```bash
qsub run_verify_model.ksh        # cycling model-space stats
qsub run_verify_model_fcst.ksh   # extended-forecast model-space stats
qsub run_verify_obs.ksh          # OMB/OMA stats + figures
qsub run_verify_obs_fcst.ksh     # OMF stats + figures
```

Edit the `export ...` lines at the top of each script for your experiments,
directories, sensors, and cycle range, and replace the
`#PBS -P your_project_id` directive with your own PBS allocation ID.

## Repository layout

```
MPAS-STAT/
├── README.md
├── LICENSE
├── requirements.txt / environment.yml
├── input.yaml, graphic.yaml          # model-space configuration
├── mpas_stats_diag_*.py              # model-space statistics stage
├── mpas_stats_graphic_*.py           # shared graphics engine
├── obs_sensor_config.py              # obs-type metadata registry
├── obsstat_common.py                 # obs-space config/path conventions
├── diag_ioda_obsout_stats.py         # obs-space statistics stage
├── plot_ioda_obsout_stats.py         # obs-space graphics stage
├── radar_*.py                        # optional radar product
├── run_verify_*.ksh                  # PBS drivers
├── docs/USER_GUIDE.md                # full documentation
├── examples/                         # annotated configs
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

The smoke tests cover bin-center/edge consistency, domain masking with
missing QC, the percent-reduction sign convention, pressure-level binning,
and the Python 3.9 import path. They need `numpy`, `netCDF4`, `h5py`, and
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
