# Installation

## Requirements

- Python 3.9 or newer (CI tests 3.10 and 3.13)
- Core: `numpy`, `netCDF4`, `matplotlib`, `PyYAML`
- Observation space: `h5py`; radar product: `scipy`
- Optional: `numba` (faster vertical interpolation), `cartopy` (map
  projections and coastlines), MET/METplus (radar neighbourhood scores)

## Install

```bash
git clone https://github.com/mos3r3n/MPAS-STAT.git
cd MPAS-STAT

# Option A: conda
conda env create -f environment.yml
conda activate mpas-stat

# Option B: pip / venv
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .            # add ".[dev]" for the test and lint tools
```

The package uses a `src/` layout. Installing provides these commands:

| Command | Stage |
|---|---|
| `mpas-stat-model` | model-space statistics |
| `mpas-stat-graphics` | model-space figures |
| `mpas-stat-obs` | observation-space statistics |
| `mpas-stat-obs-plot` | observation-space figures (+ scorecards, error tables) |
| `mpas-stat-obs-errors` | Desroziers observation-error tables only |
| `mpas-stat-radar` | radar reflectivity statistics |
| `mpas-stat-radar-plot` | radar reflectivity figures |

Every stage also runs without installing:

```bash
PYTHONPATH=src python -m mpas_stat.cli.model_stats --help
```

## On a PBS cluster

The batch drivers in `scripts/` set `PYTHONPATH` to the repository's `src/`
directory, so no install step is needed on compute nodes. If a job activates
a conda environment, keep the activation inside the scripts' "Python
environment" section: it relaxes `set -u`, which conda activation hooks
often trip over (e.g. `GEOMETRIC_DATA_DIR: unbound variable`).

## Building this documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```
