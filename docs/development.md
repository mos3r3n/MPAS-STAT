# Development

## Set up

```bash
pip install -e ".[dev,docs]"
```

## Tests and lint

```bash
PYTHONPATH=src python -m pytest -q tests
python -m pyflakes src tests
```

The suite builds small synthetic MPAS and IODA files, so it needs no real
data. `tests/test_examples.py` loads every YAML in `examples/` and
`configs/`, so a renamed option fails the tests until the examples follow.

## Continuous integration

`.github/workflows/tests.yml` runs pyflakes, the test suite and every
command-line entry point on Python 3.10 and 3.13 for every push and pull
request. `.github/workflows/docs.yml` builds this documentation with
warnings treated as errors and, on `main`, publishes it to GitHub Pages
(enable once under Settings → Pages → Source: GitHub Actions).
`.readthedocs.yaml` lets Read the Docs build the same site.

## Documentation

```bash
sphinx-build -W -b html docs docs/_build/html
```

Pages are Markdown (MyST). The API reference is generated from docstrings
(NumPy or Google style, via napoleon), so new public functions should have
one. The configuration and example pages include the YAML files directly,
so they never go stale.

## Code layout

See {doc}`guide/layout`. In short: `core/` holds statistics, pooling and
I/O shared by every pipeline; `model/`, `obs/` and `radar/` are the
statistics stages; `graphics/` draws the figures; `cli/` has the thin
command-line wrappers.

## Conventions

- Pools are `{kind, w, m1, m2}` (+ optional `cycles`); combine them with
  the functions in `mpas_stat.core.pooling` rather than averaging
  statistics.
- Paired comparisons align experiments by cycle and on a common sample
  (`align_pools`); never compare by array position.
- Missing inputs are reported, not silently skipped; catch only
  `MISSING_DATA_ERRORS` around reads.
- Labels follow the NCAR mpas-jedi-graphics style: `T (K)`, `rms`,
  `rms: % diff. from Control`.
