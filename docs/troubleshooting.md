# Troubleshooting

Common messages and surprises, with what to check. Messages are printed
to the log of the stage that hits them.

## No figures, or figures with missing experiments

`WARNING: missing file: <path>` (model space)
: The graphics stage looked for a statistics file that is not there.
  Compare the path with what the statistics stage wrote: check
  `paths.base_dir`, the experiment names and the cycle range in
  `time`. A missing cycle is skipped; a missing experiment leaves its
  panels empty.

`WARNING: no diagnosis files found for <sensor> under <dir>; run the statistics stage … first. Skipping.`
: No experiment has any statistics file for that sensor. Run
  `mpas-stat-obs` (or the PBS script) for it, and check that `diag_dir`
  is the same in both configurations.

`WARNING: no valid experiments for <stat>`
: Every experiment's value is missing for that statistic, often because
  the variable or statistic is not in the files (e.g. `spread` from a
  deterministic run).

`ERROR in <job>:` followed by a traceback
: One figure failed; the others are still drawn. The traceback names the
  cause.

## Comparisons

The right-hand panels or the difference plots are empty
: Comparisons need at least two experiments; the first is the reference.
  Cells where the reference is close to zero are blanked
  (`relative_min_reference`, and `qv_relative_min_reference` for
  water vapour).

No ▲/▼ significance markers
: Bootstrap intervals need at least two cycles. The observation-space
  scorecard also needs `scorecard.min_cycles` (default 5).
  Cycle–level sections (`time_level_2d`) never show significance, since
  each column is one cycle.

`WARNING: common sample … keeps only N of M valid observations`
: The compared runs share few observations, for example because they
  assimilated different data or use different files. Check the
  experiments, or set `statistics.common_sample: false` to verify each on
  its own sample (comparisons are then less fair, §5.4).

## Observation space

No O−A
: O−A needs `types: [da]` (not `hofx`), `plot_settings.plot_oma: true`
  and the right `outerloop`. For ensembles O−A is read from
  `obsout_da_<obs>_ana.h5` (§2.7).

No error-consistency figures or error tables
: They need a `da` experiment with O−A, `statistics.method: weighted`
  and `error_diagnostics: true`. Surface observations have no profile
  figure.

No rank histograms, spread–skill or spread figures
: These need an ensemble (`n_member > 1`). For a deterministic run they
  are not written or drawn by design (§5.3).

No `continuous/` figures, CSV or `.stat` files
: They need `statistics.met_statistics` (and `met_stat_files` for the
  `.stat` files) with `method: weighted`, and statistics files written
  with them on: re-run the statistics stage after turning them on. GNSS-RO
  has no partial sums.

Radiosondes appear only every 12 hours
: Radiosondes report at 00 and 12 UTC (`cycle_valid_hours`). In O−F, leads
  that are not valid at those hours are empty (§2.7).

`WARNING: plot_settings.latlon_domain is 'auto' for a single-cycle run …`
: Per-cycle PBS runs would each find a different map box, and the
  graphics stage cannot combine them. Use a named box from `domains`.

`WARNING: latlon_domain: auto found no valid observations … falling back to full-globe bounds`
: No observation of that sensor passed QC in the scanned files; check the
  input paths and cycle range.

Very large O−F values in a regional run
: Should not happen: missing HofX values (NaN, Inf or fill values) are
  dropped. If you see it, check that the files are the expected HofX
  output and report it.

## Maps

Maps without coastlines or projection
: cartopy is not installed; maps fall back to plain axes. Install the
  `maps` extra (`pip install -e ".[maps]"`).

The map shows a small patch in a large frame
: Set `map_extent: auto` (the default) to zoom to the data, or give a
  box.

## Running

`GEOMETRIC_DATA_DIR: unbound variable` (or another unbound variable) in a PBS job
: A conda activation hook ran under `set -u`. Activate conda in the
  scripts' "Python environment" section, which relaxes `set -u`.

Statistics are slow
: Install `numba` (the `accelerate` extra) for vertical interpolation, and
  raise `n_processes` (model) or `n_workers` (observation graphics).

Radar: nothing happens
: The radar product runs only when the configuration has a top-level
  `radar:` block. If `run_metplus.py` is not found, set
  `radar.run_metplus_cmd` (§2.6).
