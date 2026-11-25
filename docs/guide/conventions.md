# 3. Shared conventions & how the two sides connect

**Figure labels** follow NCAR mpas-jedi-graphics conventions, in model and
observation space alike:
- **Panel titles:** the variable with plain-text units, e.g. `T (K)`,
  `U (m/s)`, `Q (g/kg)`, `Ps (Pa)`, `Channel 7 (K)`.
- **Axis and colour-bar labels:** the statistic, e.g. `rms`, `mean`, `STD`,
  `Count`, `spread`, `total spread`.
- **Comparisons:** `rms: % diff. from Control`.

Units come from the variable tables (`mpas_stat/model/variables.py`,
`mpas_stat/obs/sensor_config.py`). `mpas_stat/graphics/labels.py` converts
them to this notation.

The observation-space package was built to *reuse* the model-space package
rather than duplicate it:

- **Same diagnosis-file shape.** Both `mpasstat.<date>.nc` and
  `obsstat_<obs>.<date>.nc` use the identical nested-group convention:
  `<variable>/<0D|1D|2D>/<key>/<statistic>`. Anyone who knows how to read
  one file family with `ncdump`/Panoply immediately recognizes the other.
- **Same binning kernel.** The obs-space 2-D lat-lon gridding calls
  `grouped_stats_1d` straight out of `src/mpas_stat/model/kernels.py` — the
  exact routine the model side uses for its own `2D/lat_lon` product.
- **Same plotting functions.** `src/mpas_stat/obs/plot.py` imports
  `plot_time_series_0d`, `plot_profiles_1d`, `plot_lat_lon_2d`,
  `plot_time_level_2d`, `plot_lat_vert_2d`, `plot_forecast_lead_0d`, and
  `plot_forecast_time_level_2d` directly from `src/mpas_stat/graphics/functions.py`
  — obs-space contributes no plotting logic of its own. Figure styling, colorbars, panel layout,
  and filenames all come from the same code path as the model-space
  figures. One small bug fix WAS made to this shared file while wiring up
  the lat-lon map's real-experiment comparison convention (see the callout
  box in §2.3): `_draw_2d_panels`'s non-Cartopy fallback path (used when
  the `cartopy` package isn't installed) hardcoded `transpose=True` in its
  call to `add_significance`, which is correct for `plot_time_level_2d` /
  `plot_lat_vert_2d` (`transpose_data=True`) but wrong for `plot_lat_lon_2d`
  (`transpose_data=False`) — a shape mismatch that silently dropped
  significance markers (a printed warning, not a crash). Fixed by passing
  `transpose=transpose_data` instead of a hardcoded `True`; behavior is
  unchanged for every existing caller (all already had `transpose_data`
  match the hardcoded value) and for any environment with `cartopy`
  installed (that code path isn't taken). This is the first time obs-space
  exercises `plot_lat_lon_2d`'s relative-reduction/significance path at
  all, which is how the bug surfaced.
- **Same 2-D comparison convention.** The lat-lon/lev-time/lev-lat
  real-experiment comparison figures (§2.3) work by feeding
  `src/mpas_stat/graphics/functions.py` a `cfg["experiments"]` built from the
  REAL configured experiments (not OMB/OMA pseudo-experiments) and setting
  `cfg["relative_reduction_statistics"]` — the exact same mechanism
  (`is_relative_stat`, `_compute_panel_data`, `_draw_2d_panels`) the
  model-space package uses for its own cross-experiment 2-D comparisons.
  Obs-space did not need to add any new comparison/bootstrap logic of its
  own for this.
- **Same forecast-lead machinery.** The OMF "forecast skill" reader
  (`collect_obs_forecast_lead_0d`) calls `mpas_stat.graphics.io`'s own
  `generate_forecast_leads` and `bootstrap_mean_ci` — the same lead-hour
  list and bootstrap-CI aggregation the model side's
  `collect_forecast_lead_0d` uses for its `ExtendedFC` files — so `config`'s
  `forecast:` block means exactly the same thing on both sides.
- **Same OMF forecast-lead products, one level deeper.** The four
  `generate_forecast_*_graphics` functions (§2.4: forecast profile,
  lat-lon map, lev-lat cross-section, lev-time Hovmoller) are the obs-space
  counterparts of the model-space package's own `forecast_profiles_1d` /
  `forecast_lat_lon_2d` / `forecast_lat_vert_2d` / `forecast_time_level_2d`
  plot types (`src/mpas_stat/graphics/driver.py`), built the same way: three of
  the four (profile, lat-lon, lat-vert) just call their non-forecast
  `plot_*` function once per lead hour with a `_lead_suffix`-carrying cfg
  (`_lead_cfg`, ported from `mpas_stat.graphics.driver._lead_cfg` verbatim),
  landing in the SAME output folder as the non-forecast figures; the fourth
  (lev-time) is genuinely different, since averaging every initialization
  cycle into each lead-hour column makes bootstrap significance available
  where the plain per-cycle OMB/OMA Hovmoller has none — so it gets its own
  `plot_forecast_time_level_2d` call and output folder, exactly mirroring
  the model-space package's own `forecast_time_level_2d`/`time_level_2d`
  split.
- **Parallel plotting, at sensor granularity.** `src/mpas_stat/obs/plot.py`'s
  `_process_sensor()` collects (reads that sensor's diagnosis netCDF files)
  AND renders every figure for ONE sensor, and `main()` runs one
  `_process_sensor()` call per sensor — sequentially by default, or spread
  across `config["n_workers"]` worker processes via `ProcessPoolExecutor`
  (the same top-level `n_workers` config key the model-space package uses in
  `run_graphics()`, applied at sensor rather than figure granularity here).
  Each `generate_*_graphics` function still collects its data (netCDF reads)
  and returns a flat list of `(plot_func, args, kwargs)` jobs instead of
  calling `plot_*` directly (run by `_process_sensor()`'s ported
  `_run_job()` helper), but that collection happens per-sensor inside
  whichever process is handling that sensor, not up front for every sensor
  in the main process — so with `n_workers > 1`, different sensors'
  statistics are never all resident in memory at once, and one sensor's
  figures can start rendering before another sensor's diagnosis files have
  even been read. Sequential and parallel runs produce byte-identical
  figures (verified: `n_workers: 1` vs. `n_workers: 4` over the same
  2-sensor synthetic configs produced 84/84 and 132/132 pixel-identical
  PNGs, with wall time roughly halving to match the 2-worker/2-sensor
  split).
- **The "pseudo-experiment" trick.** `src/mpas_stat/graphics/functions.py`'s
  plotting functions draw one line/column per "experiment" — they have no
  built-in idea of "background vs. analysis". To reuse them unmodified,
  `src/mpas_stat/obs/plot.py` exposes each (real experiment, OMB|OMA) pair
  as its own pseudo-experiment — `"<label> - OMB"` (solid) and
  `"<label> - OMA"` (dashed), sharing the real experiment's colour — so the
  familiar OMB-solid/OMA-dashed convention falls out of the shared plotting
  code for free. OMF needs no such trick (`hofx_forecast` experiments map
  one-to-one onto `plot_forecast_lead_0d`'s notion of "experiment"), so
  `_build_forecast_experiments()` skips the doubling entirely.
- **Stat-name convention.** Obs-space stats are labelled `RMS`, `Bias`,
  `STD`, `Count`, `Spread`, `TotalSpread` to read consistently alongside the
  model side's `rms`, `mean`, `std`, `spread`.
- **One module for sensor metadata.** All per-sensor knowledge (variables,
  units, labels, bin method, and quirks like shared IODA files or
  non-standard cycle cadence) lives only in `src/mpas_stat/obs/sensor_config.py`. Neither
  pipeline stage hard-codes a sensor name anywhere else — adding a new obs
  type never requires touching `src/mpas_stat/obs/diag.py`,
  `src/mpas_stat/obs/plot.py`, or `src/mpas_stat/obs/common.py`.

## Known limitations (inherited from the shared graphic functions)

- Axis/legend labels show only the **statistic name**, not physical units —
  true on both the model and obs sides, since `plot_time_series_0d` /
  `plot_profiles_1d` don't carry a units argument.
- `relative_reduction_statistics` is disabled for the 0-D/1-D OMB/OMA
  products, which draw OMB and OMA as two absolute-valued lines/panels on
  the same figure. The 2-D comparison products (lat-lon map, lev-lat,
  lev-time) instead use `comparison_relative_reduction_statistics` and draw
  non-reference experiments as a bootstrap percent difference
  `100*(exp-ref)/ref` from the first experiment. Every experiment-vs-control
  figure (2-D comparison panels, `profile_diff_1d`,
  `forecast_lead_diff_0d`, `forecast_scorecard`) uses this sign:
  **negative = lower RMS/STD than the control = improvement**, drawn with
  an upward triangle (▲) and in blue; positive = degradation (▼, red).
- Obs-space absolute profiles (1D) and the lev-time Hovmoller are plain
  time averages / cycle-resolved values without confidence intervals. The
  experiment-vs-control profiles (`omb/` and `oma/profile_diff_1d/`), the
  lat-lon map and the lev-lat cross-section use bootstrap CI/significance,
  via the real-experiment comparison convention (§2.3).
- OMF (`hofx_forecast` experiments) now has its own forecast-lead
  counterpart of every OMB/OMA product — profile, lat-lon map, lev-lat
  cross-section, and lev-time Hovmoller — in addition to the domain-averaged
  `forecast_lead_0d` "skill vs. lead time" product (§2.4). Nothing is
  missing from obs-space's own OMB/OMA/OMF figure set any more; only the
  model-space-only products noted above (3-D binned fields, scorecards)
  remain unimplemented for obs-space.


