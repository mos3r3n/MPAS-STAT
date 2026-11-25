# 2.6 Radar composite-reflectivity verification (separate opt-in product)

Radar reflectivity is verified fundamentally differently from every other
sensor in this package: instead of per-observation binned statistics
(RMS/Bias/STD vs. pressure/height/latitude/lat-lon), scattered obs and HofX
reflectivity are first **gridded** onto a regular lat/lon "composite
reflectivity" field (max value within a search radius of each grid-cell
center), then verified with neighborhood/categorical scores — Fractions
Skill Score (FSS), Threat Score (TS), Equitable Threat Score (ETS),
Frequency Bias (FBIAS), Probability of Detection (POD), False Alarm Ratio
(FAR) — plus RMSE of the gridded field itself. This does not fit the
`get_levels()`/`bin_method` framework the rest of §2 uses at all, so it
gets its own pair of scripts (`src/mpas_stat/radar/diag.py` /
`src/mpas_stat/radar/plot.py`) and its own shared module
(`src/mpas_stat/radar/common.py`) rather than being folded into
`src/mpas_stat/obs/diag.py`/`src/mpas_stat/obs/plot.py`.

**The FSS/categorical scores are computed by MET's `grid_stat` (via
METplus), not by a native Python kernel.** Gridding (scattered obs/HofX →
regular lat/lon composite reflectivity) and RMSE are still computed
natively, exactly as before; see "METplus / `grid_stat` integration"
below for the full architecture, required external dependencies, and
important caveats — read it before turning this product on. (This
replacement applies ONLY to this integrated, config-driven product. The
original standalone `legacy/radar_reflectivity_verify.py` script this product
was lifted from is untouched and keeps its own independent, fully-native
FSS/categorical kernels — use it if you want radar verification with no
MET/METplus dependency at all.)

It is **OMF-only** — only `hofx_forecast`-typed experiments are processed
(same `<hofx_dir>/<exp>/HofX/<init_cycle>/f<HHH>/dbOut/obsout_hofx_<obs>.h5`
file convention as the rest of §2's OMF product, via
`build_hofx_forecast_filename()`) — there is no cycling/OMB-OMA
counterpart. It is entirely **opt-in**: both stages no-op immediately if
the config has no top-level `radar:` section (`resolve_radar_settings()`
returns `None`), so adding it never affects a config that doesn't use it.

```bash
# Statistics: for every radar.sensors entry and every hofx_forecast
# experiment, grids obs/hofx composite reflectivity and writes ONE
# radarstat_<obs>.<date>.nc file per (experiment, init cycle, lead)
python -m mpas_stat.cli.radar_stats my_radar_config.yaml

# Graphics: reads those files back, aggregates across init cycles, draws
# FSS/TS/ETS/FBIAS/RMSE vs. lead time and a Performance Diagram
python -m mpas_stat.cli.radar_graphics my_radar_config.yaml
```

Config (a new top-level `radar:` block; everything else — `paths`,
`experiments`, `time_settings`, `forecast`, `n_workers` — is the SAME
config section the rest of §2 already uses, so a radar sensor can share one
config file with the point-wise pipeline):

```yaml
radar:
  sensors: ["radar_zh"]              # obs-type name(s); resolved to an HDF5
                                      # filename via the SAME src/mpas_stat/obs/sensor_config.py
                                      # registry the point-wise pipeline uses
                                      # (e.g. the "radar_zh*" wildcard entry) --
                                      # but see the filename-collision note below
  diag_var: equivalentReflectivityFactor   # HDF5 ObsValue/hofx group variable
  thresholds: [20, 40]               # dBZ thresholds for FSS/TS/ETS/FBIAS/POD/FAR
  radii: [3, 5, 9, 15]               # neighborhood radii, in GRID CELLS of the
                                      # sample_bin-spaced composite-reflectivity grid
  sample_bin: 0.027                  # composite-reflectivity gridding radius, degrees
  n_workers: null                    # file-processing pool for the (expensive)
                                      # statistics stage; null -> falls back to the
                                      # top-level n_workers
  aggregated_score: false            # graphics-stage aggregation convention -- see below

  # --- METplus / grid_stat settings (see the subsection below) -----------
  nbrhd_shape: SQUARE                 # MET neighborhood shape (grid_stat default)
  nbrhd_cov_thresh: ">=0.5"           # MET neighborhood coverage threshold (grid_stat default)
  run_metplus_cmd: run_metplus.py     # assumed on PATH; override with a full path,
                                      # or a list like [conda, run, -n, metplus, run_metplus.py]
  metplus_work_dir: null              # scratch dir for conf/log/npz-cache/MET-output files;
                                      # null -> <diag_dir>/metplus_scratch
  metplus_timeout_sec: 600            # per-cycle run_metplus.py subprocess timeout
  keep_metplus_scratch: false         # keep each cycle's .npz cache after a successful run
                                      # (for debugging; conf/log/MET ASCII outputs are always kept)
  metplus_model_name: null            # MET MODEL field; null -> the experiment name
  metplus_obtype: null                # MET OBTYPE field; null -> radar.sensors entry, upper-cased
```

**No filename collision, even for the same sensor name.** A radar sensor
may be listed in BOTH the top-level `sensors:` list (point-wise pipeline;
e.g. `radar_zh*`'s existing HEIGHT-binned products) and `radar.sensors`
(this product) for the identical `diag_obs` name — the two products' files
never collide because this one uses a distinct `radarstat_` filename
prefix instead of `obsstat_`, in the same
`<diag_base_dir>/<exp>/VerifyOBS/<init_cycle>/f<HHH>/` directory (see
`radar_composite_diag_file()` in `src/mpas_stat/obs/common.py` — the path-convention
function lives there, alongside `obs_diag_forecast_file()`, for
consistency with every other path helper in the package, even though the
rest of this product's code lives in its own `src/mpas_stat/radar/common.py`).

**Both aggregation conventions are always available, from one statistics
run.** Every diagnosis file `src/mpas_stat/radar/diag.py` writes carries
BOTH the per-cycle derived scores (`FSS`/`TS`/`ETS`/`FBIAS`/`POD`/`FAR`)
AND the raw accumulator components they're built from
(`FSS_num`/`FSS_denom`/`NA`/`NB`/`NC`/`ND`) unconditionally. `radar.
aggregated_score` only controls how `src/mpas_stat/radar/plot.py`
*aggregates across init cycles* at the graphics stage:
  - `false` (default): bootstrap CI over each cycle's own derived score
    (matches every other bootstrap-CI product in this package). POD/FAR are
    plain cycle-means (never bootstrapped).
  - `true`: sum each cycle's raw counts/FSS numerator+denominator across
    cycles FIRST, then compute one score from the totals (no CI) — the
    conventional way to combine FSS/categorical scores over a verification
    period without giving equal weight to a cycle that happened to have
    very little reflectivity coverage.

Changing `aggregated_score` never requires re-running the statistics stage
— only `src/mpas_stat/radar/plot.py` reads it.

**Figures** (mirrors the standalone `legacy/radar_reflectivity_verify.py` script
this product was integrated from), written to
`<graphic_dir>/<sensor>/radar_reflectivity/`:

| Figure | Filename |
|---|---|
| FSS / TS / ETS / FBIAS vs. forecast lead time (one panel per threshold, one figure per radius) | `refl_fcst_<metric>_radius<r>.<fmt>` |
| Performance Diagram (POD vs. 1-FAR, one panel per threshold, one figure per radius, points labeled by lead) | `refl_fcst_performance_diagram_radius<r>.<fmt>` |
| RMSE vs. forecast lead time | `refl_fcst_rmse.<fmt>` |

**Known limitation:** unlike every other product in §2, these figures are
NOT drawn with `src/mpas_stat/graphics/functions.py` (`apply_pub_style`,
`auto_layout`, etc.) — `src/mpas_stat/radar/plot.py` carries its own small
`plot_time_series`/`plot_performance_diagram` functions (lifted unchanged
from the standalone script), so radar figures won't visually match the
publication styling of the rest of the package. Folding them onto the
shared styling helpers is a reasonable follow-on, not done here.

## METplus / `grid_stat` integration

Per-cycle FSS and categorical (TS/ETS/FBIAS/POD/FAR) neighborhood scores
are computed by MET's `grid_stat` tool, invoked through the METplus
`GridStat` wrapper (`run_metplus.py`) — **not** by a native Python kernel.
Composite-reflectivity gridding and RMSE are still fully native Python,
unchanged.

**Requires MET and METplus already installed** on the system that runs
`src/mpas_stat/radar/diag.py`, with `run_metplus.py` on `PATH` (or
`radar.run_metplus_cmd` pointed at it). This package does not install or
configure MET/METplus itself.

**How one cycle is processed**, for every (experiment, init cycle, lead):
  1. Obs/HofX composite reflectivity is gridded natively (unchanged), and
     RMSE is computed natively (unchanged — RMSE is NOT delegated to MET's
     `CNT` line type).
  2. The gridded obs/forecast fields are written to a small scratch
     `.npz` cache file under `radar.metplus_work_dir` (default
     `<diag_dir>/metplus_scratch/<exp>_<obs>_<init>_f<lead>/data.npz`).
  3. A per-cycle METplus `GridStat` conf is generated, scoped to exactly
     that one (init, lead) pair, requesting EVERY configured threshold
     and EVERY neighborhood width (`radii` converted to MET's own
     `width = 2*radius + 1` convention) in one `grid_stat` call. The
     forecast/obs inputs are MET **Python-embedding** script invocations
     (`src/mpas_stat/radar/metplus_embed.py <cache_path> FCST|OBS`) that just load the
     cache file back — this deliberately avoids depending on METplus's
     own filename-into-script-argument templating behavior.
  4. `run_metplus.py` is invoked on that conf as a subprocess, inside the
     same `ProcessPoolExecutor` that already parallelizes across cycles.
  5. `grid_stat` is configured to write BOTH the combined ASCII `.stat`
     file (kept per cycle for your own independent inspection/
     `stat_analysis` use — its path is recorded as the diagnosis file's
     `metplus_stat_file` global attribute) AND dedicated, self-describing
     per-line-type `NBRCTC`/`NBRCTS`/`NBRCNT` `.txt` files
     (`GRID_STAT_OUTPUT_FLAG_*=BOTH`). Only the dedicated `.txt` files are
     parsed — by column NAME, matched back to a (threshold, radius) by
     each row's own `FCST_THRESH`/`INTERP_PNTS` values — never by
     position in the combined `.stat` file (which is not self-describing
     past its common columns).
  6. `NBRCTC` counts (`FY_OY`/`FY_ON`/`FN_OY`/`FN_ON`) and `NBRCTS` scores
     (`CSI`/`GSS`/`FBIAS`/`PODY`/`FAR`) map directly onto this product's
     existing `na`/`nb`/`nc`/`nd` and `ts`/`ets`/`fbias`/`pod`/`far`.
     `NBRCNT`'s `FBS`/`FSS`/`TOTAL` are used directly for the per-cycle
     `FSS`, and algebraically back-derived into the SAME
     `fss_num`/`fss_denom` accumulator components this product has always
     summed across cycles for `aggregated_score: true` (see below).

**Cross-cycle aggregation is UNCHANGED** — `src/mpas_stat/radar/plot.py`'s
aggregation (bootstrap CI, or sum-then-ratio) still happens entirely in
this package's own Python code, exactly as before METplus was
introduced. This was a deliberate choice, not an oversight: MET's
`stat_analysis -job aggregate_stat` has a documented `NBRCTC`→`NBRCTS`
conversion for combining categorical counts across cases, but **no**
documented `aggregate_stat` conversion for `NBRCNT` (FSS) — it is only
listed under the separate, generic `-job aggregate` job, whose exact
combination semantics for `NBRCNT` could not be confirmed with
confidence against the MET User's Guide. Rather than depend on that, this
package keeps pooling FSS itself from `fss_num`/`fss_denom` (sum first,
then one ratio), just fed by MET's own per-cycle `FBS`/`FSS`/`TOTAL`
instead of a native kernel's.

**Caveats — please read before trusting results:**
  - **Not tested against a real MET/METplus install.** This integration
    was developed and unit-tested in a sandbox with no MET/METplus
    available — only against fabricated, MET-shaped stand-ins for its
    ASCII output (conf-text generation, ASCII-table parsing, the FBS/FSS
    back-derivation algebra, and `src/mpas_stat/radar/metplus_embed.py`'s `met_data`/
    `attrs` construction were all exercised directly; the actual
    `run_metplus.py`/`grid_stat` invocation was not). **Run a single
    cycle on your own system first** and check its
    `metplus_scratch/.../logs/run_metplus.log` and output files before
    trusting a full run.
  - `src/mpas_stat/radar/metplus_embed.py` relies on MET's Python-embedding support
    treating a non-finite (NaN) `met_data` value as missing/bad data —
    documented behavior, but not exercised against a real MET build here
    either; see that script's own docstring.
  - RMSE is still computed natively (not via MET's `CNT` line type) — by
    design, see above, not an oversight.
  - A cycle whose FSS is numerically indistinguishable from 1.0 (a
    perfect match for that one cycle) cannot be exactly back-divided into
    the `fss_num`/`fss_denom` this package pools across cycles (it's a
    0/0) and is treated as contributing 0 to both — a very slight
    understatement of that cycle's weight, expected to be rare for a
    continuous field like reflectivity. See `_parse_nbrcnt()`'s docstring
    in `src/mpas_stat/radar/common.py`.


